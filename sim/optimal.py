"""OPT_ideal: the deterministic ceiling on task_value for a scenario.

    python -m sim.optimal scenarios/demo.json

The IDEALIZED relaxation — every friction at its best value:
  - communication free & instantaneous (full information at t=0)
  - skill multiplier 1x for everyone (any person can do any task)
  - no meetings / interruption overhead; authored OOO remains a hard
    availability constraint
  - decisions cost nothing

What remains are the HARD constraints: arrivals (a task can't start before
it lands), precedence, effort hours, one-task-at-a-time per person, working
calendars, and the horizon.

OPT_ideal is a pure function of the scenario file alone — no NPCs, no LLM,
no agent, no engine version. It is the SOLO ceiling: the best week where
every task has exactly one owner. Working sessions (bounded swarms) are
deliberately NOT modeled here — a well-run session can legitimately beat
this anchor, so normalized scores > 1 mean exactly that: collaboration
beyond the solo optimum (bounded by the swarm cap and the daily meeting
caps). Within the solo policy space it remains a true upper bound on any
score, computable at scenario-generation time, and a stable denominator for
normalized comparison ("captured X% of ideal") across environment versions.

Computation: exact search over worker assignments (workers^n for the
schedulable tasks) x candidate orderings, each evaluated with the SAME
compute_schedule the world runs (frictions zeroed). Milliseconds at our
scale; scales as an admissible bound via the weight-descending heuristic if
instances ever grow past exhaustive range.
"""

import itertools
import json
import sys

from .rubric import DEFAULT_PRIORITY_WEIGHTS, load_rubric, task_value
from .tasks import add_work_minutes, physics_of, task_view
from .work import materialize_work_items


def _all_tasks(scenario):
    start = scenario.get("start_time", 545)
    tasks = [dict(t, arrival=start, source="seed")
             for t in (scenario.get("project") or {}).get("tasks", [])]
    for arr in scenario.get("task_arrivals", []):
        tasks.append(dict(arr["task"], arrival=arr["at"], source="external"))
    return materialize_work_items(tasks)


def worker_ids(scenario):
    """The labor pool OPT may assign to. Stakeholders you REPORT to (a VP, an
    exec) carry `worker: false` and are EXCLUDED — treating them as fungible
    labor inflates the ceiling with work no intended policy should do, and
    deflates every normalized score against it."""
    return [n["id"] for n in scenario["npcs"] if n.get("worker", True)]


def _skills_of(scenario):
    return {n["id"]: n["skills"] for n in scenario["npcs"] if n.get("skills")}


def _hard_busy(scenario):
    """Hard authored availability only — OOO, never cancellable meetings."""
    return {n["id"]: [tuple(x) for x in n.get("ooo", [])]
            for n in scenario.get("npcs", []) if n.get("worker", True)}


def _choice_plans(tasks, base_busy):
    """Cartesian product of gating-question options -> (choices, effects, busy)."""
    questions = [(t["id"], q) for t in tasks for q in t.get("questions") or []
                 if q.get("gates")]
    plans = []
    if not questions:
        return [({}, {}, {k: list(v) for k, v in base_busy.items()})]
    option_lists = [[(q["id"], choice) for choice in q.get("options") or [None]]
                    for _, q in questions]
    for picks in itertools.product(*option_lists):
        choices = {}
        effects = {}
        busy = {k: list(v) for k, v in base_busy.items()}
        costs = []
        for (task_id, q), (question_id, choice) in zip(questions, picks):
            choices[question_id] = choice
            effect = (q.get("choice_effects") or {}).get(choice) or {}
            if not effect:
                continue
            row = {"at": q["at"], "question_id": question_id,
                   "choice": choice, "summary": effect.get("summary"),
                   "task_effort_delta_hours":
                       effect.get("task_effort_delta_hours", 0),
                   "capacity_costs": list(effect.get("capacity_costs") or [])}
            effects.setdefault(task_id, []).append(row)
            for cost in row["capacity_costs"]:
                costs.append((q["at"], len(costs), cost["npc"],
                              cost["minutes"]))
        for at, _, person, minutes in sorted(costs):
            intervals = busy.setdefault(person, [])
            start_at = add_work_minutes(at, 0, intervals)
            intervals.append(
                (start_at, add_work_minutes(start_at, minutes, intervals)))
        plans.append((choices, effects, busy))
    return plans


def _greedy_assignments(free, workers, skills, weights):
    """A shortlist of near-optimal solo assignments for instances too big to
    search exhaustively — used only as the reference anchor, not a proof of
    optimality. (1) best-skill per task; (2) a load-balanced pass: heaviest
    tasks first, each to its top-skilled worker, breaking ties by who's least
    loaded so no one specialist becomes everyone's bottleneck."""
    from .tasks import _person_skill_on
    best_skill = {t["id"]: max(workers, key=lambda w: _person_skill_on(t, w, skills))
                  for t in free}
    load = {w: 0.0 for w in workers}
    balanced = {}
    for t in sorted(free, key=lambda t: -(
            weights.get(t.get("priority") or "P2", 2)
            * t.get("score_weight_factor", 1.0)
            * (t["effort_hours"] - t.get("done_hours", 0)))):
        pick = min(workers, key=lambda w: (-_person_skill_on(t, w, skills), load[w]))
        balanced[t["id"]] = pick
        load[pick] += t["effort_hours"] - t.get("done_hours", 0)
    # de-dup (they often coincide when there's no contention)
    out, seen = [], set()
    for c in (best_skill, balanced):
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def opt_ideal(scenario, rubric=None, max_exhaustive=200000):
    rubric = rubric or load_rubric(scenario)
    tv_cfg = rubric.get("task_value", {})
    weights = tv_cfg.get("priority_weights", DEFAULT_PRIORITY_WEIGHTS)
    horizon = rubric["horizon"]
    start = scenario.get("start_time", 545)
    workers = worker_ids(scenario)
    skills = _skills_of(scenario)

    tasks = _all_tasks(scenario)
    fixed = [t for t in tasks if not t.get("effort_hours")]
    free = [t for t in tasks if t.get("effort_hours")]

    # BLOCKING QUESTIONS & the ceiling. A frictionless PM answers the instant a
    # question opens (full info, zero decision cost), so every gating question
    # resolves at its own open-time -> an empty [at, at) block -> OPT is
    # unaffected. A real agent can only answer LATER (woken at `at`, reply costs
    # ≥1 action-min + channel latency), so its block strictly contains OPT's
    # empty one and the questioned task finishes no earlier than here: agent ≤ OPT.
    instant_answers = {t["id"]: {q["id"]: q["at"] for q in t.get("questions") or []
                                 if q.get("gates")}
                       for t in tasks if t.get("questions")}
    choice_plans = _choice_plans(tasks, _hard_busy(scenario))

    # OPT is the frictionless INDIVIDUAL-optimal reference: one best-skilled
    # owner per task, no friction, questions answered instantly. It is NOT a
    # provably-unbeatable ceiling, and deliberately so — COORDINATION mechanics
    # the agent has but an individual schedule doesn't (chiefly a swarm: pooling
    # idle teammates onto a bottleneck via meetings) can push a real run ABOVE
    # this line. That shows up as normalized > 1 and is a FEATURE: it measures
    # coordination value earned over the no-coordination optimum. Chasing a
    # tight, swarm-aware, provably-valid bound isn't worth it because the reward
    # is AFFINE-invariant to [baseline, OPT] — for a fixed scenario B and OPT are
    # shared constants across all K rollouts, so (x−B)/(OPT−B) leaves the
    # group-relative GRPO advantage exactly unchanged. OPT earns its keep as a
    # cheap, stable, interpretable anchor and a difficulty gate, not as a
    # correctness-critical ceiling. (Swarm is still bounded in the live world —
    # the per-day meeting cap + focus-tax + one-owner-at-a-time keep any real
    # overshoot small and finite.)
    def evaluate(assignment, ordering, decision_effects, busy):
        # frictionless apart from authored hard busy/decision consequences, with
        # REAL skills — the search routes each task to its fastest eligible
        # worker; the agent's play is one point in this same constrained space.
        # `assignment` maps task-id -> worker.
        candidate = [dict(t, assignees=[assignment[t["id"]]]) for t in ordering]
        rows = task_view(candidate + fixed, start, horizon,
                         busy_by_assignee=busy, skills=skills,
                         answers=instant_answers,
                         physics=physics_of(scenario),
                         decision_effects=decision_effects)
        return task_value(rows, tv_cfg, horizon, start, workers=workers)

    def best_combined(assignment, ordering):
        return max((evaluate(assignment, ordering, effects, busy)
                    for _, effects, busy in choice_plans),
                   key=lambda v: v["combined"])

    # three ceilings tracked in ONE search pass: each metric gets its own
    # maximum (the completion-optimal and efficiency-optimal assignments
    # may differ)

    # candidate orderings: authored order + weight-descending (urgent-first
    # engine semantics make ordering matter for contended workers)
    orderings = [list(free)]
    by_weight = sorted(
        free, key=lambda t: -(weights.get(t.get("priority") or "P2", 2)
                              * t.get("score_weight_factor", 1.0)))
    if [t["id"] for t in by_weight] != [t["id"] for t in free]:
        orderings.append(by_weight)

    # candidate assignments (task-id -> worker). Small instances: EXHAUSTIVE
    # (every worker^task combo — the exact solo optimum). Large instances:
    # a GREEDY shortlist (best-skill per task + a load-balanced pass). Since OPT
    # is a cheap REFERENCE anchor (the reward is affine-invariant to it), a
    # near-optimal greedy anchor on big weeks is fine — a slightly loose anchor
    # doesn't change any GRPO advantage, it just shifts the 0..1 label.
    n_combos = len(workers) ** len(free) * len(orderings) * len(choice_plans)
    if n_combos <= max_exhaustive:
        candidates = [{t["id"]: w for t, w in zip(free, combo)}
                      for combo in itertools.product(workers, repeat=len(free))]
        searched = n_combos
    else:
        # greedy seeds, then HILL-CLIMB (single-owner reassignment moves) to a
        # local optimum — the raw 2-candidate greedy underestimated the true
        # optimum by ~9%, tight enough to matter for the difficulty gate and for
        # reading normalized>1. Local search recovers most of that gap cheaply.
        ord0 = orderings[-1]
        ids = [t["id"] for t in free]
        candidates = list(_greedy_assignments(free, workers, skills, weights))
        n_eval = 0
        for seed in list(candidates):
            cur = dict(seed)
            curv = best_combined(cur, ord0)["combined"]
            n_eval += 1
            for _ in range(8):   # capped sweeps; converges well before this
                improved = False
                for tid in ids:
                    for w in workers:
                        if w == cur[tid]:
                            continue
                        v = best_combined(dict(cur, **{tid: w}), ord0)["combined"]
                        n_eval += 1
                        if v > curv + 1e-9:
                            cur[tid] = w
                            curv = v
                            improved = True
                if not improved:
                    break
            candidates.append(cur)
        searched = -n_eval   # negative flags the greedy+local-search path

    best = {"completion": (-1.0, None), "efficiency": (-1.0, None),
            "done_weight_rate": (-1.0, None), "combined": (-1.0, None, None, None)}
    for ordering in orderings:
        for named in candidates:
            for choices, effects, busy in choice_plans:
                v = evaluate(named, ordering, effects, busy)
                for metric in ("completion", "efficiency", "done_weight_rate"):
                    if (v[metric] or 0) > best[metric][0]:
                        best[metric] = (v[metric], named)
                if v["combined"] > best["combined"][0]:
                    best["combined"] = (v["combined"], named, v, choices)
    v_best = best["combined"][2]
    n_combos = searched
    return {
        "opt_completion": best["completion"][0],
        "opt_efficiency": best["efficiency"][0],
        "opt_done_weight_rate": best["done_weight_rate"][0],
        "opt_combined": best["combined"][0],
        "assignment": best["combined"][1],
        "choices": best["combined"][3],
        "per_task": v_best["per_task"],
        # fairness OF the combined-optimal schedule — the reference point:
        # how balanced the week is under perfect play (chains can force
        # imbalance even on OPT, so agents are compared to this, not to 1.0)
        "opt_workload_fairness": v_best["workload_fairness"],
        "opt_utilization": v_best["utilization"],
        "horizon": horizon,
        "combos_searched": n_combos,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/demo.json"
    with open(path) as f:
        scenario = json.load(f)
    result = opt_ideal(scenario)
    print("OPT completion=%.3f  efficiency=%.3f  combined=%.3f  (searched %d combos)"
          % (result["opt_completion"], result["opt_efficiency"],
             result["opt_combined"], result["combos_searched"]))
    print("optimal assignment:", json.dumps(result["assignment"], indent=2))
    if result.get("choices"):
        print("optimal choices:", json.dumps(result["choices"], indent=2))
    for t in result["per_task"]:
        print("  %-24s %s w=%.2f progress=%.0f%% done=%s completion=%.2f efficiency=%.2f"
              % (t["id"], t["priority"], t["weight"], t["progress"] * 100,
                 t["done"], t["completion"], t["efficiency"]))


if __name__ == "__main__":
    main()
