"""Task-intent audit: WHICH decisions carry the winnable band?

    python -m sim.intent scenarios/demo.json

A task is authored with an INTENT — the decisions a winning PM must make
(act the moment information lands, allocate better than the org volunteer,
keep noise off people's focus). The trajectory space is huge, but a
well-authored task collapses it onto the [baseline, OPT] band where a
trajectory's position equals which intended mechanisms it captured.

This audit VERIFIES the intent causally, with zero LLM calls: it plays a
family of deterministic policies that differ in exactly one decision and
measures each one's score. share(mechanism) = score(oracle) − score(without
that mechanism): the slice of the band that decision owns.

Reading the report:
  - an intended mechanism with share ~0 is NOT taught by this task
    (physics lets you skip it for free — re-author)
  - one mechanism owning nearly everything = a one-bit task: sparse
    learning signal, saturates after a single insight
  - several comparable shares = dense signal across skill levels

Policies (all friction-respecting; acting only on knowable information).
Under the owned-arrivals contract every arrival already has a default
holder; the PM's lever is REARRANGING, so the ablations are about when and
whether to rearrange:
  oracle   seed moves Monday; REASSIGN each arrival to its OPT owner at the
           earliest instant it is knowable (chat announce / email batch tick)
  late     the same reassignments, one day later — isolates TIMING
  noisy    oracle + 3 chat pings to each touched assignee per decision —
           isolates CHANNEL DISCIPLINE (the serialized focus tax)
Shares: timing = oracle − late; channel = oracle − noisy; allocation =
late's own score (the value owner-correction retains even when slow —
do-nothing is 0 by construction, so late's residual is pure allocation).
"""

import json
import shutil
import sys

from .engine import Engine
from .eval import StubClient, evaluate
from .optimal import opt_ideal
from .tools import call_tool


def _knowable_at(arr, scenario):
    """Earliest instant the PM can know this arrival exists."""
    if arr.get("via", "chat") == "email":
        batch = scenario.get("email_batch_minutes", 30)
        return (arr["at"] // batch + 1) * batch + 1
    return arr["at"] + 1


def _gating_questions(scenario, choices=None):
    """(at, task_id, question_id, choice) for every blocking question."""
    choices = choices or {}
    out = []
    for t in scenario["project"]["tasks"]:
        for q in t.get("questions") or []:
            if q.get("gates"):
                choice = choices.get(q["id"],
                                     (q.get("options") or ["proceed"])[0])
                out.append((q["at"], t["id"], q["id"], choice))
    return out


def _run_policy(scenario, name, opt, timing, noisy=False, answer_qs=True):
    """One deterministic trajectory. timing: 'early'|'late'. answer_qs: whether
    the policy answers blocking questions (chat to the holder the moment each
    opens) — a lever an oracle MUST pull, since an unanswered gate stalls the
    critical path exactly as the do-nothing baseline does."""
    tag = (scenario.get("project") or {}).get("id", "x")
    run_dir = "runs/intent-%s-%s" % (tag, name)
    # a fresh dir ALWAYS: reusing one appends to the old event log and the
    # evaluation replays a franken-history (this once made oracle "beat" OPT)
    shutil.rmtree(run_dir, ignore_errors=True)
    eng = Engine(scenario, client=StubClient(), verbose=False,
                 run_dir=run_dir)
    w = eng.world

    def goto(t):
        while w.clock < t:
            call_tool(eng, "advance_time", {"minutes": t - w.clock})

    def ping(npc):
        for i in range(3):
            call_tool(eng, "send_chat",
                      {"npc": npc, "text": "quick check-in %d — how is it "
                                           "going?" % i})

    # seed reassignments OPT wants: possible Monday, no information needed
    for t in scenario["project"]["tasks"]:
        want = opt["assignment"].get(t["id"])
        have = (t.get("assignees") or [None])[0]
        if want and have and want != have:
            call_tool(eng, "assign_task", {"task_id": t["id"], "npc": want})
            if noisy:
                ping(want)

    horizon_guess = load_horizon(scenario)
    # one timeline mixing arrival-reassignments and blocking-question answers,
    # executed in time order
    plan = []
    for arr in scenario.get("task_arrivals", []):
        when = _knowable_at(arr, scenario)
        if timing == "late":
            when = min(when + 1440, horizon_guess - 120)  # a day of dithering
        who = opt["assignment"].get(arr["task"]["id"])
        default = (arr["task"].get("assignees") or [None])[0]
        if who and who != default:   # same-owner reassign is pure loss
            plan.append((when, "assign", arr["task"]["id"], who))
    if answer_qs:
        for at, task_id, question_id, choice in _gating_questions(
                scenario, opt.get("choices") or {}):
            when = at + 1 if timing != "late" else min(at + 1440, horizon_guess - 120)
            plan.append((when, "answer", task_id, question_id, choice))
    for when, kind, a, b, *rest in sorted(plan, key=lambda x: (x[0], x[1])):
        goto(when)
        if kind == "assign":
            call_tool(eng, "assign_task", {"task_id": a, "npc": b})
            if noisy:
                ping(b)
        else:
            call_tool(eng, "answer_question",
                      {"task_id": a, "question_id": b, "choice": rest[0],
                       "rationale": "Fastest safe path for the committed outcome",
                       "via": "chat"})

    goto(load_horizon(scenario))
    return evaluate(run_dir)


def load_horizon(scenario):
    try:
        from .rubric import load_rubric
        return load_rubric(scenario)["horizon"]
    except SystemExit:
        return scenario.get("start_time", 545) + 7 * 1440


def audit(scenario):
    opt = opt_ideal(scenario)
    runs = {
        "oracle": _run_policy(scenario, "oracle", opt, "early"),
    }
    if runs["oracle"]["score"] is None:
        return {"degenerate": True, "scores": {}, "mechanism_shares": {},
                "note": "band is empty — OPT == baseline; nothing for any "
                        "policy to win. Re-author or regenerate (sim.generate "
                        "gates this out automatically)."}
    runs.update({
        "late":   _run_policy(scenario, "late", opt, "late"),
        "noisy":  _run_policy(scenario, "noisy", opt, "early", noisy=True),
        # blind = oracle allocation/timing but NEVER answers the blockers:
        # isolates how much of the band the gating decisions own
        "blind":  _run_policy(scenario, "blind", opt, "early", answer_qs=False),
    })
    top = runs["oracle"]["score"]
    mechanisms = {
        "answering blockers (unblock the gate)": top - runs["blind"]["score"],
        "timing (rearrange early, not late)":    top - runs["late"]["score"],
        "allocation (owner + skill routing)":    runs["late"]["score"],
        "channel discipline (focus tax)":        top - runs["noisy"]["score"],
    }
    return {"scores": {k: v["score"] for k, v in runs.items()},
            "odds": {k: v.get("score_odds") for k, v in runs.items()},
            "oracle_score": top,
            # 1 - oracle = the relaxation gap: band no feasible policy reaches
            "relaxation_gap": round(1 - top, 3),
            "mechanism_shares": {k: round(v, 3)
                                 for k, v in mechanisms.items()}}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/demo.json"
    with open(path) as f:
        scenario = json.load(f)
    r = audit(scenario)
    print("intent audit: %s" % path)
    if r.get("degenerate"):
        print("  DEGENERATE: %s" % r["note"])
        return
    print("  policy scores:  %s" % "  ".join(
        "%s=%.3f" % (k, v) for k, v in r["scores"].items()))
    print("  relaxation gap: %.1f%% of the band is unreachable by any policy"
          % (r["relaxation_gap"] * 100))
    print("  mechanism shares of the band (oracle minus ablation):")
    for k, v in sorted(r["mechanism_shares"].items(), key=lambda x: -x[1]):
        flag = "   <- NOT TAUGHT by this task" if v < 0.02 else ""
        print("    %-34s %6.3f%s" % (k, v, flag))


if __name__ == "__main__":
    main()
