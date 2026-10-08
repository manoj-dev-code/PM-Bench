"""Rubric loading + the dense task-value score.

A rubric is a separate, versioned artifact under rubrics/, referenced by the
scenario ("rubric": "rubrics/demo.json"). Engines copy it into the run dir so
recorded runs stay self-contained and re-gradeable forever.

task_value = the scheduling objective itself:

    value(task) = w_priority x (progress + alpha*done) / (1 + alpha)

summed over AUTHORED tasks only (seed + external; agent-created tasks carry
zero weight — guard against add-fake-tasks reward hacking). Fully
deterministic: a pure function of derived task state at the horizon.
"""

import json
import os

from .sim_time import working_minutes_between

DEFAULT_PRIORITY_WEIGHTS = {"P0": 8, "P1": 4, "P2": 2, "P3": 1}


def load_rubric(scenario, run_dir=None):
    """Resolution order: run_dir/rubric.json -> scenario['rubric'] path ->
    inline scenario['evaluation'] (backward compat)."""
    if run_dir:
        p = os.path.join(run_dir, "rubric.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    ref = scenario.get("rubric")
    if ref and os.path.exists(ref):
        with open(ref) as f:
            return json.load(f)
    if "evaluation" in scenario:
        return scenario["evaluation"]
    raise SystemExit("no rubric found (rubrics/ file or inline 'evaluation')")


def task_value(task_rows, cfg, horizon, sim_start, workers=None,
               busy_hours=None):
    """TWO first-class metrics over a tasks_view snapshot:

      completion = Σ w · (progress + α·done)/(1+α)      — did the work get done
      efficiency = Σ w · done · (horizon − done_at)/span — how EARLY it got done

    plus `combined` = (completion + γ·efficiency)/(1+γ) as the single scalar
    for ranking/RL. The efficiency metric is what makes noise self-punishing:
    meetings and chat interruptions consume capacity → completions land
    later → efficiency drops. No behavioral penalties needed anywhere.

    FAIRNESS IS A RUBRIC TOO (pass `workers`): per-person utilization =
    hours worked / hours available; workload_fairness = 1 − σ(utilization),
    the population std across the team — variance sees EVERYONE's imbalance,
    not just the extreme pair. A pure outcome of assignment decisions —
    riding one person while others idle is visible in the schedule itself.
    """
    alpha = cfg.get("alpha", 0.5)
    gamma = cfg.get("gamma", 0.5)
    weights = cfg.get("priority_weights", DEFAULT_PRIORITY_WEIGHTS)
    span = float(max(1, horizon - sim_start))
    completion_total, efficiency_total, per = 0.0, 0.0, []
    for t in task_rows:
        if t.get("source") == "agent":
            continue  # zero weight: agents can't mint value
        if not t.get("effort_hours"):
            continue  # tracking-only items carry no schedulable value
        w = (weights.get(t.get("priority") or "P2", 2)
             * t.get("score_weight_factor", 1.0))
        progress = min(1.0, (t.get("true_done_hours") or 0.0) / t["effort_hours"])
        done = t["status"] == "done"
        completion = w * (progress + alpha * (1.0 if done else 0.0)) / (1 + alpha)
        earliness = 0.0
        if done and t.get("projected_done") is not None:
            earliness = max(0.0, horizon - t["projected_done"]) / span
        efficiency = w * earliness
        completion_total += completion
        efficiency_total += efficiency
        per.append({"id": t["id"], "priority": t.get("priority") or "P2",
                    "weight": w, "progress": round(progress, 3), "done": done,
                    "completion": round(completion, 3),
                    "efficiency": round(efficiency, 3)})
    combined = (completion_total + gamma * efficiency_total) / (1 + gamma)
    w_total = sum(p["weight"] for p in per)
    w_done = sum(p["weight"] for p in per if p["done"])

    utilization, utilization_std, workload_fairness = None, None, None
    if workers:
        avail_h = working_minutes_between(sim_start, horizon) / 60.0
        if busy_hours is not None:
            # CALENDAR mode (live/replayed worlds): the caller supplies each
            # person's true calendar occupancy — work segments + meetings +
            # interruption taxes, overlap-merged (world.calendar_load). Never
            # mixed with effort-hours: a specialist's 12 effort-hours at 1.3x
            # occupy 9 1/4 calendar-hours, and a swarm session counts once.
            worked = {w: busy_hours.get(w, 0.0) for w in workers}
        else:
            # EFFORT approximation (config-only paths like OPT, where no
            # schedule segments exist): task effort-hours per current owner
            worked = dict.fromkeys(workers, 0.0)
            for t in task_rows:
                if t.get("source") == "agent" or not t.get("effort_hours"):
                    continue
                a = (t.get("assignees") or [None])[0]
                if a in worked:
                    done = min(t.get("true_done_hours") or 0.0, t["effort_hours"])
                    worked[a] += max(0.0, done - (t.get("seed_done_hours") or 0.0))
        if avail_h > 0:
            utilization = {w: round(h / avail_h, 3) for w, h in worked.items()}
            vals = list(utilization.values())
            mean = sum(vals) / len(vals)
            utilization_std = round(
                (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5, 3)
            workload_fairness = round(1.0 - utilization_std, 3)
    return {
        "completion": round(completion_total, 3),
        "efficiency": round(efficiency_total, 3),
        "combined": round(combined, 3),
        # W(tasks completed) / W(all authored tasks) — the crisp headline:
        # what weighted fraction of the week's work actually SHIPPED
        "done_weight_rate": round(w_done / w_total, 3) if w_total else None,
        "utilization": utilization,
        "utilization_std": utilization_std,
        "workload_fairness": workload_fairness,
        "per_task": per,
    }
