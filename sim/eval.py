"""Evaluation: two outcome metrics, nothing else.

    python -m sim.eval runs/<run-dir>

    COMPLETION = Σ w·(progress + α·done)/(1+α)        — did the work get done
    EFFICIENCY = Σ w·done·(horizon − done_at)/span     — how early it got done

Each is reported as a ladder:  null baseline  ≤  agent  ≤  OPT_ideal

- The BASELINE (same scenario, agent does nothing; stub LLM — NPC text never
  mutates world state) is the floor: outcomes that happen anyway earn zero,
  regressions score negative.
- OPT_ideal (sim/optimal.py) is the frictionless ceiling, a pure function of
  the scenario file: headline = % of available value captured.
- COMBINED = (completion + γ·efficiency)/(1+γ) is the single scalar for
  ranking / RL.

There are NO checks, NO penalties, NO keyword matching, NO LLM judging.
Communication, discovery, and record hygiene earn nothing directly — they
matter only through their consequences on the work (allocation quality,
capacity, timing). Noise punishes itself through physics: interruptions and
meetings consume capacity, completions land later, efficiency drops.

Every number is deterministic and reproducible from the run dir forever.
"""

import glob
import copy
import hashlib
import json
import os
import sys

from .optimal import opt_ideal, worker_ids
from .replay import load_run, replay
from .rubric import load_rubric, task_value
from .sim_time import fmt
from .temporal import temporal_metrics


def scoring_version():
    """A short hash of the SCORING-relevant source (the schedule + the metrics
    + OPT/baseline). Stamped into every scorecard so a cached score is
    self-identifying: if this differs from the current code's, the scorecard
    was written by different physics and must NOT be trusted — re-score from
    the immutable events.jsonl instead. (This is the guard against the stale-
    scorecard scare; the aggregator `sim.bench` always re-scores regardless.)"""
    h = hashlib.sha256()
    here = os.path.dirname(__file__)
    for name in ("tasks.py", "rubric.py", "optimal.py", "eval.py",
                 "world.py", "replay.py", "engine.py", "npc.py", "sim_time.py",
                 "signals.py", "validate.py", "temporal.py", "work.py"):
        try:
            with open(os.path.join(here, name), "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:12]


class _StubUsage:
    input_tokens = 0
    output_tokens = 0


class _StubResp:
    content = [type("B", (), {"type": "text", "text": "(baseline stub)"})()]
    usage = _StubUsage()
    stop_reason = "end_turn"


class StubClient:
    """LLM stand-in for the null-agent baseline: text is irrelevant to
    world state, so the baseline never needs a real model call."""
    class messages:
        @staticmethod
        def create(**kw):
            return _StubResp()


def build_baseline(scenario, rubric, run_dir):
    """Deterministic null-agent run: advance to horizon, take no actions.
    Built in a UNIQUE temp dir, discarded afterwards: evaluations can run
    concurrently (the hub's server threads poll while the harness scores)
    and a shared runs/<id>/baseline path let them append to — or delete —
    each other's event logs mid-write. The baseline world we need lives in
    memory; the disk artifact was only ever a byproduct."""
    import shutil
    import tempfile
    from .engine import Engine
    tmp = tempfile.mkdtemp(prefix="pmsim-baseline-")
    eng = None
    try:
        # Regrading a copied run must not depend on today's working directory
        # or on a mutable rubric at the original source path.
        pinned = copy.deepcopy(scenario)
        pinned.pop("rubric", None)
        pinned["evaluation"] = rubric
        eng = Engine(pinned, client=StubClient(), verbose=False, run_dir=tmp)
        eng.advance_until(rubric["horizon"], max_events=2000)
        return eng.world
    finally:
        if eng is not None:
            eng.world.store.close()
        shutil.rmtree(tmp, ignore_errors=True)


def evaluate(run_dir):
    scenario, events = load_run(run_dir)
    rubric = load_rubric(scenario, run_dir)
    horizon = rubric["horizon"]
    tv_cfg = rubric.get("task_value", {})
    start = scenario.get("start_time", 545)

    # Later assignments/cancellations must not rewrite the world being graded.
    agent_world = replay(scenario, [e for e in events if e["t"] <= horizon])
    baseline_world = build_baseline(scenario, rubric, run_dir)

    workers = worker_ids(scenario)
    v_agent = task_value(agent_world.tasks_view(at=horizon), tv_cfg, horizon,
                         start, workers=workers,
                         busy_hours=agent_world.calendar_load(horizon))
    v_base = task_value(baseline_world.tasks_view(at=horizon), tv_cfg, horizon,
                        start, workers=workers,
                        busy_hours=baseline_world.calendar_load(horizon))
    opt = opt_ideal(scenario, rubric)

    def ladder(metric, opt_key):
        a, b, o = v_agent[metric], v_base[metric], opt[opt_key]
        delta = round(a - b, 3)
        available = round(o - b, 3)
        regret = round(available - delta, 3)
        return {
            # normalized is THE reading: 0 = null baseline, 1 = the frictionless
            # INDIVIDUAL-optimal reference, negative = worse than not existing.
            # NOT clamped: >1 is legitimate and desirable — it means the agent
            # out-COORDINATED the individual optimum (e.g. swarming idle people
            # onto a bottleneck), value an individual schedule can't reach. The
            # reward is affine-invariant to [baseline, OPT] for GRPO, so OPT need
            # not be a tight or unbeatable ceiling — it's a cheap stable anchor.
            "normalized": (round(delta / available, 3)
                           if available > 0.001 else None),
            # gain_over_regret = delta/(opt − agent) = N/(1−N), the odds form;
            # re-expands top-end differences near/above the reference.
            "gain_over_regret": (round(delta / regret, 3)
                                 if available > 0.001 and regret > 0.001
                                 else None),
            "baseline": b, "agent": a, "opt_ideal": o,
            "delta": delta, "available": available,
            "regret": regret if available > 0.001 else None,
        }

    # THE RL reward: normalized combined delta over baseline (0 = do-nothing,
    # 1 = OPT). `score` stays None for a degenerate scenario (nothing winnable)
    # so tooling can flag it, but `reward` is ALWAYS numeric — a training loop
    # can never be handed a None. Degenerate instances should be rejected by
    # the difficulty gate before training; the raw delta is the safe fallback.
    winnable = opt["opt_combined"] - v_base["combined"]
    score = (round((v_agent["combined"] - v_base["combined"]) / winnable, 3)
             if winnable > 0.001 else None)
    reward = score if score is not None else round(v_agent["combined"]
                                                   - v_base["combined"], 3)

    return {
        "run_dir": run_dir,
        "scored_with": scoring_version(),   # stale-cache guard (see above)
        "horizon": horizon,
        "horizon_fmt": fmt(horizon),
        "observed_through": max([start] + [e["t"] for e in events]),
        "complete": any(e["t"] >= horizon for e in events),
        "temporal": temporal_metrics(agent_world, horizon),
        "degenerate": score is None,
        "reward": reward,
        "completion": ladder("completion", "opt_completion"),
        "efficiency": ladder("efficiency", "opt_efficiency"),
        "done_weight_rate": ladder("done_weight_rate", "opt_done_weight_rate"),
        "combined": ladder("combined", "opt_combined"),
        # fairness is a rubric too: 1 − σ(hours worked / hours available)
        # across the team. Raw triplet, not normalized — OPT itself can be
        # forced into imbalance (dependency chains), so OPT is the reference,
        # not 1.0.
        "workload_fairness": {"baseline": v_base["workload_fairness"],
                              "agent": v_agent["workload_fairness"],
                              "opt_ideal": opt["opt_workload_fairness"]},
        "utilization": v_agent["utilization"],
        "per_task_agent": v_agent["per_task"],
        "opt_assignment": opt["assignment"],
        "opt_choices": opt.get("choices", {}),
        # the headline scalar, in [<=1], anchored 0=baseline 1=OPT
        "score": score,
        # odds-form headline: gain per unit of remaining regret (monotone in
        # `score`: odds = N/(1−N)) — resolves top-end compression from the
        # unreachable frictionless ceiling
        "score_odds": (round((v_agent["combined"] - v_base["combined"])
                             / (opt["opt_combined"] - v_agent["combined"]), 3)
                       if winnable > 0.001
                       and opt["opt_combined"] - v_agent["combined"] > 0.001
                       else None),
        "score_raw_delta": round(v_agent["combined"] - v_base["combined"], 3),
    }


def print_scorecard(result):
    print("=== scorecard: %s (graded at %s) ===\n"
          % (result["run_dir"], result["horizon_fmt"]))

    def line(name, m):
        if m["normalized"] is None:
            print("%-11s (nothing available beyond baseline)" % name)
            return
        print("%-11s %6.3f   [0 = do-nothing baseline, 1 = solo optimum; "
              ">1 = collaboration beat it]"
              "   (raw: %.2f -> %.2f -> %.2f)"
              % (name, m["normalized"], m["baseline"], m["agent"], m["opt_ideal"]))

    line("COMPLETION", result["completion"])
    line("EFFICIENCY", result["efficiency"])
    line("COMBINED", result["combined"])
    d = result["done_weight_rate"]
    print("DONE-RATE   W(shipped)/W(all): baseline %.2f -> agent %.2f -> OPT %.2f"
          % (d["baseline"], d["agent"], d["opt_ideal"]))
    f = result["workload_fairness"]
    if f["agent"] is not None:
        print("FAIRNESS    1-sigma(util):     baseline %.2f -> agent %.2f -> OPT %.2f"
              "   util: %s"
              % (f["baseline"], f["agent"], f["opt_ideal"],
                 ", ".join("%s %.0f%%" % (w, u * 100)
                           for w, u in sorted(result["utilization"].items()))))

    print("\nper task (agent run):")
    for t in result["per_task_agent"]:
        print("  %-24s %s w=%.2f %3.0f%%%s  compl=%.2f eff=%.2f"
              % (t["id"], t["priority"], t["weight"], t["progress"] * 100,
                 " done" if t["done"] else "     ", t["completion"], t["efficiency"]))
    if result["score"] is not None:
        odds = ("  |  ODDS %.2f (gain per remaining regret)" % result["score_odds"]
                if result.get("score_odds") is not None else "")
        print("\nSCORE: %.3f  (0 = do-nothing, 1 = optimal; <0 = made things worse)%s"
              % (result["score"], odds))
    else:
        print("\nSCORE: n/a (degenerate scenario: nothing winnable beyond baseline)")
    timing = result["temporal"]
    mae = ("%.1f work-min" % timing["p50_final_mae_work_minutes"]
           if timing["p50_final_mae_work_minutes"] is not None else "n/a")
    p90 = ("%.1f%%" % (timing["p90_coverage"] * 100)
           if timing["p90_coverage"] is not None else "n/a")
    width = ("%.1f work-min" % timing["mean_interval_width_work_minutes"]
             if timing["mean_interval_width_work_minutes"] is not None else "n/a")
    accuracy = ("%.1f%%" % (timing["horizon_accuracy"] * 100)
                if timing["horizon_accuracy"] is not None else "n/a")
    revision = ("%.1f%%" % (timing["belief_revision_coverage"] * 100)
                if timing["belief_revision_coverage"] is not None else "n/a")
    print("TIME: coverage %d/%d | forecasts %d (%d revisions) | p50 final MAE %s | "
          "p90 coverage %s | mean interval %s | belief-update coverage %s | "
          "by-horizon accuracy %s | censored %d"
          % (timing["forecasted_tasks"], timing["eligible_tasks"],
             timing["forecast_count"], timing["revision_count"], mae, p90,
             width, revision, accuracy, timing["censored_forecasts"]))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    result = evaluate(sys.argv[1])
    print_scorecard(result)
    out = sys.argv[1].rstrip("/") + "/scorecard.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
