"""Aggregate completed runs without mixing Max@k across different tasks.

    python -m sim.bench [runs]

Each task/model cell reports its own rollout distribution. The final macro
view averages task-level means; it never takes a maximum across tasks.
"""

import glob
import json
import os
import sys

from .eval import evaluate
from .identity import (legacy_run_identity, resolved_model_id,
                       with_resolved_model)

METRICS = ["completion", "efficiency", "done_weight_rate", "combined"]


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return {"n": len(values), "mean": mean, "max": max(values),
            "min": min(values), "std": variance ** 0.5}


def _load(runs_root):
    cells, skipped = {}, []
    for meta_path in glob.glob(os.path.join(runs_root, "*", "meta.json")):
        run_dir = os.path.dirname(meta_path)
        if not os.path.exists(os.path.join(run_dir, "events.jsonl")):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("rollout_status", "completed") != "completed":
                skipped.append((run_dir, meta["rollout_status"]))
                continue
            scorecard = evaluate(run_dir)
            if not scorecard["complete"]:
                skipped.append((run_dir, "did not reach evaluation horizon"))
                continue
        except Exception as error:
            skipped.append((run_dir, str(error)))
            continue
        row = {m: (scorecard.get(m) or {}).get("normalized") for m in METRICS}
        if row["combined"] is None:
            skipped.append((run_dir, "degenerate scenario"))
            continue
        row["temporal"] = scorecard["temporal"]
        identity = meta.get("benchmark_identity") or legacy_run_identity(
            run_dir, meta, scorecard)
        model = resolved_model_id(run_dir, meta)
        if not model:
            model = (identity.get("model") or meta.get("agent_model") or "?")
            if meta.get("probe") == "devin":
                model = "unresolved:" + model
        identity = with_resolved_model(identity, model)
        row["benchmark_coordinate"] = identity["coordinate"]
        row["task_version"] = identity["task_version"]
        key = (meta.get("task", "?"), identity["task_version"], model)
        cells.setdefault(key, []).append(row)
    return cells, skipped


def aggregate(runs_root):
    cells, skipped = _load(runs_root)
    versions = {}
    for task, task_version, _model in cells:
        versions.setdefault(task, set()).add(task_version)
    output = {}
    for (task, task_version, model), rows in cells.items():
        display = (task if len(versions.get(task, ())) == 1
                   else "%s@%s" % (task, task_version[:8]))
        output.setdefault(display, {})[model] = {
            "task_name": task,
            "task_version": task_version,
            "coordinates": sorted({r["benchmark_coordinate"] for r in rows}),
            "k": len(rows),
            "metrics": {m: _stats([r[m] for r in rows]) for m in METRICS},
            "temporal": {
                "coverage": _stats([r["temporal"]["coverage"] for r in rows]),
                "mae_work_minutes": _stats(
                    [r["temporal"]["mae_work_minutes"] for r in rows]),
                "p50_final_mae_work_minutes": _stats(
                    [r["temporal"].get("p50_final_mae_work_minutes")
                     for r in rows]),
                "p90_coverage": _stats(
                    [r["temporal"].get("p90_coverage") for r in rows]),
                "mean_interval_width_work_minutes": _stats(
                    [r["temporal"].get("mean_interval_width_work_minutes")
                     for r in rows]),
                "horizon_accuracy": _stats(
                    [r["temporal"]["horizon_accuracy"] for r in rows]),
                "belief_revision_coverage": _stats(
                    [r["temporal"]["belief_revision_coverage"] for r in rows]),
                "time_weighted_mae_work_minutes": _stats(
                    [r["temporal"]["time_weighted_mae_work_minutes"]
                     for r in rows]),
                "revision_count": sum(
                    r["temporal"]["revision_count"] for r in rows),
                "censored_forecasts": sum(
                    r["temporal"]["censored_forecasts"] for r in rows),
            },
        }
    return output, skipped


def main():
    runs_root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    tasks, skipped = aggregate(runs_root)
    if not tasks:
        print("no completed scored runs under %s/" % runs_root)
        for path, reason in skipped:
            print("  skipped %s: %s" % (os.path.basename(path), reason))
        sys.exit(1)
    print("Per-task Max@k (0 = do-nothing baseline, 1 = ideal reference)")
    print("Incompatible task versions are separated; @<hash> marks each version.")
    task_means = {}
    for task in sorted(tasks):
        print("\nTASK %s" % task)
        for model in sorted(tasks[task]):
            cell = tasks[task][model]
            combined = cell["metrics"]["combined"]
            if combined is None:
                continue
            temporal = cell["temporal"]
            print("  %-24s k=%-3d combined %+.3f [%+.3f .. %+.3f] sigma=%.3f"
                  % (model, cell["k"], combined["mean"], combined["min"],
                     combined["max"], combined["std"]))
            coverage = temporal["coverage"]
            mae = temporal["p50_final_mae_work_minutes"]
            p90 = temporal["p90_coverage"]
            width = temporal["mean_interval_width_work_minutes"]
            accuracy = temporal["horizon_accuracy"]
            revision = temporal["belief_revision_coverage"]
            print("  %-24s       time coverage=%s  p50-final-MAE=%s  "
                  "p90-coverage=%s  mean-interval=%s  belief-revision=%s  "
                  "horizon-acc=%s"
                  % ("",
                     "%.3f" % coverage["mean"] if coverage else "n/a",
                     "%.1f min" % mae["mean"] if mae else "n/a",
                     "%.3f" % p90["mean"] if p90 else "n/a",
                     "%.1f min" % width["mean"] if width else "n/a",
                     "%.3f" % revision["mean"] if revision else "n/a",
                     "%.3f" % accuracy["mean"] if accuracy else "n/a"))
            task_means.setdefault(model, []).append(combined["mean"])
    print("\nMacro outcome (equal weight per task; task means, no cross-task Max@k)")
    for model in sorted(task_means):
        stats = _stats(task_means[model])
        print("  %-24s tasks=%-3d mean=%+.3f range=[%+.3f .. %+.3f]"
              % (model, stats["n"], stats["mean"], stats["min"], stats["max"]))
    if skipped:
        print("\nSkipped failed/incomplete runs: %d" % len(skipped))


if __name__ == "__main__":
    main()
