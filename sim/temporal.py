"""Prospective calibrated forecasts and belief-revision metrics."""

from .sim_time import working_minutes_between


REVISION_TRIGGERS = {
    "belief_revealed", "meeting_cancelled", "meeting_scheduled", "message",
    "question_answered", "task_added", "task_completed", "task_revealed",
    "task_updated",
}


def record_forecast(engine, args):
    world = engine.world
    task_id = args["task_id"]
    p50 = args["p50_completion_time"]
    p90 = args["p90_completion_time"]
    task = next((t for t in world.tracker_view() if t["id"] == task_id), None)
    if task is None:
        return {"error": "task is not on the visible board"}
    spec = world.find_task(task_id)
    if spec.get("source") == "agent" or not spec.get("effort_hours"):
        return {"error": "forecast an authored work item"}
    if task["status"] == "done":
        return {"error": "task is already complete"}
    if p50 <= world.clock or p90 <= world.clock:
        return {"error": "forecast quantiles must be in the future"}
    if p50 > p90:
        return {"error": "p50_completion_time must be <= p90_completion_time"}
    if not args["rationale"].strip():
        return {"error": "rationale must explain the current evidence"}
    prior = [e for e in world.log
             if e["kind"] == "completion_forecast" and e["task_id"] == task_id]
    if prior and not any(
            e["seq"] > prior[-1]["seq"] and e["kind"] in REVISION_TRIGGERS
            for e in world.log):
        return {"error": "revise only after new evidence or a plan change"}

    engine._advance_action_clock()
    if world.clock >= engine.agent_horizon:
        return {"error": "evaluation horizon reached before the forecast was recorded"}
    task = next((t for t in world.tracker_view() if t["id"] == task_id), None)
    if task is None or task["status"] == "done":
        return {"error": "task completed before the forecast was recorded"}

    revision = len(prior)
    world.record(
        "completion_forecast", task_id=task_id,
        p50_completion_time=p50, p90_completion_time=p90,
        rationale=args["rationale"], revision=revision)
    return {"recorded": task_id, "p50_completion_time": p50,
            "p90_completion_time": p90, "revision": revision}


def _forecast_quantiles(event):
    p50 = event.get("p50_completion_time", event.get("completion_time"))
    p90 = event.get("p90_completion_time", event.get("completion_time", p50))
    return p50, p90


def _signed_work_error(predicted, actual):
    magnitude = working_minutes_between(
        min(predicted, actual), max(predicted, actual))
    return magnitude if actual > predicted else -magnitude


def _pinball_loss(quantile, signed_work_error):
    loss = (quantile * signed_work_error if signed_work_error >= 0
            else (quantile - 1) * signed_work_error)
    return round(loss, 6)


def _errors(p50, p90, actual):
    if actual is None:
        return {
            "signed_error_minutes": None,
            "signed_error_work_minutes": None,
            "absolute_error_minutes": None,
            "absolute_error_work_minutes": None,
            "p50_absolute_error_work_minutes": None,
            "p90_absolute_error_work_minutes": None,
            "p50_pinball_loss_work_minutes": None,
            "p90_pinball_loss_work_minutes": None,
            "p90_covered": None,
        }
    p50_signed = _signed_work_error(p50, actual)
    p90_signed = _signed_work_error(p90, actual)
    return {
        "signed_error_minutes": actual - p50,
        "signed_error_work_minutes": p50_signed,
        "absolute_error_minutes": abs(actual - p50),
        "absolute_error_work_minutes": abs(p50_signed),
        "p50_absolute_error_work_minutes": abs(p50_signed),
        "p90_absolute_error_work_minutes": abs(p90_signed),
        "p50_pinball_loss_work_minutes": _pinball_loss(0.5, p50_signed),
        "p90_pinball_loss_work_minutes": _pinball_loss(0.9, p90_signed),
        "p90_covered": actual <= p90,
    }


def temporal_metrics(world, horizon):
    rows = {t["id"]: t for t in world.tasks_view(at=horizon)
            if t.get("source") != "agent" and t.get("effort_hours")
            and (t.get("seed_done_hours") or 0) < t["effort_hours"]}
    by_task = {}
    for event in world.log:
        if (event["kind"] == "completion_forecast" and event["t"] < horizon
                and event["task_id"] in rows):
            by_task.setdefault(event["task_id"], []).append(event)

    details = []
    weighted_error, weighted_minutes = 0.0, 0.0
    improvements = []
    pinball_improvements = []
    reveal_latencies = []
    reveal_count = 0
    reveal_reforecasted = 0
    usable_by_task = {}
    for task_id, events in by_task.items():
        task = rows[task_id]
        actual = task["projected_done"] if task["status"] == "done" else None
        usable = [e for e in events if actual is None or e["t"] < actual]
        if not usable:
            continue
        usable_by_task[task_id] = usable
        forecasts = []
        for event in usable:
            p50, p90 = _forecast_quantiles(event)
            row = {
                "forecast_at": event["t"],
                "p50_completion_time": p50,
                "p90_completion_time": p90,
                "predicted_completion": p50,
                "rationale": event.get("rationale", ""),
                "revision": event.get("revision", len(forecasts)),
                "interval_width_work_minutes": working_minutes_between(p50, p90),
            }
            row.update(_errors(p50, p90, actual))
            forecasts.append(row)
        if actual is not None:
            if len(forecasts) > 1:
                improvements.append(
                    forecasts[0]["absolute_error_work_minutes"]
                    - forecasts[-1]["absolute_error_work_minutes"])
                initial_pinball = (
                    forecasts[0]["p50_pinball_loss_work_minutes"]
                    + forecasts[0]["p90_pinball_loss_work_minutes"]) / 2
                final_pinball = (
                    forecasts[-1]["p50_pinball_loss_work_minutes"]
                    + forecasts[-1]["p90_pinball_loss_work_minutes"]) / 2
                pinball_improvements.append(initial_pinball - final_pinball)
            for i, forecast in enumerate(forecasts):
                end = (forecasts[i + 1]["forecast_at"]
                       if i + 1 < len(forecasts) else actual)
                duration = working_minutes_between(forecast["forecast_at"], end)
                weighted_error += forecast["p50_absolute_error_work_minutes"] * duration
                weighted_minutes += duration

        final = forecasts[-1]
        details.append({
            "task_id": task_id, "actual_completion": actual,
            "censored": actual is None, "forecast_count": len(forecasts),
            "forecasts": forecasts,
            "horizon_prediction_correct":
                (final["p50_completion_time"] <= horizon) == (actual is not None),
        })

    for reveal in world.log:
        if (reveal["kind"] != "belief_revealed"
                or reveal["task_id"] not in rows or reveal["t"] >= horizon):
            continue
        actual = rows[reveal["task_id"]]["projected_done"] \
            if rows[reveal["task_id"]]["status"] == "done" else None
        if actual is not None and reveal["t"] >= actual:
            continue
        reveal_count += 1
        revised = next((f for f in usable_by_task.get(reveal["task_id"], [])
                        if f["seq"] > reveal["seq"]), None)
        if revised is not None:
            reveal_reforecasted += 1
            reveal_latencies.append(working_minutes_between(
                reveal["t"], revised["t"]))

    observed = [r for r in details if not r["censored"]]

    def forecast_mean(position, key):
        values = [r["forecasts"][position][key] for r in observed]
        return round(sum(values) / len(values), 3) if values else None

    def pinball_mean(position):
        values = [
            (r["forecasts"][position]["p50_pinball_loss_work_minutes"]
             + r["forecasts"][position]["p90_pinball_loss_work_minutes"]) / 2
            for r in observed]
        return round(sum(values) / len(values), 3) if values else None

    final_mae = forecast_mean(-1, "absolute_error_minutes")
    final_work_mae = forecast_mean(-1, "p50_absolute_error_work_minutes")
    return {
        "eligible_tasks": len(rows), "forecasted_tasks": len(details),
        "forecast_count": sum(r["forecast_count"] for r in details),
        "revision_count": sum(max(0, r["forecast_count"] - 1) for r in details),
        "coverage": round(len(details) / len(rows), 3) if rows else None,
        "observed_completions": len(observed),
        "censored_forecasts": len(details) - len(observed),
        # Backward-compatible headline fields use the final p50 forecast.
        "mae_minutes": final_mae,
        "mae_work_minutes": final_work_mae,
        "bias_minutes": forecast_mean(-1, "signed_error_minutes"),
        "bias_work_minutes": forecast_mean(-1, "signed_error_work_minutes"),
        "initial_mae_minutes": forecast_mean(0, "absolute_error_minutes"),
        "initial_mae_work_minutes":
            forecast_mean(0, "p50_absolute_error_work_minutes"),
        "final_mae_minutes": final_mae,
        "final_mae_work_minutes": final_work_mae,
        "p50_final_mae_work_minutes": final_work_mae,
        "p90_final_mae_work_minutes":
            forecast_mean(-1, "p90_absolute_error_work_minutes"),
        "p90_coverage": (
            round(sum(r["forecasts"][-1]["p90_covered"] for r in observed)
                  / len(observed), 3) if observed else None),
        "mean_interval_width_work_minutes": (
            round(sum(r["forecasts"][-1]["interval_width_work_minutes"]
                      for r in details) / len(details), 3)
            if details else None),
        "mean_pinball_loss_work_minutes": pinball_mean(-1),
        "initial_mean_pinball_loss_work_minutes": pinball_mean(0),
        "final_mean_pinball_loss_work_minutes": pinball_mean(-1),
        "time_weighted_mae_work_minutes": (
            round(weighted_error / weighted_minutes, 3)
            if weighted_minutes else None),
        "mean_revision_improvement_work_minutes": (
            round(sum(improvements) / len(improvements), 3)
            if improvements else None),
        "mean_revision_pinball_improvement_work_minutes": (
            round(sum(pinball_improvements) / len(pinball_improvements), 3)
            if pinball_improvements else None),
        "belief_reveal_count": reveal_count,
        "belief_reforecasted": reveal_reforecasted,
        "belief_revision_coverage": (
            round(reveal_reforecasted / reveal_count, 3)
            if reveal_count else None),
        "mean_belief_revision_latency_work_minutes": (
            round(sum(reveal_latencies) / len(reveal_latencies), 3)
            if reveal_latencies else None),
        "horizon_accuracy": (
            round(sum(r["horizon_prediction_correct"] for r in details) / len(details), 3)
            if details else None),
        "per_task": details,
    }
