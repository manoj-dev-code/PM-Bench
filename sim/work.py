"""Deterministic staged work items and pre-materialized quality outcomes."""

import copy


def _stage_row(parent, stage):
    row = copy.deepcopy(parent)
    row.pop("stages", None)
    row.pop("quality_gate", None)
    row.update(copy.deepcopy(stage))
    row.pop("quality_gate", None)
    row["id"] = "%s::%s" % (parent["id"], stage["id"])
    row["title"] = "%s — %s" % (parent["title"], stage["title"])
    row["work_item_id"] = parent["id"]
    row["work_item_title"] = parent["title"]
    row["stage_id"] = stage["id"]
    row["stage_kind"] = stage["kind"]
    return row


def _expand(parent):
    rows = []
    previous = None
    for stage in parent.get("stages") or []:
        row = _stage_row(parent, stage)
        if previous is None:
            row["blocked_by"] = list(parent.get("blocked_by") or [])
        else:
            row["blocked_by"] = [previous]
        rows.append(row)
        previous = row["id"]
        gate = stage.get("quality_gate") or {}
        if gate.get("outcome") == "rework":
            rework = _stage_row(parent, gate["rework"])
            rework["blocked_by"] = [row["id"]]
            rework["contingent"] = True
            rework["hidden_until"] = [row["id"]]
            rework["quality_reason"] = gate.get("reason", "")
            rows.append(rework)
            previous = rework["id"]
    total = sum(float(r.get("effort_hours") or 0) for r in rows)
    running = 0.0
    for i, row in enumerate(rows):
        if total:
            if i + 1 == len(rows):
                factor = 1.0 - running
            else:
                factor = float(row.get("effort_hours") or 0) / total
                running += factor
            row["score_weight_factor"] = factor
    return rows


def work_item_terminals(tasks):
    """Return {staged_parent_id: terminal_materialized_stage_id}."""
    terminals = {}
    for task in tasks:
        stages = task.get("stages") or []
        if not stages:
            continue
        terminal_id = stages[-1]["id"]
        for stage in stages:
            terminal_id = stage["id"]
            gate = stage.get("quality_gate") or {}
            if gate.get("outcome") == "rework":
                terminal_id = gate["rework"]["id"]
        terminals[task["id"]] = "%s::%s" % (task["id"], terminal_id)
    return terminals


def materialize_work_items(tasks):
    """Expand authored staged work items into ordinary scheduler tasks."""
    expanded = []
    terminal = work_item_terminals(tasks)
    for task in tasks:
        if not task.get("stages"):
            expanded.append(copy.deepcopy(task))
            continue
        expanded.extend(_expand(task))

    for task in expanded:
        task["blocked_by"] = [terminal.get(dep, dep)
                              for dep in task.get("blocked_by", [])]
    return expanded
