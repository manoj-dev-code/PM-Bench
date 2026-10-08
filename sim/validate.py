"""Scenario validation + world-mutation bounds.

Two layers of control on the levers:

1. LOAD-TIME: `validate_scenario` rejects a scenario whose config exceeds
   bounds (too many tasks, unreasonable efforts, dangling references) —
   authored or generated, every scenario passes through the same gate.
2. RUNTIME: `check_task_bounds` / meeting bounds guard the mutation paths
   (external arrivals, agent-added tasks, meetings), so no actor — including
   a future LLM-driven scenario generator — can push the world outside the
   authored envelope.

The LLM itself has NO mutation channel: it produces message/transcript text
only. These bounds exist so that the things which DO mutate (config, agent
tools) stay reasonable too.
"""

import itertools
import json
import math

DEFAULT_BOUNDS = {
    "max_tasks": 50,               # total tasks ever on the board
    "max_task_effort_hours": 80,   # one task can't be a month of work
    "min_task_effort_hours": 0.5,
    "max_external_arrivals": 10,   # OOD injections per scenario
    "max_beats": 20,
    "max_meeting_minutes": 240,    # no all-day meetings
    "max_horizon_days": 14,        # events must land inside the run window
    "max_questions_per_task": 3,   # a task can't be an endless blocking series
}


def bounds_of(scenario):
    b = dict(DEFAULT_BOUNDS)
    b.update(scenario.get("bounds", {}))
    return b


def check_task_bounds(task, existing_count, bounds):
    """Runtime gate for every task addition (seed/external/agent)."""
    if existing_count >= bounds["max_tasks"]:
        return "task limit reached (%d)" % bounds["max_tasks"]
    eh = task.get("effort_hours")
    if eh is not None and not (bounds["min_task_effort_hours"] <= eh
                               <= bounds["max_task_effort_hours"]):
        return ("effort_hours %.1f outside [%s, %s]"
                % (eh, bounds["min_task_effort_hours"],
                   bounds["max_task_effort_hours"]))
    return None


def _numeric(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _stage_violations(stage, label, bounds, worker_ids):
    """Phase-1 stage fields: ordinary bounded work, no nested local ephemera."""
    errors = []
    if not isinstance(stage, dict):
        return [label + ": stage must be an object"]
    sid = stage.get("id")
    if not isinstance(sid, str) or not sid.strip():
        errors.append(label + ": stage id must be a non-empty string")
    elif "::" in sid:
        errors.append(label + ": stage id may not contain '::'")
    for key in ("title", "kind"):
        if not isinstance(stage.get(key), str) or not stage[key].strip():
            errors.append("%s: %s must be a non-empty string" % (label, key))
    assignees = stage.get("assignees")
    if (not isinstance(assignees, list) or not assignees
            or any(not isinstance(a, str) for a in assignees)):
        errors.append(label + ": assignees must name at least one worker")
    else:
        for a in assignees:
            if a not in worker_ids:
                errors.append("%s: assignee %r is not a worker" % (label, a))
    effort = stage.get("effort_hours")
    if not _numeric(effort) or not (
            bounds["min_task_effort_hours"] <= effort
            <= bounds["max_task_effort_hours"]):
        errors.append(label + ": effort_hours outside task bounds")
    done = stage.get("done_hours", 0)
    if not _numeric(done) or done < 0 or (_numeric(effort) and done > effort):
        errors.append(label + ": done_hours must be within 0..effort_hours")
    for key in ("questions", "belief", "stages", "blocked_by", "contingent",
                "hidden_until",
                "work_item_id", "work_item_title", "stage_id", "stage_kind",
                "score_weight_factor", "quality_reason"):
        if key in stage:
            errors.append("%s: %s is not supported inside stages in Phase 1"
                          % (label, key))
    return errors


def _validate_staged_parent(task, where, bounds, worker_ids, authored_ids):
    errors = []
    label = "%s[%s]" % (where, task.get("id"))
    stages = task.get("stages")
    if not isinstance(task.get("title"), str) or not task["title"].strip():
        errors.append(label + ": staged work needs a non-empty title")
    for forbidden in ("assignees", "effort_hours", "done_hours", "questions",
                      "belief"):
        if forbidden in task:
            errors.append("%s: staged work item may not have top-level %s"
                          % (label, forbidden))
    if not isinstance(stages, list) or len(stages) < 2:
        return errors + [label + ": staged work needs at least two stages"]
    local_ids = set()
    generated = set()
    for i, stage in enumerate(stages):
        slabel = "%s.stages[%d]" % (label, i)
        errors.extend(_stage_violations(stage, slabel, bounds, worker_ids))
        sid = stage.get("id")
        if isinstance(sid, str) and sid.strip() and "::" not in sid:
            if sid in local_ids:
                errors.append(slabel + ": duplicate stage id %r" % sid)
            local_ids.add(sid)
            gid = "%s::%s" % (task.get("id"), sid)
            if gid in authored_ids or gid in generated:
                errors.append(slabel + ": generated id %r collides" % gid)
            generated.add(gid)
        gate = stage.get("quality_gate")
        if gate is None:
            continue
        glabel = slabel + ".quality_gate"
        if (not isinstance(gate, dict) or set(gate) - {"outcome", "reason", "rework"}
                or "outcome" not in gate or "reason" not in gate):
            errors.append(glabel + ": bad fields")
            continue
        outcome = gate.get("outcome")
        if outcome not in ("pass", "rework"):
            errors.append(glabel + ": outcome must be pass|rework")
        if not isinstance(gate.get("reason"), str) or not gate["reason"].strip():
            errors.append(glabel + ": reason must be non-empty")
        if outcome == "pass" and "rework" in gate:
            errors.append(glabel + ": pass may not carry rework")
        if outcome != "rework":
            continue
        rework = gate.get("rework")
        if not isinstance(rework, dict):
            errors.append(glabel + ": rework outcome requires a rework stage")
            continue
        errors.extend(_stage_violations(
            rework, glabel + ".rework", bounds, worker_ids))
        rid = rework.get("id")
        if "quality_gate" in rework:
            errors.append(glabel + ".rework: nested quality_gate is not allowed")
        if isinstance(rid, str) and rid.strip() and "::" not in rid:
            if rid in local_ids:
                errors.append(glabel + ".rework: duplicate stage id %r" % rid)
            gid = "%s::%s" % (task.get("id"), rid)
            if gid in authored_ids or gid in generated:
                errors.append(glabel + ".rework: generated id %r collides" % gid)
            generated.add(gid)
            local_ids.add(rid)
    return errors


def _materialized_cycles(tasks):
    known = {t["id"] for t in tasks}
    deps = {t["id"]: [d for d in t.get("blocked_by", []) if d in known]
            for t in tasks}
    visiting, visited = set(), set()

    def visit(tid):
        if tid in visited:
            return False
        if tid in visiting:
            return True
        visiting.add(tid)
        cyclic = any(visit(dep) for dep in deps.get(tid, []))
        visiting.remove(tid)
        visited.add(tid)
        return cyclic

    return any(visit(t["id"]) for t in tasks)


def _validate_materialized(tasks, bounds, worker_ids):
    errors = []
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        errors.append("materialized work has duplicate task ids")
    if len(ids) > bounds["max_tasks"]:
        errors.append("materialized work exceeds max_tasks=%d"
                      % bounds["max_tasks"])
    known = set(ids)
    by_item = {}
    for task in tasks:
        for dep in task.get("blocked_by", []):
            if dep not in known:
                errors.append("materialized task %r has unknown blocker %r"
                              % (task["id"], dep))
        for a in task.get("assignees", []):
            if a not in worker_ids:
                errors.append("materialized task %r assignee %r is not a worker"
                              % (task["id"], a))
        msg = check_task_bounds(task, 0, bounds)
        if msg:
            errors.append("materialized task %r: %s" % (task["id"], msg))
        if "work_item_id" in task:
            by_item.setdefault(task["work_item_id"], []).append(
                task.get("score_weight_factor", 0))
    for item, factors in by_item.items():
        if abs(sum(factors) - 1.0) > 1e-9:
            errors.append("work item %r score factors sum to %s, not 1.0"
                          % (item, sum(factors)))
    if _materialized_cycles(tasks):
        errors.append("materialized task dependencies contain a cycle")
    return errors


def validate_scenario(scenario):
    """Return a list of violations (empty = valid). Called at engine init."""
    errors = []
    b = bounds_of(scenario)
    start = scenario.get("start_time", 545)
    horizon = start + b["max_horizon_days"] * 1440
    npc_ids = {n["id"] for n in scenario.get("npcs", [])}
    worker_ids = {n["id"] for n in scenario.get("npcs", []) if n.get("worker", True)}

    if not npc_ids:
        # LIVENESS: NPC heartbeats are what keep the event queue non-empty.
        errors.append("scenario has no NPCs — the event queue would starve")
    if not worker_ids:
        # OPT and the baseline both need someone who actually does work; an
        # all-stakeholder cast has an empty labor pool (no winnable value).
        errors.append("scenario has no workers — every NPC is worker:false")

    tasks = (scenario.get("project") or {}).get("tasks", [])
    arrivals = scenario.get("task_arrivals", [])
    beats = scenario.get("beats", [])

    if len(tasks) + len(arrivals) > b["max_tasks"]:
        errors.append("too many tasks: %d seed + %d arrivals > max_tasks=%d"
                      % (len(tasks), len(arrivals), b["max_tasks"]))
    if len(arrivals) > b["max_external_arrivals"]:
        errors.append("too many external arrivals: %d > %d"
                      % (len(arrivals), b["max_external_arrivals"]))
    if len(beats) > b["max_beats"]:
        errors.append("too many beats: %d > %d" % (len(beats), b["max_beats"]))

    seen_ids = set()
    seen_question_ids = set()
    all_specs = [(t, "project.tasks") for t in tasks] + \
                [(a["task"], "task_arrivals") for a in arrivals]
    known_ids = {t["id"] for t, _ in all_specs}
    for t, where in all_specs:
        if t["id"] in seen_ids:
            errors.append("%s: duplicate task id %r" % (where, t["id"]))
        seen_ids.add(t["id"])
        msg = check_task_bounds(t, 0, b)
        if msg:
            errors.append("%s[%s]: %s" % (where, t["id"], msg))
        # a task MAY have several assignees: when more than one works it at once
        # it runs at the damped parallel_rate (a meeting lifts that to a swarm).
        # More hands is sublinear and costs their own work, so it's a real
        # tradeoff, not a free win. assignees[0] remains the accountable owner
        # (belief holder / question owner default).
        for a in t.get("assignees", []):
            if a not in npc_ids:
                errors.append("%s[%s]: unknown assignee %r" % (where, t["id"], a))
            elif a not in worker_ids:
                errors.append("%s[%s]: assigned to %r, a stakeholder "
                              "(worker:false) — not on the delivery team"
                              % (where, t["id"], a))
        for dep in t.get("blocked_by", []):
            if dep not in known_ids:
                errors.append("%s[%s]: unknown blocker %r" % (where, t["id"], dep))
        for j, bel in enumerate(t.get("belief") or []):
            if j == 0 and "at" in bel:
                errors.append("%s[%s]: belief[0] is the initial picture — no 'at'"
                              % (where, t["id"]))
            if j > 0 and not (start <= bel.get("at", -1) <= horizon):
                errors.append("%s[%s]: belief[%d].at outside run window"
                              % (where, t["id"], j))
            held = bel.get("held_by")
            if held is not None and held not in worker_ids:
                errors.append("%s[%s]: belief[%d].held_by=%r must be a worker "
                              "(the person carrying the stale estimate)"
                              % (where, t["id"], j, held))
        # blocking questions: a gating question suspends the task until the PM
        # replies to its OWNER. Must open in-window, be owned by the task's sole
        # worker, and carry an id (completability is checked in the fairness gate).
        qs = t.get("questions") or []
        if len(qs) > b["max_questions_per_task"]:
            errors.append("%s[%s]: %d questions > max_questions_per_task=%d"
                          % (where, t["id"], len(qs), b["max_questions_per_task"]))
        owner = (t.get("assignees") or [None])[0]
        question_deltas = []
        effort_numeric = isinstance(t.get("effort_hours"), (int, float)) \
            and not isinstance(t.get("effort_hours"), bool) \
            and math.isfinite(t["effort_hours"])
        for j, q in enumerate(qs):
            qid = q.get("id")
            if not isinstance(qid, str) or not qid.strip():
                errors.append("%s[%s]: questions[%d] needs a non-empty string id"
                              % (where, t["id"], j))
            elif q.get("gates"):
                if qid in seen_question_ids:
                    errors.append("%s[%s]: duplicate gating question id %r"
                                  % (where, t["id"], qid))
                seen_question_ids.add(qid)
            if not (start <= q.get("at", -1) <= horizon):
                errors.append("%s[%s]: questions[%d].at outside run window"
                              % (where, t["id"], j))
            held = q.get("held_by", owner)
            if held != owner:
                errors.append("%s[%s]: questions[%d].held_by=%r is not the task's "
                              "owner %r — only the doer can be blocked on it"
                              % (where, t["id"], j, held, owner))
            if owner is not None and owner not in worker_ids:
                errors.append("%s[%s]: a blocking question needs a worker owner, "
                              "got %r" % (where, t["id"], owner))
            if q.get("gates"):
                options = q.get("options")
                if (not isinstance(options, list) or len(options) < 2
                        or len(set(options)) != len(options)
                        or any(not isinstance(x, str) or not x.strip()
                               for x in options)):
                    errors.append(
                        "%s[%s]: questions[%d] needs at least two unique, "
                        "non-empty string options" % (where, t["id"], j))
                effects = q.get("choice_effects")
                if effects is not None:
                    option_set = set(options or [])
                    if not isinstance(effects, dict) or set(effects) != option_set:
                        errors.append("%s[%s]: questions[%d] choice_effects keys "
                                      "must match options" % (where, t["id"], j))
                    else:
                        shapes = set()
                        deltas = []
                        for choice, effect in effects.items():
                            if not isinstance(effect, dict) or set(effect) - {
                                    "summary", "task_effort_delta_hours",
                                    "capacity_costs"} or "summary" not in effect:
                                errors.append("%s[%s]: questions[%d] choice %r "
                                              "effect has bad fields"
                                              % (where, t["id"], j, choice))
                                continue
                            if not isinstance(effect.get("summary"), str) \
                                    or not effect["summary"].strip():
                                errors.append("%s[%s]: questions[%d] choice %r "
                                              "effect summary must be non-empty"
                                              % (where, t["id"], j, choice))
                            delta = effect.get("task_effort_delta_hours", 0)
                            if not isinstance(delta, (int, float)) \
                                    or isinstance(delta, bool) \
                                    or not math.isfinite(delta):
                                errors.append("%s[%s]: questions[%d] choice %r "
                                              "effort delta must be finite"
                                              % (where, t["id"], j, choice))
                            else:
                                deltas.append(delta)
                                if "task_effort_delta_hours" in effect \
                                        and not effort_numeric:
                                    errors.append("%s[%s]: questions[%d] choice %r "
                                                  "effort delta requires numeric effort_hours"
                                                  % (where, t["id"], j, choice))
                                elif effort_numeric and not (
                                        b["min_task_effort_hours"]
                                        <= t["effort_hours"] + delta
                                        <= b["max_task_effort_hours"]):
                                    errors.append("%s[%s]: questions[%d] choice %r "
                                                  "effort delta leaves effort out of bounds"
                                                  % (where, t["id"], j, choice))
                            costs = effect.get("capacity_costs", [])
                            if not isinstance(costs, list):
                                errors.append("%s[%s]: questions[%d] choice %r "
                                              "capacity_costs must be a list"
                                              % (where, t["id"], j, choice))
                                costs = []
                            for cost in costs:
                                if not isinstance(cost, dict) or set(cost) != {
                                        "npc", "minutes", "reason"}:
                                    errors.append("%s[%s]: questions[%d] choice %r "
                                                  "capacity cost has bad fields"
                                                  % (where, t["id"], j, choice))
                                    continue
                                if cost["npc"] not in worker_ids:
                                    errors.append("%s[%s]: questions[%d] choice %r "
                                                  "capacity cost targets non-worker %r"
                                                  % (where, t["id"], j, choice,
                                                     cost["npc"]))
                                if not isinstance(cost["minutes"], int) \
                                        or isinstance(cost["minutes"], bool) \
                                        or not 1 <= cost["minutes"] <= 480:
                                    errors.append("%s[%s]: questions[%d] choice %r "
                                                  "capacity cost minutes must be 1..480"
                                                  % (where, t["id"], j, choice))
                                if not isinstance(cost["reason"], str) \
                                        or not cost["reason"].strip():
                                    errors.append("%s[%s]: questions[%d] choice %r "
                                                  "capacity cost reason must be non-empty"
                                                  % (where, t["id"], j, choice))
                            shapes.add(json.dumps({
                                "task_effort_delta_hours":
                                    effect.get("task_effort_delta_hours", 0),
                                "capacity_costs":
                                    effect.get("capacity_costs", []),
                            }, sort_keys=True))
                        if len(deltas) == len(effects):
                            question_deltas.append(deltas)
                        if len(shapes) < 2:
                            errors.append("%s[%s]: questions[%d] choice_effects "
                                          "must differ across options"
                                          % (where, t["id"], j))
        if question_deltas and effort_numeric:
            for combo in itertools.product(*question_deltas):
                effective = t["effort_hours"] + sum(combo)
                if not (b["min_task_effort_hours"] <= effective
                        <= b["max_task_effort_hours"]):
                    errors.append("%s[%s]: question choice deltas leave effort "
                                  "out of bounds" % (where, t["id"]))
                    break
        if "stages" in t:
            errors.extend(_validate_staged_parent(
                t, where, b, worker_ids, known_ids))

    try:
        from .work import materialize_work_items
        materialized = materialize_work_items(
            [t for t in tasks] + [a["task"] for a in arrivals])
        errors.extend(_validate_materialized(materialized, b, worker_ids))
    except Exception as error:
        errors.append("staged work materialization failed: %s" % error)

    costs = scenario.get("costs") or {}
    for key in ("handoff_minutes", "helper_onboarding_minutes"):
        if key in costs and (not isinstance(costs[key], int)
                             or isinstance(costs[key], bool)
                             or not 0 <= costs[key] <= 480):
            errors.append("costs.%s must be an integer in 0..480" % key)

    for i, arr in enumerate(arrivals):
        if "stages" in arr.get("task", {}):
            errors.append("task_arrivals[%d]: staged work is not supported "
                          "in Phase 1" % i)
        if not (start <= arr["at"] <= horizon):
            errors.append("task_arrivals[%d]: at=%d outside run window" % (i, arr["at"]))
        if arr.get("npc") and arr["npc"] not in npc_ids:
            errors.append("task_arrivals[%d]: unknown npc %r" % (i, arr["npc"]))
        if arr.get("via") not in (None, "chat", "email"):
            errors.append("task_arrivals[%d]: via must be chat|email, got %r"
                          % (i, arr["via"]))
        # Real work always has an OWNER — an arrival must land already assigned
        # to a worker (no unowned backlog). A complete no-PM baseline: the
        # default holder works it, badly if they're the wrong/swamped choice;
        # the PM earns credit by REARRANGING to a better owner, not by merely
        # filing something nobody was on.
        owners = arr["task"].get("assignees") or []
        if not owners:
            errors.append("task_arrivals[%d]: task %r has no owner — arrivals "
                          "must land assigned to a default holder"
                          % (i, arr["task"]["id"]))
        for o in owners:
            osp = next((n for n in scenario.get("npcs", []) if n["id"] == o), None)
            if osp is None:
                errors.append("task_arrivals[%d]: owner %r unknown" % (i, o))
            elif not osp.get("worker", True):
                errors.append("task_arrivals[%d]: owner %r is a stakeholder — "
                              "not on the delivery team" % (i, o))
        if arr.get("announce") and not arr.get("npc"):
            errors.append("task_arrivals[%d]: announce needs an npc messenger" % i)
        if not arr.get("announce"):
            # unfiled + unannounced = a task nobody could ever learn about
            errors.append("task_arrivals[%d]: arrivals reach the PM only via "
                          "announce (chat/email) — silent arrivals are "
                          "undiscoverable now that filing is required" % i)

    # FAIRNESS IS A RUBRIC TOO: a scored ask must be physically completable
    # from the moment the PM could first KNOW it (chat: instant; email: the
    # next batch tick). Anything tighter scores luck, not skill — rejected
    # at authoring time, same as any other invalid scenario.
    grade_horizon = None
    try:
        from .rubric import load_rubric
        grade_horizon = load_rubric(scenario).get("horizon")
    except SystemExit:
        pass  # no rubric resolvable here — the fingerprint still reports
    if grade_horizon:
        from .sim_time import working_minutes_between
        batch = scenario.get("email_batch_minutes", 30)
        for i, arr in enumerate(arrivals):
            delivery = (arr["at"] if arr.get("via", "chat") == "chat"
                        else (arr["at"] // batch + 1) * batch)
            need = arr["task"].get("effort_hours", 0) * 60
            window = working_minutes_between(delivery, grade_horizon)
            if need and window < need:
                errors.append(
                    "task_arrivals[%d]: UNFAIR — %r needs %d work-min but only "
                    "%d remain after the PM can first know of it (%s %s)"
                    % (i, arr["task"]["id"], need, window,
                       arr.get("via", "chat"), "delivery"))
        # A blocking question is fair BY CONSTRUCTION: answered the instant it
        # opens it is identical to no question at all (empty [at, at) block), so
        # a prompt reply always recovers the no-question outcome — the winnable
        # value is exactly the stall the PM saves. The only trap is a question
        # opening in dead time (no working minutes left to matter), which the
        # in-window check above already forbids. So no extra effort-fits gate.
    for i, beat in enumerate(beats):
        if not (start <= beat["at"] <= horizon):
            errors.append("beats[%d]: at=%d outside run window" % (i, beat["at"]))
        if beat.get("npc") not in npc_ids:
            errors.append("beats[%d]: unknown npc %r" % (i, beat.get("npc")))
        arms = beat.get("arms")
        if arms is not None:
            if not arms or any("intent" not in a for a in arms):
                errors.append("beats[%d]: arms must be non-empty, each with an intent" % i)
        elif "intent" not in beat:
            errors.append("beats[%d]: needs an intent or arms" % i)
    return errors
