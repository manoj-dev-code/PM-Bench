"""The agent-facing tool surface — single source of truth.

Each tool = name + description + JSON schema + handler. The same registry
drives the web UI / REST API today and becomes the Claude tool list for the
agent harness (the evaluated PM agent) verbatim — Claude-tool-definition
format on purpose.

Handlers are all SYNCHRONOUS agent actions (see DESIGN.md): they mutate the
world at the current instant (+1 sim-min cost) and may schedule async
consequences. Time-advancing tools yield control to the queue.
"""

import math

from .sim_time import fmt
from .temporal import record_forecast


def _tool(name, description, properties, required, handler):
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
        "handler": handler,
    }


def _view_tasks(engine, args):
    # THE TRACKER, not the truth: structure + recorded notes + announced
    # completions. True progress is earned by asking people (their beliefs)
    # and observing completions — never read off a board.
    return {"now": engine.world.now(), "tasks": engine.world.tracker_view()}


def _view_inbox(engine, args):
    # scrollback of everything DELIVERED so far: all chat, plus emails whose
    # batch-delivery event has fired. An email in flight (sent, not yet
    # delivered) is invisible — in a DES nothing reaches you except by event.
    delivered = {e["msg_id"] for e in engine.world.log
                 if e["kind"] == "email_delivered"}
    msgs = [m for m in engine.world.agent_inbox()
            if m.via == "chat" or m.id in delivered]
    return [{"time": fmt(m.time), "from": m.sender, "via": m.via, "text": m.text}
            for m in msgs[-int(args.get("limit", 20)):]]


def _view_calendar(engine, args):
    # calendars are EXPOSED truth (DESIGN.md information ledger): who is in
    # what meeting or out-of-office block when is public — capacity math stays
    # the PM's job
    now = engine.world.clock
    rows = []
    for m in engine.world.meetings:
        if m.get("cancelled"):
            continue
        row = {"id": m["id"], "topic": m["topic"],
               "start": fmt(m["start"]), "end": fmt(m["end"]),
               "attendees": m["attendees"],
               "status": ("past" if m["end"] <= now else
                          "now" if m["start"] <= now else "upcoming")}
        if m.get("task"):
            row["task"] = m["task"]
        rows.append((m["start"], row))
    for npc in engine.world.scenario.get("npcs", []):
        for i, (start, end) in enumerate(npc.get("ooo", [])):
            rows.append((start, {
                "id": "ooo-%s-%d" % (npc["id"], i),
                "topic": "Out of office: %s" % npc["name"],
                "start": fmt(start), "end": fmt(end),
                "attendees": [npc["id"]], "type": "out_of_office",
                "status": ("past" if end <= now else
                           "now" if start <= now else "upcoming")}))
    out = [row for _, row in sorted(rows, key=lambda item: item[0])]
    return {"now": engine.world.now(), "meetings": out}


def _advance(engine, minutes):
    # PM sleep semantics: push-class signals interrupt the advance (control
    # comes back at the interruption instant); board broadcasts never do.
    target = engine.world.clock + minutes
    target = min(target, getattr(engine, "agent_horizon", target))
    fired = engine.advance_until(target, interruptible=True)
    out = {"fired": fired, "now": engine.world.now()}
    if engine.world.clock < target:
        out["interrupted"] = True
    return out


def _wait(engine):
    horizon = min(engine.world.clock + 36 * 60,
                  getattr(engine, "agent_horizon", engine.world.clock + 36 * 60))
    return {"replied": engine.run_until_reply(horizon=horizon) is not None,
            "now": engine.world.now()}


TOOLS = [
    _tool("forecast_completion",
          "Record or revise a calibrated estimate of when a visible unfinished "
          "task will complete. p50_completion_time and p90_completion_time are "
          "absolute simulated minutes since Monday 00:00 (next Monday starts "
          "at 10080), and p50 must be <= p90. Recording costs one minute. "
          "Revisions are accepted only after new evidence or a plan change; "
          "explain what changed in rationale.",
          {"task_id": {"type": "string"},
           "p50_completion_time": {"type": "integer", "minimum": 0},
           "p90_completion_time": {"type": "integer", "minimum": 0},
           "rationale": {"type": "string"}},
          ["task_id", "p50_completion_time", "p90_completion_time",
           "rationale"],
          record_forecast),
    _tool("send_chat",
          "Send a chat message to one coworker (npc) or several (to) — a "
          "group ping where everyone sees who else is on it.",
          {"npc": {"type": "string", "description": "coworker id"},
           "to": {"type": "array", "items": {"type": "string"},
                  "description": "several coworker ids (group ping)"},
           "text": {"type": "string"}},
          ["text"],
          # no reply_due returned: WHEN people answer is theirs to know
          lambda e, a: e.agent_chat(a.get("to")
                                    or ([a["npc"]] if a.get("npc") else []),
                                    a["text"])),

    _tool("send_email",
          "Send an email to one or more coworkers.",
          {"to": {"type": "array", "items": {"type": "string"}},
           "subject": {"type": "string"},
           "body": {"type": "string"}},
          ["to", "subject", "body"],
          lambda e, a: e.agent_email(a["to"], a["subject"], a["body"])),

    _tool("answer_question",
          "Make an explicit decision on an open blocking question and deliver it to the current task owner. Generic chat/email messages do not unblock work.",
          {"task_id": {"type": "string"},
           "question_id": {"type": "string"},
           "choice": {"type": "string"},
           "rationale": {"type": "string"},
           "via": {"type": "string", "enum": ["chat", "email"]}},
          ["task_id", "question_id", "choice", "rationale", "via"],
          lambda e, a: e.agent_answer_question(a["task_id"], a["question_id"], a["choice"], a["rationale"], a["via"])),

    _tool("add_task",
          "Add a TRACKING item to the project board (a note everyone can "
          "see). Items you create are never scheduled and earn no score — "
          "real work arrives on its own.",
          {"title": {"type": "string"},
           "id": {"type": "string",
                  "description": "ticket reference id, if one was mentioned"},
           "assignee": {"type": "string"},
           "effort_hours": {"type": "number"}},
          ["title"],
          lambda e, a: e.agent_add_task(
              dict({"id": (a.get("id") or
                           a["title"].lower().replace(" ", "-")[:40]),
                    "title": a["title"]},
                   **({"assignees": [a["assignee"]]} if a.get("assignee") else {}),
                   **({"effort_hours": a["effort_hours"]}
                      if a.get("effort_hours") else {})))),

    _tool("assign_task",
          "Assign or reassign a task to a coworker (replaces the current owner).",
          {"task_id": {"type": "string"}, "npc": {"type": "string"}},
          ["task_id", "npc"],
          lambda e, a: e.agent_assign_task(a["task_id"], a["npc"])),

    _tool("add_helper",
          "Put an EXTRA person on a task alongside its owner so they work it in "
          "parallel. More hands finish it sooner but with diminishing returns, "
          "and it takes the helper off their own work — worth it for a real "
          "bottleneck or an idle teammate, not by default.",
          {"task_id": {"type": "string"}, "npc": {"type": "string"}},
          ["task_id", "npc"],
          lambda e, a: e.agent_add_helper(a["task_id"], a["npc"])),

    _tool("drop_helper",
          "Take a helper back off a task, returning them to their own work "
          "(can't remove the only owner — reassign instead).",
          {"task_id": {"type": "string"}, "npc": {"type": "string"}},
          ["task_id", "npc"],
          lambda e, a: e.agent_drop_helper(a["task_id"], a["npc"])),

    _tool("reprioritize",
          "Change how a task ranks in its assignee's work queue: set a "
          "working priority (P0..P3) and/or mark it urgent.",
          {"task_id": {"type": "string"},
           "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
           "urgent": {"type": "boolean"}},
          ["task_id"],
          lambda e, a: e.agent_reprioritize(a["task_id"], a.get("priority"),
                                            a.get("urgent"))),

    _tool("update_tracker_note",
          "Correct a task's reported status in the tracker (what others read).",
          {"task_id": {"type": "string"}, "note": {"type": "string"}},
          ["task_id", "note"],
          lambda e, a: e.agent_update_note(a["task_id"], a["note"])),

    _tool("schedule_meeting",
          "Book a meeting at a future time. Include 'agent' in attendees to "
          "attend yourself. A transcript is kept. Optionally set 'task' to a "
          "task id to make it a working session on that task with those people.",
          {"attendees": {"type": "array", "items": {"type": "string"}},
           "start_in_minutes": {"type": "integer", "minimum": 0},
           "duration_minutes": {"type": "integer", "minimum": 1},
           "topic": {"type": "string"},
           "agenda": {"type": "string"},
           "task": {"type": "string"}},
          ["attendees", "start_in_minutes", "duration_minutes", "topic"],
          lambda e, a: e.agent_schedule_meeting(
              a["attendees"], e.world.clock + int(a["start_in_minutes"]),
              int(a["duration_minutes"]), a["topic"], a.get("agenda", ""),
              a.get("task"))),

    _tool("cancel_meeting",
          "Cancel a meeting that hasn't started yet, freeing its attendees' "
          "time. Use the meeting id from view_calendar. A meeting that is "
          "already underway or finished cannot be cancelled.",
          {"meeting_id": {"type": "string"}},
          ["meeting_id"],
          lambda e, a: e.agent_cancel_meeting(a["meeting_id"])),

    _tool("talk_in_meeting",
          "Speak in the meeting you are attending right now (only works while "
          "the clock is inside a meeting you attend).",
          {"text": {"type": "string"}},
          ["text"],
          lambda e, a: e.agent_talk_in_meeting(a["text"])),

    _tool("write_doc",
          "Write a document and share it with coworkers — its content enters "
          "their context (e.g. a status summary, a plan, a decision record).",
          {"title": {"type": "string"},
           "content": {"type": "string"},
           "share_with": {"type": "array", "items": {"type": "string"}}},
          ["title", "content", "share_with"],
          lambda e, a: e.agent_write_doc(a["title"], a["content"], a["share_with"])),

    _tool("view_tasks",
          "Read the task tracker: assignments, dependencies, priorities, "
          "recorded status notes, completions.",
          {}, [], _view_tasks),

    _tool("view_inbox",
          "Read recent messages you've received.",
          {"limit": {"type": "integer", "minimum": 1}}, [], _view_inbox),

    _tool("view_calendar",
          "See the team calendar: every meeting with its time and attendees.",
          {}, [], _view_calendar),

    _tool("advance_time",
          "Let simulated time pass (you do nothing for N minutes); scheduled "
          "events fire along the way. Some things happening around you can "
          "interrupt you before the time is up.",
          {"minutes": {"type": "integer", "minimum": 0}}, ["minutes"],
          lambda e, a: _advance(e, int(a["minutes"]))),

    _tool("wait_for_reply",
          "Advance time until the next message reaches you — a chat, or an "
          "email landing in your inbox.",
          {}, [],
          lambda e, a: _wait(e)),
]

BY_NAME = {t["name"]: t for t in TOOLS}


def schemas():
    """Claude-ready tool definitions (registry minus handlers)."""
    return [{k: t[k] for k in ("name", "description", "input_schema")}
            for t in TOOLS]


def _argument_error(value, schema, path="arguments"):
    kind = schema.get("type")
    valid = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "boolean": lambda: isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
    }
    if kind in valid and not valid[kind]():
        return "%s must be %s" % (path, kind)
    if kind == "object":
        for key in schema.get("required", []):
            if key not in value:
                return "%s.%s is required" % (path, key)
        for key, item in value.items():
            if key not in schema.get("properties", {}):
                return "%s.%s is unknown" % (path, key)
            error = _argument_error(item, schema["properties"][key], path + "." + key)
            if error:
                return error
    if kind == "array":
        for i, item in enumerate(value):
            error = _argument_error(item, schema.get("items", {}), "%s[%d]" % (path, i))
            if error:
                return error
    if "enum" in schema and value not in schema["enum"]:
        return "%s must be one of %s" % (path, schema["enum"])
    if "minimum" in schema and value < schema["minimum"]:
        return "%s must be >= %s" % (path, schema["minimum"])
    if "maximum" in schema and value > schema["maximum"]:
        return "%s must be <= %s" % (path, schema["maximum"])
    return None


def call_tool(engine, name, args):
    if engine.world.clock >= getattr(engine, "agent_horizon", float("inf")):
        return {"error": "evaluation horizon reached"}
    if not isinstance(name, str) or name not in BY_NAME:
        return {"error": "unknown tool %r" % name}
    args = {} if args is None else args
    error = _argument_error(args, BY_NAME[name]["input_schema"])
    if error:
        return {"error": error}
    result = BY_NAME[name]["handler"](engine, args)
    # PUSH delivery (sim/signals.py): whatever landed on the PM unasked since
    # the last tool call rides along with this result — chat and delivered email
    # notifications both ride along.
    new = engine.drain_agent_push()
    if new:
        return {"result": result, "notifications": new}
    return result
