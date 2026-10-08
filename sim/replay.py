"""Replay: rebuild world state by folding the event log — no LLM calls.

    world_at(seq) = fold(scenario, events[:seq+1])

The only nondeterminism in a run is LLM reply text, and that text is captured
in the log, so replay is instant, free, and byte-identical.

CLI:  python -m sim.replay runs/run-XXXXXXXX-XXXXXX [until_seq]
"""

import json
import os
import sys

from .sim_time import fmt
from .world import Message, World


def load_run(run_dir):
    with open(os.path.join(run_dir, "scenario.json")) as f:
        scenario = json.load(f)
    events = []
    with open(os.path.join(run_dir, "events.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return scenario, events


def replay(scenario, events, until_seq=None):
    """Fold events into a World. Pass until_seq to time-travel to any point."""
    world = World(scenario)
    for e in events:
        if until_seq is not None and e["seq"] > until_seq:
            break
        world.advance_clock_to(max(world.clock, e["t"]))
        if e["kind"] == "message":
            world.messages.append(
                Message(e["msg_id"], e["t"], e["sender"], e["recipient"],
                        e.get("text", ""), e.get("via", "chat"),
                        e.get("group"))
            )
        elif e["kind"] == "task_added":
            world.tasks.append(e["task"])
            if not e["task"].get("contingent"):
                world.revealed_tasks.add(e["task"]["id"])
        elif e["kind"] == "task_revealed":
            world.revealed_tasks.add(e["task_id"])
        elif e["kind"] == "task_updated":
            t = world.find_task(e["task_id"])
            if t is not None:
                t.update(e["changes"])
        elif e["kind"] == "task_completed":
            world.completed_announced.add(e["task_id"])
        elif e["kind"] == "meeting_scheduled":
            m = {k: e[k] for k in ("id", "start", "end", "attendees", "topic", "agenda")}
            m["task"] = e.get("task")
            world.meetings.append(m)
        elif e["kind"] == "meeting_cancelled":
            for m in world.meetings:
                if m["id"] == e["meeting_id"]:
                    m["cancelled"] = True
        elif e["kind"] == "transcript":
            world.transcripts.append(
                {k: e[k] for k in ("meeting_id", "t", "attendees", "topic", "text")})
        elif e["kind"] == "doc_added":
            world.docs.append(
                {k: e[k] for k in ("id", "title", "content", "shared_with", "t")})
        world.log.append(e)
    return world


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dir = sys.argv[1]
    until = int(sys.argv[2]) if len(sys.argv) > 2 else None
    scenario, events = load_run(run_dir)
    world = replay(scenario, events, until)
    names = {n["id"]: n["name"] for n in scenario["npcs"]}
    names["agent"] = scenario.get("agent_name", "PM")

    print("replayed %d events -> world at %s" % (len(world.log), fmt(world.clock)))
    for m in world.messages:
        print("  [%s] %s -> %s: %s" % (
            fmt(m.time), names.get(m.sender, m.sender),
            names.get(m.recipient, m.recipient), m.text))


if __name__ == "__main__":
    main()
