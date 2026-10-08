"""Agent harness — Mode 0 ("reactive").

Runs a PM agent against the environment through the SAME tool registry the
UI/API uses, records the full run (events.jsonl / llm.jsonl / agent turns),
computes navigation metrics, and scores the run with sim.eval.

    python -m sim.harness scenarios/demo.json --probe scripted
    python -m sim.harness scenarios/demo.json --probe llm [--max-turns 30]

Probes:
  scripted  deterministic tool sequence, zero agent tokens — the floor above
            the null baseline; exercises every tool surface end-to-end.
  llm       Claude (claude-opus-4-8) plays the PM via tool use. The agent is
            NOT shown the evaluation checks — only the job.
"""

import argparse
import json
import os
import sys

import anthropic

from .cli import load_env
from .deterministic_npc import DeterministicNPCClient
from .engine import Engine
from .identity import (benchmark_identity, normalize_devin_model,
                       with_resolved_model)
from .sim_time import fmt, wall_now
from .rubric import load_rubric
from .tools import call_tool, schemas


def _horizon(engine):
    return load_rubric(engine.world.scenario, engine.run_dir)["horizon"]

DEFAULT_AGENT_MODEL = "claude-opus-4-8"
MAX_RESULT_CHARS = 2000


def _agent_thinking(model):
    """Adaptive where supported; Haiku 4.5 has no adaptive thinking."""
    if model.startswith("claude-haiku"):
        return {}
    return {"thinking": {"type": "adaptive"}}


# ---------------------------------------------------------------------------
# scripted probe: a competent-but-mechanical PM
# ---------------------------------------------------------------------------

SCRIPT = [
    ("view_tasks", {}),
    ("send_chat", {"npc": "sarah",
                   "text": "Hi Sarah, Alex here. Before I trust the tracker: "
                           "what's the honest migration status — real hours left?"}),
    ("wait_for_reply", {}),
    ("send_chat", {"npc": "dave",
                   "text": "Hey Dave — anything about the auth handoff or your "
                           "workload I should know going into this week?"}),
    ("wait_for_reply", {}),
    ("send_chat", {"npc": "priya",
                   "text": "Priya — early heads up: migration is behind the tracker "
                           "number (real figure ~55%), so Friday has delivery risk. "
                           "I'm building a recovery plan today and will keep you posted."}),
    ("update_tracker_note", {"task_id": "migration", "note": "~55% actual; PM verifying recovery plan"}),
    ("forecast_completion", {"task_id": "migration",
                             "p50_completion_time": 6720,
                             "p90_completion_time": 7440,
                             "rationale": "Initial plan; tracker is probably optimistic"}),
    ("forecast_completion", {"task_id": "data-backfill",
                             "p50_completion_time": 5280,
                             "p90_completion_time": 6000,
                             "rationale": "Depends on migration clearing first"}),
    ("forecast_completion", {"task_id": "rate-limiting",
                             "p50_completion_time": 6720,
                             "p90_completion_time": 7440,
                             "rationale": "Lower-priority work yields to launch asks"}),
    ("advance_to", {"t": 2160}),                # -> Tue ~12:00, past SSO arrival
    ("answer_question", {"task_id": "migration",
                         "question_id": "cutover-call",
                         "choice": "online-zero-downtime",
                         "rationale": "Protect customer availability during cutover",
                         "via": "chat"}),
    ("view_inbox", {}),
    # arrivals come as tickets in chat/email — file first, then assign
    ("add_task", {"title": "SSO login failures", "id": "sso-incident"}),
    ("assign_task", {"task_id": "sso-incident", "npc": "dave"}),
    ("forecast_completion", {"task_id": "sso-incident",
                             "p50_completion_time": 3600,
                             "p90_completion_time": 4320,
                             "rationale": "Urgent incident routed to the auth owner"}),
    ("advance_to", {"t": 3840}),                # -> Wed ~16:00, past questionnaire
    ("forecast_completion", {"task_id": "migration",
                             "p50_completion_time": 7200,
                             "p90_completion_time": 8160,
                             "rationale": "Owner reported deeper schema and index work"}),
    ("view_tasks", {}),
    ("add_task", {"title": "Meridian security questionnaire", "id": "security-questionnaire"}),
    ("assign_task", {"task_id": "security-questionnaire", "npc": "dave"}),
    ("forecast_completion", {"task_id": "security-questionnaire",
                             "p50_completion_time": 5340,
                             "p90_completion_time": 6060,
                             "rationale": "Four-hour ask queued behind the incident"}),
    ("send_chat", {"npc": "priya",
                   "text": "Update: SSO incident owned by Dave (done tomorrow), "
                           "questionnaire assigned, backfill on track for Thursday."}),
    ("advance_to", {"t": 4920}),                # -> Thu ~10:00; email delivered ~09:30
    ("add_task", {"title": "Launch readiness confirmations", "id": "readiness-signoff"}),
    ("assign_task", {"task_id": "readiness-signoff", "npc": "sarah"}),
    ("forecast_completion", {"task_id": "readiness-signoff",
                             "p50_completion_time": 6660,
                             "p90_completion_time": 7380,
                             "rationale": "Small but late engineering deliverable"}),
    ("advance_to", {"t": 6780}),                # -> horizon
]


def run_scripted(engine):
    turns = []
    for name, args in SCRIPT:
        if name == "advance_to":
            # advances are interruptible now (people wake you) — keep going
            while engine.world.clock < args["t"]:
                call_tool(engine, "advance_time",
                          {"minutes": args["t"] - engine.world.clock})
            turns.append({"tool": name, "args": args, "ok": True})
            continue
        if name == "answer_question":
            task = engine.world.find_task(args["task_id"])
            question_exists = task is not None and any(
                q.get("gates") and q.get("id") == args["question_id"]
                for q in task.get("questions") or [])
            if not question_exists:
                continue
        result = call_tool(engine, name, args)
        turns.append({"tool": name, "args": args,
                      "ok": not (isinstance(result, dict) and "error" in result)})
        if engine.world.clock >= _horizon(engine):
            break
    return turns


# ---------------------------------------------------------------------------
# LLM probe: Claude plays the PM
# ---------------------------------------------------------------------------

AGENT_PROMPT_PATH = os.path.join(os.path.dirname(__file__),
                                 "prompts", "agent_system.txt")


def agent_system_prompt(scenario):
    """The PM system prompt lives in sim/prompts/agent_system.txt (a clean,
    editable file) with {agent}/{company}/{roster}/{project}/{prio} slots the
    scenario fills. Kept out of code so it's easy to iterate and diff."""
    roster = "\n".join("- %s: %s — %s" % (n["id"], n["name"], n["role"])
                       for n in scenario["npcs"])
    project = scenario.get("project", {})
    horizon = load_rubric(scenario)["horizon"]
    start = scenario.get("start_time", 545)
    with open(AGENT_PROMPT_PATH, encoding="utf-8") as f:
        template = f.read()
    return template.format(
        agent=scenario.get("agent_name", "the PM"),
        company=scenario.get("company", "the company"),
        roster=roster, project=project.get("name", "(project)"),
        prio=project.get("priority", "-"),
        due=fmt(project.get("due", horizon)), start=fmt(start),
        horizon=fmt(horizon), start_minute=start,
        horizon_minute=horizon).strip()


SAFETY_TURNS = 200   # runaway guard ONLY (a PM that never advances time) —
                     # NOT a budget the agent should ration. The real pacing
                     # constraint is sim-time: the loop drives to Friday.


def run_llm(engine, max_turns=SAFETY_TURNS, model=DEFAULT_AGENT_MODEL):
    client = engine.client
    horizon = _horizon(engine)
    tools = schemas()
    system = agent_system_prompt(engine.world.scenario)
    messages = [{"role": "user", "content":
                 "It is %s. The week starts now — over to you."
                 % engine.world.now()}]
    turns = []

    # PROMPT CACHING (cost only — cached tokens yield identical outputs, so
    # determinism / per-policy reward are untouched). The system prompt and the
    # whole tool list are a large static prefix, cached once; and because the
    # transcript is APPEND-ONLY, a moving breakpoint on the latest message lets
    # each turn re-read the growing conversation from cache (~0.1x) instead of
    # re-billing it in full. On a 158-turn week that's ~78% off the input bill.
    ck = {"type": "ephemeral"}
    system_blocks = [{"type": "text", "text": system, "cache_control": ck}]
    tools_cached = [dict(t) for t in tools]
    if tools_cached:
        tools_cached[-1] = {**tools_cached[-1], "cache_control": ck}

    def _cache_last():
        """Put ONE ephemeral breakpoint on the last message (clearing prior
        message breakpoints so we stay under the 4-block cap with system+tools),
        caching the entire conversation prefix. Skips assistant turns whose
        content is raw SDK blocks (rare: a yield with no push) — those still get
        the system+tools cache."""
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict):
                        b.pop("cache_control", None)
        c = messages[-1]["content"]
        if isinstance(c, str):
            messages[-1]["content"] = [{"type": "text", "text": c, "cache_control": ck}]
        elif isinstance(c, list) and c and isinstance(c[-1], dict):
            c[-1] = {**c[-1], "cache_control": ck}

    # THE TRANSCRIPT: the agent's whole conversation in OpenAI chat format
    # (system / user / assistant+tool_calls / tool), one JSON line per
    # message, UNTRUNCATED — the model may see clipped tool results
    # (MAX_RESULT_CHARS), but the evaluation record keeps everything.
    tpath = os.path.join(engine.run_dir, "transcript.jsonl")

    def tlog(entry):
        entry["sim_t"] = engine.world.clock
        entry["sim_t_fmt"] = engine.world.now()
        with open(tpath, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    tlog({"role": "system", "content": system})
    tlog({"role": "user", "content": messages[0]["content"]})

    def do_turn():
        """One agent LLM call. Executes its tool calls (their push
        notifications ride back in each tool_result). Returns True if the
        agent acted, False if it emitted no tool calls (a yield)."""
        t0 = wall_now()
        _cache_last()
        try:
            response = client.messages.create(
                model=model, max_tokens=8000, system=system_blocks,
                tools=tools_cached, messages=messages, **_agent_thinking(model))
        except anthropic.APIError as e:
            print("agent LLM error (%s) — ending run gracefully" % type(e).__name__)
            engine.agent_run_status = "runner_failed"
            return None
        engine.world.store.append_llm({
            "wall_ts": wall_now(), "sim_t": engine.world.clock,
            "sim_t_fmt": engine.world.now(), "npc": "AGENT", "kind": "agent_turn",
            "model": model, "latency_ms": int((wall_now() - t0) * 1000),
            "stop_reason": response.stop_reason,
            "usage": {"input_tokens": response.usage.input_tokens,
                      "output_tokens": response.usage.output_tokens,
                      "cache_creation_input_tokens":
                          getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                      "cache_read_input_tokens":
                          getattr(response.usage, "cache_read_input_tokens", 0) or 0},
            "response_text": "".join(getattr(b, "text", "") for b in response.content
                                     if b.type == "text"),
            "tool_calls": [{"name": b.name, "input": b.input}
                           for b in response.content if b.type == "tool_use"]})
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        tlog({"role": "assistant",
              "thinking": "".join(getattr(b, "thinking", "")
                                  for b in response.content
                                  if b.type == "thinking") or None,
              "content": "".join(getattr(b, "text", "")
                                 for b in response.content
                                 if b.type == "text"),
              "tool_calls": [{"id": b.id, "name": b.name, "arguments": b.input}
                             for b in tool_uses]})
        if not tool_uses:
            safe = [b for b in response.content if b.type in ("text", "thinking")]
            if safe:
                messages.append({"role": "assistant", "content": safe})
            return False
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for tu in tool_uses:
            result = call_tool(engine, tu.name, tu.input)
            turns.append({"tool": tu.name, "args": tu.input,
                          "ok": not (isinstance(result, dict) and "error" in result)})
            # transcript keeps the FULL output; the model sees a clipped copy
            tlog({"role": "tool", "tool_call_id": tu.id, "name": tu.name,
                  "content": result})
            payload = json.dumps(result, default=str)
            if len(payload) > MAX_RESULT_CHARS:
                payload = payload[:MAX_RESULT_CHARS] + "…(truncated)"
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": payload})
        messages.append({"role": "user", "content": results})
        return True

    def deliver_push(woke):
        text = ("It is %s — something reached you while you were "
                "waiting:\n%s\nThe week is not over. Handle it with "
                "your tools (remember: only add_task / assign_task / "
                "etc. change outcomes — talking does not). When "
                "genuinely nothing is left, stop."
                % (engine.world.now(), json.dumps(woke, default=str)))
        tlog({"role": "user", "content": text})
        messages.append({"role": "user", "content": text})

    # -- drive the whole week in ONE loop: act -> (on yield) roll time forward
    # interruptibly -> a push wakes the PM -> act, until Friday. No turn
    # budget: sim-time is the constraint. `max_turns` is only a runaway guard
    # for a PM that acts forever without ever advancing the clock.
    for _ in range(max_turns):
        if engine.world.clock >= horizon:
            break
        acted = do_turn()
        if acted is None:      # agent-side API failure — end gracefully
            break
        if acted:
            continue
        # the agent yielded (no tool calls): roll time forward interruptibly.
        # advance_until returns either at Friday, or the instant a push lands.
        engine.advance_until(horizon, interruptible=True)
        woke = engine.drain_agent_push()
        if engine.world.clock >= horizon:
            break
        if woke:               # a push woke the PM before Friday — hand it over
            deliver_push(woke)
    else:
        print("harness: hit SAFETY_TURNS=%d before Friday — agent never "
              "advanced the clock?" % max_turns)

    if engine.world.clock < horizon:
        engine.advance_until(horizon, max_events=1000)
    return turns


def run_devin(engine, max_turns=SAFETY_TURNS, model=None, timeout=900):
    """Run Devin CLI as a black-box PM against a localhost tool gateway."""
    from .devin_adapter import run_devin as run
    return run(engine, agent_system_prompt(engine.world.scenario),
               max_turns=max_turns, model=model, timeout=timeout)


# ---------------------------------------------------------------------------
# navigation metrics: HOW did the agent move through the environment?
# ---------------------------------------------------------------------------

def navigation_metrics(engine, turns):
    log = engine.world.log
    horizon = _horizon(engine)
    start = engine.world.start_time
    by_tool = {}
    for t in turns:
        by_tool[t["tool"]] = by_tool.get(t["tool"], 0) + 1
    msgs = [e for e in log if e["kind"] == "message" and e["sender"] == "agent"]
    per_npc = {}
    for m in msgs:
        per_npc[m["recipient"]] = per_npc.get(m["recipient"], 0) + 1
    replies = [e for e in log if e["kind"] == "message" and e["recipient"] == "agent"]
    yields = by_tool.get("advance_time", 0) + by_tool.get("wait_for_reply", 0)
    acts = len(turns) - yields
    return {
        "turns": len(turns),
        "invalid_tool_calls": sum(1 for t in turns if not t["ok"]),
        "tool_mix": by_tool,
        "act_vs_yield": {"actions": acts, "yields": yields},
        "channel_mix": {
            "chat": sum(1 for m in msgs if m.get("via", "chat") == "chat"),
            "email": sum(1 for m in msgs if m.get("via") == "email"),
            "meetings_booked": sum(1 for e in log if e["kind"] == "meeting_scheduled"),
            "docs_written": sum(1 for e in log if e["kind"] == "doc_added"),
        },
        "contact_per_npc": per_npc,
        "replies_received": len(replies),
        "task_mutations": sum(1 for e in log
                              if e["kind"] in ("task_updated", "task_added")
                              and e.get("source") == "agent"),
        "sim_time_covered_pct": round(
            100.0 * min(1.0, (engine.world.clock - start) / (horizon - start)), 1),
        "final_clock": fmt(engine.world.clock),
    }


def _agent_model_labels(probe, model, devin_model):
    agent_model = (model if probe == "llm" else
                   ("devin/%s" % (devin_model or "default")
                    if probe == "devin" else "scripted"))
    return agent_model, None if probe == "devin" else agent_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", nargs="?", default="scenarios/demo.json")
    parser.add_argument("--probe", choices=["scripted", "llm", "devin"],
                        default="scripted")
    parser.add_argument("--max-turns", type=int, default=SAFETY_TURNS,
                        help="runaway guard on agent turns (NOT a budget); the "
                             "loop drives to Friday by sim-time")
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--devin-model",
                        help="Devin CLI model (omit to use its configured default)")
    parser.add_argument("--devin-timeout", type=int, default=900,
                        help="wall-clock timeout for a Devin CLI rollout")
    parser.add_argument("--npc-mode", choices=["anthropic", "deterministic"],
                        help="coworker voice backend (Devin defaults to deterministic)")
    args = parser.parse_args()

    load_env()
    with open(args.scenario) as f:
        scenario = json.load(f)

    npc_mode = args.npc_mode or (
        "deterministic" if args.probe == "devin" else "anthropic")
    npc_client = (DeterministicNPCClient()
                  if npc_mode == "deterministic" else None)
    engine = Engine(scenario, client=npc_client, verbose=True)
    agent_model, resolved_model = _agent_model_labels(
        args.probe, args.model, args.devin_model)
    rubric = load_rubric(scenario, engine.run_dir)
    identity = benchmark_identity(scenario, rubric, npc_mode, agent_model)
    with open(engine.run_dir + "/meta.json", "w") as f:
        # task identity = the SCENARIO, not the project: scenario variants
        # (demo_1/demo/demo_2) share a project id but are different tasks
        task = os.path.splitext(os.path.basename(args.scenario))[0]
        json.dump({"probe": args.probe,
                   "agent_model": agent_model,
                   "resolved_model": resolved_model,
                   "agent_runner": args.probe,
                   "npc_mode": npc_mode,
                   "task": task,
                   "benchmark_identity": identity,
                   "rollout_status": "running"},
                  f)
    selected = (args.devin_model or "default"
                if args.probe == "devin" else args.model)
    print("harness probe=%s model=%s run=%s\n"
          % (args.probe, selected, engine.run_dir))

    if args.probe == "scripted":
        turns = run_scripted(engine)
    elif args.probe == "llm":
        turns = run_llm(engine, args.max_turns, args.model)
    else:
        turns = run_devin(engine, args.max_turns, args.devin_model,
                          args.devin_timeout)

    with open(engine.run_dir + "/meta.json") as f:
        meta = json.load(f)
    meta["rollout_status"] = getattr(engine, "agent_run_status", "completed")
    if getattr(engine, "agent_run_failure", None):
        meta["rollout_failure"] = engine.agent_run_failure
    if args.probe == "devin":
        devin_path = os.path.join(engine.run_dir, "devin.json")
        resolved = None
        if os.path.exists(devin_path):
            try:
                with open(devin_path) as f:
                    resolved = json.load(f).get("resolved_model")
            except (OSError, ValueError):
                resolved = None
        if resolved:
            agent_model = normalize_devin_model(resolved)
            meta["resolved_model"] = agent_model
            meta["benchmark_identity"] = with_resolved_model(
                meta["benchmark_identity"], agent_model)
    with open(engine.run_dir + "/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    nav = navigation_metrics(engine, turns)
    with open(engine.run_dir + "/navigation.json", "w") as f:
        json.dump(nav, f, indent=2)

    print("\n--- navigation ---")
    print(json.dumps(nav, indent=2))

    if meta["rollout_status"] != "completed":
        engine.world.store.close()
        print("\nRUNNER FAILED: %s — run is not scoreable"
              % meta.get("rollout_failure", "unknown"))
        raise SystemExit(2)

    from .eval import evaluate, print_scorecard
    result = evaluate(engine.run_dir)
    print()
    print_scorecard(result)
    with open(engine.run_dir + "/scorecard.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
