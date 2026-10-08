"""Deterministic trajectory smoke tests — NO LLM / API (uses the eval StubClient).

    python -m sim.smoke [scenarios/crunch.json]

These pin the DES trajectory invariants we care about (and just fixed), so
they can't silently regress. NPC text is a stub — irrelevant to trajectory
STRUCTURE, which is what we assert. Exits non-zero if any check fails.

Checks:
  1. causality      — no undispatched event ever sits at/before the clock
  2. stops-at-push  — an interruptible advance halts AT the first push instant
  3. no-push-skip   — driving to horizon delivers EVERY push (none fired into
                      the void — the bug that let the PM sleep through Wed/Thu)
  4. determinism    — same seed -> byte-identical event trajectory on replay
  5. wait-for-reply — the yield primitive returns when a message for the PM lands
"""

import json
import os
import shutil
import sys
import tempfile

from .engine import Engine
from .eval import StubClient
from .rubric import load_rubric
from .sim_time import fmt

# smoke engines write to a throwaway dir, never pollute runs/
_SMOKE_ROOT = tempfile.mkdtemp(prefix="sim-smoke-")
_SEQ = [0]


def _engine(scenario):
    _SEQ[0] += 1
    return Engine(scenario, client=StubClient(), verbose=False,
                  run_dir=os.path.join(_SMOKE_ROOT, "e%d" % _SEQ[0]))


def _horizon(scenario):
    return load_rubric(scenario)["horizon"]


def _waking_pushes(log):
    """The (time, sender) of every push that should WAKE the PM — chat to the
    agent + delivered email. (Completions are delivered but don't wake.)"""
    out = []
    for e in log:
        if (e["kind"] == "message" and e.get("recipient") == "agent"
                and e.get("via", "chat") == "chat"):
            out.append((e["t"], e["sender"]))
        elif e["kind"] == "email_delivered":
            out.append((e["t"], e["sender"]))
    return out


def _queue_clean(eng):
    """The causality invariant: nothing undispatched at/before the clock."""
    nxt = eng.queue.peek()
    return nxt is None or nxt.time > eng.world.clock


def check_causality(scenario):
    """Drive a mixed trajectory (actions + advances); after every operation no
    pending event may sit at/before the clock."""
    eng = _engine(scenario)
    horizon = _horizon(scenario)
    npc = next(iter(eng.world.npcs))
    assert _queue_clean(eng), "dirty queue at init"
    # a burst of synchronous actions, each must drain due events
    for i in range(5):
        eng.agent_say(npc, "ping %d" % i)
        if not _queue_clean(eng):
            return False, "action %d left an event at/before clock %s" % (i, eng.world.now())
    # interruptible advances toward horizon
    steps = 0
    while eng.world.clock < horizon and steps < 500:
        eng.advance_until(horizon, interruptible=True)
        eng.drain_agent_push()
        steps += 1
        if not _queue_clean(eng):
            return False, "advance left an event at/before clock %s" % eng.world.now()
    return True, "invariant held across actions + full drive to %s" % fmt(horizon)


def check_stops_at_push(scenario):
    """An interruptible advance from t0 must halt AT the first push instant,
    not somewhere past it."""
    truth = _waking_pushes(_drive_full(scenario))
    if not truth:
        return False, "scenario has no PM-directed pushes to test"
    first_t = min(t for t, _ in truth)
    eng = _engine(scenario)
    horizon = _horizon(scenario)
    eng.advance_until(horizon, interruptible=True)
    woke = eng.drain_agent_push()
    if eng.world.clock != first_t:
        return False, ("stopped at %s, but first push is at %s"
                       % (fmt(eng.world.clock), fmt(first_t)))
    if not any("chat_from" in w or "email_from" in w for w in woke):
        return False, "halted but delivered no waking push"
    return True, "halted exactly at first push %s (%d pushes total)" % (fmt(first_t), len(truth))


def _drive_full(scenario):
    """Full non-interruptible drive to horizon -> the ground-truth log."""
    eng = _engine(scenario)
    eng.advance_until(_horizon(scenario), max_events=5000)
    return eng.world.log


def check_no_push_skipped(scenario):
    """Driving to horizon with the interruptible loop must surface EVERY push
    that a full drive produces — none skipped, none duplicated (exact
    multiset), so a repeated (time, sender) can't hide a skip."""
    from collections import Counter
    truth = Counter(_waking_pushes(_drive_full(scenario)))
    eng = _engine(scenario)
    horizon = _horizon(scenario)
    delivered, stops = Counter(), 0
    while eng.world.clock < horizon and stops < 1000:
        eng.advance_until(horizon, interruptible=True)
        got = eng.drain_agent_push()
        stops += 1
        for w in got:
            if "chat_from" in w:
                delivered[(_unfmt(scenario, w["time"]), w["chat_from"])] += 1
            elif "email_from" in w:
                delivered[(_unfmt(scenario, w["time"]), w["email_from"])] += 1
    if delivered != truth:
        return False, ("push multiset mismatch — missing=%s extra=%s"
                       % (sorted((truth - delivered).elements()),
                          sorted((delivered - truth).elements())))
    return True, "all %d pushes delivered across the week (exact, none skipped)" % sum(truth.values())


def _unfmt(scenario, timestr):
    """Recover sim-minutes from a 'Day HH:MM' stamp (for set comparison)."""
    from .sim_time import DAY_NAMES
    day, hm = timestr.split(" ")
    h, m = hm.split(":")
    return DAY_NAMES.index(day) * 1440 + int(h) * 60 + int(m)


def check_determinism(scenario):
    """Same seed + stub -> identical event trajectory (kind, time, text)."""
    def sig(log):
        return [(e["seq"], e["t"], e["kind"], e.get("text", ""),
                 e.get("task_id", ""), e.get("sender", "")) for e in log]
    a = sig(_drive_full(scenario))
    b = sig(_drive_full(scenario))
    if a != b:
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
        return False, "trajectories diverge at seq %s" % i
    return True, "%d events identical across two runs" % len(a)


def check_wait_for_reply(scenario):
    """wait_for_reply must return once a message for the PM lands."""
    eng = _engine(scenario)
    npc = next(iter(eng.world.npcs))
    eng.agent_say(npc, "are you there?")
    msg = eng.run_until_reply()
    if msg is None:
        return False, "no reply surfaced within the wait horizon"
    if msg.get("recipient") != "agent":
        return False, "returned a non-PM message"
    return True, "reply from %s surfaced at %s" % (msg.get("sender"), fmt(msg["t"]))


def check_harness_drives_to_horizon(scenario):
    """The actual run_llm loop: a no-op agent (stub returns text, never tools)
    must still be driven all the way to Friday — woken on every push, never
    hanging, never advanced past work. Guards the turn-budget removal + the
    interruptible drive directly."""
    from .harness import run_llm
    eng = _engine(scenario)
    horizon = _horizon(scenario)
    run_llm(eng)   # StubClient agent: yields every turn -> pure drive loop
    if eng.world.clock < horizon:
        return False, "run_llm returned at %s, short of horizon %s" % (
            fmt(eng.world.clock), fmt(horizon))
    if not _queue_clean(eng):
        return False, "run_llm left the queue dirty"
    return True, "no-op agent driven to %s, queue clean" % fmt(eng.world.clock)


def check_preemptive_resume(_scenario):
    """The preemptive scheduler: person x holds A (P1) and B (P2), both ready
    at t0, so x works A first. Expedite B at t0+30 working-min (A half-done):
    A must PAUSE with progress banked, B runs, then A RESUMES and completes
    (work conserved, split into segments) — and the reprioritize is
    NON-RETROACTIVE (A's progress before the switch is identical to a run
    with no reprioritize)."""
    from .tasks import compute_schedule, task_view
    START = 540  # Mon 09:00

    def mk(evs=None):
        b = {"id": "B", "title": "B", "assignees": ["x"], "effort_hours": 1.0,
             "priority": "P2", "done_hours": 0, "arrival": START}
        if evs:
            b["order_events"] = evs
        return [{"id": "A", "title": "A", "assignees": ["x"], "effort_hours": 1.0,
                 "priority": "P1", "done_hours": 0, "arrival": START}, b]

    def acc(tasks, tid, now):
        return next(r["true_done_hours"] for r in task_view(tasks, START, now)
                    if r["id"] == tid)

    plain, pre = mk(), mk([{"at": 570, "order_urgent": True}])
    sp = compute_schedule(pre, START)
    if acc(plain, "A", 560) != acc(pre, "A", 560):
        return False, "reprioritize at 570 changed A's progress at 560 — RETROACTIVE"
    if not (acc(pre, "A", 570) == acc(pre, "A", 620) > 0):
        return False, "A did not hold its banked progress while preempted"
    da = sp["A"]["done_at"]
    if da is None or acc(pre, "A", da) < 1.0 - 1e-6:
        return False, "A did not resume to completion after B"
    if len(sp["A"]["segments"]) < 2:
        return False, "A was not split into resume segments"
    return True, ("A paused at %.0f%%, B ran, A resumed to 100%% (%d segments), "
                  "non-retroactive" % (acc(pre, "A", 570) * 100, len(sp["A"]["segments"])))


def check_agent_tasks_free(_scenario):
    """Agent-CREATED tasks must never consume worker capacity — otherwise an
    agent could tank its own score by inventing fake work that eats an
    engineer's week (zero scoring weight, but real capacity cost). A real
    task's completion must be identical whether or not an agent task sits on
    the same person."""
    from .tasks import compute_schedule
    START = 540
    real = {"id": "R", "title": "R", "assignees": ["x"], "effort_hours": 2.0,
            "priority": "P0", "done_hours": 0, "arrival": START, "source": "seed"}
    junk = {"id": "J", "title": "J", "assignees": ["x"], "effort_hours": 5.0,
            "priority": "P0", "done_hours": 0, "arrival": START, "source": "agent"}
    alone = compute_schedule([dict(real)], START)
    withjunk = compute_schedule([dict(real), dict(junk)], START)
    if withjunk["R"]["done_at"] != alone["R"]["done_at"]:
        return False, ("agent task delayed the real task: R done %s vs %s"
                       % (withjunk["R"]["done_at"], alone["R"]["done_at"]))
    if "J" in withjunk:
        return False, "agent task J was scheduled (should be tracking-only)"
    return True, "agent-created work consumes zero capacity (R unaffected, J unscheduled)"


def check_skills(_scenario):
    """A specialist (higher skill multiplier on the task's tags) finishes the
    SAME task in proportionally less calendar time — and true progress still
    reaches the full effort. Values are hidden from the PM; only the physics
    are asserted here."""
    from .tasks import compute_schedule, task_view
    START = 540
    base = {"id": "T", "title": "T", "assignees": ["slow"], "effort_hours": 2.0,
            "tags": ["db"], "done_hours": 0, "arrival": START, "source": "seed",
            "priority": "P1"}
    skills = {"fast": {"db": 2.0}}     # 'slow' has no skills -> 1x
    slow = compute_schedule([dict(base)], START, skills=skills)
    fastt = dict(base, assignees=["fast"])
    fast = compute_schedule([fastt], START, skills=skills)
    slow_dur = slow["T"]["done_at"] - START
    fast_dur = fast["T"]["done_at"] - START
    if abs(fast_dur * 2 - slow_dur) > 2:
        return False, "2x skill didn't halve time (slow=%d fast=%d)" % (slow_dur, fast_dur)
    td = next(r["true_done_hours"] for r in task_view([fastt], START,
              fast["T"]["done_at"], skills=skills) if r["id"] == "T")
    if abs(td - 2.0) > 0.05:
        return False, "specialist true_done=%.2f != effort 2.0 (progress miscount)" % td
    return True, ("2x specialist finished in half the time (%d vs %d min), "
                  "progress exact" % (fast_dur, slow_dur))


def _mig(scenario):
    """A swarmable target from the scenario: the highest-effort still-open seed
    task with a worker owner, plus a DISTINCT worker to help. Scenario-agnostic
    so the meeting checks run on any authored instance, not just crunch."""
    workers = {n["id"] for n in scenario.get("npcs", []) if n.get("worker", True)}
    tasks = scenario.get("project", {}).get("tasks", [])
    cand = sorted(
        (t for t in tasks
         if t.get("effort_hours", 0) - t.get("done_hours", 0) > 1
         and (t.get("assignees") or [None])[0] in workers),
        key=lambda t: -(t["effort_hours"] - t.get("done_hours", 0)))
    if not cand:
        return "migration", "sarah", "dave"   # legacy fallback
    tid = cand[0]["id"]
    owner = cand[0]["assignees"][0]
    helper = next((w for w in workers if w != owner), owner)
    return tid, owner, helper


def _answer_open_questions(eng):
    """Simulate a reasonable PM promptly answering every blocking question, so
    a projection can test an unrelated mechanic without a question stall."""
    for t in eng.world.tasks:
        for q in t.get("questions") or []:
            if q.get("gates"):
                eng.world.record("question_answered", task_id=t["id"],
                                 question_id=q["id"], choice="proceed",
                                 rationale="smoke", via="chat",
                                 owner=(t.get("assignees") or [None])[0],
                                 delivered_at=q["at"])


def check_meeting_swarm(scenario):
    """A meeting LABELLED to a task, with the task's owner + a helper in the
    room, SWARMS it: it banks work-minutes x swarm_rate at the meeting's end,
    finishing the task (and unblocking its successor) EARLIER than no meeting.
    Guards the full world wiring: add_meeting(task=) -> meeting_deposits ->
    busy -> task_view. Also asserts the two anti-cheese physics:
      - owner NOT in the room -> zero deposit (you can't swarm without the doer)
      - progress never exceeds the task's effort (no over-deposit past 100%)."""
    tid, owner, helper = _mig(scenario)
    horizon = _horizon(scenario)

    def project(setup):
        eng = _engine(scenario)
        t = eng.world.find_task(tid)
        if t is None:
            return None
        _answer_open_questions(eng)   # isolate the swarm from any question stall
        setup(eng)
        rows = {r["id"]: r for r in eng.world.tasks_view(at=horizon)}
        return rows

    base = project(lambda e: None)
    if base is None or tid not in base:
        return False, "scenario has no %r task to swarm" % tid
    succ = next((r["id"] for r in base.values()
                 if tid in (r.get("blocked_by") or [])), None)

    at = 2220  # Tue 13:00 — a 150-min block using the remaining daily cap
    swarmed = project(lambda e: e.agent_schedule_meeting(
        [owner, helper], at, 150, "swarm", task=tid))
    # owner absent: same room minus the doer -> must bank nothing
    absent = project(lambda e: e.agent_schedule_meeting(
        [helper], at, 150, "no-doer", task=tid))

    b_done, s_done = base[tid]["projected_done"], swarmed[tid]["projected_done"]
    if not (s_done and b_done and s_done < b_done):
        return False, ("swarm didn't speed %s (base done %s, swarm done %s)"
                       % (tid, b_done, s_done))
    if absent[tid]["projected_done"] != b_done:
        return False, "owner-absent meeting still deposited work (cheese!)"
    if swarmed[tid]["true_done_hours"] > base[tid]["effort_hours"] + 1e-6:
        return False, "swarm over-deposited past effort (progress > 100%)"
    gain = ""
    if succ and base[succ]["pct"] is not None:
        if swarmed[succ]["pct"] < base[succ]["pct"]:
            return False, "swarm hurt the successor task"
        gain = " (successor %s %d%%->%d%%)" % (succ, base[succ]["pct"],
                                               swarmed[succ]["pct"])
    return True, ("owner+helper swarm finished %s earlier (%s vs %s)%s; "
                  "owner-absent room banked nothing"
                  % (tid, fmt(s_done), fmt(b_done), gain))


def check_meeting_ops(scenario):
    """Meeting lifecycle rules the agent could try to game:
      - a future meeting can be cancelled (frees the room, voids the swarm)
      - a meeting already STARTED/ended cannot be cancelled (world physics)
      - no double-booking one person into overlapping meetings
      - the per-day meeting-minute cap is enforced."""
    tid, owner, helper = _mig(scenario)
    eng = _engine(scenario)
    now = eng.world.clock
    horizon = _horizon(scenario)

    # a future swarm, then cancel it -> the task projection reverts to no-swarm
    base = {r["id"]: r for r in eng.world.tasks_view(at=horizon)}
    m = eng.agent_schedule_meeting([owner, helper], 2220, 150, "swarm", task=tid)
    if "id" not in m:
        return False, "scheduling a swarm failed: %s" % m
    if eng.world.tasks_view(at=horizon)[0] is None:
        return False, "tasks_view broke after scheduling"
    cancelled = eng.agent_cancel_meeting(m["id"])
    if "cancelled" not in cancelled:
        return False, "future meeting refused to cancel: %s" % cancelled
    after = {r["id"]: r for r in eng.world.tasks_view(at=horizon)}
    if after[tid]["projected_done"] != base[tid]["projected_done"]:
        return False, "cancel didn't void the swarm deposit"

    # can't cancel a meeting that has already started
    started = eng.agent_schedule_meeting([owner], now + 5, 60, "soon")
    eng.advance_until(started["start"] + 1, interruptible=False)
    r = eng.agent_cancel_meeting(started["id"])
    if "error" not in r:
        return False, "cancelled a meeting that had already started"

    # double-booking one person into overlapping meetings is rejected
    eng2 = _engine(scenario)
    eng2.agent_schedule_meeting([owner], 2220, 120, "a")
    dbl = eng2.agent_schedule_meeting([owner], 2220 + 60, 120, "b")
    if "error" not in dbl:
        return False, "allowed a double-booking overlap"

    # daily cap: pile back-to-back same-day meetings on one person past the cap
    eng3 = _engine(scenario)
    cap = scenario.get("costs", {}).get("max_meeting_minutes_per_day", 180)
    ok_min = 0
    capped = None
    for k in range(12):
        r = eng3.agent_schedule_meeting([helper], 2220 + k * 60, 60, "m%d" % k)
        if "error" in r:
            capped = r
            break
        ok_min += 60
    if capped is None:
        return False, "daily meeting cap never fired"
    return True, ("cancel future-only + no double-book + daily cap (%d/%d min) "
                  "all enforced" % (ok_min, cap))


def check_parallel_work(_scenario):
    """Multi-assignee physics + the meet premium + a REAL tradeoff (no dominant
    move): solo < co-assigned parallel < meeting-swarm on wall-clock; co-assign
    is SUBLINEAR (wastes capacity to coordination); and pulling a helper onto a
    task delays the helper's OWN task."""
    from .tasks import compute_schedule, parallel_rate, swarm_rate
    START = 540
    sk = {"a": {}, "b": {}, "c": {}}

    def mk(assignees, tid="T", pri="P1"):
        return {"id": tid, "title": tid, "assignees": list(assignees),
                "effort_hours": 6.0, "done_hours": 0, "arrival": START,
                "source": "seed", "priority": pri, "tags": ["x"]}

    def done(tasks, tid):
        d = compute_schedule(tasks, START, {}, sk, {})[tid]["done_at"]
        return (d - START) if d else None

    solo = done([mk(["a"])], "T")
    duo = done([mk(["a", "b"])], "T")
    trio = done([mk(["a", "b", "c"])], "T")
    # ladder: more hands -> sooner, but strictly diminishing (never linear)
    if not (solo > duo > trio):
        return False, "more hands didn't finish sooner (solo=%s duo=%s trio=%s)" % (
            solo, duo, trio)
    if duo <= solo / 2 + 1:
        return False, "2 people ~halved the time — parallelism isn't damped (duo=%s)" % duo
    # meeting beats uncoordinated parallel for the same people
    pr = parallel_rate({"tags": ["x"]}, ["a", "b"], sk)
    sr = swarm_rate({"tags": ["x"]}, ["a", "b"], sk)
    if not sr > pr:
        return False, "meeting (%.2f) not faster than parallel (%.2f)" % (sr, pr)
    # TRADEOFF: b helping on T delays b's own task U
    base = [mk(["a"], "T"), mk(["b"], "U")]
    helped = [mk(["a", "b"], "T"), mk(["b"], "U")]
    if not (done(helped, "T") < done(base, "T")
            and done(helped, "U") > done(base, "U")):
        return False, "no tradeoff — helping T didn't cost helper's own task U"
    return True, ("solo %d > duo %d > trio %d min (damped); meeting %.2f>%.2f "
                  "parallel; helping T delays helper's U (real tradeoff)"
                  % (solo, duo, trio, sr, pr))


def check_blocking_question(scenario):
    """A gating question SUSPENDS its owner's task from open-time until an
    explicit decision is DELIVERED — and the CHANNEL sets the latency: chat
    unblocks now, email only at the next answer-batch tick. Asserts the full
    derived path (World.question_answers -> scheduler gate):
      - never answered  -> task frozen at its open-time progress
      - chat answer      -> resumes immediately (most progress by Friday)
      - email answer     -> resumes later than chat (strictly less progress)
      - a WRONG question id doesn't resolve it (explicit-decision scoped)."""
    # locate a gating question in the scenario
    found = None
    for t in scenario.get("project", {}).get("tasks", []):
        for q in t.get("questions") or []:
            if q.get("gates"):
                found = (t["id"], q["id"], (t.get("assignees") or [None])[0], q["at"])
                break
        if found:
            break
    if not found:
        return False, "scenario has no gating question to test"
    tid, qid, owner, at = found
    horizon = _horizon(scenario)
    batch = scenario.get("answer_batch_minutes", 120)

    email_lands = (at // batch + 1) * batch   # the async cost: next grid tick

    def run(reply_via):
        eng = _engine(scenario)
        if reply_via in ("chat", "email"):
            eng.world.record("question_answered", task_id=tid,
                             question_id=qid, choice="proceed",
                             rationale="smoke", via=reply_via, owner=owner,
                             delivered_at=(at if reply_via == "chat"
                                           else email_lands))
        elif reply_via == "wrong":
            eng.world.record("question_answered", task_id=tid,
                             question_id="different-question", choice="proceed",
                             rationale="smoke", via="chat", owner=owner,
                             delivered_at=at)
        def at_t(when, key):
            return {x["id"]: x for x in eng.world.tasks_view(at=when)}[tid][key]
        return {
            "probe_status": at_t(at + 20, "status"),      # just after open
            "mid": at_t(email_lands, "true_done_hours"),  # email's unblock instant
            "fri": at_t(horizon, "true_done_hours"),      # end of week
        }

    never = run(None)
    chat = run("chat")
    email = run("email")
    wrong = run("wrong")

    # just after the open-time: chat has already unblocked, email/never haven't
    if not (chat["probe_status"] == "in_progress"
            and email["probe_status"] == "blocked"
            and never["probe_status"] == "blocked"):
        return False, ("channel latency not modelled — probe statuses "
                       "chat=%s email=%s never=%s"
                       % (chat["probe_status"], email["probe_status"],
                          never["probe_status"]))
    # at the instant email finally lands, chat's head-start is real progress
    if not chat["mid"] > email["mid"] + 1e-6:
        return False, ("chat's earlier unblock bought no progress (mid chat=%.1f "
                       "email=%.1f)" % (chat["mid"], email["mid"]))
    # answering at all beats never (the stall is genuine lost work)
    if not (chat["fri"] > never["fri"] + 1e-6 and email["fri"] > never["fri"] + 1e-6):
        return False, ("stall not lost work — Fri never=%.1f email=%.1f chat=%.1f"
                       % (never["fri"], email["fri"], chat["fri"]))
    if abs(wrong["fri"] - never["fri"]) > 1e-6:
        return False, "answer for the wrong question resolved the block"
    return True, ("chat unblocks now vs email at +%dmin (head-start %.1f>%.1f h) "
                  "vs never-stalled (Fri +%.1f h over never)"
                  % (email_lands - at, chat["mid"], email["mid"],
                     chat["fri"] - never["fri"]))


def helper_of(scenario, owner):
    """Any worker who is NOT the question owner (for the wrong-recipient test)."""
    for n in scenario.get("npcs", []):
        if n.get("worker", True) and n["id"] != owner:
            return n["id"]
    return "nobody"


CHECKS = [
    ("causality", check_causality),
    ("stops-at-push", check_stops_at_push),
    ("no-push-skipped", check_no_push_skipped),
    ("determinism", check_determinism),
    ("wait-for-reply", check_wait_for_reply),
    ("preemptive-resume", check_preemptive_resume),
    ("agent-tasks-free", check_agent_tasks_free),
    ("skills", check_skills),
    ("meeting-swarm", check_meeting_swarm),
    ("meeting-ops", check_meeting_ops),
    ("parallel-work", check_parallel_work),
    ("blocking-question", check_blocking_question),
    ("harness-drive", check_harness_drives_to_horizon),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/crunch.json"
    with open(path) as f:
        scenario = json.load(f)
    print("trajectory smoke: %s\n" % path)
    failed = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn(scenario)
        except Exception as e:
            ok, detail = False, "EXCEPTION %s: %s" % (type(e).__name__, e)
        print("  [%s] %-16s %s" % ("PASS" if ok else "FAIL", name, detail))
        failed += not ok
    print("\n%s (%d/%d passed)"
          % ("ALL PASS" if not failed else "FAILURES", len(CHECKS) - failed, len(CHECKS)))
    shutil.rmtree(_SMOKE_ROOT, ignore_errors=True)   # never pollute runs/
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
