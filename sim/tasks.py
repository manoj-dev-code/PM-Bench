"""Project/task world model: analytic work-calendar scheduling.

Task progress is a PURE FUNCTION of (task list, logged mutations, sim time) —
no ticks. Tasks arrive three ways: seeded in the scenario (arrival = sim
start), injected on a schedule by external beats (OOD arrivals), or added by
the agent for tracking. Every arrival is a logged `task_added` mutation, so
rewind/replay reconstruct the board at any time.

Scheduling semantics (deterministic, PREEMPTIVE, work-conserving):
- Each assignee works one task at a time, only during working hours
  (Mon-Fri 09:00-17:30).
- A task is READY once it has arrived, somebody holds it
  (max(arrival, assigned_at)), and its blockers are done.
- At every instant each person works their highest-priority ready task
  (urgent -> priority -> creation order, with the PM's timestamped
  `order_events` applied forward-only). When that changes — a higher-priority
  task arrives or unblocks, or a reprioritization fires — they SWITCH; the
  paused task keeps its accrued work and resumes exactly where it stopped.
- Tasks without an assignee or effort are "tracking" items: visible on the
  board, never scheduled.
"""

import math

from .sim_time import MIN_PER_DAY, WORK_END, WORK_START


def _day_segments(day, busy):
    """Available working segments on one day, minus busy (meeting) intervals."""
    if day % 7 >= 5:
        return []
    segs = [(day * MIN_PER_DAY + WORK_START, day * MIN_PER_DAY + WORK_END)]
    for bs, be in busy:
        nxt = []
        for s, e in segs:
            if be <= s or bs >= e:
                nxt.append((s, e))
                continue
            if bs > s:
                nxt.append((s, bs))
            if be < e:
                nxt.append((be, e))
        segs = nxt
    return segs


def work_minutes_between(a, b, busy=()):
    """Available working minutes (Mon-Fri 09:00-17:30, minus busy) in [a, b)."""
    if b <= a:
        return 0
    total = 0
    day = a // MIN_PER_DAY
    while day * MIN_PER_DAY < b:
        for s, e in _day_segments(day, busy):
            total += max(0, min(b, e) - max(a, s))
        day += 1
    return total


def add_work_minutes(t, minutes, busy=()):
    """Earliest time u >= t with work_minutes_between(t, u, busy) == minutes."""
    remaining = minutes
    cur = t
    day = cur // MIN_PER_DAY
    while True:
        for s, e in _day_segments(day, busy):
            start = max(cur, s)
            if start < e:
                if remaining <= e - start:
                    return start + remaining
                remaining -= e - start
        day += 1
        cur = day * MIN_PER_DAY


_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _order_at(t, clock):
    """The scheduling order (urgent-rank, priority-rank) in effect for task t
    at time `clock`. Authored `urgent`/`priority` are the defaults; the PM's
    reprioritizations are TIMESTAMPED events (`order_events`, appended by the
    reprioritize tool) applied FORWARD-ONLY — an event at time T changes the
    order only for clock >= T, never retroactively. Scoring weight always comes
    from the authored `priority`, so the PM reorders work but cannot relabel a
    task to mint value."""
    urgent, prio = t.get("urgent"), t.get("priority")
    for ev in t.get("order_events") or []:
        if ev.get("at", 0) <= clock:
            if "order_urgent" in ev:
                urgent = ev["order_urgent"]
            if "order_priority" in ev:
                prio = ev["order_priority"]
    return (0 if urgent else 1, _PRIORITY_RANK.get(prio, 2))


def _earliest(t, sim_start):
    """Work can't start before the task EXISTS (arrival) or before somebody
    actually HOLDS it (assigned_at, stamped by assignment mutations). Without
    the latter, assignment is retroactive — picking a task up at 16:00 would
    be credited work since its 11:30 arrival, and acting early earns nothing."""
    return max(t.get("arrival", sim_start), t.get("assigned_at") or 0)


def _primary(t):
    return t["assignees"][0] if t.get("assignees") else None


def _schedulable(t):
    # Agent-CREATED tasks are tracking-only: they never consume worker
    # capacity. Symmetric with scoring (agent tasks carry zero weight) — an
    # agent can neither mint value NOR destroy it by inventing fake work that
    # eats an engineer's week. Real work reaches the board as seed tasks or
    # FILED external tickets (source stays "external"); only invented
    # source=="agent" items are excluded.
    return (t.get("assignees") and t.get("effort_hours") is not None
            and t.get("source") != "agent")


# Pooling physics are PART OF THE TASK (scenario config), not magic constants —
# a scenario can dial how much coordination overhead its work carries. These are
# only fallbacks when a scenario doesn't override `physics`.
DEFAULT_PHYSICS = {
    "parallel_cap": 1.5,    # uncoordinated parallel work barely helps (Brooks)
    "parallel_decay": 0.35,  # 3 people on one task ≈ 1.5x, not 3x
    "swarm_cap": 2.5,       # a MEETING (synced) can push ~2.5x, never explode
    "swarm_decay": 0.6,     # meeting collaborators add diminishing but real help
}


def merge_minutes(spans, lo, hi):
    """Total minutes covered by `spans` within [lo, hi), overlaps merged —
    the calendar-occupancy primitive (a ping inside a meeting counts once)."""
    clipped = sorted((max(s, lo), min(e, hi)) for s, e in spans
                     if e > lo and s < hi)
    total, cur_s, cur_e = 0, None, None
    for s, e in clipped:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def physics_of(scenario):
    p = dict(DEFAULT_PHYSICS)
    p.update((scenario or {}).get("physics", {}))
    return p


def belief_hours_left(entry, effort_hours):
    """A belief entry's estimate of work-hours LEFT. Authored as a FRACTION of
    total effort (`remaining_frac`, 0..1) — scale-invariant, so a generator can
    rescale a task's effort and the belief tracks automatically — and surfaced
    to people as hours. `remaining_hours` (absolute) is still accepted for
    hand-authoring; None means the entry is the legacy %-done form."""
    if entry.get("remaining_frac") is not None:
        return entry["remaining_frac"] * (effort_hours or 0)
    return entry.get("remaining_hours")


def _person_skill_on(t, person, skills):
    """One person's effective speed on task t: geomean over t's tags of their
    skill factors (default 1.0 for tags they have no listed skill in)."""
    tags = t.get("tags") or []
    if not tags:
        return 1.0
    sk = (skills or {}).get(person) or {}
    prod = 1.0
    for tag in tags:
        prod *= sk.get(tag, 1.0)
    return prod ** (1.0 / len(tags))


def _capped_additive(rates, decay, cap):
    """Diminishing-returns pooling: contributions sorted desc, weighted
    1, decay, decay², … The cap bounds the POOLING gain, never a single
    worker's own rate — so one worker always works at exactly their skill
    (even a >cap specialist), and the floor is the best individual."""
    ordered = sorted(rates, reverse=True)
    if not ordered:
        return 0.0
    total, w = 0.0, 1.0
    for c in ordered:
        total += c * w
        w *= decay
    return max(ordered[0], min(cap, total))


def parallel_rate(t, workers, skills, physics=None):
    """Rate when several people are assigned to task t and work it AT THE SAME
    TIME but WITHOUT a meeting — heavily damped: unsynchronized hands on one task
    conflict and duplicate, so extra people add little. A single worker just gets
    their own skill rate (so single-owner scheduling — OPT, the baseline, most
    tasks — is unchanged). A MEETING turns these same people into an effective
    swarm (swarm_rate); that gap is exactly what a meeting buys."""
    p = physics or DEFAULT_PHYSICS
    return _capped_additive((_person_skill_on(t, w, skills) for w in workers),
                            p["parallel_decay"], p["parallel_cap"])


def swarm_rate(t, attendees, skills, physics=None):
    """Rate a task runs at when `attendees` SWARM it in a meeting: the same
    capped-additive pooling as parallel_rate but LIGHTLY damped and a higher cap
    — the meeting is the synchronization that makes extra hands actually land.
    So meeting > uncoordinated parallel > solo, and each is bounded."""
    p = physics or DEFAULT_PHYSICS
    return _capped_additive((_person_skill_on(t, a, skills) for a in attendees),
                            p["swarm_decay"], p["swarm_cap"])


def compute_schedule(tasks, sim_start, busy_by_assignee=None, skills=None,
                     deposits=None, answers=None, physics=None,
                     decision_effects=None):
    """Fully PREEMPTIVE, event-sourced scheduler. At every instant each person
    works their highest-priority READY task; the moment that changes — a
    higher-priority task arrives or unblocks, or the PM reprioritizes at a
    timestamp — they SWITCH, and the paused task keeps its accrued work and
    resumes later from exactly where it stopped (work is conserved).

    Returns, per schedulable task:
        {start, done_at, segments: [(seg_start, seg_end, person), ...]}
    where progress is the FOLD of the segments — so `task_view` can ask for
    true progress at any past/future instant. Deterministic and replay-safe:
    the only inputs are the task list (incl. timestamped `order_events`),
    sim_start, and busy intervals.

    busy_by_assignee: {person: [(start, end), ...]} — meetings/interruptions
    that consume that person's working time.
    """
    busy_by_assignee = busy_by_assignee or {}
    creation = {t["id"]: i for i, t in enumerate(tasks)}
    sched = {t["id"]: t for t in tasks if _schedulable(t)}
    # a task may have MULTIPLE assignees; each person appears against every task
    # they're on and, at any instant, works their single highest-priority ready
    # one. When several land on the SAME task at once it runs at parallel_rate
    # (damped); a meeting lifts that to swarm_rate via a deposit.
    persons = {}
    for tid, t in sched.items():
        for a in (t.get("assignees") or []):
            persons.setdefault(a, []).append(tid)

    # `remaining` is EFFORT-minutes still needed. A task completes when live work
    # (its active workers × parallel_rate, banked per interval) PLUS meeting
    # deposits (a swarm banked at the meeting's end) reach the effort.
    remaining = {tid: max(0.0, round((sched[tid]["effort_hours"]
                                      - sched[tid].get("done_hours", 0)) * 60.0))
                 for tid in sched}
    deposits = deposits or {}
    # answers: {task_id: {question_id: answered_at}} — when the PM's reply to a
    # BLOCKING question was delivered. A gating question suspends its task from
    # its open-time until answered_at (never, if absent). Derived upstream from
    # the message log (real run) / instant (OPT) / empty (no-PM baseline).
    answers = answers or {}
    decision_effects = decision_effects or {}
    effects = {tid: sorted(decision_effects.get(tid, []),
                          key=lambda e: (e["at"], e.get("question_id", "")))
               for tid in sched}
    segments = {tid: [] for tid in sched}
    done_at, started, applied = {}, {}, set()
    applied_effects = set()
    for tid in sched:
        if remaining[tid] <= 0:
            done_at[tid] = _earliest(sched[tid], sim_start)

    statics = set()
    for tid, t in sched.items():
        statics.add(_earliest(t, sim_start))
        for ev in t.get("order_events") or []:
            statics.add(ev.get("at", sim_start))
        for (end, _dep) in deposits.get(tid, []):
            statics.add(end)   # a meeting's output lands at its end
        for ev in effects.get(tid, []):
            statics.add(ev["at"])
        for q in t.get("questions") or []:
            if q.get("gates"):
                statics.add(q["at"])                     # block opens
                ans = answers.get(tid, {}).get(q["id"])
                if ans is not None:
                    statics.add(ans)                     # block clears
    # busy-interval edges are decision points too: a person's availability (and
    # thus a task's active-worker set and rate) can only change at these.
    for iv in busy_by_assignee.values():
        for (s, e) in iv:
            statics.add(s)
            statics.add(e)

    def free_at(pers, clock):
        for (s, e) in busy_by_assignee.get(pers, ()):
            if s <= clock < e:
                return False   # in a meeting / eating a refocus-tax interval
        return True

    def apply_deposits(clock):
        # a meeting on task T banks `dep` effort-minutes into T when it ends
        for tid in sched:
            for i, (end, dep) in enumerate(deposits.get(tid, [])):
                if end <= clock and (tid, i) not in applied and remaining[tid] > 0:
                    applied.add((tid, i))
                    remaining[tid] -= dep
                    if remaining[tid] <= 1e-6:
                        remaining[tid] = 0
                        done_at.setdefault(tid, end)

    def apply_effects(clock):
        # A choice effect is pinned into the question_answered event and becomes
        # active at its delivery instant — never when the PM merely sent it.
        for tid in sched:
            for i, ev in enumerate(effects.get(tid, [])):
                if ev["at"] > clock or (tid, i) in applied_effects:
                    continue
                applied_effects.add((tid, i))
                if remaining[tid] <= 0:
                    continue
                remaining[tid] += ev.get("task_effort_delta_hours", 0) * 60.0
                if remaining[tid] <= 1e-6:
                    remaining[tid] = 0
                    done_at.setdefault(tid, ev["at"])

    def ready(tid, clock):
        t = sched[tid]
        if remaining[tid] <= 0 or _earliest(t, sim_start) > clock:
            return False
        for b in t.get("blocked_by", []):
            if b in sched and (b not in done_at or done_at[b] > clock):
                return False  # a scheduled blocker isn't finished by `clock`
        for q in t.get("questions") or []:
            # a gating question the owner raised suspends the task from its
            # open-time until the PM's answer is DELIVERED (answered_at). No
            # answer, or one not yet delivered by `clock` -> the owner is stuck.
            if q.get("gates") and q["at"] <= clock:
                ans = answers.get(tid, {}).get(q["id"])
                if ans is None or ans > clock:
                    return False
        return True

    clock, INF, guard = sim_start, float("inf"), 0
    apply_effects(clock)
    apply_deposits(clock)
    while any(r > 0 for r in remaining.values()) and guard < 100000:
        guard += 1
        # each person works their single highest-priority ready assigned task
        chosen = {}
        for pers, tids in persons.items():
            avail = [tid for tid in tids if ready(tid, clock)]
            chosen[pers] = (min(avail, key=lambda x: (_order_at(sched[x], clock),
                                                      creation[x]))
                            if avail else None)
        # group the FREE workers landing on each task -> its live rate this step
        active = {}
        for pers, tid in chosen.items():
            if tid is not None and free_at(pers, clock):
                active.setdefault(tid, []).append(pers)
        rate = {tid: parallel_rate(sched[tid], ws, skills, physics)
                for tid, ws in active.items()}
        # next decision point: earliest completion at the current rates, or a
        # static event (arrival / reprioritization / meeting end / busy edge)
        nxt, comp = INF, {}
        for tid, r in rate.items():
            if r <= 0:
                continue
            # active workers are free across [clock, nxt) (busy edges are static)
            c = add_work_minutes(clock, math.ceil(remaining[tid] / r), ())
            comp[tid] = c
            nxt = min(nxt, c)
        for s in statics:
            if s > clock:
                nxt = min(nxt, s)
        if nxt == INF:
            break  # everyone idle and no future event — remaining tasks stuck
        for tid, r in rate.items():
            w = work_minutes_between(clock, nxt, ())
            if w <= 0:
                continue  # [clock, nxt) had no working minutes
            segments[tid].append((clock, nxt, r))
            started.setdefault(tid, clock)
            remaining[tid] -= w * r              # EFFORT accrued at the live rate
            if remaining[tid] <= 1e-6:
                remaining[tid] = 0
                done_at.setdefault(tid, comp.get(tid, nxt))
        clock = nxt
        apply_effects(clock)    # delivered choices landing at `nxt`
        apply_deposits(clock)   # meeting outputs landing at `nxt`

    return {tid: {"start": started.get(tid), "done_at": done_at.get(tid),
                  "segments": segments[tid]}
            for tid in sched}


def _accrued_hours(seg_row, sim_start_seed, now, task_deposits=()):
    """True EFFORT-hours done on a task by `now`: seed progress + live work
    (each segment's effort = its live rate × the working minutes it covered,
    with the workers free by construction) + any MEETING deposits landed by
    `now` (a swarm banked at the meeting's end)."""
    acc = 0.0
    for (s, e, rate) in seg_row["segments"]:
        if s >= now:
            break
        acc += rate * work_minutes_between(s, min(e, now), ())
    dep = sum(d for (end, d) in task_deposits if end <= now)
    return sim_start_seed + (acc + dep) / 60.0


def task_view(tasks, sim_start, now, busy_by_assignee=None, skills=None,
              deposits=None, answers=None, physics=None,
              decision_effects=None):
    """Ground-truth task states at `now`. Tasks not yet arrived are omitted."""
    busy_by_assignee = busy_by_assignee or {}
    deposits = deposits or {}
    answers = answers or {}
    decision_effects = decision_effects or {}
    sched = compute_schedule(tasks, sim_start, busy_by_assignee, skills,
                             deposits, answers, physics, decision_effects)
    out = []
    for t in tasks:
        if t.get("arrival", sim_start) > now:
            continue  # doesn't exist yet at this (possibly rewound) time
        effective_effort = t.get("effort_hours")
        if effective_effort is not None:
            effective_effort += sum(
                ev.get("task_effort_delta_hours", 0)
                for ev in decision_effects.get(t["id"], [])
                if ev["at"] <= now)
        row = {
            "id": t["id"],
            "title": t["title"],
            "assignees": t.get("assignees", []),
            "effort_hours": effective_effort,
            "blocked_by": t.get("blocked_by", []),
            "reported": t.get("reported", ""),
            "source": t.get("source", "seed"),
            "priority": t.get("priority"),
            "urgent": bool(t.get("urgent")),
            # progress that predates the week — utilization must not credit it
            "seed_done_hours": t.get("done_hours", 0),
        }
        for key in ("score_weight_factor", "work_item_id", "work_item_title",
                    "stage_id", "stage_kind", "contingent"):
            if key in t:
                row[key] = t[key]
        if not _schedulable(t):
            row.update(status="tracking", pct=None, true_done_hours=None,
                       projected_done=None)
        elif t["id"] not in sched:
            row.update(status="stuck", pct=0,
                       true_done_hours=t.get("done_hours", 0), projected_done=None)
        else:
            s = sched[t["id"]]
            effort, base = effective_effort, t.get("done_hours", 0)
            done_at = s["done_at"]
            true_done = _accrued_hours(s, base, now, deposits.get(t["id"], ()))
            true_done = min(true_done, effort)
            if done_at is not None and now >= done_at:
                status, true_done = "done", effort
            else:
                # coarse board column (no % / ETA leaked): is a work segment
                # live at `now` (in_progress), waiting on a blocker (blocked),
                # or ready-but-behind / not-yet-begun (queued)?
                active = any(s0 <= now < e0 for (s0, e0, _) in s["segments"])
                blocked = any(b in sched and (sched[b]["done_at"] is None
                                              or now < sched[b]["done_at"])
                              for b in t.get("blocked_by", []))
                # an open, still-unanswered gating question also shows as blocked
                q_blocked = any(
                    q.get("gates") and q["at"] <= now
                    and (answers.get(t["id"], {}).get(q["id"]) is None
                         or answers[t["id"]][q["id"]] > now)
                    for q in t.get("questions") or [])
                status = ("in_progress" if active
                          else "blocked" if (blocked or q_blocked) else "queued")
            row.update(
                status=status,
                true_done_hours=round(true_done, 1),
                pct=int(round(100.0 * true_done / effort)) if effort else 100,
                projected_done=done_at,
            )
        out.append(row)
    return out


def task_done(tasks, sim_start, now, task_id, busy_by_assignee=None, skills=None,
              deposits=None, answers=None, physics=None,
              decision_effects=None):
    sched = compute_schedule(tasks, sim_start, busy_by_assignee or {}, skills,
                             deposits or {}, answers or {}, physics,
                             decision_effects)
    s = sched.get(task_id)
    return s is not None and s["done_at"] is not None and now >= s["done_at"]
