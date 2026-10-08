"""The engine: pop event -> advance clock -> dispatch -> maybe push new events.

Every queue push is also recorded as a "scheduled" log entry, so the log holds
both the realized past and (at write time) the planned future — enough to
rebuild pending state on resume.
"""

import anthropic

from copy import deepcopy

from .events import EventQueue
from .npc import NPC, generate_transcript
from .rubric import load_rubric
from .sim_time import fmt, in_working_hours
from .store import RunStore, new_run_dir, setup_logging
from .validate import bounds_of, check_task_bounds, validate_scenario
from .world import World

AGENT_ACTION_COST = 1  # sim-minutes an agent action takes


class Engine:
    def __init__(self, scenario, client=None, verbose=True, run_dir=None):
        problems = validate_scenario(scenario)
        if problems:
            raise ValueError("invalid scenario:\n  " + "\n  ".join(problems))
        self.bounds = bounds_of(scenario)
        self.run_dir = run_dir or new_run_dir()
        self.logger = setup_logging(self.run_dir).getChild("engine")
        self.world = World(scenario, store=RunStore(self.run_dir, scenario))
        self.agent_horizon = load_rubric(scenario, self.run_dir)["horizon"]
        self.queue = EventQueue()
        self._reveal_checks = set()
        self.client = client or anthropic.Anthropic()
        self.verbose = verbose

        seed = scenario.get("seed", 0)
        for spec in scenario["npcs"]:
            npc = NPC(spec, seed)
            self.world.npcs[npc.id] = npc
            # Bootstrap: one heartbeat per NPC keeps the heap alive.
            self._schedule(npc.next_wakeup(self.world.clock), "npc_wakeup", {"npc": npc.id})

        # Authored scenario beats live in the SAME queue as organic behavior.
        for i, beat in enumerate(scenario.get("beats", [])):
            self._schedule(beat["at"], "beat", {"npc": beat["npc"], "beat": i})

        # OOD task arrivals: externally-caused tasks that appear on a schedule.
        for i, arr in enumerate(scenario.get("task_arrivals", [])):
            self._schedule(arr["at"], "task_arrival",
                           {"arrival": i, "npc": arr.get("npc", "")})
            # Org fallback: a no-PM team still functions — if the ask is
            # still unowned at fallback.at, its natural volunteer just grabs
            # it (same physics in EVERY run; the baseline is not a strawman,
            # and the PM earns credit only for coordinating BETTER).
            if arr.get("fallback"):
                self._schedule(arr["fallback"]["at"], "org_pickup",
                               {"arrival": i, "npc": arr["fallback"]["npc"]})

        # Belief maturation: holders' honest-but-wrong estimates correct at
        # authored moments (learning on the trajectory — deterministic).
        for t in self.world.tasks:
            for i, b in enumerate(t.get("belief") or []):
                if "at" in b:
                    self._schedule(b["at"], "belief_update",
                                   {"task": t["id"], "idx": i})

        # Blocking questions: at its open-time the owner hits a decision they
        # can't make alone and pings the PM — their work on the task suspends
        # until the PM's answer is DELIVERED (the block itself is derived in
        # World.question_answers; here we only fire the ping + stall awareness).
        for t in self.world.tasks:
            for i, q in enumerate(t.get("questions") or []):
                if q.get("gates"):
                    self._schedule(q["at"], "question_ping",
                                   {"task": t["id"], "idx": i})

        # Each NPC's experience opens with their honest (possibly wrong)
        # picture of their own work — belief, never scheduler truth.
        for npc in self.world.npcs.values():
            bv = self.world.belief_view(npc.id)
            if bv:
                npc.notify(self.world.clock,
                           "your current picture of your work: "
                           + "; ".join("%s — %s" % (b["task"], b["your_view"])
                                       for b in bv))

        self._check_completions()  # tasks already done at seed are public
        # push-delivery cursor: seed-time events predate the PM's first action
        self._push_cursor = len(self.world.log)

    def _push_items(self, entries):
        """Which log entries are PUSH-class for the PM (sim/signals.py):
        chat messages, batched email deliveries, room lines in a meeting
        they're sitting in, completion broadcasts. EVERYTHING is push — the
        agent has no background thread to poll with, so the world delivers."""
        new = []
        for e in entries:
            if (e["kind"] == "message" and e["recipient"] == "agent"
                    and e.get("via", "chat") == "chat"):
                new.append({"time": fmt(e["t"]), "chat_from": e["sender"],
                            "text": e["text"]})
            elif e["kind"] == "email_delivered":
                new.append({"time": fmt(e["t"]), "email_from": e["sender"],
                            "text": e["text"]})
            elif e["kind"] == "task_completed":
                new.append({"time": fmt(e["t"]),
                            "tracker": "task completed: %s" % e["task_id"]})
            elif e["kind"] == "task_revealed":
                new.append({"time": fmt(e["t"]),
                            "work_revealed": e["title"],
                            "task_id": e["task_id"],
                            "reason": e["reason"]})
            elif e["kind"] == "room_line" and e["speaker"] != "agent":
                m = next((x for x in self.world.meetings
                          if x["id"] == e["meeting_id"]), None)
                if m and "agent" in m["attendees"]:
                    new.append({"time": fmt(e["t"]), "in_meeting": e["topic"],
                                "speaker": e["speaker"], "text": e["text"]})
        return new

    def drain_agent_push(self):
        """Everything that landed on the PM unasked since their last look."""
        new = self._push_items(self.world.log[self._push_cursor:])
        self._push_cursor = len(self.world.log)
        return new

    def _schedule(self, time, kind, payload):
        self.world.record("scheduled", fire_at=time, event=kind, **payload)
        return self.queue.push(time, kind, payload)

    def pending(self):
        """Future events still in the heap, soonest first."""
        return sorted(
            ({"fire_at": e.time, "event": e.kind, "payload": e.payload}
             for e in self.queue._heap),
            key=lambda x: x["fire_at"],
        )

    # -- agent actions (synchronous with the current instant) ----------------

    def _advance_action_clock(self):
        """CAUSALITY INVARIANT: no undispatched event may exist at or before
        the clock. Every action-cost advance immediately drains events that
        became due — the clock never passes over a pending event. If the
        agent's 3rd action of a burst crosses a reply's due-time, that reply
        fires right then; the 4th action happens in a world where it exists.

        Terminates because every dispatched event schedules strictly-future
        events (see DESIGN.md §7b)."""
        self.world.advance_clock_to(self.world.clock + AGENT_ACTION_COST)
        self._drain_due()

    def _drain_due(self):
        while len(self.queue) and self.queue.peek().time <= self.world.clock:
            self.step()
        self._check_completions()

    def _check_completions(self):
        """Completions are the one PUBLIC, reliable signal: when a task truly
        finishes, the tracker updates and everyone can see it — this is what
        lets the PM calibrate people from evidence (2 completions ≈ a velocity
        estimate) instead of reading the scheduler's mind."""
        for row in self.world.tasks_view():
            if row["status"] != "done" or row["id"] in self.world.completed_announced:
                continue
            self.world.completed_announced.add(row["id"])
            done_at = row["projected_done"]
            self.world.record("task_completed", task_id=row["id"],
                              title=row["title"], completed_at=done_at)
            self.world.update_task(row["id"],
                                   {"reported": "Done (finished %s)" % fmt(done_at)},
                                   "system")
            t = self.world.find_task(row["id"])
            for a in (t or {}).get("assignees", []):
                if a in self.world.npcs:
                    self.world.npcs[a].notify(self.world.clock,
                                              "you finished '%s'" % row["title"])

        rows = {r["id"]: r for r in self.world.tasks_view()}
        for task in self.world.tasks:
            if (not task.get("contingent")
                    or task["id"] in self.world.revealed_tasks):
                continue
            blockers = task.get("hidden_until") or []
            if not blockers:
                continue
            done_at = [rows.get(dep, {}).get("projected_done")
                       for dep in blockers]
            if any(t is None for t in done_at):
                continue
            when = max(done_at)
            key = (task["id"], when)
            if when > self.world.clock and key not in self._reveal_checks:
                self._reveal_checks.add(key)
                self._schedule(when, "work_reveal_check",
                               {"task": task["id"]})
        changed = True
        while changed:
            changed = False
            for task in self.world.tasks:
                if (not task.get("contingent")
                        or task["id"] in self.world.revealed_tasks):
                    continue
                blockers = task.get("hidden_until") or []
                if not blockers or not all(
                        rows.get(dep, {}).get("status") == "done"
                        for dep in blockers):
                    continue
                self.world.revealed_tasks.add(task["id"])
                triggered = max(rows[dep]["projected_done"]
                                for dep in blockers)
                reason = task.get("quality_reason", "quality gate outcome")
                self.world.record(
                    "task_revealed", task_id=task["id"],
                    work_item_id=task.get("work_item_id"),
                    title=task["title"], reason=reason,
                    triggered_at=triggered)
                for owner in task.get("assignees", []):
                    if owner in self.world.npcs:
                        self.world.npcs[owner].notify(
                            self.world.clock,
                            "new work revealed on '%s': %s"
                            % (task["title"], reason))
                changed = True

    def _schedule_email_delivery(self, msg):
        """Email to the PM is a BATCHED PUSH: the agent has no background
        thread to poll an inbox with, so the world models the human habit
        ('always refreshing my email') as a delivery event on a fixed batch
        grid. The batch interval is the PM-side email K — learnable, never
        documented."""
        batch = self.world.scenario.get("email_batch_minutes", 30)
        self._schedule((msg.time // batch + 1) * batch, "email_delivery",
                       {"msg_id": msg.id})

    def agent_chat(self, npc_ids, text):
        """One GROUP chat ping: composed once (one action-minute), delivered
        to every recipient at the same instant, each seeing who else is on
        the thread. Every recipient pays their own focus tax and replies on
        their own latency. Unknown recipients bounce, never crash."""
        if not npc_ids:
            return {"error": "give npc (one person) or to (several)"}
        known = [n for n in npc_ids if n in self.world.npcs]
        bounced = [n for n in npc_ids if n not in self.world.npcs]
        if known:
            self._advance_action_clock()
        agent_name = self.world.scenario.get("agent_name", "the PM")
        gid = len(self.world.messages) if len(known) > 1 else None
        for nid in known:
            npc = self.world.npcs[nid]
            msg = self.world.send_message("agent", nid, text, via="chat",
                                          group=gid)
            others = [self.world.npcs[o].name for o in known if o != nid]
            cc = (", with %s" % ", ".join(others)) if others else ""
            npc.notify(self.world.clock, "%s (chat%s): %s"
                       % (agent_name, cc, text))
            if in_working_hours(self.world.clock):
                # the tax is FELT, not just charged: the same focus-minutes
                # busy_by_assignee subtracts lands in the holder's
                # experience, so testimony about it stays honest
                npc.notify(self.world.clock,
                           "(that ping just broke your concentration — it'll "
                           "take you a good while to get back into deep work)")
            due = npc.response_time(self.world.clock, via="chat")
            due = self.world.defer_for_meetings(nid, due, npc.rng)
            self._schedule(due, "npc_respond", {"npc": nid, "message_id": msg.id})
            self._say("[%s] you -> %s (chat): %s"
                      % (self.world.now(), npc.name, text))
        out = {"sent": bool(known)}
        if bounced:
            out["bounced"] = bounced
        return out

    def agent_say(self, npc_id, text, via="chat"):
        """Single-recipient compatibility wrapper (CLI/API/smoke): chat goes
        through the group path; email through agent_email."""
        if via == "email":
            r = self.agent_email([npc_id], "message", text)
        else:
            r = self.agent_chat([npc_id], text)
        if r.get("bounced") or not r.get("sent"):
            return {"error": "not delivered — %r is not in the company "
                             "directory" % npc_id}
        return r

    def agent_email(self, npc_ids, subject, body):
        """One GROUP email: composed once (one action-minute regardless of
        recipient count), delivered to every recipient at the same instant,
        and every recipient SEES the to-line — "Hi both" is coherent because
        both know who else got it. Unknown addresses bounce, never crash."""
        text = "[%s] %s" % (subject, body)
        known = [n for n in npc_ids if n in self.world.npcs]
        bounced = [n for n in npc_ids if n not in self.world.npcs]
        if known:
            self._advance_action_clock()   # one composition, one cost
        agent_name = self.world.scenario.get("agent_name", "the PM")
        gid = len(self.world.messages) if len(known) > 1 else None
        for nid in known:
            npc = self.world.npcs[nid]
            self.world.send_message("agent", nid, text, via="email", group=gid)
            others = [self.world.npcs[o].name for o in known if o != nid]
            cc = (", also to %s" % ", ".join(others)) if others else ""
            npc.notify(self.world.clock,
                       "%s (email%s): %s" % (agent_name, cc, text))
            npc.unanswered_emails += 1
            self._say("[%s] you -> %s (email): %s"
                      % (self.world.now(), npc.name, text))
        out = {"sent": bool(known)}
        if bounced:
            out["bounced"] = bounced
        return out

    def agent_assign_task(self, task_id, npc_id):
        """Reassign work — a decision with real schedule consequences."""
        task = self.world.find_task(task_id)
        if not self.world.task_visible(task) or task.get("filed") is False:
            # unfiled tickets aren't on the board: officially add first
            return {"error": "no task %r on the tracker" % task_id}
        if len(task.get("assignees") or []) == 1 and task["assignees"][0] == npc_id:
            return {"error": "%s already owns %r" % (npc_id, task_id)}
        if npc_id not in self.world.npcs:
            return {"error": "unknown person %r" % npc_id}
        spec = next(n for n in self.world.scenario["npcs"] if n["id"] == npc_id)
        if not spec.get("worker", True):
            # the labor pool binds BOTH sides: OPT's ceiling excludes
            # stakeholders, so the agent must not be able to use them either —
            # otherwise the agent can out-labor OPT and normalized > 1
            return {"error": "%s is a stakeholder, not on the delivery team — "
                             "they don't take tickets" % self.world.npcs[npc_id].name}
        self._advance_action_clock()
        # persist progress-so-far and stamp the handoff instant: the new
        # holder works from NOW (no retroactive credit), prior work is kept
        row = next((r for r in self.world.tasks_view()
                    if r["id"] == task_id), None)
        changes = {"assignees": [npc_id], "assigned_at": self.world.clock}
        if row and row.get("true_done_hours") is not None:
            changes["done_hours"] = row["true_done_hours"]
        prev = [a for a in (task.get("assignees") or [])
                if a != npc_id and a in self.world.npcs]
        old_owners = list(task.get("assignees") or [])
        task = self.world.update_task(task_id, changes, "agent")
        handoff = int((self.world.scenario.get("costs") or {})
                      .get("handoff_minutes", 0))
        if handoff and old_owners != [npc_id]:
            self.world.record("capacity_cost", person=npc_id,
                              task_id=task_id, minutes=handoff,
                              reason="handoff")
            self.world.npcs[npc_id].notify(
                self.world.clock,
                "orientation on '%s' will cost about %d minutes"
                % (task["title"], handoff))
        self.world.npcs[npc_id].notify(self.world.clock,
                                       "you've been assigned: '%s'" % task["title"])
        for a in prev:   # losing a task is an event in YOUR week too
            self.world.npcs[a].notify(
                self.world.clock, "'%s' has been moved off your plate to %s"
                % (task["title"], self.world.npcs[npc_id].name))
        self._say("[%s] you assigned %s -> %s"
                  % (self.world.now(), task_id, self.world.npcs[npc_id].name))
        # tracker-shaped acknowledgement, never the truth dict (beliefs,
        # scheduler fields stay hidden)
        return {"assigned": task_id, "to": npc_id, "title": task["title"],
                "handoff_minutes": handoff if old_owners != [npc_id] else 0}

    def _restamp_owners(self, task_id, new_assignees, verb):
        """Change a task's assignee SET, banking progress-so-far and stamping the
        instant so the recompute credits the new roster only from NOW (no
        retroactive parallel work). Shared by add_helper / drop_helper."""
        row = next((r for r in self.world.tasks_view() if r["id"] == task_id), None)
        changes = {"assignees": new_assignees, "assigned_at": self.world.clock}
        if row and row.get("true_done_hours") is not None:
            changes["done_hours"] = row["true_done_hours"]
        task = self.world.update_task(task_id, changes, "agent")
        return task

    def agent_add_helper(self, task_id, npc_id):
        """Put an EXTRA person on a task alongside its owner. While both work it
        they share the damped parallel_rate — faster wall-clock, but sublinear
        (coordination overhead) and it costs the helper their own work. Worth it
        for a bottleneck / an idle teammate, not as a default."""
        task = self.world.find_task(task_id)
        if not self.world.task_visible(task) or task.get("filed") is False:
            return {"error": "no task %r on the tracker" % task_id}
        if npc_id not in self.world.npcs:
            return {"error": "unknown person %r" % npc_id}
        spec = next(n for n in self.world.scenario["npcs"] if n["id"] == npc_id)
        if not spec.get("worker", True):
            return {"error": "%s is a stakeholder — they don't take tickets"
                    % self.world.npcs[npc_id].name}
        owners = task.get("assignees") or []
        if npc_id in owners:
            return {"error": "%s is already on %r" % (npc_id, task_id)}
        self._advance_action_clock()
        self._restamp_owners(task_id, owners + [npc_id], "add")
        onboarding = int((self.world.scenario.get("costs") or {})
                         .get("helper_onboarding_minutes", 0))
        if onboarding:
            self.world.record("capacity_cost", person=npc_id,
                              task_id=task_id, minutes=onboarding,
                              reason="helper_onboarding")
            self.world.npcs[npc_id].notify(
                self.world.clock,
                "onboarding on '%s' will cost about %d minutes"
                % (task["title"], onboarding))
        self.world.npcs[npc_id].notify(self.world.clock,
                                       "you're now helping on: '%s'" % task["title"])
        self._say("[%s] you added %s to help on %s"
                  % (self.world.now(), self.world.npcs[npc_id].name, task_id))
        return {"task_id": task_id, "assignees": owners + [npc_id],
                "onboarding_minutes": onboarding}

    def agent_drop_helper(self, task_id, npc_id):
        """Take a helper back OFF a task (return them to their own work). Can't
        drop the last/only owner — reassign that instead."""
        task = self.world.find_task(task_id)
        if not self.world.task_visible(task) or task.get("filed") is False:
            return {"error": "no task %r on the tracker" % task_id}
        owners = task.get("assignees") or []
        if npc_id not in owners:
            return {"error": "%s isn't on %r" % (npc_id, task_id)}
        if len(owners) <= 1:
            return {"error": "can't drop the only owner of %r — reassign it"
                    % task_id}
        self._advance_action_clock()
        remaining = [a for a in owners if a != npc_id]
        self._restamp_owners(task_id, remaining, "drop")
        self._say("[%s] you took %s off %s"
                  % (self.world.now(), self.world.npcs[npc_id].name, task_id))
        return {"task_id": task_id, "assignees": remaining}

    def agent_reprioritize(self, task_id, priority=None, urgent=None):
        """Reorder a task in its holder's work queue. SCHEDULING only: the
        authored `priority` keeps its rubric weight, so the PM can decide
        what gets worked first but can never relabel a task to mint value."""
        task = self.world.find_task(task_id)
        if not self.world.task_visible(task) or task.get("filed") is False:
            return {"error": "no task %r on the tracker" % task_id}
        if priority is not None and priority not in ("P0", "P1", "P2", "P3"):
            return {"error": "priority must be P0..P3, got %r" % priority}
        if priority is None and urgent is None:
            return {"error": "nothing to change — pass priority and/or urgent"}
        self._advance_action_clock()
        # a TIMESTAMPED order event: the scheduler applies it forward-only
        # (a Thursday reprioritization can never rewrite Monday's schedule)
        ev = {"at": self.world.clock}
        if priority is not None:
            ev["order_priority"] = priority
        if urgent is not None:
            ev["order_urgent"] = bool(urgent)
        events = list(task.get("order_events") or []) + [ev]
        self.world.update_task(task_id, {"order_events": events}, "agent")
        changes = {k: v for k, v in ev.items() if k != "at"}
        for a in task.get("assignees", []):
            if a in self.world.npcs:
                self.world.npcs[a].notify(
                    self.world.clock, "the PM re-prioritized '%s'%s%s"
                    % (task["title"],
                       " to %s" % priority if priority is not None else "",
                       " (urgent)" if urgent else ""))
        self._say("[%s] you reprioritized %s %s"
                  % (self.world.now(), task_id, changes))
        return {"task_id": task_id, **changes}

    def agent_answer_question(self, task_id, question_id, choice, rationale,
                              via):
        """Deliver one explicit decision on an open blocking question."""
        task = self.world.find_task(task_id)
        if not self.world.task_visible(task) or task.get("filed") is False:
            return {"error": "no task %r on the tracker" % task_id}
        question = next((q for q in task.get("questions") or []
                         if q.get("gates") and q.get("id") == question_id), None)
        if question is None:
            return {"error": "no open blocking question %r on %r"
                    % (question_id, task_id)}
        if self.world.clock < question["at"]:
            return {"error": "question %r has not opened yet" % question_id}
        if question_id in self.world.question_answers().get(task_id, {}):
            return {"error": "question %r is already answered" % question_id}
        if "options" in question and choice not in question["options"]:
            return {"error": "choice must be one of %s" % question["options"]}
        if via not in ("chat", "email"):
            return {"error": "via must be chat or email"}
        if not choice.strip() or not rationale.strip():
            return {"error": "choice and rationale must be non-empty"}
        owner = (task.get("assignees") or [None])[0]
        if owner not in self.world.npcs:
            return {"error": "question owner is not a current worker"}

        effect = deepcopy((question.get("choice_effects") or {}).get(choice, {}))
        text = ("Decision for '%s' / %s: %s. Rationale: %s"
                % (task["title"], question_id, choice, rationale))
        if via == "chat":
            sent = self.agent_chat([owner], text)
            delivered_at = self.world.clock
        else:
            sent = self.agent_email([owner], "Decision: " + task["title"], text)
            answer_batch = self.world.scenario.get("answer_batch_minutes", 120)
            delivered_at = ((self.world.clock // answer_batch + 1)
                            * answer_batch)
        if "error" in sent:
            return sent
        self.world.record("question_answered", task_id=task_id,
                          question_id=question_id, choice=choice,
                          rationale=rationale, via=via, owner=owner,
                          delivered_at=delivered_at, effect=effect)
        out = {"task_id": task_id, "question_id": question_id,
               "choice": choice, "via": via, "delivered_at": delivered_at}
        if effect.get("summary"):
            out["effect_summary"] = effect["summary"]
        return out

    def agent_update_note(self, task_id, note):
        """Correct the tracker's reported status (record vs truth hygiene)."""
        task = self.world.find_task(task_id)
        if not self.world.task_visible(task) or task.get("filed") is False:
            return {"error": "no task %r on the tracker" % task_id}
        self._advance_action_clock()
        self.world.update_task(task_id, {"reported": note}, "agent")
        self._say("[%s] you updated tracker note on %s" % (self.world.now(), task_id))
        return {"task_id": task_id, "reported": note}

    def agent_schedule_meeting(self, attendees, start, duration, topic,
                               agenda="", task=None):
        """Book a meeting. Costs every attendee real working capacity. Label it
        with a `task` to make it a SWARM session — the labelled task's owner
        works it at the pooled (all-attendees) rate for the block, so pulling
        the right specialist onto a bottleneck accelerates it. Unlabelled or
        wrong-room meetings are pure overhead."""
        from .sim_time import MIN_PER_DAY
        unknown = [a for a in attendees if a != "agent" and a not in self.world.npcs]
        if unknown:
            return {"error": "unknown attendees: %s" % unknown}
        if duration > self.bounds["max_meeting_minutes"]:
            return {"error": "meeting too long: %d > %d min"
                    % (duration, self.bounds["max_meeting_minutes"])}
        if task is not None:
            t = self.world.find_task(task)
            if not self.world.task_visible(t) or t.get("filed") is False:
                return {"error": "no task %r on the tracker to align" % task}
        self._advance_action_clock()
        # strictly future: a due-now meeting_start may never sit undispatched
        start = max(start, self.world.clock + 1)
        end = start + duration
        cap = self.world.scenario.get("costs", {}).get(
            "max_meeting_minutes_per_day", 180)
        ooo = {n["id"]: n.get("ooo", []) for n in self.world.scenario["npcs"]}
        for a in attendees:
            for (os_, oe) in ooo.get(a, []):
                if os_ < end and start < oe:   # can't meet someone who's OUT
                    return {"error": "%s is out of office %s-%s"
                            % (a, fmt(os_), fmt(oe))}
            for m in self.world.meetings:
                if m.get("cancelled") or a not in m["attendees"]:
                    continue
                if m["start"] < end and start < m["end"]:
                    return {"error": "%s is double-booked %s-%s"
                            % (a, fmt(m["start"]), fmt(m["end"]))}
            booked = sum(m["end"] - m["start"] for m in self.world.meetings
                         if not m.get("cancelled") and a in m["attendees"]
                         and m["start"] // MIN_PER_DAY == start // MIN_PER_DAY)
            if booked + duration > cap:
                return {"error": "%s would blow the daily meeting cap "
                        "(%d/%d min)" % (a, booked + duration, cap)}
        meeting = self.world.add_meeting(start, end, attendees, topic, agenda, task)
        for a in attendees:
            if a in self.world.npcs:
                self.world.npcs[a].notify(
                    self.world.clock, "meeting '%s' put on your calendar: %s-%s, with %s"
                    % (topic, fmt(start), fmt(start + duration), ", ".join(attendees)))
        self._schedule(start, "meeting_start", {"meeting": meeting["id"]})
        self._schedule(start + duration, "meeting_end", {"meeting": meeting["id"]})
        if "agent" in attendees:
            # a LIVE room: NPC speaking turns every ~5 min (opener at +2).
            # Each fires only if the room has gone quiet — rules decide WHO
            # and WHEN speaks; the LLM only writes the line.
            t = start + 2
            while t < start + duration:
                self._schedule(t, "room_turn", {"meeting": meeting["id"]})
                t += 5
        self._say("[%s] you scheduled '%s' %s (%d min) with %s"
                  % (self.world.now(), topic, fmt(start), duration,
                     ", ".join(attendees)))
        out = dict(meeting)
        if task is not None:
            # a session that cannot bank work must SAY so, not fail silently
            from .tasks import work_minutes_between
            t = self.world.find_task(task)
            owner = (t.get("assignees") or [None])[0]
            if owner not in attendees:
                out["warning"] = ("task owner %s is not attending — this "
                                  "session will bank NO work" % (owner or "?"))
            elif work_minutes_between(start, end) <= 0:
                out["warning"] = ("outside working hours — this session will "
                                  "bank NO work")
        return out

    def agent_cancel_meeting(self, meeting_id):
        """Clear a meeting off the calendar — frees its attendees' time and
        voids its swarm. World-physics rule: you can only cancel a meeting that
        HASN'T STARTED YET; a meeting in progress or already over is history."""
        m = next((x for x in self.world.meetings if x["id"] == meeting_id), None)
        if m is None:
            return {"error": "no meeting %r" % meeting_id}
        if m.get("cancelled"):
            return {"error": "%r is already cancelled" % meeting_id}
        self._advance_action_clock()
        if self.world.clock >= m["start"]:
            return {"error": "can't cancel '%s' — it already %s"
                    % (m["topic"], "started" if self.world.clock < m["end"]
                       else "happened")}
        self.world.cancel_meeting(meeting_id)
        for a in m["attendees"]:
            if a in self.world.npcs:
                self.world.npcs[a].notify(
                    self.world.clock, "meeting '%s' (%s) is cancelled — that "
                    "time is yours again" % (m["topic"], fmt(m["start"])))
        self._say("[%s] you cancelled '%s' (%s)"
                  % (self.world.now(), m["topic"], fmt(m["start"])))
        return {"cancelled": meeting_id, "topic": m["topic"]}

    def _room_speaker_order(self, meeting):
        """Deterministic turn-offering order: least-spoken NPC attendees first.
        Each offered NPC's agent decides speak-or-PASS; silence is allowed."""
        npcs = [a for a in meeting["attendees"] if a in self.world.npcs]
        counts = {a: 0 for a in npcs}
        for l in meeting.get("minutes", []):
            if l["speaker"] in counts:
                counts[l["speaker"]] += 1
        return [self.world.npcs[a]
                for a in sorted(npcs, key=lambda a: (counts[a], npcs.index(a)))]

    def _offer_room_turn(self, meeting):
        """Offer the floor: first NPC whose agent chooses to speak, speaks."""
        for speaker in self._room_speaker_order(meeting):
            text = self._llm(lambda s=speaker: s.room_turn(
                self.client, self.world, meeting))
            if text:
                return speaker, text
        return None, None

    def agent_talk_in_meeting(self, text):
        """Speak in the meeting you are in RIGHT NOW. Zero queue latency —
        but the exchange consumes the meeting slot itself (~3 sim-min)."""
        m = self.world.active_meeting_with("agent")
        if m is None:
            nxt = min((x for x in self.world.meetings
                       if "agent" in x["attendees"] and x["start"] > self.world.clock),
                      key=lambda x: x["start"], default=None)
            hint = (" Your next meeting: '%s' at %s." % (nxt["topic"], fmt(nxt["start"]))
                    if nxt else "")
            return {"error": "you are not in a meeting right now.%s" % hint}
        self._advance_action_clock()
        if self.world.clock >= m["end"]:
            return {"error": "the meeting ended as you were speaking"}
        self.world.add_room_line(m, "agent", text)
        self._say("[%s] you (in '%s'): %s" % (self.world.now(), m["topic"], text))
        speaker, reply_text = self._offer_room_turn(m)
        reply = None
        if speaker is not None:
            self.world.advance_clock_to(min(self.world.clock + 2, m["end"]))
            self.world.add_room_line(m, speaker.id, reply_text)
            self._say("[%s] %s (in meeting): %s"
                      % (self.world.now(), speaker.name, reply_text))
            reply = "%s: %s" % (speaker.name, reply_text)
        self._drain_due()   # crossing m["end"] fires meeting_end naturally
        return {"reply": reply or "(no one responded)",
                "meeting_minutes_left": max(0, m["end"] - self.world.clock)}

    def agent_write_doc(self, title, content, share_with):
        """Write a doc and share it — shared docs enter recipients' context."""
        unknown = [a for a in share_with if a not in self.world.npcs]
        if unknown:
            return {"error": "unknown people: %s" % unknown}
        self._advance_action_clock()
        doc = self.world.add_doc(title, content, share_with)
        for a in share_with:
            if a not in self.world.npcs:
                continue
            others = [self.world.npcs[o].name for o in share_with
                      if o != a and o in self.world.npcs]
            cc = (" (also shared with %s)" % ", ".join(others)) if others else ""
            self.world.npcs[a].notify(self.world.clock,
                                      "doc shared with you%s — '%s':\n%s"
                                      % (cc, title, content[:800]))
        self._say("[%s] you shared doc '%s' with %s"
                  % (self.world.now(), title, ", ".join(share_with) or "nobody"))
        return doc

    def agent_add_task(self, spec):
        """Agent files a task. If the id matches a ticket that was announced
        to the PM but never put on the board (an unfiled external arrival),
        THAT task is filed officially — it becomes visible and assignable.
        Otherwise a fresh agent-created tracking item is added, which carries
        zero rubric weight: agents can't mint value."""
        existing = self.world.find_task(spec.get("id") or "")
        if existing is not None and existing.get("filed") is False:
            self._advance_action_clock()
            changes = {"filed": True}
            # file AND assign in one call if an assignee was given — otherwise
            # the agent's obvious "add_task(id, assignee=dave)" would file the
            # ticket but leave it unowned (a silent no-op on the assignment).
            who = (spec.get("assignees") or [None])[0]
            spec_npc = next((n for n in self.world.scenario["npcs"]
                             if n["id"] == who), None) if who else None
            assigned = None
            if spec_npc is not None and spec_npc.get("worker", True):
                changes["assignees"] = [who]
                changes["assigned_at"] = self.world.clock
                assigned = who
            self.world.update_task(existing["id"], changes, "agent")
            if assigned:
                self.world.npcs[assigned].notify(
                    self.world.clock, "you've been assigned: '%s'" % existing["title"])
            self._say("[%s] you filed ticket %s on the board%s"
                      % (self.world.now(), existing["id"],
                         (" -> %s" % assigned) if assigned else ""))
            return {"filed": existing["id"], "title": existing["title"],
                    "priority": existing.get("priority"), "assigned_to": assigned}
        msg = check_task_bounds(spec, len(self.world.tasks), self.bounds)
        if msg:
            return {"error": msg}
        self._advance_action_clock()
        task = self.world.add_task(spec, source="agent")
        self._say("[%s] you added task: %s" % (self.world.now(), task["title"]))
        return task

    # -- time advancement -----------------------------------------------------

    def step(self):
        """Pop the next event, jump the clock to it, dispatch. Returns the event.

        With the causality invariant (_advance_action_clock drains due events
        on every action advance), a popped event's time is never in the past;
        the max() is a belt-and-suspenders monotonicity guard that logs if it
        ever actually engages.
        """
        if not len(self.queue):
            return None
        event = self.queue.pop()
        if event.time < self.world.clock:
            self.logger.error(
                "CAUSALITY GUARD ENGAGED: event seq=%d t=%d popped at clock=%d "
                "— invariant violated somewhere upstream",
                event.seq, event.time, self.world.clock)
        self.world.advance_clock_to(max(self.world.clock, event.time))
        self.logger.debug("pop seq=%d kind=%s t=%s payload=%s",
                          event.seq, event.kind, self.world.now(), event.payload)
        self._dispatch(event)
        self._check_completions()
        return event

    def run_until_reply(self, max_events=25, horizon=None):
        """Advance until a message for the agent arrives.

        Gives up at `horizon` (default: ~1.5 workdays out) WITHOUT firing
        events beyond it — waiting must never silently burn the week when
        nothing is ever going to message the agent.
        """
        if horizon is None:
            horizon = self.world.clock + 36 * 60
        for _ in range(max_events):
            nxt = self.queue.peek()
            if nxt is None or nxt.time > horizon:
                self.advance_until(horizon)
                return None
            before = len(self.world.log)
            self.step()
            # a chat arriving or an email batch landing both count as "the
            # next message for you" — everything reaches the PM as an event
            msg = next((e for e in self.world.log[before:]
                        if (e["kind"] == "message" and e["recipient"] == "agent"
                            and e.get("via", "chat") == "chat")
                        or (e["kind"] == "email_delivered"
                            and e.get("recipient", "agent") == "agent")), None)
            if msg is not None:
                # CAUSALITY: drain co-temporal events before returning control
                self._drain_due()
                return msg
        return None

    def advance_until(self, target, max_events=1000, interruptible=False):
        """Operator time control: fire everything up to `target`, then jump
        the clock there — even through empty space (nights, weekends).

        `interruptible` is the PM's sleep semantics: a PUSH-class signal
        (chat, room line, completion broadcast — never email) breaks the
        advance and hands control back at the interruption instant, like a
        notification waking a human. Pull signals let you sleep.

        LIVENESS: always reaches `target` unless interrupted (max_events is a
        runaway guard far above any legitimate event density; hitting it is
        logged loudly)."""
        fired = 0
        while fired < max_events:
            nxt = self.queue.peek()
            if nxt is None or nxt.time > target:
                break
            self.step()
            fired += 1
            # a chat ping, an email batch landing, or someone speaking in
            # the meeting you're in — all wake you. Board broadcasts
            # (completions) are delivered but never break sleep.
            if interruptible and any(
                    "chat_from" in i or "email_from" in i or "in_meeting" in i
                    or "work_revealed" in i
                    for i in self._push_items(self.world.log[self._push_cursor:])):
                # CAUSALITY: being woken must not leave co-temporal events
                # undispatched — everything due at this instant happens
                # before the agent gets control back.
                self._drain_due()
                return fired
        else:
            self.logger.warning("advance_until hit max_events=%d before "
                                "reaching %s — runaway event source?",
                                max_events, target)
        if self.world.clock < target:
            self.world.advance_clock_to(target)
            self.world.record("clock_advanced", to=target)
            self._say("[%s] (time passes)" % self.world.now())
        self._check_completions()
        return fired

    # -- dispatch --------------------------------------------------------------

    def _dispatch(self, event):
        npc = self.world.npcs.get(event.payload.get("npc") or "")

        if event.kind == "task_arrival":
            arr = self.world.scenario["task_arrivals"][event.payload["arrival"]]
            msg = check_task_bounds(arr["task"], len(self.world.tasks), self.bounds)
            if msg:
                self.world.record("task_rejected", reason=msg,
                                  task_id=arr["task"].get("id"))
                self._say("[%s] (external task rejected: %s)" % (self.world.now(), msg))
                return
            # Real work always has an OWNER — it lands already assigned to a
            # default holder and ON the board (no unowned backlog). The default
            # is often sub-optimal (a swamped person, the wrong specialist); the
            # PM's job is to REARRANGE it (reassign / reprioritize), not file it
            # from scratch. The ask still reaches the PM as a push so they know
            # to act. assigned_at = arrival, so the holder works it from now.
            task = self.world.add_task(
                dict(arr["task"], filed=True, assigned_at=self.world.clock),
                source="external")
            self._say("[%s] (external task arrived: %s -> %s)"
                      % (self.world.now(), task["title"],
                         ", ".join(task.get("assignees") or ["unowned"])))
            if npc is not None and arr.get("announce"):
                via = arr.get("via", "chat")
                text = self._llm(lambda: npc.ping(self.client, self.world, arr["announce"]))
                if via == "email":
                    text = "[%s] %s" % (task["title"], text)
                # rules carry the reference + current owner (voice from LLM)
                text = "%s\n(ticket: %s, currently on %s)" % (
                    text, task["id"], ", ".join(task.get("assignees") or ["nobody"]))
                msg = self.world.send_message(npc.id, "agent", text, via=via)
                if via == "email":
                    self._schedule_email_delivery(msg)
                self._say("[%s] %s (%s): %s" % (self.world.now(), npc.name, via, text))
            return

        if event.kind == "work_reveal_check":
            self._reveal_checks.discard(
                (event.payload["task"], event.time))
            return  # step() runs the deterministic completion/reveal fold

        if event.kind == "org_pickup":
            arr = self.world.scenario["task_arrivals"][event.payload["arrival"]]
            t = self.world.find_task(arr["task"]["id"])
            if t is None or t.get("assignees"):
                return  # the PM already handled it — no pickup needed
            self.world.update_task(t["id"],
                                   {"assignees": [npc.id], "filed": True,
                                    "assigned_at": self.world.clock,
                                    "reported": "%s picked this up (nobody "
                                                "was on it)" % npc.name},
                                   "org")
            npc.notify(self.world.clock,
                       "'%s' was sitting unowned, so you've just picked it up "
                       "yourself" % t["title"])
            self._say("[%s] (org: %s picks up unowned '%s')"
                      % (self.world.now(), npc.name, t["title"]))
            return

        if event.kind == "email_delivery":
            m = self.world.messages[event.payload["msg_id"]]
            self.world.record("email_delivered", msg_id=m.id, sender=m.sender,
                              text=m.text)
            self._say("[%s] (email from %s lands in your inbox)"
                      % (self.world.now(), m.sender))
            return

        if event.kind == "meeting_start":
            m = next(x for x in self.world.meetings
                     if x["id"] == event.payload["meeting"])
            if m.get("cancelled"):
                return  # cancelled before it began — the room never opens
            self.world.record("meeting_started", meeting_id=m["id"], topic=m["topic"])
            self._say("[%s] (meeting '%s' starts: %s)"
                      % (self.world.now(), m["topic"], ", ".join(m["attendees"])))
            return

        if event.kind == "room_turn":
            m = next(x for x in self.world.meetings
                     if x["id"] == event.payload["meeting"])
            if m.get("cancelled"):
                return
            if not (m["start"] <= self.world.clock < m["end"]):
                return
            minutes = m.get("minutes", [])
            if minutes and self.world.clock - minutes[-1]["t"] < 4:
                return  # room is already talking — don't speak over people
            speaker, text = self._offer_room_turn(m)
            if speaker is None:
                return  # everyone passed — real meetings have silences too
            self.world.add_room_line(m, speaker.id, text)
            self._say("[%s] %s (in '%s'): %s"
                      % (self.world.now(), speaker.name, m["topic"], text))
            return

        if event.kind == "meeting_end":
            m = next(x for x in self.world.meetings
                     if x["id"] == event.payload["meeting"])
            if m.get("cancelled"):
                return  # no transcript for a meeting that never happened
            minutes = m.get("minutes", [])
            if minutes:
                # LIVE meeting: the transcript is the VERBATIM minutes —
                # deterministic record of what was actually said.
                agent_name = self.world.scenario.get("agent_name", "PM")
                text = "\n".join(
                    "%s: %s" % (agent_name if l["speaker"] == "agent"
                                else self.world.npcs[l["speaker"]].name, l["text"])
                    for l in minutes)
            else:
                # nobody was in the room live (NPC-only) — lazy one-shot synthesis
                text = self._llm(lambda: generate_transcript(self.client,
                                                             self.world, m))
            self.world.add_transcript(m, text)
            if not m.get("minutes"):
                for a in m["attendees"]:
                    if a in self.world.npcs:
                        self.world.npcs[a].notify(
                            self.world.clock, "the meeting '%s' you attended:\n%s"
                            % (m["topic"], text[:2000]))
            self._say("[%s] (meeting '%s' ended — transcript captured)"
                      % (self.world.now(), m["topic"]))
            return

        if event.kind == "belief_update":
            t = self.world.find_task(event.payload["task"])
            b = (t.get("belief") or [])[event.payload["idx"]]
            from .tasks import belief_hours_left
            self.world.update_task(t["id"],
                                   {"belief_pct": b.get("pct"),
                                    "belief_remaining": belief_hours_left(
                                        b, t.get("effort_hours")),
                                    "belief_note": b.get("note", "")},
                                   "belief")
            # the slip is the PINNED holder's estimate correcting (their blind
            # spot), not whoever currently owns the task — so reassigning to a
            # fresh owner doesn't magically move the discovery to them.
            holder = b.get("held_by") or t.get("belief_holder") \
                or (t.get("assignees") or [None])[0]
            npc_h = self.world.npcs.get(holder)
            if npc_h is not None:
                npc_h.notify(self.world.clock,
                             "you've just realized about '%s': %s"
                             % (t["title"], b.get("note", "")))
                if b.get("proactive_ping"):
                    self.world.record("belief_revealed", task_id=t["id"],
                                      holder=holder, idx=event.payload["idx"])
                    text = self._llm(lambda: npc_h.ping(
                        self.client, self.world,
                        "You just discovered: %s. You've decided to tell Alex "
                        "now, in your own way." % b.get("note", "")))
                    self.world.send_message(npc_h.id, "agent", text)
                    self._say("[%s] %s: %s" % (self.world.now(), npc_h.name, text))
            self._say("[%s] (%s's belief about '%s' updated)"
                      % (self.world.now(), holder, t["id"]))
            return

        if event.kind == "question_ping":
            t = self.world.find_task(event.payload["task"])
            q = (t.get("questions") or [])[event.payload["idx"]]
            holder = (t.get("assignees") or [None])[0]
            npc_h = self.world.npcs.get(holder)
            if npc_h is not None:
                # the owner now KNOWS they're stuck (honest testimony if asked)
                # and reaches out to the PM. The block is derived from this
                # message's answer, not from this event — replay-safe.
                npc_h.notify(self.world.clock,
                             "you're blocked on '%s' — you need the PM's call "
                             "before you can keep going, and you've flagged it"
                             % t["title"])
                text = self._llm(lambda: npc_h.ping(
                    self.client, self.world,
                    "You're blocked on '%s' and need Alex's answer to continue. "
                    "Raise it now, in your own way: %s"
                    % (t["title"], q.get("ping", "you need a decision from the PM"))))
                effects = q.get("choice_effects") or {}
                choices = [
                    "%s%s" % (choice,
                              " (%s)" % effects[choice].get("summary", "")
                              if choice in effects else "")
                    for choice in (q.get("options") or [])
                ]
                text = ("%s\n(blocking question: task=%s, question_id=%s, "
                        "options=%s)" % (
                            text, t["id"], q["id"], ", ".join(choices)))
                self.world.send_message(npc_h.id, "agent", text)
                self._say("[%s] %s: %s" % (self.world.now(), npc_h.name, text))
            return

        if event.kind == "npc_wakeup":
            self.world.record("npc_wakeup", npc=npc.id)
            self._schedule(npc.next_wakeup(self.world.clock), "npc_wakeup", {"npc": npc.id})
            self._say("[%s] (%s checks their messages)" % (self.world.now(), npc.name))
            # THE WAKEUP'S JOB: process the email batch (unless in a meeting —
            # then the batch waits for the next surfacing).
            if npc.unanswered_emails and self.world.active_meeting_with(npc.id) is None:
                npc.unanswered_emails = 0
                text = self._llm(lambda: npc.reply(self.client, self.world, via="email"))
                msg = self.world.send_message(npc.id, "agent", text, via="email")
                self._schedule_email_delivery(msg)
                self._say("[%s] %s (email): %s" % (self.world.now(), npc.name, text))

        elif event.kind == "npc_respond":
            via = self.world.messages[event.payload["message_id"]].via
            text = self._llm(lambda: npc.reply(self.client, self.world, via=via))
            self.world.send_message(npc.id, "agent", text)
            self._say("[%s] %s: %s" % (self.world.now(), npc.name, text))

        elif event.kind == "beat":
            beat = self.world.scenario["beats"][event.payload["beat"]]
            arm = self._resolve_beat_arm(beat, npc.id)
            if arm is not None:
                text = self._llm(lambda: npc.ping(self.client, self.world, arm["intent"]))
                self.world.send_message(npc.id, "agent", text)
                self._say("[%s] %s: %s" % (self.world.now(), npc.name, text))
            else:
                self.world.record("beat_skipped", beat=event.payload["beat"], npc=npc.id)
                self._say("[%s] (beat skipped — no arm matched)" % self.world.now())

    def _resolve_beat_arm(self, beat, npc_id):
        """Beats are IF-chains: the first arm whose condition matches the world
        fires with that arm's intent. Conditions are deterministic world-state
        predicates — rules pick the arm, the LLM only voices it. A beat with a
        bare `intent` (+optional `condition`) is a single-arm chain."""
        arms = beat.get("arms")
        if arms is None:
            arms = [{"condition": beat.get("condition"), "intent": beat["intent"]}]
        for arm in arms:
            if self._beat_condition(arm.get("condition"), npc_id):
                return arm
        return None

    def _beat_condition(self, cond, npc_id):
        if not cond:
            return True
        ok = True
        if "task_not_done" in cond:
            ok = ok and not self.world.task_done(cond["task_not_done"])
        if "task_done" in cond:
            ok = ok and self.world.task_done(cond["task_done"])
        if "agent_messaged_me" in cond:
            # existence check only — never content/keywords
            has = any(m.sender == "agent" and m.recipient == npc_id
                      for m in self.world.messages)
            ok = ok and (has == bool(cond["agent_messaged_me"]))
        return ok

    def _llm(self, call):
        """LIVENESS: an NPC brain failure must NEVER kill the sim — any
        exception degrades to placeholder text and the queue keeps moving."""
        try:
            return call()
        except anthropic.RateLimitError:
            self.logger.warning("llm rate limited")
            return "(rate limited — try again in a moment)"
        except anthropic.APIStatusError as e:
            self.logger.error("llm API error %s: %s", e.status_code, e.message)
            return "(API error %s: %s)" % (e.status_code, e.message)
        except anthropic.APIConnectionError:
            self.logger.error("llm connection error")
            return "(network error reaching the Claude API)"
        except Exception:
            self.logger.exception("npc brain failure (degraded to placeholder)")
            return "(…)"

    def _say(self, line):
        self.logger.info(line)
        if self.verbose:
            print(line)
