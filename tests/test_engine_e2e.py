"""End-to-end engine flows on the stub LLM: arrivals as communication,
filing, the org fallback, batched email, wake semantics, liveness, replay."""

import unittest
from unittest.mock import patch

from sim.replay import load_run, replay
from sim.tools import call_tool

from tests.fixtures import HORIZON, goto, make_engine, scenario, unwrap


def advance_collect(eng, target):
    """Advance to an absolute time, collecting every push notification."""
    notifs = []
    while eng.world.clock < target:
        r = call_tool(eng, "advance_time",
                      {"minutes": target - eng.world.clock})
        _, n = unwrap(r)
        notifs += n
    return notifs


class TestArrivals(unittest.TestCase):
    def test_chat_announce_wakes_with_ticket_and_owner(self):
        eng = make_engine()
        notifs = advance_collect(eng, 2100)  # incident announced at 2040
        self.assertTrue(any(n.get("chat_from") == "bo"
                            and "(ticket: incident, currently on bo)"
                            in n.get("text", "")
                            for n in notifs))
        # the proactive belief confession (t=2000) is a push too
        self.assertTrue(any(n.get("chat_from") == "ana" for n in notifs))
        self.assertEqual(eng.world.find_task("core")["belief_pct"], 40)

    def test_arrival_lands_owned_and_on_the_board(self):
        """The contract: real work always has an owner — an arrival is on the
        board, assigned to its default holder, worked from the instant it
        lands (assigned_at = arrival). The PM's lever is rearranging."""
        eng = make_engine()
        goto(eng, 2100)
        t = eng.world.find_task("incident")
        self.assertEqual(t["assignees"], ["bo"])
        self.assertTrue(t["filed"])
        self.assertEqual(t["assigned_at"], 2040)
        self.assertIn("incident", {r["id"] for r in eng.world.tracker_view()})

    def test_reassignment_stamps_handoff_and_keeps_progress(self):
        eng = make_engine()
        goto(eng, 2100)
        r, _ = unwrap(call_tool(eng, "assign_task",
                                {"task_id": "incident", "npc": "ana"}))
        self.assertEqual(r.get("assigned"), "incident")
        # tracker-shaped ack only — never the truth dict
        self.assertNotIn("belief", r)
        self.assertNotIn("effort_hours", r)
        t = eng.world.find_task("incident")
        self.assertEqual(t["assignees"], ["ana"])
        self.assertGreaterEqual(t["assigned_at"], 2100)  # handoff is NOW
        # bo's ~hour of work before the handoff is persisted, not re-credited
        self.assertGreater(t.get("done_hours", 0), 0)

    def test_configured_handoff_cost_delays_new_owner(self):
        scn = scenario()
        scn["costs"] = {"handoff_minutes": 45}
        charged = make_engine(scn)
        free = make_engine()
        for eng in (charged, free):
            goto(eng, 2100)
        before = charged.world.find_task("incident").get("done_hours", 0)
        result, _ = unwrap(call_tool(
            charged, "assign_task", {"task_id": "incident", "npc": "ana"}))
        unwrap(call_tool(free, "assign_task",
                         {"task_id": "incident", "npc": "ana"}))
        self.assertEqual(result["handoff_minutes"], 45)
        event = next(e for e in charged.world.log
                     if e["kind"] == "capacity_cost")
        self.assertEqual((event["person"], event["minutes"], event["reason"]),
                         ("ana", 45, "handoff"))
        self.assertIn((2101, 2146), charged.world.busy_by_assignee()["ana"])
        after = charged.world.find_task("incident")
        self.assertGreater(after["done_hours"], before)
        charged_eta = next(r for r in charged.world.tasks_view()
                           if r["id"] == "incident")["projected_done"]
        free_eta = next(r for r in free.world.tasks_view()
                        if r["id"] == "incident")["projected_done"]
        self.assertGreater(charged_eta, free_eta)

    def test_configured_helper_onboarding_cost(self):
        scn = scenario()
        scn["costs"] = {"helper_onboarding_minutes": 30}
        eng = make_engine(scn)
        result, _ = unwrap(call_tool(
            eng, "add_helper", {"task_id": "core", "npc": "bo"}))
        self.assertEqual(result["onboarding_minutes"], 30)
        event = next(e for e in eng.world.log if e["kind"] == "capacity_cost")
        self.assertEqual((event["person"], event["task_id"],
                          event["minutes"], event["reason"]),
                         ("bo", "core", 30, "helper_onboarding"))
        self.assertIn((546, 576), eng.world.busy_by_assignee()["bo"])

    def test_same_owner_assignment_is_noop(self):
        eng = make_engine()
        before_clock, before_log = eng.world.clock, len(eng.world.log)
        result = call_tool(eng, "assign_task",
                           {"task_id": "core", "npc": "ana"})
        self.assertIn("error", result)
        self.assertEqual(eng.world.clock, before_clock)
        self.assertEqual(len(eng.world.log), before_log)


class TestChannels(unittest.TestCase):
    def test_email_arrival_delivers_on_batch_grid(self):
        eng = make_engine()
        notifs = advance_collect(eng, 3600)  # email arrival 3540 -> tick 3570
        mail = [n for n in notifs if "email_from" in n]
        self.assertTrue(any("(ticket: form" in n["text"] for n in mail))
        delivered = [e for e in eng.world.log if e["kind"] == "email_delivered"]
        self.assertEqual(delivered[0]["t"], 3570)

    def test_completions_deliver_but_never_wake(self):
        """Board broadcasts are delivered with tool results but only PEOPLE
        (chat, delivered email, room lines) may interrupt an advance."""
        eng = make_engine()
        saw_completion = False
        while eng.world.clock < 2600:   # ana's core completes ~Tue 12:45
            r = call_tool(eng, "advance_time",
                          {"minutes": 2600 - eng.world.clock})
            res, notifs = unwrap(r)
            saw_completion = saw_completion or any("tracker" in n for n in notifs)
            if res.get("interrupted"):
                self.assertTrue(any("chat_from" in n or "email_from" in n
                                    or "in_meeting" in n for n in notifs),
                                "woken with no person-signal in the batch")
        self.assertTrue(saw_completion)

    def test_chat_wakes_mid_advance(self):
        eng = make_engine()
        r, _ = unwrap(call_tool(eng, "advance_time", {"minutes": 3000}))
        self.assertTrue(r.get("interrupted"))       # woken by a person
        self.assertLess(eng.world.clock, 545 + 3000)

    def test_calendar_exposes_out_of_office_blocks(self):
        scn = scenario()
        scn["npcs"][0]["ooo"] = [[600, 660]]
        eng = make_engine(scn)
        result, _ = unwrap(call_tool(eng, "view_calendar", {}))
        ooo = next(m for m in result["meetings"]
                   if m.get("type") == "out_of_office")
        self.assertEqual(ooo["id"], "ooo-ana-0")
        self.assertEqual(ooo["topic"], "Out of office: Ana")
        self.assertEqual(ooo["start"], "Mon 10:00")
        self.assertEqual(ooo["end"], "Mon 11:00")
        self.assertEqual(ooo["attendees"], ["ana"])
        self.assertEqual(ooo["status"], "upcoming")

        goto(eng, 600)
        result, _ = unwrap(call_tool(eng, "view_calendar", {}))
        ooo = next(m for m in result["meetings"]
                   if m.get("type") == "out_of_office")
        self.assertEqual(ooo["status"], "now")
        result = call_tool(eng, "schedule_meeting",
                           {"attendees": ["ana"], "start_in_minutes": 30,
                            "duration_minutes": 30, "topic": "overlap"})
        self.assertIn("error", result)


class TestToolValidation(unittest.TestCase):
    def test_invalid_calls_return_errors_without_mutating_world(self):
        eng = make_engine()
        invalid = [
            ("missing required", "send_email",
             {"to": ["ana"], "subject": "status"}),
            ("list arguments", "view_tasks", []),
            ("wrong array item", "send_email",
             {"to": ["ana", 7], "subject": "status", "body": "update"}),
            ("boolean minutes", "advance_time", {"minutes": True}),
            ("nonfinite minutes", "advance_time", {"minutes": float("inf")}),
            ("negative minutes", "advance_time", {"minutes": -1}),
            ("negative meeting start", "schedule_meeting",
             {"attendees": ["ana"], "start_in_minutes": -1,
              "duration_minutes": 30, "topic": "status"}),
            ("zero meeting duration", "schedule_meeting",
             {"attendees": ["ana"], "start_in_minutes": 10,
              "duration_minutes": 0, "topic": "status"}),
            ("zero inbox limit", "view_inbox", {"limit": 0}),
            ("nonfinite number", "add_task",
             {"title": "bad estimate", "effort_hours": float("nan")}),
            ("legacy forecast fields", "forecast_completion",
             {"task_id": "side", "completion_time": 4000,
              "confidence": 1.1, "rationale": "Initial estimate"}),
            ("wrong boolean", "reprioritize",
             {"task_id": "side", "urgent": "false"}),
            ("invalid enum", "reprioritize",
             {"task_id": "side", "priority": "P9"}),
            ("unknown argument", "view_tasks", {"extra": True}),
            ("unknown tool", "not_a_tool", {}),
            ("non-string tool", ["view_tasks"], {}),
        ]
        for label, name, args in invalid:
            with self.subTest(label=label):
                before_clock = eng.world.clock
                before_log = list(eng.world.log)
                result = call_tool(eng, name, args)
                self.assertIn("error", result)
                self.assertEqual(eng.world.clock, before_clock)
                self.assertEqual(eng.world.log, before_log)

    def test_valid_calls_remain_valid(self):
        eng = make_engine()
        self.assertIn("tasks", call_tool(eng, "view_tasks", None))
        result = call_tool(eng, "advance_time", {"minutes": 0})
        self.assertNotIn("error", result)
        self.assertIn("now", result)
        self.assertIsInstance(call_tool(eng, "view_inbox", {"limit": 1}), list)

    def test_handler_error_still_wraps_push_notifications(self):
        eng = make_engine()
        notifications = [{"time": eng.world.now(), "chat_from": "ana",
                          "text": "update"}]
        with patch.object(eng, "drain_agent_push", return_value=notifications):
            result = call_tool(eng, "send_chat", {"text": "hello"})
        self.assertIn("error", result["result"])
        self.assertEqual(result["notifications"], notifications)


class TestWaitForReply(unittest.TestCase):
    def test_reply_is_found_before_completion_and_same_time_events_are_drained(self):
        eng = make_engine()
        call_tool(eng, "send_chat", {"npc": "ana", "text": "status?"})
        reply = next(e for e in eng.queue._heap if e.kind == "npc_respond")
        eng._schedule(reply.time, "belief_update", {"task": "core", "idx": 0})

        real_step = eng.step
        completion_added = [False]

        def step_with_trailing_completion():
            before = len(eng.world.log)
            event = real_step()
            delivered = any(e["kind"] == "message"
                            and e["recipient"] == "agent"
                            for e in eng.world.log[before:])
            if delivered and not completion_added[0]:
                eng.world.record("task_completed", task_id="synthetic",
                                 title="Synthetic", completed_at=eng.world.clock)
                completion_added[0] = True
            return event

        with patch.object(eng, "step", side_effect=step_with_trailing_completion):
            msg = eng.run_until_reply(max_events=100, horizon=reply.time)

        self.assertEqual(msg["kind"], "message")
        self.assertEqual(msg["recipient"], "agent")
        self.assertEqual(msg["t"], reply.time)
        self.assertEqual(eng.world.clock, reply.time)
        self.assertTrue(completion_added[0])
        self.assertFalse(any(e.time <= eng.world.clock for e in eng.queue._heap))

    def test_empty_queue_wait_advances_to_explicit_horizon(self):
        eng = make_engine()
        while len(eng.queue):
            eng.queue.pop()
        horizon = eng.world.clock + 90
        self.assertIsNone(eng.run_until_reply(horizon=horizon))
        self.assertEqual(eng.world.clock, horizon)


class TestBlockingQuestions(unittest.TestCase):
    def _engine(self):
        scn = scenario()
        scn["project"]["tasks"][0]["questions"] = [
            {"id": "scope", "at": 600, "gates": True, "held_by": "ana",
             "options": ["cut", "keep"]}]
        return make_engine(scn)

    def _effect_engine(self, capacity=True):
        scn = scenario()
        cut = {"summary": "cut scope", "task_effort_delta_hours": -2}
        if capacity:
            cut["capacity_costs"] = [
                {"npc": "ana", "minutes": 60, "reason": "customer note"}]
        scn["project"]["tasks"][0]["questions"] = [{
            "id": "scope", "at": 600, "gates": True, "held_by": "ana",
            "options": ["cut", "keep"],
            "choice_effects": {
                "cut": cut,
                "keep": {"summary": "keep scope",
                         "task_effort_delta_hours": 2}}}]
        return make_engine(scn)

    def test_choice_effects_change_effort_and_completion(self):
        slow = self._effect_engine()
        fast = self._effect_engine()
        for eng, choice in ((slow, "keep"), (fast, "cut")):
            goto(eng, 600)
            result, _ = unwrap(call_tool(
                eng, "answer_question",
                {"task_id": "core", "question_id": "scope", "choice": choice,
                 "rationale": "deterministic tradeoff", "via": "chat"}))
            self.assertEqual(result["effect_summary"],
                             "keep scope" if choice == "keep" else "cut scope")
        slow_view = next(r for r in slow.world.tasks_view(at=601)
                         if r["id"] == "core")
        fast_view = next(r for r in fast.world.tasks_view(at=601)
                         if r["id"] == "core")
        self.assertEqual(slow_view["effort_hours"], 22)
        self.assertEqual(fast_view["effort_hours"], 18)
        fast_eta = next(r for r in fast.world.tasks_view()
                        if r["id"] == "core")["projected_done"]
        slow_eta = next(r for r in slow.world.tasks_view()
                        if r["id"] == "core")["projected_done"]
        self.assertLess(fast_eta, slow_eta)

    def test_email_choice_effect_activates_at_delivery(self):
        eng = self._effect_engine()
        goto(eng, 600)
        result, _ = unwrap(call_tool(
            eng, "answer_question",
            {"task_id": "core", "question_id": "scope", "choice": "cut",
             "rationale": "cut after review", "via": "email"}))
        before = next(r for r in eng.world.tasks_view(
            at=result["delivered_at"] - 1) if r["id"] == "core")
        after = next(r for r in eng.world.tasks_view(
            at=result["delivered_at"]) if r["id"] == "core")
        self.assertEqual(before["effort_hours"], 20)
        self.assertEqual(after["effort_hours"], 18)
        self.assertEqual(before["status"], "blocked")
        self.assertNotEqual(after["status"], "blocked")

    def test_choice_capacity_cost_and_tradeoff_observability(self):
        eng = self._effect_engine()
        twin = self._effect_engine(capacity=False)
        for e in (eng, twin):
            goto(e, 600)
        row = next(t for t in call_tool(eng, "view_tasks", {})["tasks"]
                   if t["id"] == "core")
        self.assertEqual(row["open_questions"][0]["tradeoffs"]["cut"],
                         "cut scope")
        for e in (eng, twin):
            unwrap(call_tool(
                e, "answer_question",
                {"task_id": "core", "question_id": "scope", "choice": "cut",
                 "rationale": "cut scope", "via": "chat"}))
        busy = eng.world.busy_by_assignee()["ana"]
        self.assertIn((601, 621), busy)   # explicit decision is itself a chat
        self.assertIn((621, 681), busy)   # authored 60-min consequence after it
        effect = eng.world.decision_effects()["core"][0]
        self.assertEqual(effect["choice"], "cut")
        self.assertEqual(effect["capacity_costs"][0]["minutes"], 60)
        projected = next(r for r in eng.world.tasks_view(at=1200)
                         if r["id"] == "core")["projected_done"]
        twin_projected = next(r for r in twin.world.tasks_view(at=1200)
                              if r["id"] == "core")["projected_done"]
        self.assertGreater(projected, 681)
        self.assertGreater(projected, twin_projected)

    def test_delivered_choice_updates_accurate_belief_remaining(self):
        eng = self._effect_engine()
        goto(eng, 600)
        unwrap(call_tool(
            eng, "answer_question",
            {"task_id": "core", "question_id": "scope", "choice": "cut",
             "rationale": "cut scope", "via": "chat"}))
        call_tool(eng, "add_helper", {"task_id": "core", "npc": "bo"})
        view = next(r for r in eng.world.belief_view("bo")
                    if r["task"] == "Core work")
        self.assertEqual(view["your_view"], "~9h of work left")

    def test_replayed_world_preserves_decision_physics(self):
        eng = self._effect_engine()
        goto(eng, 600)
        unwrap(call_tool(
            eng, "answer_question",
            {"task_id": "core", "question_id": "scope", "choice": "cut",
             "rationale": "cut scope", "via": "chat"}))
        goto(eng, 900)
        scn, events = load_run(eng.run_dir)
        replayed = replay(scn, events)
        self.assertEqual(replayed.decision_effects(), eng.world.decision_effects())
        self.assertEqual(replayed.busy_by_assignee(),
                         eng.world.busy_by_assignee())
        self.assertEqual(replayed.tasks_view(), eng.world.tasks_view())

    def test_generic_chat_does_not_answer_and_explicit_chat_unblocks(self):
        eng = self._engine()
        goto(eng, 600)
        call_tool(eng, "send_chat", {"npc": "ana", "text": "keep going"})
        self.assertEqual(eng.world.question_answers(), {})
        tracker = call_tool(eng, "view_tasks", {})["tasks"]
        core = next(t for t in tracker if t["id"] == "core")
        self.assertEqual(core["open_questions"], [{
            "id": "scope", "prompt": "", "options": ["cut", "keep"]}])
        rows = {r["id"]: r for r in eng.world.tasks_view(at=700)}
        self.assertEqual(rows["core"]["status"], "blocked")

        before_clock = eng.world.clock
        invalid = call_tool(eng, "answer_question",
                            {"task_id": "core", "question_id": "scope",
                             "choice": "delay", "rationale": "bad", "via": "chat"})
        self.assertIn("error", invalid)
        self.assertEqual(eng.world.clock, before_clock)

        result, _ = unwrap(call_tool(
            eng, "answer_question",
            {"task_id": "core", "question_id": "scope", "choice": "cut",
             "rationale": "ship the committed outcome", "via": "chat"}))
        self.assertEqual(result["delivered_at"], eng.world.clock)
        event = next(e for e in reversed(eng.world.log)
                     if e["kind"] == "question_answered")
        self.assertEqual(event["question_id"], "scope")
        self.assertEqual(event["choice"], "cut")
        self.assertEqual(event["owner"], "ana")
        self.assertEqual(eng.world.question_answers()["core"]["scope"],
                         result["delivered_at"])
        tracker = call_tool(eng, "view_tasks", {})["tasks"]
        core = next(t for t in tracker if t["id"] == "core")
        self.assertNotIn("open_questions", core)
        rows = {r["id"]: r for r in eng.world.tasks_view(at=700)}
        self.assertEqual(rows["core"]["status"], "in_progress")

    def test_email_decision_unblocks_at_next_answer_batch(self):
        eng = self._engine()
        goto(eng, 600)
        result, _ = unwrap(call_tool(
            eng, "answer_question",
            {"task_id": "core", "question_id": "scope", "choice": "cut",
             "rationale": "ship the committed outcome", "via": "email"}))
        expected = (eng.world.clock // 120 + 1) * 120
        self.assertEqual(result["delivered_at"], expected)
        self.assertEqual(eng.world.question_answers()["core"]["scope"], expected)
        rows = {r["id"]: r for r in eng.world.tasks_view(at=expected - 1)}
        self.assertEqual(rows["core"]["status"], "blocked")
        rows = {r["id"]: r for r in eng.world.tasks_view(at=expected)}
        self.assertEqual(rows["core"]["status"], "in_progress")


class TestLiveness(unittest.TestCase):
    def test_unknown_recipients_bounce_not_crash(self):
        eng = make_engine()
        n_msgs = len(eng.world.messages)
        r, _ = unwrap(call_tool(eng, "send_email",
                                {"to": ["ghost@nowhere"], "subject": "s",
                                 "body": "b"}))
        self.assertEqual(r, {"sent": False, "bounced": ["ghost@nowhere"]})
        r, _ = unwrap(call_tool(eng, "send_chat", {"npc": "ghost", "text": "x"}))
        # chat and email bounce with the SAME shape — one contract
        self.assertEqual(r, {"sent": False, "bounced": ["ghost"]})
        self.assertEqual(len(eng.world.messages), n_msgs)  # nothing delivered

    def test_group_chat_is_one_send(self):
        """A group ping: one action-minute total, delivered simultaneously,
        every recipient sees the to-line, every copy shares a group id,
        and EACH recipient pays their own focus tax."""
        eng = make_engine()
        t0 = eng.world.clock
        r, _ = unwrap(call_tool(eng, "send_chat",
                                {"to": ["ana", "bo"], "text": "standup in 5"}))
        self.assertTrue(r.get("sent"))
        self.assertEqual(eng.world.clock, t0 + 1)        # ONE composition
        msgs = eng.world.messages[-2:]
        self.assertEqual(msgs[0].group, msgs[1].group)   # explicit group id
        self.assertIsNotNone(msgs[0].group)
        ana = "\n".join(m["content"] for m in eng.world.npcs["ana"].context
                        if m["role"] == "user")
        self.assertIn("with Bo", ana)                    # coherent to-line
        busy = eng.world.busy_by_assignee()
        self.assertIn("ana", busy)                       # both taxed
        self.assertIn("bo", busy)

    def test_reprioritize_tool(self):
        eng = make_engine()
        r, _ = unwrap(call_tool(eng, "reprioritize",
                                {"task_id": "side", "priority": "P0"}))
        self.assertEqual(r.get("order_priority"), "P0")
        self.assertEqual(eng.world.find_task("side")["priority"], "P2")
        r, _ = unwrap(call_tool(eng, "reprioritize",
                                {"task_id": "side", "priority": "P9"}))
        self.assertIn("error", r)
        r, _ = unwrap(call_tool(eng, "reprioritize", {"task_id": "nope"}))
        self.assertIn("error", r)  # unknown task

    def test_stakeholders_take_no_tickets(self):
        eng = make_engine()
        r, _ = unwrap(call_tool(eng, "assign_task",
                                {"task_id": "side", "npc": "vp"}))
        self.assertIn("stakeholder", r.get("error", ""))


class TestReplay(unittest.TestCase):
    def test_replay_reconstructs_the_live_world(self):
        """Regression: replay once dropped Message.via, so the evaluator taxed
        email runs as chat. The fold must reproduce the live world exactly."""
        eng = make_engine()
        goto(eng, 2100)
        call_tool(eng, "add_task", {"title": "Incident", "id": "incident"})
        call_tool(eng, "assign_task", {"task_id": "incident", "npc": "bo"})
        call_tool(eng, "send_email", {"to": ["ana"], "subject": "s", "body": "b"})
        goto(eng, HORIZON)

        world2 = replay(*load_run(eng.run_dir))
        self.assertEqual(eng.world.messages, world2.messages)  # incl. via
        self.assertEqual(eng.world.tracker_view(), world2.tracker_view())
        self.assertEqual([e for e in eng.world.log],
                         [e for e in world2.log])
        # every interpersonal signal is stamped with delivery semantics
        for e in world2.log:
            if e["kind"] == "message":
                self.assertIn("delivery", e)


if __name__ == "__main__":
    unittest.main()
