"""Scoring anchors and authoring gates: the band, the ceiling, the fairness
gate, the fingerprint, and the anti-minting guard."""

import json
import os
import tempfile
import unittest

from sim.difficulty import fingerprint, stamp
from sim.eval import evaluate
from sim.tools import call_tool
from sim.validate import validate_scenario

from tests.fixtures import HORIZON, goto, make_engine, scenario


class TestBandAnchors(unittest.TestCase):
    def test_lazy_pm_scores_exactly_zero(self):
        """The baseline IS the do-nothing world — including the org fallback
        picking up both arrivals. A PM who adds nothing scores 0, not credit
        for work that happens anyway."""
        eng = make_engine()
        goto(eng, HORIZON)
        res = evaluate(eng.run_dir)
        self.assertAlmostEqual(res["score"], 0.0, places=3)
        self.assertIsInstance(res["reward"], float)
        # unfiled-or-not, every authored task stays in the denominators
        ids = {t["id"] for t in res["per_task_agent"]}
        self.assertLessEqual({"core", "side", "incident", "form"}, ids)

    def test_active_pm_lands_in_the_band(self):
        eng = make_engine()
        goto(eng, 2050)  # incident landed on bo at 2040
        goto(eng, 3580)  # form email delivered at 3570 — also on bo
        # the PM's lever: REARRANGE the pile-on-the-volunteer default
        call_tool(eng, "assign_task", {"task_id": "form", "npc": "ana"})
        # a made-up task must never mint value (weight 0), only cost capacity
        call_tool(eng, "add_task", {"title": "Fake thing", "effort_hours": 3})
        goto(eng, HORIZON)
        res = evaluate(eng.run_dir)
        self.assertGreaterEqual(res["score"], 0.0)   # preempting can't hurt
        self.assertLessEqual(res["score"], 1.0)      # OPT is a true ceiling
        ids = {t["id"] for t in res["per_task_agent"]}
        self.assertNotIn("fake-thing", ids)
        # odds is the same measurement reshaped: odds = N/(1-N). `score` is
        # rounded to 3dp while odds is computed unrounded, and the transform
        # amplifies rounding by 1/(1-N)^2 near the ceiling — tolerate that.
        if res["score_odds"] is not None and res["score"] < 1:
            slack = 0.001 / (1 - res["score"]) ** 2 + 0.01
            self.assertAlmostEqual(res["score_odds"],
                                   res["score"] / (1 - res["score"]),
                                   delta=slack)


class TestUtilizationSeesBusyLoad(unittest.TestCase):
    def test_meetings_count_as_load(self):
        """Regression: utilization/fairness once ignored meetings entirely —
        a PM could bury someone in sessions and still read as 'fair'."""
        quiet = make_engine()
        goto(quiet, HORIZON)
        loaded = make_engine()
        call_tool(loaded, "schedule_meeting",
                  {"attendees": ["ana"], "start_in_minutes": 30,
                   "duration_minutes": 120, "topic": "sync"})
        goto(loaded, HORIZON)
        rq = evaluate(quiet.run_dir)
        rl = evaluate(loaded.run_dir)
        self.assertGreater(rl["utilization"]["ana"],
                           rq["utilization"]["ana"])
        # calendar_load counts the meeting block itself (>= 2h occupied)
        self.assertGreaterEqual(loaded.world.calendar_load(HORIZON)["ana"],
                                rq.get("utilization")["ana"] and 2.0)

    def test_merge_minutes_never_double_counts(self):
        """Overlapping spans (a chat tax inside a meeting) count once."""
        from sim.tasks import merge_minutes
        self.assertEqual(
            merge_minutes([(600, 720), (700, 760)], 545, 6780), 160)

    def test_utilization_is_calendar_time_not_effort(self):
        """A swarm session must count ONCE for the owner (the meeting block),
        never block + banked deposit — and load can't exceed hours present."""
        eng = make_engine()
        call_tool(eng, "schedule_meeting",
                  {"attendees": ["ana", "bo"], "start_in_minutes": 30,
                   "duration_minutes": 120, "topic": "pair", "task": "side"})
        goto(eng, HORIZON)
        load = eng.world.calendar_load(HORIZON)
        avail = (HORIZON - 545) / 60.0
        for person, hours in load.items():
            self.assertLessEqual(hours, avail + 0.01,
                                 "%s occupied more hours than exist" % person)


class TestGates(unittest.TestCase):
    def test_valid_fixture_passes(self):
        self.assertEqual(validate_scenario(scenario()), [])

    def test_blocking_question_requires_explicit_options(self):
        s = scenario()
        s["project"]["tasks"][0]["questions"] = [{
            "id": "scope", "at": 600, "gates": True, "held_by": "ana",
        }]
        self.assertTrue(any("options" in e for e in validate_scenario(s)))
        s["project"]["tasks"][0]["questions"][0]["options"] = ["cut", "keep"]
        self.assertFalse(any("options" in e for e in validate_scenario(s)))

    def test_choice_effects_must_be_real_distinct_consequences(self):
        base = scenario()
        base["project"]["tasks"][0]["questions"] = [{
            "id": "scope", "at": 600, "gates": True, "held_by": "ana",
            "options": ["cut", "keep"],
            "choice_effects": {
                "cut": {"summary": "cut", "task_effort_delta_hours": -2},
                "keep": {"summary": "keep", "task_effort_delta_hours": 2}}}]
        self.assertEqual(validate_scenario(base), [])

        bad = scenario()
        bad["project"]["tasks"][0]["questions"] = [
            dict(base["project"]["tasks"][0]["questions"][0])]
        bad["project"]["tasks"][0]["questions"][0]["choice_effects"] = {
            "cut": {"summary": "cut", "task_effort_delta_hours": -2}}
        self.assertTrue(any("keys" in e for e in validate_scenario(bad)))

        same = scenario()
        same["project"]["tasks"][0]["questions"] = [
            dict(base["project"]["tasks"][0]["questions"][0])]
        same["project"]["tasks"][0]["questions"][0]["choice_effects"] = {
            "cut": {"summary": "different prose"},
            "keep": {"summary": "but the same physical consequence"}}
        self.assertTrue(any("differ" in e for e in validate_scenario(same)))

        worker = scenario()
        worker["project"]["tasks"][0]["questions"] = [
            dict(base["project"]["tasks"][0]["questions"][0])]
        worker["project"]["tasks"][0]["questions"][0]["choice_effects"]["cut"][
            "capacity_costs"] = [{"npc": "vp", "minutes": 0, "reason": "x"}]
        errors = validate_scenario(worker)
        self.assertTrue(any("non-worker" in e for e in errors))
        self.assertTrue(any("minutes" in e for e in errors))

        nonfinite = scenario()
        nonfinite["project"]["tasks"][0]["questions"] = [
            dict(base["project"]["tasks"][0]["questions"][0])]
        nonfinite["project"]["tasks"][0]["questions"][0]["choice_effects"][
            "cut"]["task_effort_delta_hours"] = float("nan")
        self.assertTrue(any("finite" in e for e in validate_scenario(nonfinite)))

        bounds = scenario()
        bounds["project"]["tasks"][0]["questions"] = [
            dict(base["project"]["tasks"][0]["questions"][0])]
        bounds["project"]["tasks"][0]["questions"][0]["choice_effects"][
            "keep"]["task_effort_delta_hours"] = 100
        self.assertTrue(any("out of bounds" in e
                            for e in validate_scenario(bounds)))

    def test_cumulative_choice_deltas_are_bounded(self):
        s = scenario()
        s["project"]["tasks"][0]["questions"] = [
            {"id": "first", "at": 600, "gates": True, "held_by": "ana",
             "options": ["cut-a", "keep-a"],
             "choice_effects": {
                 "cut-a": {"summary": "cut first",
                           "task_effort_delta_hours": -19},
                 "keep-a": {"summary": "keep first",
                            "task_effort_delta_hours": 0}}},
            {"id": "second", "at": 601, "gates": True, "held_by": "ana",
             "options": ["cut-b", "keep-b"],
             "choice_effects": {
                 "cut-b": {"summary": "cut second",
                           "task_effort_delta_hours": -19},
                 "keep-b": {"summary": "keep second",
                            "task_effort_delta_hours": 0}}}]
        self.assertTrue(any("choice deltas" in e
                            for e in validate_scenario(s)))

    def test_choice_delta_on_tracking_task_is_rejected(self):
        s = scenario()
        task = s["project"]["tasks"][0]
        task.pop("effort_hours")
        task["questions"] = [{
            "id": "scope", "at": 600, "gates": True, "held_by": "ana",
            "options": ["cut", "keep"],
            "choice_effects": {
                "cut": {"summary": "cut",
                        "task_effort_delta_hours": -1},
                "keep": {"summary": "keep",
                         "task_effort_delta_hours": 0}}}]
        self.assertTrue(any("numeric effort_hours" in e
                            for e in validate_scenario(s)))

    def test_gating_question_ids_are_globally_unique(self):
        s = scenario()
        s["project"]["tasks"][0]["questions"] = [{
            "id": "scope", "at": 600, "gates": True, "held_by": "ana",
            "options": ["cut", "keep"]}]
        s["project"]["tasks"][1]["questions"] = [{
            "id": "scope", "at": 600, "gates": True, "held_by": "bo",
            "options": ["cut", "keep"]}]
        self.assertTrue(any("duplicate gating question id" in e
                            for e in validate_scenario(s)))

    def test_question_ids_must_be_nonempty_strings(self):
        for bad_id in ("", ["scope"]):
            s = scenario()
            s["project"]["tasks"][0]["questions"] = [{
                "id": bad_id, "at": 600, "gates": True, "held_by": "ana",
                "options": ["cut", "keep"]}]
            errors = validate_scenario(s)
            self.assertTrue(any("non-empty string id" in e
                                for e in errors), (bad_id, errors))

    def test_setup_cost_knobs_are_bounded(self):
        s = scenario()
        s["costs"] = {"handoff_minutes": 45, "helper_onboarding_minutes": 30}
        self.assertEqual(validate_scenario(s), [])
        s["costs"]["handoff_minutes"] = 481
        self.assertTrue(any("handoff_minutes" in e
                            for e in validate_scenario(s)))

    def test_opt_returns_best_choice_and_intent_consumes_it(self):
        from sim.intent import _gating_questions
        from sim.optimal import opt_ideal
        s = scenario()
        s["project"]["tasks"][0]["questions"] = [{
            "id": "scope", "at": 600, "gates": True, "held_by": "ana",
            "options": ["slow", "fast"],
            "choice_effects": {
                "slow": {"summary": "adds work",
                         "task_effort_delta_hours": 2},
                "fast": {"summary": "removes work",
                         "task_effort_delta_hours": -2}}}]
        opt = opt_ideal(s)
        self.assertEqual(opt["choices"], {"scope": "fast"})
        self.assertEqual(_gating_questions(s, opt["choices"]),
                         [(600, "core", "scope", "fast")])

    def test_opt_ignores_cancellable_meetings_but_not_ooo(self):
        from sim.optimal import opt_ideal
        base = scenario()
        meeting = scenario()
        meeting["meetings"] = [{
            "id": "sync", "start": 600, "end": 780,
            "attendees": ["ana", "bo"], "topic": "cancellable sync",
            "agenda": ""}]
        self.assertEqual(opt_ideal(meeting)["opt_combined"],
                         opt_ideal(base)["opt_combined"])
        ooo = scenario()
        ooo["npcs"][0]["ooo"] = [[545, 1085]]
        self.assertNotEqual(opt_ideal(ooo)["opt_combined"],
                            opt_ideal(base)["opt_combined"])

    def test_crunch_review_gate_validates(self):
        with open("scenarios/crunch.json") as f:
            s = json.load(f)
        self.assertEqual(validate_scenario(s), [])
        backfill = next(t for t in s["project"]["tasks"]
                        if t["id"] == "data-backfill")
        self.assertEqual(backfill["blocked_by"], ["migration-review"])

    def test_unowned_arrival_rejected(self):
        """The contract: real work always has an owner — arrivals must land
        assigned to a default holder, or the baseline is a strawman."""
        s = scenario()
        del s["task_arrivals"][0]["task"]["assignees"]
        self.assertTrue(any("owner" in e for e in validate_scenario(s)))

    def test_stakeholder_owner_rejected(self):
        s = scenario()
        s["task_arrivals"][0]["task"]["assignees"] = ["vp"]
        self.assertTrue(any("stakeholder" in e for e in validate_scenario(s)))

    def test_silent_arrival_rejected(self):
        s = scenario()
        del s["task_arrivals"][0]["announce"]
        self.assertTrue(any("announce" in e for e in validate_scenario(s)))

    def test_unfair_arrival_rejected(self):
        """Fairness is a rubric too: an ask that cannot be completed after
        the PM could first know it scores luck, not skill."""
        s = scenario()
        s["task_arrivals"].append({
            "at": 6600, "npc": "bo", "announce": "Impossible Friday ask.",
            "task": {"id": "impossible", "title": "x", "effort_hours": 8,
                     "priority": "P0"},
            "fallback": {"npc": "bo", "at": 6700}})
        self.assertTrue(any("UNFAIR" in e for e in validate_scenario(s)))

    def test_bad_via_rejected(self):
        s = scenario()
        s["task_arrivals"][0]["via"] = "fax"
        self.assertTrue(any("via" in e for e in validate_scenario(s)))


class TestFingerprintAndStamp(unittest.TestCase):
    def test_fingerprint_vector(self):
        fp = fingerprint(scenario())
        for key in ("winnable_combined", "capacity_utilization",
                    "opt_done_weight_rate", "min_reaction_ratio", "fair",
                    "log10_trajectory_classes"):
            self.assertIn(key, fp)
        self.assertTrue(fp["fair"])
        self.assertGreaterEqual(fp["winnable_combined"], 0.0)  # OPT >= baseline

    def test_stamp_embeds_band_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scn.json")
            with open(path, "w") as f:
                json.dump(scenario(), f)
            stamp(path)
            with open(path) as f:
                band = json.load(f)["band"]
            self.assertIn("no_pm_baseline", band)
            self.assertIn("opt_max", band)
            self.assertGreaterEqual(band["opt_max"]["combined"],
                                    band["no_pm_baseline"]["combined"])


if __name__ == "__main__":
    unittest.main()
