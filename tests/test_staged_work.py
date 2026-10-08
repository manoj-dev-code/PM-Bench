"""Staged work items and deterministic quality-gate reveals."""

import copy
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

from sim.engine import Engine
from sim.eval import StubClient, evaluate, print_scorecard
from sim.optimal import opt_ideal
from sim.replay import replay
from sim.rubric import task_value
from sim.server import problem_graph
from sim.validate import validate_scenario
from sim.work import materialize_work_items, work_item_terminals

from tests.fixtures import make_engine, scenario


def staged_item(outcome="rework"):
    gate = {"outcome": outcome,
            "reason": "Rollback automation missed a lock-timeout recovery path."}
    if outcome == "rework":
        gate["rework"] = {
            "id": "rollback-rework",
            "title": "Fix rollback automation findings",
            "kind": "rework",
            "assignees": ["bo"],
            "effort_hours": 3,
            "done_hours": 0,
            "reported": "Review findings need correction"}
    return {
        "id": "release", "title": "Release review", "blocked_by": ["build"],
        "priority": "P0", "tags": ["ops"],
        "stages": [
            {"id": "review", "title": "Code review", "kind": "review",
             "assignees": ["ana"], "effort_hours": 2, "done_hours": 0,
             "reported": "Waiting", "quality_gate": gate},
            {"id": "verify", "title": "Rollback verification",
             "kind": "verification", "assignees": ["ana"],
             "effort_hours": 2, "done_hours": 0,
             "reported": "Waiting"}]}


def staged_scenario():
    scn = {
        "seed": 1, "start_time": 540,
        "evaluation": {"horizon": 4000, "task_value": {}},
        "project": {"id": "p", "name": "P", "tasks": [
            {"id": "build", "title": "Build", "assignees": ["ana"],
             "effort_hours": 1, "done_hours": 0, "priority": "P0"},
            staged_item(),
            {"id": "deploy", "title": "Deploy", "assignees": ["ana"],
             "effort_hours": 1, "done_hours": 0,
             "blocked_by": ["release"], "priority": "P1"}]},
        "task_arrivals": [], "beats": [],
        "npcs": [
            {"id": "ana", "name": "Ana", "role": "Engineer",
             "persona": "x", "knowledge": []},
            {"id": "bo", "name": "Bo", "role": "Engineer",
             "persona": "x", "knowledge": []}]}
    return scn


class TestMaterialization(unittest.TestCase):
    def test_materializes_chain_hidden_rework_and_dependencies(self):
        source = [{"id": "build", "title": "Build"}, staged_item(),
                  {"id": "deploy", "title": "Deploy",
                   "blocked_by": ["release"]}]
        before = copy.deepcopy(source)
        out = materialize_work_items(source)
        self.assertEqual(source, before)
        ids = [t["id"] for t in out]
        self.assertEqual(ids[:4], [
            "build", "release::review", "release::rollback-rework",
            "release::verify"])
        self.assertEqual(out[1]["blocked_by"], ["build"])
        self.assertEqual(out[2]["blocked_by"], ["release::review"])
        self.assertTrue(out[2]["contingent"])
        self.assertEqual(out[2]["hidden_until"], ["release::review"])
        self.assertEqual(out[2]["quality_reason"],
                         "Rollback automation missed a lock-timeout recovery path.")
        self.assertEqual(out[3]["blocked_by"], ["release::rollback-rework"])
        self.assertEqual(out[4]["blocked_by"], ["release::verify"])
        self.assertEqual(work_item_terminals(source),
                         {"release": "release::verify"})
        self.assertEqual(sum(t["score_weight_factor"] for t in out[1:4]), 1.0)
        self.assertEqual(out[1]["work_item_id"], "release")
        self.assertEqual(out[1]["stage_kind"], "review")
        self.assertEqual(out[3]["priority"], "P0")

    def test_pass_gate_inserts_no_rework(self):
        item = staged_item("pass")
        out = materialize_work_items([item])
        self.assertEqual([t["id"] for t in out],
                         ["release::review", "release::verify"])
        self.assertEqual(out[1]["blocked_by"], ["release::review"])

    def test_stage_factors_conserve_priority_weight(self):
        rows = [{"id": "a", "priority": "P0", "effort_hours": 2,
                 "score_weight_factor": 0.25, "true_done_hours": 0,
                 "status": "open"},
                {"id": "b", "priority": "P0", "effort_hours": 6,
                 "score_weight_factor": 0.75, "true_done_hours": 6,
                 "status": "done", "projected_done": 600}]
        value = task_value(rows, {"alpha": 0.5, "gamma": 0.5}, 4000, 540)
        self.assertEqual([p["weight"] for p in value["per_task"]], [2.0, 6.0])
        self.assertEqual(sum(p["weight"] for p in value["per_task"]), 8.0)


class TestRevealLifecycle(unittest.TestCase):
    def test_hidden_rework_reveals_on_gate_completion(self):
        eng = make_engine(staged_scenario())
        rework = "release::rollback-rework"
        verify = "release::verify"
        tracker_ids = {t["id"] for t in eng.world.tracker_view()}
        self.assertIn("release::review", tracker_ids)
        self.assertNotIn(rework, tracker_ids)
        self.assertFalse(any("Fix rollback" in b["task"]
                             for b in eng.world.belief_view("bo")))
        self.assertNotIn(rework,
                         {t["id"] for t in eng.world.tasks
                          if t.get("contingent") and eng.world.task_visible(t)})
        before_clock, before_log = eng.world.clock, len(eng.world.log)
        guesses = [
            lambda: eng.agent_update_note(rework, "guess"),
            lambda: eng.agent_assign_task(rework, "ana"),
            lambda: eng.agent_add_helper(rework, "ana"),
            lambda: eng.agent_drop_helper(rework, "bo"),
            lambda: eng.agent_reprioritize(rework, priority="P1"),
            lambda: eng.agent_schedule_meeting(
                ["agent", "bo"], eng.world.clock + 60, 30, "hidden",
                task=rework)]
        for call in guesses:
            self.assertIn("error", call())
        self.assertEqual((eng.world.clock, len(eng.world.log)),
                         (before_clock, before_log))

        eng.advance_until(1200, interruptible=True)
        reveal = next(e for e in eng.world.log
                      if e["kind"] == "task_revealed")
        self.assertEqual(reveal["task_id"], rework)
        self.assertEqual(reveal["reason"],
                         "Rollback automation missed a lock-timeout recovery path.")
        self.assertEqual(eng.world.clock, reveal["triggered_at"])
        pushes = eng.drain_agent_push()
        self.assertTrue(any(p.get("work_revealed") == reveal["title"]
                            and p.get("task_id") == rework
                            and p.get("reason") == reveal["reason"]
                            for p in pushes))
        order = [(e["seq"], e["kind"], e.get("task_id")) for e in eng.world.log
                 if (e["kind"], e.get("task_id")) in {
                     ("task_completed", "release::review"),
                     ("task_revealed", rework),
                     ("task_completed", rework)}]
        self.assertEqual([(kind, tid) for _, kind, tid in order],
                         [("task_completed", "release::review"),
                          ("task_revealed", rework)])
        self.assertIn(rework, {t["id"] for t in eng.world.tracker_view()})
        self.assertTrue(any(reveal["reason"] in str(m.get("content"))
                            for m in eng.world.npcs["bo"].context))
        rows = {r["id"]: r for r in eng.world.tasks_view()}
        self.assertEqual(rows[verify]["true_done_hours"], 0)
        self.assertGreater(rows[verify]["projected_done"],
                           rows[rework]["projected_done"])

        eng.advance_until(2000, interruptible=False)
        lifecycle = [(e["kind"], e.get("task_id")) for e in eng.world.log
                     if (e["kind"], e.get("task_id")) in {
                         ("task_completed", "release::review"),
                         ("task_revealed", rework),
                         ("task_completed", rework)}]
        self.assertEqual(lifecycle,
                         [("task_completed", "release::review"),
                          ("task_revealed", rework),
                          ("task_completed", rework)])

    def test_ground_truth_rows_carry_fractional_stage_weight(self):
        scn = staged_scenario()
        eng = make_engine(scn)
        rows = eng.world.tasks_view()
        value = task_value(rows, {"alpha": 0.5, "gamma": 0.5},
                           scn["evaluation"]["horizon"], scn["start_time"])
        staged = [p for p in value["per_task"]
                  if p["id"].startswith("release::")]
        self.assertEqual(sum(p["weight"] for p in staged), 8.0)
        for actual, expected in zip(
                [p["weight"] for p in staged],
                [8 * 2 / 7, 8 * 3 / 7, 8 * 2 / 7]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(all("score_weight_factor" in r for r in rows
                            if r["id"].startswith("release::")))
        opt = opt_ideal(scn)
        self.assertEqual(sum(p["weight"] for p in opt["per_task"]
                             if p["id"].startswith("release::")), 8.0)
        result = evaluate(eng.run_dir)
        self.assertEqual(sum(p["weight"] for p in result["per_task_agent"]
                             if p["id"].startswith("release::")), 8.0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_scorecard(result)
        self.assertIn("w=2.29", buf.getvalue())

    def test_arrival_dependency_rewrites_to_terminal_stage(self):
        scn = staged_scenario()
        scn["task_arrivals"].append({
            "at": 700, "npc": "ana", "announce": "patch landed",
            "task": {"id": "patch", "title": "Patch", "assignees": ["bo"],
                     "effort_hours": 0.5, "done_hours": 0,
                     "blocked_by": ["release"], "priority": "P3"}})
        eng = make_engine(scn)
        eng.advance_until(900, interruptible=False)
        patch = eng.world.find_task("patch")
        self.assertEqual(patch["blocked_by"], ["release::verify"])
        row = next(r for r in eng.world.tasks_view() if r["id"] == "patch")
        self.assertEqual(row["true_done_hours"], 0)

    def test_replay_and_repeat_runs_preserve_reveal(self):
        eng_a = make_engine(staged_scenario())
        eng_b = make_engine(staged_scenario())
        eng_a.advance_until(1200, interruptible=True)
        eng_b.advance_until(1200, interruptible=True)
        reveals_a = [e for e in eng_a.world.log if e["kind"] == "task_revealed"]
        reveals_b = [e for e in eng_b.world.log if e["kind"] == "task_revealed"]
        self.assertEqual(reveals_a, reveals_b)
        world = replay(eng_a.world.scenario, eng_a.world.log)
        self.assertIn("release::rollback-rework", world.revealed_tasks)
        self.assertEqual(world.tracker_view(), eng_a.world.tracker_view())
        self.assertEqual(world.tasks_view(), eng_a.world.tasks_view())


class TestResearcherAndFormatting(unittest.TestCase):
    def test_problem_graph_uses_materialized_truth(self):
        graph = problem_graph(staged_scenario())
        tasks = {t["id"]: t for t in graph["tasks"]}
        self.assertIn("release::review", tasks)
        self.assertIn("release::rollback-rework", tasks)
        self.assertIn("release::verify", tasks)
        self.assertNotIn("release", tasks)
        self.assertTrue(tasks["release::rollback-rework"]["contingent"])
        self.assertEqual(tasks["deploy"]["blocked_by"], ["release::verify"])
        self.assertEqual(tasks["release::review"]["stage_kind"], "review")

    def test_optimal_cli_prints_fractional_weights(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "staged.json")
            with open(path, "w") as f:
                json.dump(staged_scenario(), f)
            argv = sys.argv
            try:
                sys.argv = ["sim.optimal", path]
                buf = io.StringIO()
                with redirect_stdout(buf):
                    from sim import optimal
                    optimal.main()
            finally:
                sys.argv = argv
        self.assertIn("w=2.29", buf.getvalue())


class TestStagedValidation(unittest.TestCase):
    def assert_invalid(self, scn, needle):
        errors = validate_scenario(scn)
        self.assertTrue(any(needle in e for e in errors), errors)

    def test_malformed_stage_gate_rework_and_collision(self):
        scn = staged_scenario()
        scn["project"]["tasks"][1]["stages"] = scn["project"]["tasks"][1]["stages"][:1]
        self.assert_invalid(scn, "at least two stages")

        scn = staged_scenario()
        scn["project"]["tasks"][1]["stages"][0]["quality_gate"]["rework"]["assignees"] = []
        self.assert_invalid(scn, "assignees")

        scn = staged_scenario()
        scn["project"]["tasks"][1]["stages"][0]["quality_gate"]["reason"] = ""
        self.assert_invalid(scn, "reason")

        scn = staged_scenario()
        scn["project"]["tasks"].append(
            {"id": "release::verify", "title": "Collision",
             "assignees": ["ana"], "effort_hours": 1})
        self.assert_invalid(scn, "generated id")

    def test_staged_arrival_and_materialized_cycle_rejected(self):
        scn = staged_scenario()
        scn["task_arrivals"].append(
            {"at": 700, "npc": "ana", "announce": "staged",
             "task": staged_item()})
        self.assert_invalid(scn, "staged work is not supported")

        scn = staged_scenario()
        scn["project"]["tasks"][0]["blocked_by"] = ["deploy"]
        self.assert_invalid(scn, "cycle")

    def test_crunch_materializes_expected_ids(self):
        path = os.path.join(os.path.dirname(__file__), "..",
                            "scenarios", "crunch.json")
        with open(path) as f:
            scn = json.load(f)
        self.assertEqual(validate_scenario(scn), [])
        out = materialize_work_items(scn["project"]["tasks"])
        rows = {t["id"]: t for t in out}
        self.assertIn("migration-review::review", rows)
        self.assertIn("migration-review::rollback-rework", rows)
        self.assertIn("migration-review::rollback-verification", rows)
        self.assertEqual(rows["data-backfill"]["blocked_by"],
                         ["migration-review::rollback-verification"])


if __name__ == "__main__":
    unittest.main()
