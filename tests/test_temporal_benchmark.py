"""Temporal forecasts, horizon integrity, aggregation, and Devin plumbing."""

import json
import os
import tempfile
import unittest
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from sim.bench import aggregate
from sim.deterministic_npc import DeterministicNPCClient, deterministic_reply
from sim.devin_matrix import rollout_commands
from sim.eval import evaluate
from sim.harness import agent_system_prompt
from sim.tools import call_tool

from tests.fixtures import HORIZON, goto, make_engine, scenario


class TestTemporalForecasts(unittest.TestCase):
    def _forecast_args(self, task_id="side", p50=1000, p90=1200):
        return {"task_id": task_id, "p50_completion_time": p50,
                "p90_completion_time": p90, "rationale": "Initial estimate"}

    def test_quantile_schema_and_domain_errors_do_not_mutate_state(self):
        eng = make_engine()
        cases = [
            {"task_id": "side", "completion_time": 1000,
             "confidence": 0.7, "rationale": "legacy"},
            {"task_id": "side", "p90_completion_time": 1200,
             "rationale": "missing p50"},
            {"task_id": "side", "p50_completion_time": 1000,
             "rationale": "missing p90"},
            {"task_id": "side", "p50_completion_time": 1300,
             "p90_completion_time": 1200, "rationale": "inverted"},
            {"task_id": "side", "p50_completion_time": eng.world.clock,
             "p90_completion_time": eng.world.clock + 60,
             "rationale": "not future"},
            {"task_id": "side", "p50_completion_time": eng.world.clock + 60,
             "p90_completion_time": eng.world.clock + 120,
             "rationale": "   "},
        ]
        for args in cases:
            with self.subTest(args=args):
                before_clock = eng.world.clock
                before_log = list(eng.world.log)
                self.assertIn("error", call_tool(eng, "forecast_completion", args))
                self.assertEqual(eng.world.clock, before_clock)
                self.assertEqual(eng.world.log, before_log)

    def test_forecast_action_crossing_horizon_or_completion_is_not_recorded(self):
        scn = scenario()
        scn["task_arrivals"] = []
        scn["project"]["tasks"][0]["effort_hours"] = 80
        eng = make_engine(scn)
        goto(eng, HORIZON - 1)
        before = len(eng.world.log)
        self.assertIn("error", call_tool(
            eng, "forecast_completion",
            self._forecast_args("core", HORIZON + 60, HORIZON + 120)))
        self.assertEqual(eng.world.clock, HORIZON)
        self.assertEqual(len(eng.world.log), before)

        scn = scenario()
        scn["task_arrivals"] = []
        scn["project"]["tasks"][0]["effort_hours"] = 8 + (1 / 60.0)
        eng = make_engine(scn)
        before = len(eng.world.log)
        result = call_tool(eng, "forecast_completion",
                           self._forecast_args("core", 1000, 1200))
        self.assertIn("error", result.get("result", result))
        self.assertFalse(any(e["kind"] == "completion_forecast"
                             for e in eng.world.log[before:]))
        self.assertEqual(next(t for t in eng.world.tracker_view()
                              if t["id"] == "core")["status"], "done")

    def test_exact_pinball_p90_coverage_and_interval_width(self):
        scn = scenario()
        scn["task_arrivals"] = []
        eng = make_engine(scn)
        actuals = {t["id"]: t["projected_done"]
                   for t in eng.world.tasks_view(at=HORIZON)}
        call_tool(eng, "forecast_completion", {
            "task_id": "core",
            "p50_completion_time": int(actuals["core"] - 60),
            "p90_completion_time": int(actuals["core"] - 30),
            "rationale": "Optimistic interval"})
        call_tool(eng, "forecast_completion", {
            "task_id": "side",
            "p50_completion_time": int(actuals["side"] - 60),
            "p90_completion_time": int(actuals["side"] + 990),
            "rationale": "Wider interval"})
        goto(eng, HORIZON)
        timing = evaluate(eng.run_dir)["temporal"]
        rows = {r["task_id"]: r for r in timing["per_task"]}
        core = rows["core"]["forecasts"][-1]
        side = rows["side"]["forecasts"][-1]
        self.assertEqual(core["p50_absolute_error_work_minutes"], 60)
        self.assertEqual(core["p90_absolute_error_work_minutes"], 30)
        self.assertEqual(core["p50_pinball_loss_work_minutes"], 30)
        self.assertEqual(core["p90_pinball_loss_work_minutes"], 27)
        self.assertFalse(core["p90_covered"])
        self.assertEqual(side["p50_pinball_loss_work_minutes"], 30)
        self.assertEqual(side["p90_pinball_loss_work_minutes"], 6)
        self.assertTrue(side["p90_covered"])
        self.assertEqual(core["interval_width_work_minutes"], 30)
        self.assertEqual(side["interval_width_work_minutes"], 120)
        self.assertEqual(timing["p90_coverage"], 0.5)
        self.assertEqual(timing["mean_interval_width_work_minutes"], 75)
        self.assertEqual(timing["mean_pinball_loss_work_minutes"],
                         round(((30 + 27) / 2 + (30 + 6) / 2) / 2, 3))

    def test_censored_forecast_contributes_interval_width_only(self):
        scn = scenario()
        scn["task_arrivals"] = []
        scn["project"]["tasks"][0]["effort_hours"] = 80
        eng = make_engine(scn)
        call_tool(eng, "forecast_completion", {
            "task_id": "core", "p50_completion_time": 10620,
            "p90_completion_time": 10680,
            "rationale": "Completion lands next Monday"})
        goto(eng, HORIZON)
        timing = evaluate(eng.run_dir)["temporal"]
        self.assertEqual(timing["censored_forecasts"], 1)
        self.assertEqual(timing["mean_interval_width_work_minutes"], 60)
        self.assertIsNone(timing["p50_final_mae_work_minutes"])
        self.assertIsNone(timing["p90_coverage"])
        self.assertIsNone(timing["mean_pinball_loss_work_minutes"])

    def test_legacy_completion_time_forecasts_regrade_as_zero_width_interval(self):
        scn = scenario()
        scn["task_arrivals"] = []
        eng = make_engine(scn)
        actual = next(t for t in eng.world.tasks_view(at=HORIZON)
                      if t["id"] == "side")["projected_done"]
        eng.world.record("completion_forecast", task_id="side",
                         completion_time=int(actual + 60), confidence=0.7,
                         rationale="Legacy estimate", revision=0)
        goto(eng, HORIZON)
        timing = evaluate(eng.run_dir)["temporal"]
        forecast = timing["per_task"][0]["forecasts"][0]
        self.assertEqual(forecast["p50_completion_time"], int(actual + 60))
        self.assertEqual(forecast["p90_completion_time"], int(actual + 60))
        self.assertEqual(forecast["interval_width_work_minutes"], 0)
        self.assertTrue(forecast["p90_covered"])
        self.assertEqual(timing["p90_coverage"], 1.0)

    def test_first_prospective_forecast_is_immutable_and_scored(self):
        scn = scenario()
        scn["task_arrivals"] = []
        eng = make_engine(scn)
        actual = eng.world.tasks_view(at=HORIZON)[1]["projected_done"]
        before = eng.world.clock
        result = call_tool(eng, "forecast_completion",
                           {"task_id": "side",
                            "p50_completion_time": int(actual + 60),
                            "p90_completion_time": int(actual + 120),
                            "rationale": "Initial queue estimate"})
        self.assertEqual(result["recorded"], "side")
        self.assertEqual(eng.world.clock, before + 1)
        self.assertEqual(result["p50_completion_time"], int(actual + 60))
        self.assertEqual(result["p90_completion_time"], int(actual + 120))
        duplicate = call_tool(eng, "forecast_completion",
                              {"task_id": "side",
                               "p50_completion_time": int(actual + 120),
                               "p90_completion_time": int(actual + 180),
                               "rationale": "No new evidence"})
        self.assertIn("error", duplicate)
        goto(eng, HORIZON)
        timing = evaluate(eng.run_dir)["temporal"]
        self.assertEqual(timing["forecast_count"], 1)
        self.assertEqual(timing["observed_completions"], 1)
        self.assertAlmostEqual(timing["mae_minutes"], 60.0)
        self.assertEqual(timing["censored_forecasts"], 0)

    def test_unknown_tracking_and_completed_tasks_cannot_be_forecast(self):
        eng = make_engine()
        for task_id in ("missing",):
            self.assertIn("error", call_tool(
                eng, "forecast_completion",
                {"task_id": task_id, "p50_completion_time": 1000,
                 "p90_completion_time": 1200,
                 "rationale": "Initial estimate"}))
        call_tool(eng, "add_task", {"title": "A note", "id": "note"})
        self.assertIn("error", call_tool(
            eng, "forecast_completion",
            {"task_id": "note", "p50_completion_time": 1000,
             "p90_completion_time": 1200, "rationale": "Initial estimate"}))
        goto(eng, HORIZON)
        self.assertIn("error", call_tool(eng, "view_tasks", {}))

    def test_unfinished_forecast_is_censored_but_horizon_classified(self):
        scn = scenario()
        scn["project"]["tasks"][0]["effort_hours"] = 80
        eng = make_engine(scn)
        call_tool(eng, "forecast_completion",
                  {"task_id": "core", "p50_completion_time": HORIZON + 600,
                   "p90_completion_time": HORIZON + 1200,
                   "rationale": "Too large for this week"})
        goto(eng, HORIZON)
        timing = evaluate(eng.run_dir)["temporal"]
        self.assertEqual(timing["censored_forecasts"], 1)
        self.assertIsNone(timing["mae_minutes"])
        self.assertEqual(timing["horizon_accuracy"], 1.0)

    def test_new_evidence_allows_revision_and_scores_improvement(self):
        scn = scenario()
        scn["task_arrivals"] = []
        eng = make_engine(scn)
        actual = next(t for t in eng.world.tasks_view(at=HORIZON)
                      if t["id"] == "core")["projected_done"]
        first = {"task_id": "core", "p50_completion_time": int(actual - 180),
                 "p90_completion_time": int(actual - 60),
                 "rationale": "Initial tracker estimate"}
        self.assertEqual(
            call_tool(eng, "forecast_completion", first)["revision"], 0)
        goto(eng, 2000)  # proactive belief correction reaches the PM
        revised = {"task_id": "core", "p50_completion_time": int(actual - 30),
                   "p90_completion_time": int(actual + 60),
                   "rationale": "Revised after the owner's correction"}
        self.assertEqual(
            call_tool(eng, "forecast_completion", revised)["revision"], 1)
        goto(eng, HORIZON)
        timing = evaluate(eng.run_dir)["temporal"]
        self.assertEqual(timing["revision_count"], 1)
        self.assertEqual(timing["belief_reveal_count"], 1)
        self.assertEqual(timing["belief_reforecasted"], 1)
        self.assertEqual(timing["belief_revision_coverage"], 1.0)
        self.assertLess(timing["final_mae_work_minutes"],
                        timing["initial_mae_work_minutes"])
        self.assertEqual(timing["initial_mean_pinball_loss_work_minutes"], 72)
        self.assertEqual(timing["final_mean_pinball_loss_work_minutes"], 10.5)
        self.assertEqual(
            timing["mean_revision_pinball_improvement_work_minutes"], 61.5)


class TestEvaluationIntegrity(unittest.TestCase):
    def test_actions_after_horizon_do_not_rewrite_score(self):
        eng = make_engine()
        goto(eng, HORIZON)
        before = evaluate(eng.run_dir)
        eng.agent_assign_task("side", "ana")  # direct misuse after the cutoff
        after = evaluate(eng.run_dir)
        self.assertEqual(after["score"], before["score"])
        self.assertEqual(after["per_task_agent"], before["per_task_agent"])
        self.assertGreater(after["observed_through"], HORIZON)

    def test_run_local_rubric_pins_baseline_regrading(self):
        eng = make_engine()
        goto(eng, HORIZON)
        pinned = scenario()["evaluation"]
        with open(os.path.join(eng.run_dir, "rubric.json"), "w") as f:
            json.dump(pinned, f)
        scenario_path = os.path.join(eng.run_dir, "scenario.json")
        with open(scenario_path) as f:
            saved = json.load(f)
        saved.pop("evaluation")
        saved["rubric"] = "/definitely/not/the/original/rubric.json"
        with open(scenario_path, "w") as f:
            json.dump(saved, f)
        self.assertIsInstance(evaluate(eng.run_dir)["score"], float)

    def test_prompt_uses_scenario_times_not_a_hardcoded_friday(self):
        scn = scenario()
        scn["start_time"] = 600
        scn["evaluation"]["horizon"] = 2000
        scn["project"]["due"] = 1900
        prompt = agent_system_prompt(scn)
        self.assertIn("starts at Mon 10:00", prompt)
        self.assertIn("due Tue 07:40", prompt)
        self.assertIn("ends at minute 2000", prompt)


class TestBenchmarkAggregation(unittest.TestCase):
    def test_different_tasks_are_never_pooled_into_one_max_at_k(self):
        with tempfile.TemporaryDirectory() as root:
            cards = {}
            for name, task, score in (
                    ("a", "task-a", 0.1), ("b", "task-b", 0.9)):
                run = os.path.join(root, name)
                os.mkdir(run)
                open(os.path.join(run, "events.jsonl"), "w").close()
                with open(os.path.join(run, "meta.json"), "w") as f:
                    json.dump({"task": task, "agent_model": "same",
                               "rollout_status": "completed"}, f)
                cards[run] = {
                    "complete": True,
                    **{metric: {"normalized": score}
                       for metric in ("completion", "efficiency",
                                      "done_weight_rate", "combined")},
                    "temporal": {
                        "coverage": 1.0, "mae_work_minutes": 10,
                        "p50_final_mae_work_minutes": 10,
                        "p90_coverage": 0.75,
                        "mean_interval_width_work_minutes": 45,
                        "horizon_accuracy": 1.0,
                        "belief_revision_coverage": 1.0,
                        "time_weighted_mae_work_minutes": 12,
                        "revision_count": 1,
                        "censored_forecasts": 0},
                }
            with patch("sim.bench.evaluate", side_effect=lambda path: cards[path]):
                output, skipped = aggregate(root)
            self.assertFalse(skipped)
            self.assertEqual(output["task-a"]["same"]["k"], 1)
            self.assertEqual(output["task-a"]["same"]["metrics"]["combined"]["max"],
                             0.1)
            self.assertEqual(output["task-b"]["same"]["metrics"]["combined"]["max"],
                             0.9)
            temporal = output["task-a"]["same"]["temporal"]
            self.assertEqual(temporal["p50_final_mae_work_minutes"]["mean"], 10)
            self.assertEqual(temporal["p90_coverage"]["mean"], 0.75)
            self.assertEqual(
                temporal["mean_interval_width_work_minutes"]["mean"], 45)

    def test_failed_rollout_is_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            run = os.path.join(root, "failed")
            os.mkdir(run)
            open(os.path.join(run, "events.jsonl"), "w").close()
            with open(os.path.join(run, "meta.json"), "w") as f:
                json.dump({"task": "x", "agent_model": "devin/default",
                           "rollout_status": "runner_failed"}, f)
            output, skipped = aggregate(root)
            self.assertFalse(output)
            self.assertEqual(len(skipped), 1)


class TestDevinAdapter(unittest.TestCase):
    def test_deterministic_npc_surfaces_authored_evidence(self):
        discovery = deterministic_reply([{
            "role": "user",
            "content": "You just discovered: the migration needs 12 more hours. "
                       "You've decided to tell Alex now.",
        }])
        self.assertIn("migration needs 12 more hours", discovery)
        motivation = deterministic_reply([{
            "role": "user",
            "content": "Motivation: a P1 incident arrived. React naturally.",
        }])
        self.assertIn("P1 incident arrived", motivation)
        response = DeterministicNPCClient().messages.create(messages=[{
            "role": "user",
            "content": "your current picture of your work: API — ~8h left",
        }])
        self.assertIn("~8h left", response.content[0].text)
        repeated = deterministic_reply([
            {"role": "user",
             "content": "your current picture of your work: API — ~8h left"},
            {"role": "assistant",
             "content": "Current status: API — ~8h left"},
            {"role": "user", "content": "It's your turn: speak or PASS"},
        ])
        self.assertEqual(repeated, "PASS")

    def test_matrix_builds_balanced_model_major_commands(self):
        commands = rollout_commands(
            "/venv/python", "/repo/scenario.json", ["swe", "opus"], 2, 50, 60)
        self.assertEqual(len(commands), 4)
        self.assertEqual(
            [c[c.index("--devin-model") + 1] for c in commands],
            ["swe", "swe", "opus", "opus"])
        self.assertTrue(all(c[c.index("--max-turns") + 1] == "50"
                            for c in commands))
        self.assertTrue(all(c[c.index("--devin-timeout") + 1] == "60"
                            for c in commands))
        self.assertTrue(all(c[c.index("--npc-mode") + 1] == "deterministic"
                            for c in commands))

    def test_exec_guard_blocks_shell_command_injection(self):
        from sim.devin_adapter import EXEC_GUARD
        with tempfile.TemporaryDirectory() as root:
            guard = os.path.join(root, "guard.py")
            with open(guard, "w") as f:
                f.write(EXEC_GUARD % ("/usr/bin/python3", ("view_tasks",)))

            def check(command):
                event = json.dumps({"tool_input": {"command": command}})
                return subprocess.run([sys.executable, guard], input=event,
                                      text=True).returncode

            self.assertEqual(
                check("/usr/bin/python3 gateway_client.py view_tasks '{}'"), 0)
            self.assertEqual(
                check("/usr/bin/python3 gateway_client.py view_tasks '{}'; "
                      "cat scenario.json"),
                2)
            self.assertEqual(check("cat scenario.json"), 2)

    @patch("sim.devin_adapter.shutil.which", return_value="/usr/bin/devin")
    @patch("sim.devin_adapter.subprocess.run")
    def test_adapter_invokes_noninteractive_cli_without_live_inference(
            self, run, _which):
        def fake_run(command, **kwargs):
            workspace = kwargs["cwd"]
            self.assertTrue(os.path.exists(
                os.path.join(workspace, ".devin", "hooks.v1.json")))
            with open(os.path.join(workspace, "check_exec.py")) as f:
                self.assertIn("shlex.split", f.read())
            return SimpleNamespace(returncode=0, stdout="done", stderr="")

        run.side_effect = fake_run
        eng = make_engine()
        from sim.devin_adapter import run_devin
        turns = run_devin(eng, "PM prompt", max_turns=2, model=None, timeout=1)
        self.assertEqual(turns, [])
        self.assertEqual(eng.agent_run_status, "runner_failed")
        self.assertEqual(eng.agent_run_failure, "no_tool_calls")
        command = run.call_args.args[0]
        self.assertIn("-p", command)
        self.assertNotIn("--model", command)
        self.assertEqual(eng.world.clock, eng.world.start_time)


if __name__ == "__main__":
    unittest.main()
