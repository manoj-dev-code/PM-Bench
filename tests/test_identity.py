"""Benchmark identity coordinates and version-separated aggregation."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from sim.bench import aggregate
from sim.identity import (benchmark_identity, evidence_version,
                          legacy_run_identity, rubric_version,
                          scenario_version, with_resolved_model)
from sim.rubric import load_rubric
from sim.server import RunHub

from tests.fixtures import scenario


def _identity(task_version, coordinate=None, model="devin/swe"):
    return {"scenario": "s", "physics": "p", "evidence": "e", "rubric": "r",
            "npc_mode": "deterministic", "task_version": task_version,
            "model": model, "coordinate": coordinate or "coord-" + task_version}


class TestIdentity(unittest.TestCase):
    def test_scenario_hash_ignores_stamps_and_tracks_semantics(self):
        base = scenario()
        decorated = scenario()
        decorated["band"] = {"derived": True}
        decorated["_note"] = "comment"
        decorated["rubric"] = "rubrics/other.json"
        decorated["evaluation"] = {"horizon": 9999}
        decorated["project"]["_comment"] = "ignored"
        self.assertEqual(scenario_version(base), scenario_version(decorated))
        self.assertEqual(len(scenario_version(base)), 64)

        variants = []
        changed = scenario()
        changed["project"]["tasks"][0]["effort_hours"] += 1
        variants.append(changed)
        changed = scenario()
        changed["npcs"][0]["persona"] = "blunt"
        variants.append(changed)
        changed = scenario()
        changed["task_arrivals"][0]["at"] += 1
        variants.append(changed)
        changed = scenario()
        changed["npcs"][0]["knowledge"].append("new fact")
        variants.append(changed)
        for variant in variants:
            self.assertNotEqual(scenario_version(base),
                                scenario_version(variant))

    def test_rubric_and_evidence_hashes(self):
        rubric = load_rubric(scenario())
        self.assertEqual(rubric_version(rubric), rubric_version(rubric))
        changed = dict(rubric, horizon=rubric["horizon"] + 60)
        self.assertNotEqual(rubric_version(rubric), rubric_version(changed))
        weights = dict(rubric)
        weights["task_value"] = dict(rubric["task_value"], alpha=0.7)
        self.assertNotEqual(rubric_version(rubric), rubric_version(weights))
        self.assertEqual(evidence_version("deterministic"),
                         evidence_version("deterministic"))
        self.assertNotEqual(evidence_version("deterministic"),
                            evidence_version("anthropic"))
        with self.assertRaises(ValueError):
            evidence_version("unknown")

    def test_model_changes_coordinate_not_task_version(self):
        s = scenario()
        rubric = load_rubric(s)
        a = benchmark_identity(s, rubric, "deterministic", "devin/swe")
        b = benchmark_identity(s, rubric, "deterministic", "devin/gpt")
        self.assertEqual(a["task_version"], b["task_version"])
        self.assertNotEqual(a["coordinate"], b["coordinate"])
        for value in a.values():
            self.assertIsInstance(value, str)

    def test_legacy_identity_is_marked_and_never_current(self):
        s = scenario()
        rubric = load_rubric(s)
        current = benchmark_identity(s, rubric, "deterministic", "scripted")
        with tempfile.TemporaryDirectory() as run:
            with open(os.path.join(run, "scenario.json"), "w") as f:
                json.dump(s, f)
            with open(os.path.join(run, "rubric.json"), "w") as f:
                json.dump(rubric, f)
            legacy = legacy_run_identity(
                run, {"npc_mode": "deterministic", "agent_model": "scripted"},
                {"scored_with": "abc123"})
        self.assertEqual(legacy["physics"], "legacy:abc123")
        self.assertEqual(legacy["evidence"], "legacy:unknown")
        self.assertNotEqual(legacy["task_version"], current["task_version"])
        self.assertNotEqual(legacy["coordinate"], current["coordinate"])

    def test_legacy_devin_model_recovers_from_conversation_once(self):
        with tempfile.TemporaryDirectory() as run:
            with open(os.path.join(run, "devin_conversation.json"), "w") as f:
                json.dump({"steps": [{"model_name": "old"},
                                     {"model_name": "exact-model"}]}, f)
            legacy = legacy_run_identity(
                run, {"probe": "devin", "agent_model": "devin/swe"}, {})
            self.assertEqual(legacy["model"], "devin/exact-model")

            with open(os.path.join(run, "devin.json"), "w") as f:
                json.dump({"resolved_model": "devin/status-model"}, f)
            legacy = legacy_run_identity(
                run, {"probe": "devin", "agent_model": "devin/swe"}, {})
            self.assertEqual(legacy["model"], "devin/status-model")

    def test_initial_agent_model_labels_do_not_resolve_devin_alias(self):
        from sim.harness import _agent_model_labels
        self.assertEqual(
            _agent_model_labels("devin", "ignored", "swe"),
            ("devin/swe", None))
        self.assertEqual(
            _agent_model_labels("llm", "claude-x", None),
            ("claude-x", "claude-x"))
        self.assertEqual(
            _agent_model_labels("scripted", "ignored", None),
            ("scripted", "scripted"))


class TestIdentityAggregation(unittest.TestCase):
    def _write_run(self, root, name, identity, resolved, status="completed"):
        run = os.path.join(root, name)
        os.mkdir(run)
        open(os.path.join(run, "events.jsonl"), "w").close()
        with open(os.path.join(run, "meta.json"), "w") as f:
            json.dump({"task": "task", "probe": "devin",
                       "agent_model": "devin/swe",
                       "resolved_model": resolved,
                       "rollout_status": status,
                       "benchmark_identity": identity}, f)
        return run

    def test_versions_and_resolved_models_are_not_pooled(self):
        with tempfile.TemporaryDirectory() as root:
            v1, v2 = "a" * 64, "b" * 64
            runs = [
                self._write_run(root, "a", _identity(v1), "devin/model-a"),
                self._write_run(root, "b", _identity(v1), "devin/model-a"),
                self._write_run(root, "c", _identity(v2), "devin/model-a"),
                self._write_run(root, "d", _identity(v1), "devin/model-b")]
            card = {
                "complete": True,
                **{m: {"normalized": 0.5} for m in
                    ("completion", "efficiency", "done_weight_rate", "combined")},
                "temporal": {"coverage": 1.0, "mae_work_minutes": 10,
                             "horizon_accuracy": 1.0,
                             "belief_revision_coverage": 1.0,
                             "time_weighted_mae_work_minutes": 12,
                             "revision_count": 1, "censored_forecasts": 0}}
            with patch("sim.bench.evaluate", side_effect=lambda path: card):
                output, skipped = aggregate(root)
            self.assertFalse(skipped)
            first, second = "task@" + v1[:8], "task@" + v2[:8]
            self.assertEqual(output[first]["devin/model-a"]["k"], 2)
            self.assertEqual(output[first]["devin/model-b"]["k"], 1)
            self.assertEqual(output[second]["devin/model-a"]["k"], 1)
            self.assertEqual(output[first]["devin/model-a"]["task_version"], v1)
            self.assertEqual(
                output[first]["devin/model-a"]["coordinates"],
                [with_resolved_model(_identity(v1),
                                     "devin/model-a")["coordinate"]])

    def test_unresolved_devin_alias_is_explicit_and_recoordinated(self):
        with tempfile.TemporaryDirectory() as root:
            version = "a" * 64
            identity = _identity(version, "stale-coordinate", "devin/swe")
            self._write_run(root, "unresolved", identity, None)
            self._write_run(root, "resolved", identity, "devin/exact")
            card = {
                "complete": True,
                **{m: {"normalized": 0.5} for m in
                    ("completion", "efficiency", "done_weight_rate", "combined")},
                "temporal": {"coverage": 1.0, "mae_work_minutes": 10,
                             "horizon_accuracy": 1.0,
                             "belief_revision_coverage": 1.0,
                             "time_weighted_mae_work_minutes": 12,
                             "revision_count": 1, "censored_forecasts": 0}}
            with patch("sim.bench.evaluate", side_effect=lambda path: card):
                output, skipped = aggregate(root)
            self.assertFalse(skipped)
            models = output["task"]
            self.assertIn("unresolved:devin/swe", models)
            self.assertIn("devin/exact", models)
            self.assertNotIn("devin/swe", models)
            expected = with_resolved_model(
                identity, "unresolved:devin/swe")["coordinate"]
            self.assertEqual(models["unresolved:devin/swe"]["coordinates"],
                             [expected])

    def test_legacy_conversation_recovery_matches_bench_and_runhub(self):
        with tempfile.TemporaryDirectory() as root:
            run = os.path.join(root, "run-a")
            os.mkdir(run)
            open(os.path.join(run, "events.jsonl"), "w").close()
            with open(os.path.join(run, "scenario.json"), "w") as f:
                json.dump(scenario(), f)
            with open(os.path.join(run, "meta.json"), "w") as f:
                json.dump({"task": "task", "probe": "devin",
                           "agent_model": "devin/swe",
                           "rollout_status": "completed"}, f)
            with open(os.path.join(run, "devin_conversation.json"), "w") as f:
                json.dump({"steps": [{"model_name": "exact-model"}],
                           "final_metrics": {}}, f)
            with open(os.path.join(run, "scorecard.json"), "w") as f:
                json.dump({"score": 0.4}, f)
            card = {
                "complete": True,
                **{m: {"normalized": 0.5} for m in
                    ("completion", "efficiency", "done_weight_rate", "combined")},
                "temporal": {"coverage": 1.0, "mae_work_minutes": 10,
                             "horizon_accuracy": 1.0,
                             "belief_revision_coverage": 1.0,
                             "time_weighted_mae_work_minutes": 12,
                             "revision_count": 1, "censored_forecasts": 0}}
            with patch("sim.bench.evaluate", side_effect=lambda path: card):
                output, skipped = aggregate(root)
            rec = RunHub(root).list_runs()[0]
            self.assertFalse(skipped)
            self.assertIn("devin/exact-model", output["task"])
            self.assertEqual(rec["resolved_model"], "devin/exact-model")
            with open(os.path.join(run, "meta.json")) as f:
                meta = json.load(f)
            legacy = legacy_run_identity(
                run, meta, {"scored_with": "unknown"})
            expected = with_resolved_model(
                legacy, "devin/exact-model")["coordinate"]
            self.assertEqual(output["task"]["devin/exact-model"]["coordinates"],
                             [expected])
            self.assertEqual(rec["benchmark_coordinate"], expected)

    def test_runhub_separates_task_versions_and_models(self):
        with tempfile.TemporaryDirectory() as root:
            for name, version, model in (
                    ("run-a", "a" * 64, "devin/model-a"),
                    ("run-b", "b" * 64, "devin/model-b")):
                run = os.path.join(root, name)
                os.mkdir(run)
                for filename, data in (
                        ("scenario.json", {}),
                        ("events.jsonl", None),
                        ("scorecard.json", {"score": 0.4}),
                        ("meta.json", {
                            "task": "same", "probe": "devin",
                            "agent_model": "devin/swe",
                            "resolved_model": model,
                            "rollout_status": "completed",
                            "benchmark_identity": _identity(
                                version, "coord-" + version, model)})):
                    path = os.path.join(run, filename)
                    if data is None:
                        open(path, "w").close()
                    else:
                        with open(path, "w") as f:
                            json.dump(data, f)
            hub = RunHub(root)
            recs = hub.list_runs()
            self.assertEqual([r["task_version"] for r in recs],
                             ["b" * 64, "a" * 64])
            self.assertEqual(
                recs[0]["benchmark_coordinate"],
                with_resolved_model(_identity("b" * 64),
                                    "devin/model-b")["coordinate"])
            tasks = hub.list_tasks()
            self.assertEqual([t["task"] for t in tasks],
                             ["same@" + "b" * 8, "same@" + "a" * 8])
            self.assertEqual(tasks[0]["task_version"], "b" * 64)
            self.assertIn("devin/model-b", tasks[0]["models"])

    def test_devin_export_uses_last_nonempty_model_name(self):
        from sim.devin_adapter import _exported_model
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "conversation.json")
            with open(path, "w") as f:
                json.dump({"steps": [{"model_name": "old"},
                                     {"model_name": ""},
                                     {"model_name": "new"},
                                     {"model_name": " "}]}, f)
            self.assertEqual(_exported_model(path), "new")
            with open(path, "w") as f:
                json.dump({"steps": []}, f)
            self.assertIsNone(_exported_model(path))


if __name__ == "__main__":
    unittest.main()
