"""Canonical benchmark identity coordinates.

A comparable benchmark cell is (scenario, physics, evidence, rubric, resolved
model). Launch aliases are labels only; the resolved model is the comparison
identity.
"""

import hashlib
import json
import os

from .rubric import load_rubric

PHYSICS_FILES = (
    "engine.py", "events.py", "eval.py", "optimal.py", "rubric.py",
    "signals.py", "sim_time.py", "tasks.py", "temporal.py", "validate.py",
    "work.py", "world.py",
)

EVIDENCE_COMMON = (
    "tools.py", "harness.py", "devin_adapter.py", "prompts/agent_system.txt",
    "npc.py",
)


def _clean(value):
    """Recursively drop underscore-prefixed documentation keys."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()
                if not str(k).startswith("_")}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_json(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def scenario_version(scenario):
    """Hash the semantic scenario, excluding stamps, comments, and rubric refs."""
    semantic = _clean(scenario)
    for key in ("band", "rubric", "evaluation"):
        semantic.pop(key, None)
    return _sha_json(semantic)


def rubric_version(rubric):
    return _sha_json(_clean(rubric))


def _hash_files(root, paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        with open(os.path.join(root, path), "rb") as f:
            digest.update(f.read())
        digest.update(b"\0")
    return digest.hexdigest()


def physics_version():
    root = os.path.dirname(__file__)
    return _hash_files(root, PHYSICS_FILES)


def evidence_version(npc_mode):
    if npc_mode not in ("anthropic", "deterministic"):
        raise ValueError("unknown npc_mode %r" % npc_mode)
    paths = EVIDENCE_COMMON + (
        ("deterministic_npc.py",) if npc_mode == "deterministic" else ())
    return _hash_files(os.path.dirname(__file__), paths)


def _task_version(fields):
    return _sha_json(fields)


def _coordinate(task_version, model):
    return _sha_json({"task_version": task_version, "model": model})


def benchmark_identity(scenario, rubric, npc_mode, model):
    fields = {
        "scenario": scenario_version(scenario),
        "physics": physics_version(),
        "evidence": evidence_version(npc_mode),
        "rubric": rubric_version(rubric),
        "npc_mode": npc_mode,
    }
    task_version = _task_version(fields)
    return {**fields, "task_version": task_version, "model": model,
            "coordinate": _coordinate(task_version, model)}


def with_resolved_model(identity, model):
    """Return the same task coordinate, re-sealed for a resolved model label."""
    out = dict(identity, model=model)
    out["coordinate"] = _coordinate(out["task_version"], model)
    return out


def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def normalize_devin_model(model):
    if model and not model.startswith("devin/"):
        return "devin/" + model
    return model


def _conversation_model(path):
    conversation = _read_json(path, {})
    for step in reversed(conversation.get("steps") or []):
        if step.get("model_name"):
            return step["model_name"]
    return None


def resolved_model_id(run_dir, meta):
    """Recover the exported model ID without treating a launch alias as one."""
    meta = meta or {}
    model = meta.get("resolved_model")
    if meta.get("probe") == "devin":
        if not model:
            model = _read_json(os.path.join(run_dir, "devin.json"), {}).get(
                "resolved_model")
        if not model:
            model = _conversation_model(
                os.path.join(run_dir, "devin_conversation.json"))
        model = normalize_devin_model(model)
    return model


def legacy_run_identity(run_dir, meta, scorecard):
    """Best-effort identity for pre-coordinate runs, explicitly marked legacy."""
    scenario = _read_json(os.path.join(run_dir, "scenario.json"), {})
    rubric = _read_json(os.path.join(run_dir, "rubric.json"), None)
    if rubric is None:
        try:
            rubric = load_rubric(scenario, run_dir) if scenario else {}
        except Exception:
            rubric = {}
    model = resolved_model_id(run_dir, meta) or \
        (meta or {}).get("agent_model") or "unknown"
    if (meta or {}).get("probe") == "devin":
        model = normalize_devin_model(model)
    npc_mode = (meta or {}).get("npc_mode") or "unknown"
    fields = {
        "scenario": scenario_version(scenario),
        "physics": "legacy:%s" % (scorecard or {}).get("scored_with", "unknown"),
        "evidence": "legacy:unknown",
        "rubric": rubric_version(rubric),
        "npc_mode": npc_mode,
    }
    task_version = _task_version(fields)
    return {**fields, "task_version": task_version, "model": model,
            "coordinate": _coordinate(task_version, model)}
