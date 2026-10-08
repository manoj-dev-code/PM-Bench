"""Devin CLI adapter using the simulator's canonical tool registry.

Devin runs in an isolated temporary workspace. Its only approved command is a
small client for a random localhost gateway, so benchmark source, scenario
truth, and grading code are not in its workspace.
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .sim_time import wall_now
from .tools import call_tool, schemas


CLIENT = """\
import json
import sys
import urllib.request

if len(sys.argv) != 3:
    raise SystemExit("usage: gateway_client.py TOOL 'JSON_ARGS'")
body = json.dumps({"token": %r, "name": sys.argv[1],
                   "args": json.loads(sys.argv[2])}).encode()
request = urllib.request.Request(%r, data=body,
                                 headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=180) as response:
    print(response.read().decode())
"""

EXEC_GUARD = """\
import json
import shlex
import sys

try:
    event = json.load(sys.stdin)
    command = event.get("tool_input", {}).get("command", "")
    argv = shlex.split(command)
    allowed = (len(argv) == 4 and argv[:2] == [%r, "gateway_client.py"]
               and argv[2] in %r and isinstance(json.loads(argv[3]), dict))
except Exception:
    allowed = False
sys.exit(0 if allowed else 2)
"""


def _exported_model(export_path):
    """Last non-empty model_name in a Devin CLI conversation export."""
    try:
        with open(export_path) as f:
            steps = (json.load(f).get("steps") or [])
    except (OSError, ValueError):
        return None
    for step in reversed(steps):
        model = step.get("model_name")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return None


def _handler(engine, turns, token, max_turns):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(min(size, 100000)))
                if self.path != "/tool" or data.get("token") != token:
                    raise ValueError("invalid gateway request")
                if len(turns) >= max_turns:
                    result = {"error": "runner safety limit reached"}
                else:
                    result = call_tool(engine, data.get("name"), data.get("args"))
                    turns.append({
                        "tool": data.get("name"), "args": data.get("args"),
                        "ok": not (isinstance(result, dict)
                                  and "error" in result),
                    })
                payload, status = json.dumps(result, default=str).encode(), 200
            except Exception as error:
                payload, status = json.dumps({"error": str(error)}).encode(), 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    return Handler


def _prompt(system, port, python_executable):
    public = [{k: t[k] for k in ("name", "description", "input_schema")}
              for t in schemas()]
    return (
        system + "\n\n"
        "You are operating the simulation, not editing software. Use only this "
        "exact terminal form for every action:\n"
        "  %s gateway_client.py TOOL_NAME 'JSON_OBJECT'\n"
        "For example: %s gateway_client.py view_tasks '{}'\n"
        "Do not read files, inspect processes, write code, use network tools, "
        "or run any other command. Tool results are authoritative. Start with "
        "view_tasks, operate until the evaluation horizon, then stop. The "
        "gateway is on an opaque local port; its number has no simulated-time "
        "meaning (%d).\n\nAvailable simulation tools:\n%s"
        % (python_executable, python_executable, port,
           json.dumps(public, indent=2)))


def run_devin(engine, system, max_turns=200, model=None, timeout=900):
    if shutil.which("devin") is None:
        raise RuntimeError("Devin CLI not found; install it before --probe devin")

    turns = []
    token = uuid.uuid4().hex
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), _handler(engine, turns, token, max_turns))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    workspace = tempfile.mkdtemp(prefix="pmsim-devin-")
    python_executable = "/usr/bin/python3"
    export_path = os.path.join(workspace, "conversation.json")
    started = wall_now()
    try:
        with open(os.path.join(workspace, "gateway_client.py"), "w") as f:
            f.write(CLIENT % (token, "http://127.0.0.1:%d/tool"
                              % server.server_address[1]))
        with open(os.path.join(workspace, "check_exec.py"), "w") as f:
            allowed_tools = tuple(sorted(tool["name"] for tool in schemas()))
            f.write(EXEC_GUARD % (python_executable, allowed_tools))
        os.mkdir(os.path.join(workspace, ".devin"))
        with open(os.path.join(workspace, ".devin", "config.json"), "w") as f:
            json.dump({
                "permissions": {
                    "deny": ["read", "edit", "grep", "glob"],
                    "ask": ["exec"],
                    "allow": [
                        "Exec(%s gateway_client.py)" % python_executable],
                },
                "read_config_from": {
                    "agents_standard": False, "cursor": False,
                    "windsurf": False, "claude": False, "copilot": False,
                    "opencode": False, "zed": False,
                },
            }, f, indent=2)
        with open(os.path.join(workspace, ".devin", "hooks.v1.json"), "w") as f:
            json.dump({
                "PreToolUse": [{
                    "matcher": "exec",
                    "hooks": [{"type": "command",
                               "command": "%s check_exec.py" % python_executable,
                               "timeout": 5}],
                }],
            }, f, indent=2)
        command = [
            "devin", "--permission-mode", "auto",
            "--respect-workspace-trust", "false",
            "--export", export_path,
        ]
        if model:
            command += ["--model", model]
        command += ["-p", _prompt(
            system, server.server_address[1], python_executable)]
        try:
            result = subprocess.run(command, cwd=workspace, text=True,
                                    capture_output=True, timeout=timeout)
            status = {
                "exit_code": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr, "timed_out": False,
            }
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            status = {
                "exit_code": None, "stdout": stdout,
                "stderr": stderr, "timed_out": True,
            }
        status.update({"wall_seconds": round(wall_now() - started, 3),
                       "tool_calls": len(turns), "selected_model": model,
                       "resolved_model": _exported_model(export_path)})
        if os.path.exists(export_path):
            shutil.copyfile(export_path,
                            os.path.join(engine.run_dir, "devin_conversation.json"))
        with open(os.path.join(engine.run_dir, "devin.json"), "w") as f:
            json.dump(status, f, indent=2)
        engine.agent_run_status = (
            "completed" if status["exit_code"] == 0
            and not status["timed_out"] and turns
            else "runner_failed")
        engine.agent_run_failure = (
            None if engine.agent_run_status == "completed" else
            ("timeout" if status["timed_out"] else
             "no_tool_calls" if status["exit_code"] == 0 and not turns
             else "cli_exit_%s" % status["exit_code"]))
        if status["exit_code"] not in (0, None):
            print("Devin CLI exited %s: %s"
                  % (status["exit_code"], status["stderr"][-500:]))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        shutil.rmtree(workspace, ignore_errors=True)

    horizon = engine.agent_horizon
    if engine.agent_run_status == "completed" and engine.world.clock < horizon:
        engine.advance_until(horizon, max_events=2000)
    return turns
