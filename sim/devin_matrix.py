"""Run the same PM scenario through several Devin CLI models.

Examples:
    python -m sim.devin_matrix scenarios/crunch.json --models swe gpt opus kimi
    python -m sim.devin_matrix scenarios/crunch.json --models swe gpt opus kimi -k 3
    python -m sim.devin_matrix scenarios/crunch.json --models gpt-6 opus --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys


def rollout_commands(python, scenario, models, k, max_turns, timeout,
                     npc_mode="deterministic"):
    commands = []
    for model in models:
        for _ in range(k):
            commands.append([
                python, "-m", "sim.harness", scenario,
                "--probe", "devin", "--devin-model", model,
                "--max-turns", str(max_turns),
                "--devin-timeout", str(timeout),
                "--npc-mode", npc_mode,
            ])
    return commands


def main():
    parser = argparse.ArgumentParser(
        description="Run a scenario through multiple Devin CLI models")
    parser.add_argument("scenario", nargs="?", default="scenarios/crunch.json")
    parser.add_argument("--models", nargs="+",
                        default=["swe", "gpt", "opus", "kimi"])
    parser.add_argument("-k", type=int, default=1,
                        help="independent rollouts per model")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=900,
                        help="wall-clock seconds per rollout")
    parser.add_argument("--npc-mode", choices=["deterministic", "anthropic"],
                        default="deterministic",
                        help="coworker voice backend (default is fair and key-free)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the exact commands without model calls")
    parser.add_argument("--no-bench", action="store_true",
                        help="do not aggregate runs after the matrix")
    args = parser.parse_args()

    if args.k < 1:
        parser.error("-k must be >= 1")
    if args.max_turns < 1:
        parser.error("--max-turns must be >= 1")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1")
    if len(set(args.models)) != len(args.models):
        parser.error("--models contains duplicates")

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scenario = os.path.abspath(args.scenario)
    if not os.path.exists(scenario):
        parser.error("scenario not found: %s" % args.scenario)
    if shutil.which("devin") is None:
        parser.error("Devin CLI not found")

    commands = rollout_commands(
        sys.executable, scenario, args.models, args.k,
        args.max_turns, args.timeout, args.npc_mode)
    print("Devin matrix: %d models x %d rollouts = %d runs"
          % (len(args.models), args.k, len(commands)))
    for command in commands:
        print("  " + " ".join(command))
    if args.dry_run:
        return

    auth = subprocess.run(["devin", "auth", "status"], cwd=repo,
                          text=True, capture_output=True)
    if auth.returncode:
        raise SystemExit(
            "Devin CLI is not authenticated in this shell.\n"
            + (auth.stdout or "") + (auth.stderr or ""))

    failures = 0
    for i, command in enumerate(commands, 1):
        print("\n=== rollout %d/%d: %s ==="
              % (i, len(commands), command[command.index("--devin-model") + 1]),
              flush=True)
        result = subprocess.run(command, cwd=repo)
        if result.returncode:
            failures += 1
            print("rollout failed with exit code %d" % result.returncode)

    if not args.no_bench:
        subprocess.run([sys.executable, "-m", "sim.bench", "runs"],
                       cwd=repo, check=False)
    if failures:
        raise SystemExit("%d/%d rollouts failed; failed runs are excluded "
                         "from sim.bench" % (failures, len(commands)))


if __name__ == "__main__":
    main()
