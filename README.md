# PM-Bench

A discrete-event simulation of a project manager's week at a small SaaS
company (see `Problem.txt`) — built as an evaluation and RL-training
environment: an agent PM discovers blockers, resolves conflicts, allocates
people, and is graded **only on outcomes**.

## The one-paragraph pitch

The world is a deterministic discrete-event simulation: LLMs voice the
coworkers but can never mutate state ("rules decide, LLM narrates"). Every
task instance carries a deterministically computable band
**[no-PM baseline, OPT_ideal]** — the floor is an *honest* baseline where
the org self-organizes without a PM (volunteers pick up unowned work,
people order their own queues sensibly), the ceiling is the frictionless
optimum from exhaustive search — and the agent's score is its normalized
position in that band: **0 = added nothing a functioning unmanaged team
wouldn't produce, 1 = perfect coordination, negative = made things worse**.
Everything is replayable byte-for-byte from the event log.

## What the environment tests (and how agents can learn it)

- **Channel economics** — chat is instant but costs the recipient 20
  serialized focus-minutes; email is delayed (batch delivery) but tax-free;
  meetings block every attendee's calendar. None of this is in any tool
  description: agents learn it from evidence (reply latencies, completions
  slipping), from testimony (NPCs *feel* interruptions and complain in
  persona), or in-weights (the reward gradient across rollouts).
- **Epistemics** — three layers of knowing (truth / belief / tracker),
  honest-but-wrong estimates that mature on authored schedules, red
  herrings, and a tracker that only auto-updates on completions. The
  interrogation hack is dead: asking everyone on Monday yields honest
  *wrong* answers.
- **Execution** — external asks arrive as chat pings (push, wakes the
  agent mid-`advance_time`) or email (batched push), each landing **already
  owned by a default holder** — often the wrong one: the swamped volunteer,
  the wrong specialist. The PM's levers are `assign_task` (move it) and
  `reprioritize` (reorder it, scheduling-only — the authored priority keeps
  its scoring weight). If the PM does nothing, the defaults just play out.
- **Fairness, twice** — scenarios are *gated* for fairness (every scored
  ask must be completable after the PM could first know it), and the PM is
  *scored* on fairness (1 − σ of per-person hours-worked/hours-available).

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY
```

```sh
# dashboard hub: every run, live feeds, benchmark Pareto (per-task curves,
# model-family trajectories)
.venv/bin/python -m sim.server --hub 8742          # open http://127.0.0.1:8742

# run an agent
.venv/bin/python -m sim.harness scenarios/demo.json --probe scripted     # zero agent tokens
.venv/bin/python -m sim.harness scenarios/demo_2.json --probe llm --model claude-sonnet-5

# use Devin CLI as the PM (one-time prerequisite: devin auth login)
.venv/bin/python -m sim.harness scenarios/crunch.json --probe devin
.venv/bin/python -m sim.harness scenarios/crunch.json --probe devin --devin-model opus

# compare model families on identical tasks (defaults: swe, gpt, opus, kimi)
.venv/bin/python -m sim.devin_matrix scenarios/crunch.json --models swe gpt opus kimi
.venv/bin/python -m sim.devin_matrix scenarios/crunch.json --models swe gpt opus kimi -k 3

# inspect exact commands first; use exact IDs from `devin models list --format json`
# instead of moving family aliases when you need a permanently reproducible panel
.venv/bin/python -m sim.devin_matrix scenarios/crunch.json \
  --models swe gpt opus kimi --dry-run

# grade a run (deterministic, reproducible forever)
.venv/bin/python -m sim.eval runs/run-XXXXXXXX-XXXXXX

# replay any run with zero LLM calls
.venv/bin/python -m sim.replay runs/run-XXXXXXXX-XXXXXX [until_seq]
```

Devin matrix runs use `--npc-mode deterministic` by default: coworker messages
are rendered from authored scenario intents and belief events without a second
model provider. This removes the `ANTHROPIC_API_KEY` requirement. For the same
scenario state and action history, deterministic mode produces the same
authored facts, timing, and wording; different model actions may intentionally
produce different trajectories. Use
`--npc-mode anthropic` only when intentionally testing stochastic, persona-rich
NPC dialogue with an Anthropic key configured.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests    # ~0.1s, zero LLM calls
```

Deterministic smoke tests over every rule-based component — each one pins a
bug class that actually occurred: replay dropping message fields, retroactive
assignment credit, priority-blind personal queues, unknown recipients
crashing the engine, the org fallback double-firing, unfair arrivals passing
validation, agents minting score with made-up tasks.

## Task authoring & calibration toolchain

```sh
.venv/bin/python -m sim.optimal scenarios/demo.json        # the OPT ceiling + assignment
.venv/bin/python -m sim.difficulty scenarios/demo.json     # hardness vector + fairness gate
.venv/bin/python -m sim.difficulty --stamp scenarios/*.json # embed band metadata into configs
.venv/bin/python -m sim.intent scenarios/demo_2.json       # WHICH decisions carry the band
```

- `sim.difficulty` prints the hardness **vector** (capacity utilization,
  winnable band, forced triage, assignment divergence, reaction ratios,
  log₁₀ trajectory classes) — all from config, no engine runs, no LLM.
- `sim.validate` (run at every engine init) **rejects unfair tasks**: any
  ask that can't physically be completed after the PM could first know of
  it scores luck, not skill.
- `sim.intent` plays deterministic policy ablations (oracle / late /
  no-spread / noisy) and reports each mechanism's **share of the band** — a
  mechanism with share ≈ 0 is not taught by that task.

## Generating difficulty-tuned tasks

```sh
.venv/bin/python -m sim.generate scenarios/demo.json --tier hard --n 5 --seed 7 --out scenarios/gen
```

One template carries the fiction (personas, announce texts, herrings); the
generator mints instances by remixing the *physics* — efforts, arrival
times, ticket ids (anti-memorization), default owners (the `pileup` knob:
does everything land on one volunteer?), belief errors, latency seeds.
Every draw is keyed by (template, tier, seed) and materialized into the
emitted file; every candidate passes the full gate (validation → fairness →
tier band) and is stamped with its `[baseline, OPT]` anchors. Tiers target
capacity utilization and winnable-band ranges; hardness is then *measured*
(probe-ladder score degradation), never asserted by the label. Rejected
samples cost attempts, never bad instances — an authoring bug cannot ship.

## Grading, and how it resists reward hacking

Score = the agent's normalized position in the per-task band
`[no-PM baseline, OPT_ideal]` on the COMBINED metric (completion +
γ·efficiency), with done-rate and workload fairness reported alongside.
Every number is deterministic and re-computable forever from the run dir
(`python -m sim.eval runs/<id>`); scorecards carry a `scored_with` hash of
the scoring source so stale caches self-identify.

The benchmark also measures **prospective calibrated time sense** separately
from reward. Agents record an initial `p50/p90` completion interval (absolute
simulated minutes, `p50 <= p90`) for each visible authored task before it
completes, then may revise only after new evidence or a plan mutation. Each
accepted forecast costs one simulated minute. Reports include p50 initial,
final, and work-time-weighted MAE, p90 coverage and final error, mean interval
width (including censored tasks), pinball loss for both quantiles, revision
improvement, belief-reveal coverage and latency, by-horizon accuracy, and
censored unfinished tasks. Calibration is never shown without coverage and
censoring, so an agent cannot look accurate merely by forecasting one easy
item.

`python -m sim.bench runs` reports Max@k inside a canonical benchmark
coordinate: `(semantic scenario, physics source, evidence renderer + NPC mode,
pinned rubric, resolved model)`. The first four plus NPC mode form
`task_version`; `task_version + resolved model` forms `coordinate`.
Incompatible task or environment versions never share a cell. Model aliases
such as `devin/swe` are launch labels only — the exported resolved model ID is
the comparison identity. It does not pool different tasks and call their
maximum Max@k; the cross-task summary is an equal-task macro-average of
task-level means. Failed runners and runs that never reached their evaluation
horizon are excluded and reported.

The Devin adapter runs the CLI in a temporary workspace with benchmark source
and scenario truth absent. Its only approved command calls the same canonical
tool registry used by the Claude harness and UI. The simulator remains the
clock owner: Devin inference latency is wall-clock telemetry only.

Blocking decisions are explicit state transitions: agents must call
`answer_question` with the authored task, question, option, rationale, and
delivery channel. Ordinary chat or email can discuss a blocker but cannot
silently unblock it. Options can carry deterministic consequences — effort
deltas and named capacity costs — activated only at the decision's delivery
instant, so an email choice does not change work before it lands. Meeting
throughput counts eligible workers only; the PM and stakeholder attendees
consume calendar time but never manufacture labor.

Authored work may also be staged: a work item expands deterministically into
sequential scheduler tasks, and a quality gate may pre-materialize a hidden
rework stage. The rework exists in ground truth but stays off the tracker
until its gate truly completes, at which point a `task_revealed` event wakes
the PM and tells the owner the authored reason. Each stage carries a
`score_weight_factor`, so splitting work changes sequencing and visibility
without changing the item's total priority weight.

`scenarios/crunch.json`, the recommended comparison task, now includes two
successive belief corrections on the critical migration, an explicit cutover
tradeoff, a migration review gate before backfill, handoff/helper onboarding
capacity costs, a cancellable standing meeting, an authored OOO block, and
forecast revision measurement. This makes the model update its plan and time
estimate as evidence changes rather than react to a single scripted reveal.

| attack | defense |
|---|---|
| look busy (chats, meetings, check-ins) | activity earns nothing; interruptions cost the recipient 20 serialized focus-min and meetings block calendars — measured runs of chatty PMs score **below zero** |
| mint value with made-up tasks | agent-created items are tracking-only: zero rubric weight and zero scheduler capacity |
| relabel a task's priority to score more | `reprioritize` changes scheduling order only; scoring weight stays authored |
| claim credit for work that happens anyway | the baseline is a complete unmanaged org (default owners work every arrival); score 0 = "added nothing beyond it" |
| out-labor the ceiling via stakeholders | the labor pool binds both sides: OPT excludes `worker:false` people and the engine refuses to assign them |
| game the tracker | scoring reads scheduler truth, never reported status |
| interrogate everyone for hidden truth | beliefs are honest-but-wrong; there is no oracle to extract, only completions to observe |
| farm an impossible ask for sympathy points | the fairness gate rejects any task not completable after the PM could first know of it |
| exploit judge wording | there is no judge: no LLM, no keywords, no checks anywhere in scoring |

## Invariants (the short version of DESIGN.md)

1. **LLMs produce text only** — no mutation channel, no physics, no scoring.
2. **The world is a deterministic function of (config + seed, actions).**
   Randomness must be keyed by (seed, decision id), never stream-ordered;
   draws that can be materialized pre-run are stamped into the config.
3. **Outcome-only rubrics** — no keyword checks, no behavior rules, no LLM
   judging. Noise punishes itself through capacity physics.
4. **Sim time never touches the wall clock** — `wall_now()` is the single
   permitted system-clock call site, telemetry only.
5. **Liveness** — no agent action and no NPC brain failure can kill the
   sim; unknown recipients bounce, LLM errors degrade to placeholder text.

## Layout

```
sim/
  sim_time.py    # minutes since Mon 00:00; work calendar; the only wall-clock exile
  events.py      # (sim_time, seq) min-heap — time only advances on pop
  world.py       # single clock owner; append-only log; tracker/belief/truth views
  tasks.py       # the scheduler: pure fold of (tasks, assignments, busy intervals)
  signals.py     # push/pull delivery semantics for every signal, one table
  npc.py         # pure appended-context agents: experience stream + one voice call
  engine.py      # pop → advance → dispatch; arrivals, fallbacks, beats, meetings
  tools.py       # the agent-facing tool surface (neutral descriptions, no economics)
  validate.py    # load-time scenario gate incl. the fairness gate
  rubric.py      # task_value: completion, efficiency, done-rate, workload fairness
  optimal.py     # OPT_ideal: exhaustive frictionless ceiling (labor pool = workers)
  eval.py        # band-normalized scorecard; odds form; always-numeric reward
  difficulty.py  # hardness vector + band stamping + trajectory-space estimate
  intent.py      # mechanism-share audit: which decisions carry the band
  harness.py     # scripted + LLM probes; cost/turn accounting; meta stamping
  replay.py      # world_at(seq) = fold(scenario, events[:seq+1])
  server.py      # observer-only web UI: engine / --watch / --hub modes
web/index.html   # live feeds, rewind, trajectory lanes, benchmark Pareto
scenarios/       # task configs with stamped band metadata
rubrics/         # scoring configs, copied into each run dir
```

Every run dir is self-contained and re-gradeable forever: `scenario.json`,
`rubric.json`, `events.jsonl` (source of truth), `llm.jsonl` (full traces),
`meta.json`, `scorecard.json`.

Full design rationale: `DESIGN.md`.
