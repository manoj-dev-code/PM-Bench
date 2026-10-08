# Design: systems, semantics, and scoring

A simulated week at a SaaS company, built as an evaluation and RL-training
environment for PM agents. This document pins down the semantics the problem
statement asks to be legible — what advances synchronously with an agent
action, what advances asynchronously in the background, how state is owned
and mutated, how time flows — and the scoring/authoring machinery built on
top. Earlier design iterations live in git history, not here.

1.  [The model in one paragraph](#1-the-model-in-one-paragraph)
2.  [State and time](#2-state-and-time)
3.  [Causality and liveness invariants](#3-causality-and-liveness-invariants)
4.  [The event queue](#4-the-event-queue)
5.  [Channels: delivery and friction](#5-channels-delivery-and-friction)
6.  [Work: tasks, the scheduler, filing, the org fallback](#6-work-tasks-the-scheduler-filing-the-org-fallback)
7.  [NPCs: pure appended-context agents](#7-npcs-pure-appended-context-agents)
8.  [Epistemics: three layers of knowing](#8-epistemics-three-layers-of-knowing)
9.  [The determinism boundary](#9-the-determinism-boundary)
10. [Evaluation: the band](#10-evaluation-the-band)
11. [Task authoring and difficulty](#11-task-authoring-and-difficulty)
12. [The harness](#12-the-harness)
13. [Observability](#13-observability)
14. [Tests](#14-tests)

## 1. The model in one paragraph

The world is a discrete-event simulation: a min-heap of `(sim_time, seq)`
events, a single monotonic clock, and an append-only event log from which
any past state is a pure fold. LLMs voice the coworkers but can never mutate
state — **rules decide, LLM narrates**. Every task instance carries a
deterministically computable band **[no-PM baseline, OPT_ideal]**: the floor
is an honest baseline in which the org self-organizes without a PM, the
ceiling is the frictionless optimum from exhaustive search, and the agent's
score is its normalized position in that band (0 = added nothing a
functioning unmanaged team wouldn't produce; 1 = perfect coordination;
negative = made things worse). Everything an agent can exploit is priced by
physics, not policed by checks.

## 2. State and time

### The three classes of state change

| Class | Examples | When it executes | Clock behavior |
|---|---|---|---|
| **Synchronous** (agent-caused) | send a message, file/assign/reprioritize a task, book a meeting | immediately, inside the agent's call frame | `clock += 1` (fixed action cost); the world mutates *at that instant* |
| **Asynchronous** (queue-driven) | NPC replies, wakeups, beats, arrivals, org pickups, email deliveries, belief updates, meeting start/end | only when time rolls forward and the heap pops | clock **jumps** to `event.time`; between pops, nothing executes |
| **Derived** (analytic) | task progress, projections, busy intervals, utilization | **never** — pure functions of `(config, clock, logged mutations)` | no events at all; evaluating at a later clock simply yields a later value |

The derived class is the load-bearing trick: task hours "accrue overnight"
without anything running overnight, because progress is computed on demand
from the work calendar (Mon–Fri 09:00–17:30, one task at a time per person).
The async event set stays tiny, and rewind/replay get derived state for free.

### The two mutation frames and event sourcing

World state is owned by one `World` object and mutates in exactly two stack
frames: agent tool handlers (synchronous) and `Engine._dispatch(event)`
(asynchronous). Both funnel through `world.record(...)`, which appends to
the log and flushes to `runs/<id>/events.jsonl`. Therefore:

- `state = fold(scenario, events)` — the log is the complete history.
- Replay (`sim/replay.py`) reconstructs any past state with **zero LLM
  calls**: the only sampled thing (NPC text) is captured in the log.
- The UI's rewind slider is "render the fold of a log prefix."
- Runs are self-contained forever: `scenario.json`, `rubric.json`,
  `events.jsonl`, `llm.jsonl`, `meta.json`, `scorecard.json`.

### The two places time advances

1. **Agent action cost**: +1 sim-minute per tool call (typing and deciding
   take time; you cannot do 500 things at 09:00:00).
2. **Queue pop**: `clock = event.time` — time jumps, never crawls.

Wall-clock never leaks in: when an `npc_respond` pops, the LLM call happens
*inside that frozen instant* — Sarah replies "at 09:42" whether inference
took 2 or 30 real seconds. `sim_time.wall_now()` is the single permitted
system-clock call site in the codebase, for telemetry only.

The agent owns time. There is no background thread: `advance_time` and
`wait_for_reply` are the yield primitives, and waiting is a real PM decision
the physics grades (beats, arrivals, escalations fire *during* the jump;
evaluation happens at a fixed horizon, so skipping time forfeits agency
rather than dodging judgment). The operator's `advance_until(target)` fires
everything up to the target then jumps — nights and weekends included — so
a full 7-day run is one call.

## 3. Causality and liveness invariants

Enforced in code and by the property test suite (`tests/test_des_properties.py`,
seeded fuzzing with invariants asserted after every action):

1. **100% causal clock.** *No undispatched event exists at or before the
   clock, at any instant the agent observes.* Action-cost advances drain
   everything that became due; interruptible advances and `wait_for_reply`
   drain all co-temporal events before returning control (the fuzzer found
   the one-minute hole where a wake handed control back with a same-minute
   event still pending).
2. **Monotonic clock by construction.** One mutation point
   (`World.advance_clock_to`) that raises on any backwards move.
3. **The heap never starves.** Every `npc_wakeup` reschedules itself
   strictly in the future; scenarios must have ≥1 NPC (validated). An
   emptied heap still cannot deadlock: advancing jumps the void.
4. **No agent action can kill the sim.** Unknown chat recipients error
   in-fiction; unknown email addresses bounce (`{"bounced": [...]}`) — NPC
   voices can hallucinate out-of-world contacts, and messaging one simply
   doesn't deliver. Every fuzzed nonsense call returns an error dict, never
   an exception.
5. **LLM failures never stall the world.** NPC-brain exceptions degrade to
   placeholder text; agent-side API failures end the agent's week
   gracefully — the run still advances to horizon and is scored.
6. **Waiting is bounded.** `wait_for_reply` gives up after ~1.5 workdays
   rather than silently consuming the week.
7. **Advance granularity is irrelevant.** One jump to Friday and a thousand
   7-minute hops produce the identical world (property-tested).

## 4. The event queue

| Kind | Scheduled by | On fire |
|---|---|---|
| `npc_respond` | chat latency rule at delivery (seeded per-NPC RNG, 5–45 min in-hours, deferred past meetings) | one LLM call → reply |
| `npc_wakeup` | self-rescheduling heartbeat (~90–150 min, working hours) | answers the email batch; reschedules itself |
| `email_delivery` | sending email to the PM | the batched push: delivers to the PM at the next `email_batch_minutes` grid tick (default 30) |
| `task_arrival` | scenario `task_arrivals[]` | the task lands **already owned** by its default holder (on the board, `assigned_at` = arrival, worked from that instant) + an announcement (chat or email) carrying a rules-appended `(ticket: id, currently on X)` |
| `belief_update` | authored `belief[]` schedules on tasks | corrects the holder's honest picture; optional proactive confession ping |
| `beat` | scenario `beats[]` | deterministic IF-chain over world truth picks an arm; the LLM only voices the chosen intent |
| `meeting_start` / `meeting_end` / `room_turn` | `schedule_meeting` | live rooms: speak-or-PASS NPC turns, verbatim minutes broadcast at end |

Beats and arrivals are how authored pressure lives in the same queue as
organic behavior: a scenario is data (personas + tasks + beats + arrivals +
fallbacks + seed), not code.

## 5. Channels: delivery and friction

### Everything is push, because this is a DES

A human PM "always refreshing email" is a background process; the agent has
no background thread, and in a DES nothing happens except by an event
popping. So there is **no poll tool anywhere**: push-class signals ride
along with every tool result (`notifications`), and a *person* signal — a
chat ping, a delivered email batch, someone speaking in your meeting —
**interrupts `advance_time` early**, handing control back at that instant.
Board broadcasts (completions) are delivered but never break sleep: only
people and mail wake you. `view_inbox` is scrollback of what has been
delivered; an email in flight is invisible.

Delivery semantics are one static table (`sim/signals.py`); `World.record`
stamps every logged signal with its `delivery`, so runs are self-describing:

| Signal | Sender | Recipient | Physics |
|---|---|---|---|
| chat | push | **push** | instant; interrupts — NPC pays the serialized focus tax; the PM is woken mid-advance |
| email | push | **push (batched)** | delivered by event on the recipient's cadence: NPC wakeups / the PM's batch grid. Delayed, zero tax |
| meeting invite / room line / minutes / doc share | push | push | calendar, live room, broadcast at end, shared stream |
| task added to board | push | **pull** | arrivals land on the board with their default owner; anyone must check it |
| board edit | push | pull | exception: an assignment also notifies the assignee |
| completion | push (auto) | push | the one public reliable broadcast; never wakes |
| holder belief | — (pull) | — | surfaced only when asked, or at an authored confession ping |

### Frictions, with their real-life justification

Every friction is a claim about how offices work — if a number can't be
defended by its real-world row, it doesn't belong in the physics. None of
this appears in any tool description: channel economics are learnable only.

**Chat**
| friction | value | real life |
|---|---|---|
| NPC reply latency | 5–45 min in-hours; else next morning | people see Slack between tasks; overnight pings get answered at 9am |
| focus tax on recipient | `costs.chat_interrupt_minutes` (default 20), **serialized** | an interruption costs ~20 min of refocus (attention-research classic); three pings are three broken flows |
| delivery to PM | instant push, wakes mid-sleep | a direct ping buzzes your phone |
| unknown recipient | error: not in directory | you can't DM someone outside the workspace |

**Email**
| friction | value | real life |
|---|---|---|
| focus tax | zero, both directions | email is async by social contract |
| NPC reads | next wakeup (~90–150 min) | people process inboxes in batches |
| PM receives | next 30-min grid tick, then push | inbox-refreshing is a background habit; in a DES that habit is a scheduled event |
| unknown address | bounces, reported in the result | mail to nowhere bounces; it never crashes the office |

**Meetings**
| friction | value | real life |
|---|---|---|
| capacity block | every attendee, full duration (≤240 min) | a 1-hour meeting with 4 people costs 4 person-hours |
| `talk_in_meeting` | zero queue latency; ~3 min of slot per exchange | everyone's in the room — answers are immediate, but talking consumes the shared hour |
| reply deferral | chats answered after the meeting | people don't answer DMs mid-meeting |
| minutes | verbatim broadcast to attendees at end | the record is the meeting's durable output |

### How agents learn the frictions

Three routes, deliberately layered: **evidence** (email K read directly off
reply timestamps; the tax inferred from completions slipping against
calibrated velocity), **testimony** (physics is *felt* — the tax lands in
the holder's experience stream as "that ping broke your concentration", and
whether they voice it is persona-gated, so listening to complaints is the
honest in-episode shortcut), and **in-weights** (the reward gradient across
rollouts, which needs no explicit knowledge at all).

## 6. Work: tasks, the scheduler, filing, the org fallback

### Task origins

1. **Seed** — `project.tasks`, arrival = sim start.
2. **External (OOD)** — `task_arrivals[]`: **real work always has an
   owner.** An arrival lands already assigned to a default holder (a
   validated worker), on the board, worked from the instant it lands
   (`assigned_at` = arrival). The default is often *wrong* — the swamped
   volunteer, the wrong specialist — and the announcement that reaches the
   PM (chat or email, with a rules-appended
   `(ticket: id, currently on X)`) is the cue to REARRANGE, not to file.
   There is no unowned backlog and no filing step: the PM's levers are
   `assign_task` (move it) and `reprioritize` (reorder it).
3. **Agent** — made-up tasks are trackable but carry **zero rubric weight**
   (the anti-minting guard) while still consuming real capacity if staffed.

### The scheduler (`sim/tasks.py`)

Deterministic, **preemptive**, work-conserving, event-sourced. Per person,
one task at a time, working hours only, minus busy intervals (meetings +
the serialized chat tax). A task is ready once it has arrived, somebody
holds it, and its blockers are done. At every instant each person works
their highest-priority ready task (**urgent → priority → creation order**);
the moment that changes — a higher-priority task arrives or unblocks, or
the PM reprioritizes — they switch, and the paused task keeps its accrued
work and resumes exactly where it stopped. Progress is the fold of the
resulting work segments, so true progress is answerable at any instant.

- **`assigned_at` — no retroactive credit.** Every assignment mutation
  stamps the instant; work starts at `max(arrival, assigned_at)`.
  (Assignment used to be retroactive — a 16:00 pickup was credited work
  since the 11:30 arrival — which made acting early worthless.)
  Reassignment persists progress-so-far into `done_hours` at handoff:
  prior work is kept, never re-credited.
- **Decision effects — deterministic option consequences.** A gating
  question may author `choice_effects`: an option-specific remaining-effort
  delta and/or named-person capacity cost. `answer_question` pins the whole
  effect into `question_answered`; nothing mutates until `delivered_at`, so
  a chat choice lands now while an email choice lands on the answer batch.
  At delivery the scheduler adds one static decision point, applies the
  delta exactly once, and serializes capacity costs around meetings, OOO,
  nights, and prior interruptions. Open-question tracker rows may expose
  each option's authored summary as `tradeoffs`.
- **Setup costs — coordination consumes capacity.** Optional scenario
  `costs.handoff_minutes` and `costs.helper_onboarding_minutes` record the
  same `capacity_cost` event when ownership changes or a helper joins. The
  new owner/helper pays the configured working minutes; dropping a helper
  and same-owner no-op assignment pay nothing.
- **Staged work and quality gates.** A staged seed work item materializes
  into ordinary scheduler tasks (`item::stage`) in authored order. A
  deterministic `quality_gate.outcome=rework` pre-materializes a contingent
  rework stage after the gate; it is in ground truth but hidden from tracker,
  beliefs, forecasts, meetings, and mutations until the gate truly completes.
  Completion emits `task_revealed` with the authored reason, wakes
  interruptible advance, and notifies the rework owner. Stage
  `score_weight_factor`s sum to 1.0, so a staged item keeps exactly one
  authored task's rubric weight.
- **`reprioritize` — ordering without minting.** The PM appends a
  timestamped `order_event` (working priority P0–P3 and/or urgency) to any
  filed task. Applied **forward-only** — a Thursday reprioritization can
  never rewrite Monday's schedule — and for *scheduling only*: the authored
  `priority` keeps its rubric weight, so the PM decides what gets worked
  first but can never relabel a task to score more.
- **Skill multipliers** (partially landed): per-(person × task-tag) speed
  factors, authored at generation time ("LLM authors, rules execute" —
  never a runtime LLM call, which would give GRPO rollouts different
  physics and make the multiplier promptable). The ceiling side is wired
  (`optimal.ideal_effort_hours` uses the best eligible worker, clamped ≥1×
  so OPT stays a true upper bound); the runtime lookup lands with the
  scenario generator.

### The default-owner baseline: not a strawman

A no-PM team still functions — badly. Every arrival's default holder works
it from the moment it lands, in sensible personal priority order, with
preemption. **Same physics in every run**, so the null baseline is a
complete unmanaged org, and the PM earns credit only for what coordination
adds: moving work off the overloaded volunteer, reordering queues, keeping
noise off people's focus. The corollary is sharp: under preemptive
scheduling, a scenario whose defaults happen to be adequate has an EMPTY
band (OPT == baseline — the original hand-authored `demo` went degenerate
exactly this way when this contract landed). Band width = how wrong the
default ownership is; the generator's `pileup` knob and effort multipliers
author that wrongness deliberately, and the band gate rejects instances
where the org didn't need a PM.

### The labor pool binds both sides

Stakeholders (`worker: false`, e.g. a VP) are excluded from OPT's labor pool
— treating them as fungible engineers inflates the ceiling. The engine
therefore refuses to assign them tickets, validation rejects stakeholder
seed-assignees and fallback owners, and capacity/utilization/fairness all
measure over workers only. Without the engine-side half, an agent could
out-labor OPT and score above 1.

## 7. NPCs: pure appended-context agents

**The contract:** they *do their work* (the scheduler accrues their hours —
no LLM involved), they *experience* (every stimulus is pushed into an
append-only context stream when it happens: messages, room lines, invites,
docs, belief realizations, completions, felt interruptions), and they
*speak, and only speak* — one LLM call per utterance; speech never mutates
world state. No tools of any kind. The stream is a fold of the log:
replay-safe, no hidden state, automatic change-awareness ("the meeting
moved" is simply two entries).

- **WHEN** an NPC acts is deterministic: latency rules, wakeup cadence,
  beat conditions, meeting turn offers (speak-or-PASS, least-spoken-first).
- **WHAT** it says is one LLM call over persona + knowledge + stream, at
  temperature 0 (removes sampling variance across rollouts; the NPC default
  is a non-thinking Haiku-class model via `SIM_NPC_MODEL`).

**Reasoning stays in the PM — by information asymmetry, not prompt rules.**
An NPC sees only its own slice (its tasks via belief, its threads, its
rooms). It cannot answer about others' workloads, compute global schedules,
or execute anything. The PM alone holds global observability and allocation
authority, so the optimization can only happen there — and the benchmark
measures the PM's model, never the NPCs'.

## 8. Epistemics: three layers of knowing

| Layer | Who has it | Mechanism |
|---|---|---|
| **Truth** | engine + eval only | the scheduler; nobody in the sim reads it — not even the task's holder |
| **Belief(t)** | each holder, about their own work | honest-but-possibly-wrong; matures on an authored schedule (`belief[]` → `belief_update`, optional proactive confession ping). Red herrings are the org-level version: confidently-held false knowledge |
| **Tracker** | everyone | the record — lags, inherits stale beliefs; auto-updates only on **completions**, the one public reliable signal |

Consequences (zero LLM on any information path):

- **The interrogation hack is dead**: asking everyone Monday yields honest
  *wrong* answers — there is no perfect truth to extract, only a process to
  run.
- **Exploration is first-class**: velocities are inferable only from
  observed completions (~2 data points per person); channel economics are
  absent from tool descriptions; the board must be re-checked because org
  pickups land on it silently (pull).
- **Noise resistance is measurable**: chasing an authored herring burns
  capacity the score prices; the `herrings` block makes distractors
  auditable per scenario.

**The information ledger** — every quantity classified by how it can be known:

| Class | Examples | Access | Cost |
|---|---|---|---|
| Hidden truth | true progress, projections, skill factors | inference only | — |
| Exposed truth | completions, assignments, deadlines, calendars | tracker/events | free |
| Testimony | beliefs, self-assessments, felt interruptions, org folklore | asking / listening | latency + tax + honestly-wrong risk |
| Learnable in-trajectory | velocities, testimony reliability, email K | probe + observe | time & actions |

Two learning channels, deliberately separate: **in-weights** (environment
physics — constant across tasks, learned across episodes) and **in-context**
(instance latents — this week's wrong beliefs, this team's speeds — varied
per instance, inferred fresh each trajectory). The split trains adaptivity,
not memorization; the generator must randomize instance latents (including
ticket ids) so nothing leaks into weights.

## 9. The determinism boundary

**The LLM produces text only — it has no channel to mutate the world.** All
mutations come from scenario config, engine rules, or agent tools.

**The randomness invariant.** The world must be a deterministic function of
(config + seed, action sequence). Randomness is allowed anywhere provided
(a) every draw is keyed by (seed, stable decision id) — never by
shared-stream order, where one extra agent action shifts every later draw
and reintroduces rollout variance in a determinism costume — and (b) any
draw that can be materialized before the run **is stamped into the config**
so the gates (fairness, band, intent audit) can see it. Endogenous draws
(weights depending on the trajectory, e.g. a volunteer chosen ∝ available
hours) stay lazy but keyed, and any lever they hand the agent should earn
its own intent-audit line before being trusted.

| Subsystem | Deterministic? | Notes |
|---|---|---|
| clock, queue, ordering | ✅ | seq-tiebroken heap; monotonic by construction |
| latencies, wakeups | ✅ | per-NPC seeded RNG streams |
| scheduling & progress | ✅ | pure function (calendar + busy intervals) |
| arrivals, fallbacks, beliefs, beats | ✅ | authored config, fixed times |
| email batch grid, focus tax | ✅ | static config (`email_batch_minutes`, `costs.*`) |
| meeting capacity cost | ✅ | interval subtraction |
| scoring, bands, gates | ✅ | pure functions of config + log |
| NPC / transcript text | 🤖 LLM | text only, temperature 0, captured in log → replay is LLM-free |

**Bounds on the levers** (`sim/validate.py`): ≤50 tasks, effort 0.5–80h,
≤10 arrivals, ≤240-min meetings, events inside the run window — validated
at load and enforced at runtime on every mutation path, so no author,
generator, or agent can push the world outside the envelope.

## 10. Evaluation: the band

Over **authored tasks only** (agent-created tasks carry zero weight),
graded at a fixed horizon, no checks, no keywords, no LLM anywhere:

    COMPLETION = Σ w · (progress + α·done)/(1+α)         did the work get done
    EFFICIENCY = Σ w · done · (horizon − done_at)/span    how early it got done
    COMBINED   = (completion + γ·efficiency)/(1+γ)        the scalar for RL
    DONE-RATE  = W(shipped)/W(all)                        what fraction shipped
    FAIRNESS   = 1 − σ(per-worker hours-worked/available) team health

Each is a ladder **baseline ≤ agent ≤ OPT_ideal**:

- **Baseline** = the same scenario with a do-nothing agent (stub LLM):
  default owners work every arrival from the moment it lands — a complete
  unmanaged team, not a strawman. Score 0 means "added nothing beyond it";
  negative means the agent's activity (focus taxes, meetings, busywork)
  made the world worse.
- **OPT_ideal** (`sim/optimal.py`) = frictionless exhaustive search over
  worker assignments: communication free, skills at the best clamped rate,
  no interruptions. A pure function of the scenario file — the stable
  ceiling.
- **`score`** = (agent − baseline)/(OPT − baseline), the normalized band
  position that makes 100 tasks commensurable. **`score_odds`** =
  delta/remaining-regret = N/(1−N): the same measurement with the top end
  stretched — gain per unit of regret, useful because the relaxed ceiling
  compresses top-end differences (the achievable gap is measurable per task
  by running the friction-respecting oracle policy; ~0.3% of the band on
  `demo`). **`reward`** is always numeric for training loops (falls back to
  the raw delta on degenerate instances).
- **Fairness is reported, not normalized** — dependency chains can force
  imbalance on perfect play, so OPT is the reference, not 1.0. It separates
  equal-scoring strategies (pile-on-the-volunteer vs spread) and stays out
  of `combined` by default: a second Pareto axis (task value × team
  health).

Noise punishes itself through physics — interruptions and meetings consume
capacity, completions land later, efficiency drops — so communication,
discovery, and record hygiene are deliberately unscored; their value flows
through outcomes. Behavioral readouts live in `navigation.json` (tool mix,
channel mix, contact patterns): telemetry, never reward.

**Calibrated forecasting** is a separate temporal axis, never outcome reward.
`forecast_completion` records a p50/p90 interval in absolute simulated minutes
(`p50 <= p90`) plus a rationale; recording costs one action minute and is not
accepted if that minute crosses the horizon or task completion. For an observed
completion, signed error is actual minus prediction in work minutes; p50 drives
the backward-compatible MAE/bias fields, while p90 coverage, interval width,
and q=0.5/q=0.9 pinball loss measure calibration. Censored forecasts contribute
interval width but no error, coverage, or pinball term. Legacy
`completion_time` events regrade as a zero-width p50=p90 interval.

## 11. Task authoring and difficulty

The authoring loop is config-iteration with analytic gates — seconds per
iteration, zero LLM cost:

1. **Fairness gate** (`sim/validate.py`): a scored ask must be physically
   completable from the moment the PM could first *know* it (chat: instant;
   email: next batch tick). Anything tighter scores luck — rejected at load.
   The fingerprint reports `reaction_ratio` per arrival/confession (≥1.0
   required; 1.0–1.3 is a legitimate squeeze).
2. **Difficulty fingerprint** (`sim/difficulty.py`) — hardness is a vector:
   winnable band, capacity utilization, forced triage
   (`opt_done_weight_rate < 1`: even perfect play must sacrifice),
   assignment divergence, interrupt load, reaction ratios, and
   `log10_trajectory_classes` (the outcome-distinguishable decision space:
   who^tasks × when-windows; ~10¹²–10¹⁶ for the current tasks — the space
   the band compresses to one dimension).
3. **Band stamping** (`--stamp`): the no-PM baseline and OPT anchors are
   embedded in each scenario file as documentation; recomputable in
   milliseconds, re-stamp after edits.
4. **Intent audit** (`sim/intent.py`): a task is authored with an intent —
   the decisions a winning PM must make — and the audit *proves it
   causally* by playing deterministic policy ablations (oracle / late /
   noisy) and measuring each mechanism's **share of the band**: allocation
   (owner correction), timing (rearrange early vs a day late), channel
   discipline (the focus tax). A share ≈ 0 means the task doesn't teach
   that skill; one mechanism owning everything means a one-bit task.
   Degenerate scenarios are reported as such instead of audited.
5. **The generator** (`sim/generate.py`) closes the loop —
   `python -m sim.generate scenarios/demo.json --tier hard --n 5`:
   seeded, LLM-free remixing of a template's *physics* (efforts, arrival
   times, ticket ids, default owners via the `pileup` knob, belief errors,
   NPC latency seed), with every draw keyed by (template, tier, seed) and
   materialized into the emitted file. Each candidate runs the full gate
   (validate → fairness → tier band) and gets stamped; mis-tuned knobs
   cost attempts, never bad instances. This is how the system grows to
   many scenarios without prompt spaghetti or hand-authored one-offs: the
   template carries the fiction once, the generator mints difficulty-tuned
   instances from it.
6. **Hardness is measured, never asserted**: scenario names are neutral;
   the empirical probe ladder (score degradation across models) is the
   ground truth the fingerprint must be calibrated against, and the
   dashboard colors tasks by band gap (VIBGYOR) so the difficulty axis is
   visible at a glance.

Band width is itself an authored property: under the owned-arrivals
contract it equals **how wrong the default ownership is** — a week whose
defaults are adequate has nothing for a PM to win (and the gate rejects
it); a week where every arrival pancakes onto one volunteer while a
colleague idles has a wide band where allocation is the skill.

## 12. The harness

Two probes through the SAME tool registry the UI uses (`sim/harness.py`);
the agent is never shown the evaluation.

- **scripted** — a deterministic known-good tool sequence: zero agent
  tokens, the floor above baseline, exercises every tool surface.
- **llm** — a Claude model plays the PM. The loop is **event-driven, not
  turn-budgeted**: the agent acts; when it yields (a response with no tool
  calls), the harness rolls sim-time forward interruptibly; a push (chat,
  delivered email, room line) wakes the PM and is handed over as the next
  user message. Sim-time is the constraint — `max_turns` is only a runaway
  guard for an agent that never advances the clock. Agent-side API errors
  end the week gracefully; the run still scores.

Each run writes `meta.json` (probe, model, task = scenario name),
`navigation.json` (how it moved: tool/channel mix, act-vs-yield, contact
per NPC, sim-time coverage), and `scorecard.json`.

`meta.json` also pins a `benchmark_identity`: canonical SHA-256 hashes of the
semantic scenario, physics source, evidence renderer (+ NPC mode), and pinned
rubric form `task_version`; `task_version + resolved model` forms the
benchmark `coordinate`. Aggregation and the dashboard never pool incompatible
task or environment versions. Launch aliases are labels only — the resolved
model ID is the comparison identity. Legacy runs derive `legacy:*` identity
from their pinned scorecard rather than current source.

## 13. Observability

Strict observer/driver separation: the web UI never mutates the sim — one
writer (CLI / tool API / harness), many watchers.

`sim/server.py` serves three modes: own engine, `--watch <run_dir>` (tails
any run's `events.jsonl`, live or finished), and `--hub` (every run in
`runs/`, with `GET /api/runs` and `GET /api/tasks` — runs grouped by task
with band and per-model means). The single-page UI (`web/index.html`) is
hash-routed (`#<tab>&run=<id>&task=<id>` — refresh/back/forward restore
exactly): Feed (chat + board + pending), Timeline (every event + rewind),
Trajectory (per-person swimlanes plus a **PM-trajectory panel**: every
agent action in order), and Benchmark — overview charts where every task is
one VIBGYOR-colored curve through its per-model *mean* points, above a
filterable task table that expands to model means, a mini chart, and run
rows that deep-link straight into that run's trajectory. The validation
loop is three clicks: task → behavior → run → PM trajectory.

## 14. Tests

`.venv/bin/python -m unittest discover -s tests` — deterministic, ~0.5s,
zero LLM calls, on a self-contained fixture scenario (tuning shipped
scenarios never breaks the suite).

- **Example tests** pin every bug class hit during development: replay
  dropping message fields, retroactive assignment, priority-blind queues,
  crash-on-unknown-recipient, unowned arrivals, truth leaks in tool
  returns, unfair arrivals, value minting, the odds identity.
- **Property tests** guard the DES claims under seeded random
  interleavings — causality, monotonicity, liveness, replay equality,
  granularity-irrelevance, and the solo ceiling (score ≤ 1 even for adversarial
  play) — asserted after *every* fuzzed action. Fixed seeds: a failure is a
  reproducible counterexample, never flake. The fuzzer found a real
  causality hole (co-temporal events left undispatched on wake) within its
  first 0.2 seconds of existence.
