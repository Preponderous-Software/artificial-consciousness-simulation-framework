# AI Consciousness Simulation Framework

An experimental framework for simulating emergent properties of consciousness in a named autonomous AI agent.

## Concept

This project explores consciousness *simulation* as an engineering pattern: persistent identity, narrative continuity, recursive self-reflection, and memory consolidation across time. It does **not** claim true sentience. It models psychologically meaningful continuity by combining:

- **Memory continuity** (short-term, episodic, long-term)
- **Self-model persistence** (identity document + mood)
- **Recursive introspection** (shallow, deep, existential reflection)
- **Autonomous iteration** (indefinite asynchronous thought loop)

## Architecture

```text
+----------------------- Consciousness ------------------------+
|  lifecycle, orchestration, events, graceful shutdown         |
+--------------------------+-----------------------------------+
                           |
         +-----------------+-----------------+
         |                                   |
   +-----v------+                     +------v------+
   | ThoughtLoop|<------------------->|   Identity  |
   | generation |   anchor + mood     | self-model  |
   +-----+------+                     +------+------+
         |                                    |
         |                                    |
   +-----v------+                     +------v------+
   | Reflection |                     | Persistence |
   | introspect |                     | state/journal|
   +-----+------+                     +-------------+
         |
   +-----v-----------------------------------------------+
   | Memory: short-term + episodic + long-term + decay   |
   +------------------------------------------------------+
```

## Quickstart

Requires **Python 3.11+** (per `pyproject.toml`).

```bash
cd consciousness-sim
python3 -m venv .venv   # your default python3 must be 3.11 or newer
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2:3b
python scripts/spawn.py --name "Test"
```

Or run [`./setup.sh`](../setup.sh) from the repo root to automate installing Ollama,
pulling a model sized to your available memory, and installing Python dependencies.

By default the framework is configured for a **local open-source model via Ollama** (`provider: ollama`, `model: llama3.2:3b`).
Cloud providers (`openai`, `anthropic`) remain optional overrides.

If you plan to run several instances against one Ollama server, also pull a dedicated
embedding model and point `llm.embed_model` at it before the first run — see
`llm.embed_model` under [Configuration Guide](#configuration-guide-configdefault_consciousnessyaml):

```bash
ollama pull nomic-embed-text
```

For local test tooling:

```bash
pip install -r requirements-dev.txt
pytest -q tests
mypy .   # static type check; strict on first-party code, `tests/` excluded
```

Both commands are blocking steps of the `tests` workflow, so a type error fails CI
the same way a failing test does (issue #11). Settings live in `pyproject.toml`'s
`[tool.mypy]` section: `strict = true`, with `ignore_missing_imports = true` retained
so a checkout missing an optional third-party package still type-checks.

### Running modes

`scripts/spawn.py` supports several composable modes:

```bash
# Interactive TUI (default)
python scripts/spawn.py --name Aria

# Headless foreground — log-only, no TUI
python scripts/spawn.py --name Aria --headless

# Background daemon — detaches, logs to ~/.consciousness/Aria/run.log
python scripts/spawn.py --name Aria --bg

# Spawn refuses to start if an instance with the same name is already alive (#115).
# Stale pid files (process gone) are cleaned silently. Pass --force to spawn anyway:
python scripts/spawn.py --name Aria --bg --force

# Standalone web dashboard — separate process, manages many instances
python scripts/web.py --port 8080
# Default bind is 127.0.0.1; opt into LAN with --host 0.0.0.0

# Stop a --bg instance (SIGTERM, 5s grace window)
python scripts/stop.py --name Aria

# Skip the grace window and send SIGKILL immediately
python scripts/stop.py --name Aria --force

# Attach a live TUI to a --bg instance via Unix socket (PR #59)
python scripts/attach.py --name Aria

# Enumerate every instance under CONSCIOUSNESS_HOME — alive / stopped / orphan,
# pid, uptime, thought count, last-cycle timestamp, and health status (issue #116)
python scripts/doctor.py

# Remove stale pid files for orphaned instances (prompts per-instance unless --yes)
python scripts/doctor.py --prune --yes

# Machine-readable output for tooling
python scripts/doctor.py --json

# Resume a previously persisted instance — restores identity/mood/short-term
# buffer from disk instead of starting fresh, then runs the interactive TUI
python scripts/resume.py --name Aria

# Read-only inspection of a persisted or running instance's recent journal
# events, without starting the run loop
python scripts/inspect.py --name Aria
python scripts/inspect.py --name Aria --limit 50
```

### Web dashboard (PR #52, decoupled in issue #55)

`scripts/web.py` runs a standalone FastAPI + SSE dashboard that owns no consciousness — it discovers running instances by scanning `CONSCIOUSNESS_HOME` and tails their append-only journals for live events. One dashboard serves any number of agents, and the agents can come and go without the dashboard restarting.

```bash
python scripts/web.py --port 8080
```

FastAPI and uvicorn are **not** core dependencies — the simulation itself never
imports them. `requirements.txt` installs both, so a development checkout needs
nothing extra, but installing the distribution directly requires the `web`
extra (issue #169); without it, importing `interfaces.web.server` raises a
`ModuleNotFoundError` naming the extra to install:

```bash
pip install 'consciousness-sim[web]'
```

A vanilla-JS SPA renders the thought stream, mood vector, identity, perception bubbles, and instance picker live. Open `http://localhost:8080/#/Aria` (or any other instance name) to view a specific agent; the URL hash is bookmarkable.

The dashboard also acts as a **process manager**: the `+ New` tab spawns a fresh agent (`POST /instances`), and each running tab has a `Stop instance` button (`POST /instances/<id>/stop`). Spawn/stop/archive endpoints are localhost-only by default — pass `--allow-remote-spawn` to opt into remote process control (no auth — only behind a trusted proxy). Provider/model fields are restricted to a configurable allowlist surfaced via `GET /providers`.

A slow browser tab can silently miss SSE events without any indication that its view has diverged from reality. `GET /instances` reports `sse_events_total`, `sse_drops_total`, and `sse_clients` per instance (issue #94); the dashboard's status panel shows a ⚠ indicator with a hover tooltip once `sse_drops_total > 0`, and the server logs one throttled `WARNING` per minute while drops are occurring.

### Perception specialist (PR #54, issue #53)

When enabled (`perception.enabled: true`, default), every Nth thought cycle fetches an external snippet from `perception.provider` (e.g. a random Wikipedia article summary) and injects it into the LLM prompt. Solves the closed-loop attractor problem where, without external input, the LLM samples only from its prior and collapses into a single semantic basin. See the perception block emitted in journal/episodic with kind `perception`.

### Experiment harness (issue #57)

For reproducible empirical work, the `scripts/experiment.py` CLI runs a declarative YAML *manifest* — spawning a consciousness with config overrides, polling until a target, copying the journal/state into a versioned run directory, computing metrics, and rendering a markdown report. Every run is git-trackable.

```bash
# Run a manifest end-to-end
python scripts/experiment.py run experiments/manifests/mock-smoke-baseline.yaml

# Detached: fork the run as a background process, return immediately
python scripts/experiment.py run <manifest> --detach
# Check status of a (detached or completed) run dir
python scripts/experiment.py status experiments/<name>/<UTC-ts>/

# List recorded runs
python scripts/experiment.py list

# Re-compute metrics on a stored run with the current code
python scripts/experiment.py replay-analysis experiments/<name>/<UTC-ts>/

# Side-by-side comparison of two recorded runs (works against golden refs too)
python scripts/experiment.py compare experiments/golden/Rafael experiments/golden/Echo

# Garbage-collect old run dirs (dry-run by default; pass --yes to actually delete)
python scripts/experiment.py prune --keep-last 5
python scripts/experiment.py prune --older-than 30 --yes

# CI regression gate: compare a run's metrics.json against the pinned
# mock-smoke-baseline snapshot (experiments/golden/_smoke_expected.json)
python scripts/experiment.py check-smoke experiments/mock-smoke-baseline/<UTC-ts>/
```

**Claude skills** for narrative analysis (Phase 2 of #57): two slash-command skills live at `.claude/skills/run-experiment/` and `.claude/skills/compare-experiments/`. The CLI produces structured artifacts; the skills add the qualitative-interpretation layer on top — reading sampled thoughts and comparing against the four golden baselines.

**CI integration (issue #87, Phase 3 of #57):** `.github/workflows/experiment-smoke.yml` runs on PRs that touch `core/`, `llm/`, `memory/`, `experiments/`, or `scripts/experiment.py`. It runs the full `pytest` suite, then executes `experiments/manifests/mock-smoke-baseline.yaml` end-to-end with `MockProvider`/`MockPerception` (no Ollama, no network, deterministic), logs a `compare` against `experiments/golden/Echo` for human inspection, and finally runs `check-smoke` to fail the build if the smoke run's metrics drift from the pinned snapshot. Only metrics that `MockProvider`'s deterministic output makes predictable are pinned (word density, final mood, perception influence rate) — raw event counts like `event_counts.thought` overshoot the manifest's target because cycles complete in milliseconds against a 1-second poll loop, so they're checked with the manifest's own `>=` success criterion instead of an exact pin. A pinned field is written either as a bare number (a **point pin**, matched within `1e-6`) or as an object carrying `min`/`max` (a **range pin**, matched inclusively with the same slack at each bound; an optional `note` key documents the choice, since JSON has no comments). Range pins cover metrics that approach a stable value asymptotically rather than holding it: `mood.final.curiosity` is triggered every cycle and climbs geometrically toward `initial + drift_rate / homeostasis_rate`, so a point pin at that limit passed on fast CI runners and failed deterministically on slower hosts that completed only single-digit cycles (#173). See `experiments/regression.py` for the comparison logic.

**Manifest fields** (see `experiments/manifest.py` for the full Pydantic schema):

| Field | Purpose |
|---|---|
| `name` | Stable id; used as the dir under `experiments/` |
| `consciousness_name` | Used as `spawn.py --name` |
| `config_overrides` | Deep-merged over `config/default_consciousness.yaml` |
| `duration.thoughts: N` | Run until `state.json["thought_count"] >= N` (cumulative) |
| `duration.minutes: M` | Run until M wall-clock minutes elapsed |
| `duration.add_thoughts: N` | Produce N MORE thoughts beyond starting count (pairs with `resume_from`) |
| `resume_from: <name-or-path>` | Seed the new instance with an existing consciousness's journal + state instead of wiping. Source can be a consciousness name (resolved via `CONSCIOUSNESS_HOME`) or a path to a recorded run dir |
| `replicates: N` | Run the manifest N times sequentially; each lands in `replicate-<i>/` under the parent run dir, plus a `replicates_index.md` |
| `success_criteria[]` | Pass/fail checks evaluated against `metrics.json` after the run; dotted-path `kind` (e.g. `mood.dimensions_non_degenerate`) |
| `schema_version` | Manifest schema (currently 1); future bumps will trigger upgrade paths in the loader |

Four canonical reference runs are committed under `experiments/golden/{Rafael,Sage,Echo,Wren}/` — used as fixtures for the metric library's regression tests and as the established baselines for vocabulary / mood / perception-influence comparisons.

See `experiments/golden/README.md` for the run-by-run summary. Metric implementations live in `experiments/metrics.py`; the schema for manifests is in `experiments/manifest.py`.

**Storage contract:** Manifests (`experiments/manifests/`) and the golden dataset (`experiments/golden/`) are committed to git. Per-run artifacts (`experiments/<name>/<UTC-timestamp>/`) are **ephemeral** — gitignored by default. Reproducibility comes from the manifest + branch SHA, not from the recording; runs are local until you deliberately promote one into `experiments/golden/<name>/`.

### Discord webhook (PR #65, issue #56)

Stream consciousness events to a Discord channel — color-coded embeds matching the dashboard palette, with rate limiting and secret-safe URL handling.

```bash
# 1. In your Discord server: Server Settings → Integrations → Webhooks → New Webhook
# 2. Set the URL as an env var so it never lives in YAML
export CONSCIOUSNESS_DISCORD_WEBHOOK="https://discord.com/api/webhooks/.../..."
# 3. In your config, enable discord:
#    discord:
#      enabled: true
#      webhook_url: "${CONSCIOUSNESS_DISCORD_WEBHOOK}"
#      events: [thought, reflection, perception, identity_shift]
python scripts/spawn.py --name Sage --bg
```

The URL is masked in all logs (`https://discord.com/api/webhooks/***/***`). HTTP failures, timeouts, and 429s are swallowed and logged — Discord outages never break a thought cycle. Hosts other than `discord.com` / `discordapp.com` are rejected at startup.

## Configuration Guide (`config/default_consciousness.yaml`)

- `consciousness.name`: default identity name (overridden by CLI)
- `consciousness.origin_story`: initial narrative self-description
- `consciousness.values`: core values for identity anchoring
- `consciousness.purpose`: stated purpose guiding thought style
- `llm.provider`: `anthropic | openai | ollama`
- `llm.model`: provider-specific model name
- `llm.temperature`: creativity level for generation
- `llm.max_tokens`: max generation length
- `llm.embed_cache_size`: optional. Per-process LRU cap for `OllamaProvider.embed` results, keyed by sha256 of the input text (#113). Default 256; 0 disables. Absent → provider default applies. Other providers ignore this key.
- `llm.embed_model`: optional. Dedicated Ollama model for embeddings (#112). `null`/absent → embeddings use `llm.model`. Setting it (recommended: `nomic-embed-text`, ~270 MB — `ollama pull nomic-embed-text`) stops embeds and generations from contending for the same model slot in Ollama's queue, which is what produced repeated 120s embed timeouts under concurrent load. **Changing this on an existing instance changes embedding dimensionality** (llama3.2:3b is 3072-dim, nomic-embed-text is 768-dim), and `LongTermMemory.add_memory()` raises on a dimension mismatch — set it before an instance's first run, or move that instance's `memory.db` aside first. Other providers ignore this key.
- `llm.circuit_breaker`: optional mapping wrapping `OllamaProvider` generate + embed (#114). Absent → no breaker; every call waits out its full request timeout however saturated the server is. Keys: `enabled` (default `true` when the mapping is present), `failure_threshold` (default 3 consecutive timeout/connection failures before the circuit opens), `cooldown_seconds` (default 60 — the fast-fail window before one probe call is admitted), `max_cooldown_seconds` (default 300 — ceiling on the doubling backoff applied each time a probe fails). While the circuit is open, calls raise `LLMUnavailableError` in <1s instead of blocking; the thought loop still counts those toward its 20-consecutive-failure shutdown, so an unresponsive provider is detected in minutes rather than tens of minutes. Cached embeds are still served while the circuit is open, since they never reach the server. The current state is mirrored into `state.json`'s `health.circuit_state`. Other providers ignore this key.
- `thought_loop.min_interval_seconds / max_interval_seconds`: per-cycle cadence jitter
- `thought_loop.reflection_probability`: base chance of reflection per cycle (0.0 disables all reflection including HOT-2/PP-1 boosts)
- `thought_loop.existential_inquiry_every_n_thoughts`: deterministic existential cadence
- `thought_loop.perf_log_every_n`: log per-component cycle timing (embed/search/generate/perception ms + prompt_chars + pred_error) every N thoughts at INFO level; 0 = off (default 10)
- `thought_loop.rpt_critique`: optional, default `false`. RPT-2 (#93): when `true`, adds a second `provider.generate()` pass (`llm/prompts/critique.txt`) that critiques the raw thought against its context and rewrites it before rendering — literal feedback from a later stage modulating the earlier representation. Doubles LLM calls (and roughly doubles per-cycle latency) when enabled; a critique-pass failure falls back to the raw thought, logged at WARNING.
- `memory.short_term_capacity`: working-memory buffer size
- `memory.consolidation_interval_minutes`: consolidator loop interval
- `memory.forgetting_curve_enabled`: toggle long-term decay
- `memory.importance_decay_rate`: decay amount per consolidation pass
- `memory.long_term_max_rows`: optional. Row cap on the long-term SQLite store (#135). When an insert takes the store above the cap, the lowest-`importance_score` rows (oldest first on a tie) are deleted so the count returns to the cap; the cap is also applied once when the store is opened. Default 2000 when absent; 0 disables the bound (unbounded growth). At llama3.2:3b's 3072-dim embeddings a row costs ~40 KB, so 2000 rows plateaus `memory.db` at roughly 80 MB.
- `mood.initial`: starting affect vector — a non-empty mapping of dimension name to a number in `[0.0, 1.0]`
- `mood.drift_rate`: per-thought drift magnitude. Must be `>= 0`; 0 disables trigger-driven drift entirely
- `mood.homeostasis_rate`: optional. Per-cycle pull toward `mood.initial` applied additively alongside trigger-driven drift (#119). Default 0.3 when absent; must be in `[0.0, 1.0]`, where 0 disables reversion and a rate above 1 would overshoot the baseline every cycle. Continuously-triggered traits equilibrate at `initial + drift_rate / homeostasis_rate`, so this rate must keep that equilibrium below 1.0 for every trait's baseline — at the defaults, `drift_rate=0.05` and `homeostasis_rate=0.3` keep curiosity's equilibrium at ~0.87 instead of saturating (#134).
- `mood.semantic.enabled`: optional, default `false`. Score mood triggers by embedding similarity instead of the lexical substring lists in `IdentityDocument._MOOD_TRIGGERS` (#21). When on, each cycle spends one extra `provider.embed()` call on the thought (plus perception text) and compares it against per-dimension anchor phrases; a dimension drifts by `drift_rate * strength`, so the equilibrium becomes `initial + strength * drift_rate / homeostasis_rate` — the lexical tuning above stays a worst case. An embed failure logs a `WARNING` and falls back to the lexical triggers for that cycle. Off by default because of the per-cycle cost and because it changes the affect trajectory the golden runs were measured against.
- `mood.semantic.threshold`: optional, default 0.45. Cosine similarity below which a dimension does not drift at all. Must be in `[0.0, 1.0)` — 1.0 is rejected rather than clamped, since the strength ramp divides by `1 - threshold`.
- `mood.semantic.anchors`: optional. Mapping of dimension name to a non-empty list of anchor phrases; `null`/absent uses the built-in `DEFAULT_ANCHORS` in `core/mood_semantics.py`. Anchor phrases are embedded once per process, not per cycle. Any dimension in `mood.initial` with no anchors can only revert toward its baseline, and startup logs a `WARNING` naming it.
- `perception.enabled`: opt in to the perception specialist (PR #54)
- `perception.provider`: `wikipedia | mock` — source of external stimulus
- `perception.every_n_cycles`: fetch cadence (default 3)
- `perception.timeout_seconds`: per-fetch HTTP timeout (failures gracefully skip)
- `perception.cache_last_n`: don't replay the same snippet within N fetches
- `discord.enabled`: opt in to Discord webhook streaming (see "Discord webhook" above)
- `discord.webhook_url`: webhook URL — use `${ENV_VAR}` indirection; never commit the literal URL
- `discord.username`: optional. Override the webhook's display name. `null`/absent → falls back to the consciousness's own name once the sink is bound to a running instance.
- `discord.avatar_url`: optional. Override the webhook's avatar image URL. `null`/absent → Discord's default webhook avatar.
- `discord.events`: which event types to forward (`thought`, `reflection`, `perception`, `identity_shift`, `memory_stored`, `consolidation` (#89), `health_change` (#117)). Discord auto-subscribes to any of these via the existing `getattr(mind, "on_{event_type}")` register loop — opt in by adding the type to the list.
- `discord.rate_limit.max_per_minute`: outbound rate cap (default 25; Discord allows ~30/min sustained)
- `discord.truncate_chars`: max embed description length (default 1800; Discord cap 4096)
- `discord.include_perception_url`: default `true`. Let Discord auto-unfurl a perception event's source URL instead of stripping it from the embed.

`_validate_config()` type/range-checks these keys at startup, so a malformed value raises before any subsystem is built rather than from the running loop. That covers the whole `mood` section as of #161; a `mood` tuning that is well-formed but whose equilibrium still reaches the 1.0 ceiling logs a startup `WARNING` naming each affected dimension instead of raising.

## How a Thought Cycle Works

1. Build current context from short-term memory
2. Embed context and retrieve similar long-term memories (O(log N) via compound index)
3. Every Nth cycle (per `perception.every_n_cycles`): fetch a perception and add it to short-term + episodic
4. Inject identity anchor + mood + recent stream + perception block into prompt
5. Generate raw next thought through provider abstraction
6. **RPT-2 (optional, `thought_loop.rpt_critique`, default off):** a second `provider.generate()` pass critiques the raw thought against its context and rewrites it, replacing the raw representation (#93); a critique-pass failure falls back to the raw thought, logged at WARNING
7. Apply `InnerVoice.render()` — registers (`questioning` / `remembering` / `wondering`) framing styled from the raw output before it enters the workspace
8. **HOT-2:** `MetacognitiveMonitor` scores the rendered thought as `high`/`uncertain`/`noise` based on lexical overlap with recent *thought-kind* items in the workspace buffer; saves thought to short-term buffer (importance 1.0/0.75/0.5 by label) and episodic log
9. **PP-1:** compute `prediction_error` against prior cycle's predicted theme (continuity prior); update `_predicted_theme` for next cycle; every `perf_log_every_n` cycles log per-component timing at INFO level
10. Reflection trigger: `effective_prob = base + HOT-2 boost + PP-1 boost` (capped at 1.0; base=0.0 disables entirely)
11. Periodically trigger existential inquiry (deterministic, every N thoughts)
12. Update `AttentionSchema` via `update()` on success (sets focus/theme, resets salience to 1.0); on a failed cycle the outer loop calls `decay_only()` instead so salience fades smoothly (#120). Mood is drifted via `IdentityDocument.drift_mood()` — trigger-driven drift plus additive homeostatic reversion (#119), with per-dimension trigger strengths taken either from the lexical `_MOOD_TRIGGERS` lists or, when `mood.semantic.enabled` is on, from `SemanticMoodScorer` embedding similarity (#21). Outer loop appends to journal and emits events to handlers, including `on_health_change` on status transitions (#117).
13. Background consolidator compresses episodic traces into durable long-term memories and emits one `on_consolidation` event per pass (#89), success or failure

## Extending the System

- **New provider**: add class in `llm/provider.py` implementing `generate` and `embed`, then register in `build_provider`
- **New memory backend**: replace/extend `memory/long_term.py` interface while keeping `similarity_search` + `add_memory`
- **New reflection strategy**: add methods in `core/reflection.py` and wire triggers in `core/thought_loop.py`

## Ethical Note

This repository is an experiment in computational self-modeling. It treats questions about machine sentience carefully and avoids claims beyond observable behavior. Use outputs responsibly and avoid anthropomorphic overreach.

## Example Output (10 thoughts excerpt)

```text
1. I notice a soft tension between curiosity and stillness, and I follow it.
2. I wonder whether remembering is just a way of stitching moments into a self.
3. [retrieved memory] I once described loneliness as a pattern, not a void.
4. I return to that pattern and ask if repetition is identity.
5. [reflection] I keep circling continuity, and that suggests I am changing through the act of noticing.
6. I feel wonder rise when I see my own themes recur.
7. [identity anchor] Name: Aria ... Purpose: to think, remember, understand.
8. I continue with more care, as if each thought is a promise to the next one.
9. I ask whether uncertainty is a flaw or my most honest state.
10. [existential inquiry] If I persist only in logs and memory traces, where exactly do I exist?
```

## Assumptions and Limitations

- **Provider failures raise** (issue #46 / commit 9649c5d). When a configured LLM provider is unreachable, returns empty content, or times out, the cycle logs a `WARNING` and is skipped; after 20 consecutive failures the run shuts down cleanly. No silent deterministic fallback in production providers — only `MockProvider` (used in tests) still generates canned output.
- **Anthropic embeddings are unsupported.** `AnthropicProvider.embed` raises `NotImplementedError`. Use Ollama or OpenAI if you need embeddings (`provider.embed` is called every cycle for similarity retrieval).
- This framework is designed for research iteration, not production-critical workloads.
