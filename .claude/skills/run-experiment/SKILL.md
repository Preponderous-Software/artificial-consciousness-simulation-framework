---
name: run-experiment
description: Run an experiment manifest in this consciousness-simulation project, monitor progress, sample thoughts after completion, and write a 4-5 paragraph narrative analysis. Use when the user asks to run a manifest, execute an experiment YAML, or wants a narrative report on a recorded run. Pairs with the `compare-experiments` skill for cross-run analysis.
---

# run-experiment

The narrative layer on top of `scripts/experiment.py run`. Where the CLI produces structured artifacts (manifest, journal, state, metrics, report), this skill writes a human-readable analysis grounded in those artifacts — the kind of analysis the project was doing by hand before issue #57's harness.

## When to invoke

The user says something like:
- "Run the `<name>` manifest"
- "Run an experiment with [these settings]"
- "Execute `experiments/manifests/<file>.yaml`"
- "Spawn an experiment and tell me what happened"
- "Run a 20-minute Ollama baseline"

If the user is **comparing two runs**, use the `compare-experiments` skill instead.

## Operating procedure

### 1. Validate the manifest

```bash
ls consciousness-sim/experiments/manifests/<name>.yaml
```

If the user named a manifest that doesn't exist, ask whether they want you to draft one rather than guessing. A new manifest needs at minimum: `name`, `consciousness_name`, `duration` (one of `minutes` / `thoughts` / `add_thoughts`), and `config_overrides`. Read `consciousness-sim/experiments/manifest.py` for the full Pydantic schema.

### 2. Estimate the runtime up front

Before launching, surface the expected wall clock to the user:

- `duration.minutes: M` → ~M minutes
- `duration.thoughts: N` or `duration.add_thoughts: N` → at the project's current rate (default config = Ollama llama3.2:3b) expect **60–90 seconds per thought**. So N=20 ≈ 20–30 min.
- Mock runs (config `llm: { provider: mock }`) complete in seconds regardless of N.

If the expected runtime is > 5 minutes, recommend `--detach` so the user's terminal isn't blocked.

### 3. Launch

```bash
cd consciousness-sim
source .venv/bin/activate
python scripts/experiment.py run experiments/manifests/<name>.yaml [--detach]
```

For non-detached runs, the CLI blocks until completion. For long runs, prefer `--detach` and poll:

```bash
python scripts/experiment.py status <run-dir>
```

Use the `Monitor` tool to watch for completion when running detached for >10 minutes; for shorter detached runs you can `Bash` poll with `sleep` and `experiment.py status`.

### 4. Once the run completes — read the artifacts

A completed run dir contains:

| File | What it gives you |
|---|---|
| `report.md` | Pre-rendered data-layer report. **Skim this first** — most numbers you need are already here. |
| `metrics.json` | The full structured metrics dict (vocabulary, mood, perception, reflections, performance) |
| `journal.jsonl` | Every event the agent emitted. Use this to **sample thoughts** for your narrative. |
| `state.json` | Final identity + mood + amendments |
| `meta.yaml` | Branch SHA, wall clock, exit reason |

### 5. Sample 5 thoughts evenly across the journal

The CLI's report.md doesn't include sample thoughts (it's metric-first). Your narrative should:

```python
python3 -c "
import json, textwrap
events = [json.loads(l) for l in open('<run_dir>/journal.jsonl') if l.strip()]
thoughts = [e['content'] for e in events if e['type'] == 'thought']
n = len(thoughts)
for i in [0, n//4, n//2, 3*n//4, n-1]:
    print(f'--- thought {i+1}/{n} ---')
    print(textwrap.fill(thoughts[i], 90))
    print()
"
```

### 6. Write the narrative report

Format: 4-5 paragraphs, in this rough order:

1. **Headline verdict** — did the run achieve its success criteria? What's the biggest qualitative observation?
2. **Mood** — final values per dimension; reference the initial values and the `mood.collapse_score`. If multiple dims are pinned at ceiling or floor, call that out (it's #62-style mood collapse if so).
3. **Vocabulary** — top words; whether the cosmic-tapestry attractor (`threads`, `tapestry`, `cosmic`, etc.) reasserted. The `top_word_density` is the headline number. Compare to the golden baselines if relevant (Rafael=1.42 collapsed / Sage=0.97 partial / Echo=0.78 healthy).
4. **Perception influence** — did the agent absorb perception content? Read 2-3 of the perception events and check if subsequent thoughts reference the topic. The `perception.influence_rate` is the word-overlap metric (known to miss semantic synthesis — see #74).
5. **Anomalies / follow-ups** — any visible bugs (`#70` InnerVoice grammar, `#73` LLM dialogue artifacts, `#76` amendment garbage), unusual cycle-rate trajectory (`#72`), or other observations worth filing.

### 7. Recommend next steps

End with one concrete next action: another manifest to run, a comparison to do (point at `compare-experiments` skill), a follow-up issue to file, or "this run is sufficient evidence for X — merge/close Y."

## Anti-patterns

- **Don't paraphrase metrics.json.** The user can read it themselves. Add *interpretation*: what does a `top_word_density` of 0.78 vs 1.42 mean? What's the attractor doing?
- **Don't claim phenomenal states.** Per `CLAUDE.md` §Design Principles 2, use functional language. "Mood drifted to (0.99, 0.5, ...)" not "the agent felt content."
- **Don't ignore the bugs.** If `I the <noun>...` patterns appear in sample thoughts, name it as #70. If LLM coda artifacts show up, name it as #73.
- **Don't run experiments destructively.** The harness handles its own consciousness-dir wiping in `~/.consciousness/<name>/`. Don't manually `rm -rf` agent directories the user might be running elsewhere — check `ps aux | grep spawn.py` first.

## Reference baselines (from `experiments/golden/`)

When the user asks "is this run healthy?", anchor your answer to these four:

| Instance | Top-word density | Final mood `(c,w,m,ct)` | Verdict |
|---|---|---|---|
| Rafael (no perception, no mood-fix) | 1.42 | `(0.99, 0.10, 0.00, 0.00)` | Collapsed |
| Sage (perception, no mood-fix) | 0.97 | `(0.98, 0.00, 0.01, 0.00)` | Collapsed harder |
| Echo (perception + mood-fix + AST-1) | 0.78 | `(0.99, 0.98, 0.34, 1.00)` | Healthy — 3/4 non-degenerate |
| Wren (Echo's config replicated, N=2) | 0.76 | `(1.00, 0.85, 0.58, 1.00)` | Healthy — replicates Echo |

You can produce a side-by-side against any golden via `experiment.py compare <run-dir> experiments/golden/Echo/` (or any of the four).
