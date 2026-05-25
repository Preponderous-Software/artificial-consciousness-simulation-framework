---
name: compare-experiments
description: Compare two recorded experiment runs in this consciousness-simulation project. Use when the user asks to compare runs (e.g. "compare Echo and Wren", "did the mood fix work?"), wants A/B narrative on two run directories, or wants to know if a new run is better/worse than a baseline. Combines `experiment.py compare`'s data layer with read-the-journals qualitative analysis. Pairs with the `run-experiment` skill.
---

# compare-experiments

The narrative layer on top of `scripts/experiment.py compare`. The CLI emits a structured markdown table of deltas; this skill adds qualitative interpretation by reading the journals directly — the kind of analysis the project was doing by hand for Rafael ↔ Sage ↔ Echo ↔ Wren before issue #57's harness.

## When to invoke

The user says something like:
- "Compare `<A>` and `<B>`"
- "Is `<new-run>` better than `<baseline>`?"
- "Did the mood fix actually work? Compare Sage to Echo."
- "Show me what changed between these two runs"
- "Replicate test — Echo vs Wren?"

If the user is **launching a new run** instead of comparing existing ones, use the `run-experiment` skill.

## Operating procedure

### 1. Validate both run dirs

```bash
ls <run-a>/journal.jsonl <run-a>/state.json
ls <run-b>/journal.jsonl <run-b>/state.json
```

A valid input is **any directory with `journal.jsonl` and `state.json`** — full recorded runs from `experiment.py run`, but also `experiments/golden/{Rafael,Sage,Echo,Wren}/` (which intentionally lack `metrics.json`; `compare.py` computes them on the fly).

### 2. Call the CLI for the data layer

```bash
cd consciousness-sim
source .venv/bin/activate
python scripts/experiment.py compare <run-a> <run-b>
```

This produces a markdown report with:
- Headline metrics table (thoughts, top-word density, mood non-degenerate count, reflection rate, perception influence rate, with `Δ (B − A)` column)
- Mood per dimension with deltas
- Cosmic-attractor word ranks with movement column
- 3 sample thoughts per run, evenly spaced

**Read this output carefully** before writing your narrative — most numbers you need are here.

### 3. Read both journals for qualitative differences the metrics miss

Word-overlap-style metrics catch only literal vocabulary overlap. Real differences often live in:
- **Register / tone** (literary cosmic vs grounded introspective vs anxious-clinical)
- **Perception integration depth** (name-checking vs paraphrasing vs structural metaphor)
- **Coherence over the window** (each thought self-contained vs threading across cycles)
- **Identity references** (does the agent refer to itself by name, by role, in third person?)
- **Bug surfaces** (`I the <noun>...` from #70; `Please continue...` codas from #73; meta-amendments like "Here's a rewritten version..." from #76)

Pull 3-5 thoughts from each journal (different positions than the CLI's samples), look for any of the above.

### 4. Write the narrative comparison

Structure: 4-6 paragraphs.

1. **Verdict in one sentence.** "Run B is healthier on mood / similar on vocabulary / worse on attractor reassertion." Be specific about *what changed* and *what didn't*.

2. **The clean signal** — pick the metric with the largest, most interpretable delta. Is it mood? Vocabulary density? Attractor escape? Frame the run as evidence for or against the hypothesis the experiment was meant to test.

3. **What the metrics miss** — your read of the journals' qualitative differences. Examples from past comparisons:
   - "Echo's perceptions paraphrased the source (Théodora's music) while Sage's mostly name-dropped — the word-overlap metric undercounts Echo's influence rate"
   - "Wren's 5 amendments are 4-of-5 LLM meta-text garbage (`Here's a rewritten version of...`) — a clean N=2 of #76, not just a fluke"

4. **Confounds** — if A and B differ in more than the controlled variable, name it. Echo vs Sage isn't a clean mood-fix isolation: it also has perception (#54) and AST-1 (#61). Honest comparisons name their confounds.

5. **Recommendation** — what should the user do with this comparison? Merge a PR? Open a follow-up issue? Run a third replicate? Be concrete.

### 5. (Optional) Surface follow-ups to file

If the comparison reveals a pattern worth a ticket — a new bug, a regression, a research question — name it. The user can then decide whether to file. Don't auto-file without asking.

## Reference comparisons baked into the project

These four pairs are the canonical narratives the harness is meant to make reproducible:

| Pair | What it tests | Conclusion (from manual analysis) |
|---|---|---|
| Rafael ↔ Sage | Does perception (#54) break the closed-loop attractor? | Partially — top-word density 1.42 → 0.97 (−29%), but `threads`/`tapestry` still in top 10 |
| Sage ↔ Echo | Does mood fix (#66) prevent mood collapse? | Yes, dramatically — non-degenerate dims 1 → 3-4 |
| Echo ↔ Wren | Does the Echo result replicate at N=2? | Mood replicates clean; vocabulary attractor result variable (Wren #10 vs Echo #27 for `threads`) |
| Any ↔ `experiments/golden/Echo/` | "Is this run healthier than the current best baseline?" | Use as default reference; Echo is the strongest current benchmark |

You can run any of these on demand:

```bash
python scripts/experiment.py compare experiments/golden/Rafael experiments/golden/Sage
python scripts/experiment.py compare experiments/golden/Sage experiments/golden/Echo
python scripts/experiment.py compare experiments/golden/Echo experiments/golden/Wren
```

## Anti-patterns

- **Don't just narrate the table.** The CLI already prints the table; your job is interpretation. "Top-word density dropped from 1.42 to 0.78" is restating; "the cosmic-attractor basin lost its top-2 grip on the vocabulary" is interpreting.
- **Don't overclaim from one comparison.** N=1 vs N=1 is suggestive, not conclusive. Use the phrase "this comparison suggests" or "consistent with" rather than "proves" / "shows definitively."
- **Don't ignore confounds.** Echo carries 3 changes vs Sage (mood-fix + AST-1 + minor others). Always name the confounds.
- **Don't claim phenomenal states.** Functional language only, per `CLAUDE.md` §Design Principles 2. "Mood drifted to ceiling" not "the agent felt jubilant."
- **Don't compare runs at very different N without flagging it.** Echo (200 thoughts) vs Rafael (83 thoughts) is informative but the metrics that aren't per-thought-normalized will mislead. The CLI uses normalized metrics where possible; you should mention the N gap if it's >2x.
