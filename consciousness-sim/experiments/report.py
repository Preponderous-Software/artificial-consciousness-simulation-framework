"""Render metrics + manifest + meta into a human-readable markdown report.

Mirrors the structure of the manually-written `~/.consciousness/Echo/RUN_REPORT.md`
files so anyone who's read those will recognise the shape. Uses Python f-strings
so we don't add a Jinja2 dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiments.manifest import ExperimentManifest, SuccessCriterion, evaluate_success_criterion


def _fmt_mood(mood: dict[str, float]) -> str:
    if not mood:
        return "(no mood data)"
    parts = [f"`{k}={v:.2f}`" for k, v in mood.items()]
    return ", ".join(parts)


def _fmt_table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    """Render a simple GitHub-flavored markdown table.

    Cells are stringified here, so callers may pass rows of any element type
    (ranks are ints, criteria marks are strs); the parameter is a covariant
    ``Sequence`` rather than an invariant ``list`` so those rows type-check.
    """
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _criteria_section(
    criteria: list[SuccessCriterion], metrics: dict[str, Any]
) -> str:
    if not criteria:
        return "_None defined in this manifest._"
    rows = []
    all_passed = True
    for c in criteria:
        passed, actual = evaluate_success_criterion(c, metrics)
        all_passed = all_passed and passed
        mark = "✅" if passed else "❌"
        actual_s = f"{actual:.3f}" if actual is not None else "missing"
        rows.append((mark, c.kind, f"{c.op} {c.value}", actual_s))
    table = _fmt_table(rows, ["", "Metric", "Expected", "Actual"])
    verdict = "**All success criteria passed.**" if all_passed else "**Some criteria failed — see ❌ rows.**"
    return f"{verdict}\n\n{table}"


def render_report(
    manifest: ExperimentManifest,
    meta: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    """Build the full markdown report. Returns a single string."""
    ec = metrics.get("event_counts", {})
    voc = metrics.get("vocabulary", {})
    mood = metrics.get("mood", {})
    perc = metrics.get("perception", {})
    refl = metrics.get("reflections", {})
    perf = metrics.get("performance", {})
    intervals = perf.get("cycle_interval_stats", {})

    top10 = voc.get("top_50", [])[:10]
    attractor_top10 = voc.get("attractor_ranks_in_top_10", {})
    attractor_top50 = voc.get("attractor_ranks_in_top_50", {})

    # --- header ---
    lines = [
        f"# Experiment report — {manifest.name}",
        "",
        f"**Description:** {manifest.description or '(none)'}",
        f"**Consciousness name:** `{manifest.consciousness_name}`",
        f"**Branch SHA:** `{meta.get('branch_sha', 'unknown')}`",
        f"**Started:** {meta.get('started_at', 'unknown')}",
        f"**Ended:**   {meta.get('ended_at', 'unknown')}",
        f"**Wall clock:** {meta.get('wall_clock_minutes', '?'):.1f} min" if isinstance(meta.get('wall_clock_minutes'), (int, float)) else "**Wall clock:** unknown",
        f"**Exit reason:** {meta.get('exit_reason', 'unknown')}",
        f"**Tags:** {', '.join(manifest.tags) if manifest.tags else '(none)'}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"Event counts: {ec}",
        f"Final mood: {_fmt_mood(mood.get('final', {}))}",
        f"Mood dimensions non-degenerate (eps=0.05): **{mood.get('dimensions_non_degenerate', 0)} / 4**",
        f"Top-word density: **{voc.get('top_word_density_per_thought', 0):.2f}** per thought "
        f"({top10[0][0] if top10 else '(none)'} × {top10[0][1] if top10 else 0})",
        "",
        "## Success criteria",
        "",
        _criteria_section(manifest.success_criteria, metrics),
        "",
    ]

    # --- mood ---
    # Initial values come from the metrics dict (which sources them from
    # state.identity.initial_mood when persisted, else the default config).
    # Falling back to the hardcoded default ensures the report renders cleanly
    # for older runs that predate the metrics.mood.initial field.
    initial = mood.get("initial") or {
        "curiosity": 0.7, "wonder": 0.6, "melancholy": 0.2, "contentment": 0.5,
    }
    mood_rows = []
    for k, init in initial.items():
        actual = float(mood.get("final", {}).get(k, init))
        delta = actual - init
        mood_rows.append((k, f"{init:.2f}", f"{actual:.3f}", f"{delta:+.3f}"))
    lines += [
        "## Mood",
        "",
        f"Collapse score (sum of squared deltas from initial): **{mood.get('collapse_score', 0):.3f}**",
        "",
        _fmt_table(mood_rows, ["Dimension", "Initial", "Final", "Δ"]),
        "",
    ]

    # --- vocabulary ---
    voc_rows = [(i + 1, w, c) for i, (w, c) in enumerate(top10)]
    lines += [
        "## Vocabulary",
        "",
        "### Top 10 content words",
        "",
        _fmt_table(voc_rows, ["#", "Word", "Count"]),
        "",
        "### Attractor-word ranks (lower = more attractor reassertion)",
        "",
        "_Top-10 ranks (or out-of-top-10):_",
        "",
    ]
    for word, rank in attractor_top10.items():
        rank_top50 = attractor_top50.get(word)
        if rank is not None:
            lines.append(f"- `{word}` → **rank {rank}** in top-10")
        elif rank_top50 is not None:
            lines.append(f"- `{word}` → rank {rank_top50} in top-50 (out of top-10)")
        else:
            lines.append(f"- `{word}` → out of top-50")
    lines.append("")

    # --- perception ---
    n_traces = perc.get("n_traces", 0)
    influence = perc.get("influence_rate", 0.0)
    lines += [
        "## Perception influence",
        "",
        f"Total perceptions traced: **{n_traces}**",
        f"Word-overlap influence rate (fraction with ≥1 new word in next 3 thoughts): **{influence:.2f}**",
        "",
        "_Note: word-overlap heuristic stops detecting influence as the agent matures (see #74)._",
        "",
    ]
    for trace in perc.get("sample_traces", []):
        words = trace.get("new_words_in_next_thoughts", [])
        if words:
            lines.append(f"- **{trace['title']}** → influenced words: `{', '.join(words[:10])}`")
        else:
            lines.append(f"- **{trace['title']}** → no word-level trace")
    lines.append("")

    # --- reflections / shifts ---
    lines += [
        "## Reflections & identity shifts",
        "",
        f"Reflections per thought: **{refl.get('rate_per_thought', 0):.3f}** (config default ~0.15)",
        f"Shifts per reflection: **{refl.get('shifts_per_reflection', 0):.3f}**",
        f"Total amendments in state: **{refl.get('n_amendments_in_state', 0)}**",
        "",
        "_Note: identity_shift events are read from `state.identity.amendments` because they were "
        "not journaled at the time of the reference runs — see #75._",
        "",
    ]

    # --- performance ---
    lines += [
        "## Performance",
        "",
        f"Cycle interval (n={intervals.get('n_intervals', 0)}): "
        f"mean **{intervals.get('mean_s', 0):.1f}s**, "
        f"p50 {intervals.get('p50_s', 0):.1f}s, "
        f"p95 {intervals.get('p95_s', 0):.1f}s, "
        f"p99 {intervals.get('p99_s', 0):.1f}s",
        "",
        "### Cycle-rate trajectory (rolling mean per 30-thought window)",
        "",
    ]
    trajectory = perf.get("cycle_rate_trajectory", [])
    if trajectory:
        lines += [
            _fmt_table(
                [(f"window {i+1}", f"{r:.1f}s") for i, r in enumerate(trajectory)],
                ["Window", "Avg s/thought"],
            ),
            "",
        ]
        if len(trajectory) >= 2:
            drift_pct = (trajectory[-1] - trajectory[0]) / trajectory[0] * 100
            lines.append(f"End-to-start drift: **{drift_pct:+.1f}%** "
                         f"(see #72 for cycle-rate degradation context)")
    else:
        lines.append("_(fewer than one full 30-thought window; trajectory unavailable)_")
    lines.append("")

    return "\n".join(lines)
