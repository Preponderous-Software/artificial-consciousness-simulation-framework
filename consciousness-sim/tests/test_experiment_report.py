"""Tests for experiments/report.py — markdown rendering of a recorded run.

`_fmt_table` stringifies every cell, so callers legitimately pass rows whose
elements are not all `str` (vocabulary rows lead with an int rank). These
tests pin that mixed-element contract, which the parameter annotation now
states explicitly (#11).

`render_report` is the only writer of `report.md`, for both a fresh run
(`experiments/runner.py`) and `experiment.py replay-analysis`. Its tests drive
it with dicts shaped like `metrics.compute_all()`'s return value and pin each
section, the fallbacks for older runs with sparse metrics, and the branch
choices (attractor rank buckets, trace influence, trajectory length).
"""

from __future__ import annotations

from typing import Any

from experiments.manifest import Duration, ExperimentManifest, SuccessCriterion
from experiments.report import _criteria_section, _fmt_mood, _fmt_table, render_report


def _manifest(**overrides: Any) -> ExperimentManifest:
    fields: dict[str, Any] = {
        "name": "report-test",
        "consciousness_name": "Aria",
        "duration": Duration(thoughts=10),
    }
    fields.update(overrides)
    return ExperimentManifest(**fields)


def _meta() -> dict[str, Any]:
    return {
        "branch_sha": "abc1234",
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:12:30+00:00",
        "wall_clock_minutes": 12.5,
        "exit_reason": "duration reached",
    }


def _metrics() -> dict[str, Any]:
    """A populated metrics dict, shaped like `metrics.compute_all()` output."""
    return {
        "event_counts": {"thought": 30, "reflection": 6},
        "vocabulary": {
            "top_50": [[f"word{i}", 100 - i] for i in range(1, 13)],
            "top_word_density_per_thought": 3.3,
            "attractor_ranks_in_top_10": {"threads": 2, "tapestry": None, "cosmic": None},
            "attractor_ranks_in_top_50": {"threads": 2, "tapestry": 14, "cosmic": None},
        },
        "mood": {
            "initial": {"curiosity": 0.7, "wonder": 0.6},
            "final": {"curiosity": 0.85, "wonder": 0.55},
            "dimensions_non_degenerate": 2,
            "collapse_score": 0.025,
        },
        "perception": {
            "n_traces": 2,
            "influence_rate": 0.5,
            "sample_traces": [
                {
                    "title": "Tidal locking",
                    "new_words_in_next_thoughts": [f"w{i}" for i in range(12)],
                },
                {"title": "Basalt", "new_words_in_next_thoughts": []},
            ],
        },
        "reflections": {
            "rate_per_thought": 0.2,
            "shifts_per_reflection": 0.5,
            "n_amendments_in_state": 3,
        },
        "performance": {
            "cycle_interval_stats": {
                "mean_s": 64.0,
                "p50_s": 60.0,
                "p95_s": 90.0,
                "p99_s": 99.0,
                "n_intervals": 29,
            },
            "cycle_rate_trajectory": [50.0, 75.0],
            "trajectory_window_size": 30,
        },
    }


def _render(**metric_overrides: Any) -> str:
    metrics = _metrics()
    metrics.update(metric_overrides)
    return render_report(_manifest(), _meta(), metrics)


def test_fmt_table_renders_header_separator_and_rows() -> None:
    rendered = _fmt_table([("a", "b")], ["Left", "Right"])
    assert rendered.splitlines() == [
        "| Left | Right |",
        "|---|---|",
        "| a | b |",
    ]


def test_fmt_table_stringifies_mixed_element_rows() -> None:
    """Vocabulary rows are `(int, str, int)`; ranks must render as text."""
    rendered = _fmt_table([(1, "memory", 12), (2, "self", 9)], ["#", "Word", "Count"])
    assert "| 1 | memory | 12 |" in rendered
    assert "| 2 | self | 9 |" in rendered


def test_fmt_mood_reports_missing_data() -> None:
    assert _fmt_mood({}) == "(no mood data)"
    assert _fmt_mood({"curiosity": 0.712}) == "`curiosity=0.71`"


def test_criteria_section_without_criteria_says_none_defined() -> None:
    assert _criteria_section([], {}) == "_None defined in this manifest._"


def test_criteria_section_all_passing_shows_success_verdict() -> None:
    criteria = [SuccessCriterion(kind="mood.dimensions_non_degenerate", op=">=", value=2)]
    rendered = _criteria_section(criteria, _metrics())
    assert rendered.startswith("**All success criteria passed.**")
    assert "| ✅ | mood.dimensions_non_degenerate | >= 2.0 | 2.000 |" in rendered


def test_criteria_section_one_failure_flips_the_verdict() -> None:
    criteria = [
        SuccessCriterion(kind="mood.dimensions_non_degenerate", op=">=", value=2),
        SuccessCriterion(kind="mood.collapse_score", op="<", value=0.01),
    ]
    rendered = _criteria_section(criteria, _metrics())
    assert rendered.startswith("**Some criteria failed — see ❌ rows.**")
    assert "| ✅ | mood.dimensions_non_degenerate | >= 2.0 | 2.000 |" in rendered
    assert "| ❌ | mood.collapse_score | < 0.01 | 0.025 |" in rendered


def test_criteria_section_missing_metric_fails_closed() -> None:
    criteria = [SuccessCriterion(kind="mood.not_a_metric", op=">", value=0)]
    rendered = _criteria_section(criteria, _metrics())
    assert rendered.startswith("**Some criteria failed")
    assert "| ❌ | mood.not_a_metric | > 0.0 | missing |" in rendered


def test_render_report_header_carries_manifest_and_meta() -> None:
    manifest = _manifest(description="baseline run", tags=["smoke", "mock"])
    rendered = render_report(manifest, _meta(), _metrics())
    lines = rendered.splitlines()
    assert lines[0] == "# Experiment report — report-test"
    assert "**Description:** baseline run" in lines
    assert "**Consciousness name:** `Aria`" in lines
    assert "**Branch SHA:** `abc1234`" in lines
    assert "**Started:** 2026-01-01T00:00:00+00:00" in lines
    assert "**Ended:**   2026-01-01T00:12:30+00:00" in lines
    assert "**Wall clock:** 12.5 min" in lines
    assert "**Exit reason:** duration reached" in lines
    assert "**Tags:** smoke, mock" in lines


def test_render_report_header_falls_back_when_fields_are_absent() -> None:
    lines = render_report(_manifest(), {}, _metrics()).splitlines()
    assert "**Description:** (none)" in lines
    assert "**Branch SHA:** `unknown`" in lines
    assert "**Started:** unknown" in lines
    assert "**Ended:**   unknown" in lines
    assert "**Wall clock:** unknown" in lines
    assert "**Exit reason:** unknown" in lines
    assert "**Tags:** (none)" in lines


def test_render_report_non_numeric_wall_clock_reads_unknown() -> None:
    meta = {**_meta(), "wall_clock_minutes": "12.5"}
    assert "**Wall clock:** unknown" in render_report(_manifest(), meta, _metrics()).splitlines()


def test_render_report_sections_appear_in_order() -> None:
    headings = [line for line in _render().splitlines() if line.startswith("## ")]
    assert headings == [
        "## Summary",
        "## Success criteria",
        "## Mood",
        "## Vocabulary",
        "## Perception influence",
        "## Reflections & identity shifts",
        "## Performance",
    ]


def test_render_report_tolerates_empty_metrics() -> None:
    """A run with a sparse metrics.json still renders every section."""
    lines = render_report(_manifest(), _meta(), {}).splitlines()
    assert "Event counts: {}" in lines
    assert "Final mood: (no mood data)" in lines
    assert "Mood dimensions non-degenerate (eps=0.05): **0 / 4**" in lines
    assert "Top-word density: **0.00** per thought ((none) × 0)" in lines
    assert "Total perceptions traced: **0**" in lines
    assert "Cycle interval (n=0): mean **0.0s**, p50 0.0s, p95 0.0s, p99 0.0s" in lines
    assert "_(fewer than one full 30-thought window; trajectory unavailable)_" in lines


def test_render_report_summary_names_the_top_word() -> None:
    lines = _render().splitlines()
    assert "Event counts: {'thought': 30, 'reflection': 6}" in lines
    assert "Final mood: `curiosity=0.85`, `wonder=0.55`" in lines
    assert "Mood dimensions non-degenerate (eps=0.05): **2 / 2**" in lines
    assert "Top-word density: **3.30** per thought (word1 × 99)" in lines


def test_render_report_mood_table_shows_signed_deltas() -> None:
    rendered = _render()
    assert "Collapse score (sum of squared deltas from initial): **0.025**" in rendered
    assert "| curiosity | 0.70 | 0.850 | +0.150 |" in rendered
    assert "| wonder | 0.60 | 0.550 | -0.050 |" in rendered


def test_render_report_mood_dimension_missing_from_final_uses_initial() -> None:
    rendered = _render(mood={"initial": {"curiosity": 0.7}, "final": {}})
    assert "| curiosity | 0.70 | 0.700 | +0.000 |" in rendered


def test_render_report_mood_without_initial_uses_default_dimensions() -> None:
    """Runs predating `metrics.mood.initial` fall back to the default baseline."""
    rendered = _render(mood={"final": {"melancholy": 0.25}})
    assert "| curiosity | 0.70 | 0.700 | +0.000 |" in rendered
    assert "| wonder | 0.60 | 0.600 | +0.000 |" in rendered
    assert "| melancholy | 0.20 | 0.250 | +0.050 |" in rendered
    assert "| contentment | 0.50 | 0.500 | +0.000 |" in rendered


def test_render_report_vocabulary_table_stops_at_ten_rows() -> None:
    rendered = _render()
    assert "| 1 | word1 | 99 |" in rendered
    assert "| 10 | word10 | 90 |" in rendered
    assert "word11" not in rendered


def test_render_report_attractor_lines_bucket_by_best_rank() -> None:
    lines = _render().splitlines()
    assert "- `threads` → **rank 2** in top-10" in lines
    assert "- `tapestry` → rank 14 in top-50 (out of top-10)" in lines
    assert "- `cosmic` → out of top-50" in lines


def test_render_report_perception_section_caps_words_at_ten() -> None:
    lines = _render().splitlines()
    assert "Total perceptions traced: **2**" in lines
    assert (
        "Word-overlap influence rate (fraction with ≥1 new word in next 3 thoughts): **0.50**"
        in lines
    )
    assert "- **Tidal locking** → influenced words: `w0, w1, w2, w3, w4, w5, w6, w7, w8, w9`" in lines
    assert "- **Basalt** → no word-level trace" in lines


def test_render_report_reflection_section_formats_rates() -> None:
    lines = _render().splitlines()
    assert "Reflections per thought: **0.200** (config default ~0.15)" in lines
    assert "Shifts per reflection: **0.500**" in lines
    assert "Total amendments in state: **3**" in lines


def test_render_report_performance_shows_intervals_and_trajectory() -> None:
    rendered = _render()
    assert "Cycle interval (n=29): mean **64.0s**, p50 60.0s, p95 90.0s, p99 99.0s" in rendered
    assert "| window 1 | 50.0s |" in rendered
    assert "| window 2 | 75.0s |" in rendered
    assert "End-to-start drift: **+50.0%**" in rendered


def test_render_report_negative_drift_keeps_its_sign() -> None:
    performance = {"cycle_rate_trajectory": [100.0, 75.0]}
    assert "End-to-start drift: **-25.0%**" in _render(performance=performance)


def test_render_report_single_window_has_no_drift_line() -> None:
    rendered = _render(performance={"cycle_rate_trajectory": [50.0]})
    assert "| window 1 | 50.0s |" in rendered
    assert "End-to-start drift" not in rendered


def test_render_report_zero_first_window_reports_drift_unavailable() -> None:
    """Drift has no 0.0 baseline; the rest of the report must still render (#235)."""
    rendered = _render(performance={"cycle_rate_trajectory": [0.0, 75.0]})
    assert "| window 1 | 0.0s |" in rendered
    assert "| window 2 | 75.0s |" in rendered
    assert "End-to-start drift: unavailable (first window averaged 0.0s)" in rendered.splitlines()


def test_render_report_mood_denominator_counts_final_dimensions() -> None:
    """A five-dimension instance must not report `5 / 4` (#235)."""
    final = {name: 0.5 for name in ("a", "b", "c", "d", "e")}
    lines = _render(mood={"final": final, "dimensions_non_degenerate": 5}).splitlines()
    assert "Mood dimensions non-degenerate (eps=0.05): **5 / 5**" in lines


def test_render_report_mood_denominator_falls_back_to_initial() -> None:
    mood = {"initial": {"curiosity": 0.7, "wonder": 0.6, "calm": 0.5}, "final": {}}
    lines = _render(mood=mood).splitlines()
    assert "Mood dimensions non-degenerate (eps=0.05): **0 / 3**" in lines
