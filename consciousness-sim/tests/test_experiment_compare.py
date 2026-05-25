"""Tests for experiments/compare.py — diff math + markdown rendering.

Pinned against the four golden journals so any future metric change that
silently breaks the comparison gets caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.compare import (
    compare_runs,
    compute_diff,
    load_run,
    render_comparison,
    sample_thoughts,
)


GOLDEN = Path(__file__).resolve().parents[1] / "experiments" / "golden"


# ---------------------------------------------------------------------------
# load_run — handles full run dirs AND golden (journal+state only) dirs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_load_run_handles_golden_dir_without_metrics_json() -> None:
    """Golden dirs ship only journal.jsonl + state.json; compare should
    compute metrics on the fly rather than failing."""
    run = load_run(GOLDEN / "Echo")
    assert run.label == "Echo"
    assert run.thoughts >= 195            # Echo had 200; conservative floor
    assert run.metrics.get("event_counts", {}).get("thought", 0) >= 195


def test_load_run_raises_on_directory_without_journal(tmp_path: Path) -> None:
    empty = tmp_path / "empty-run"
    empty.mkdir()
    with pytest.raises(ValueError, match="no metrics.json"):
        load_run(empty)


def test_load_run_raises_on_nonexistent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "definitely-not-there")


# ---------------------------------------------------------------------------
# compute_diff — the headline math against the golden Rafael / Echo pair
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_diff_rafael_echo_shows_top_word_density_drop() -> None:
    """Echo's top-word density (~0.78) should be lower than Rafael's (~1.42)."""
    rafael = load_run(GOLDEN / "Rafael")
    echo = load_run(GOLDEN / "Echo")
    diff = compute_diff(rafael, echo)
    assert diff["top_word_density"]["a"] > 1.0   # Rafael ≈ 1.42
    assert diff["top_word_density"]["b"] < 1.0   # Echo ≈ 0.78
    assert diff["top_word_density"]["delta"] < -0.3   # at least a notable drop


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_diff_rafael_echo_shows_threads_escape() -> None:
    """`threads` was rank 2 in Rafael, escaped Echo's top-10."""
    rafael = load_run(GOLDEN / "Rafael")
    echo = load_run(GOLDEN / "Echo")
    diff = compute_diff(rafael, echo)
    threads = diff["attractor_ranks"]["threads"]
    assert threads["a"] == 2
    assert threads["b"] is None   # out of top-10


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_diff_includes_mood_per_dimension() -> None:
    rafael = load_run(GOLDEN / "Rafael")
    echo = load_run(GOLDEN / "Echo")
    diff = compute_diff(rafael, echo)
    assert "wonder" in diff["mood"]
    # Echo's wonder is ~0.98; Rafael's is ~0.10
    assert diff["mood"]["wonder"]["delta"] > 0.5


# ---------------------------------------------------------------------------
# sample_thoughts — evenly spaced
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_sample_thoughts_returns_k_evenly_spaced() -> None:
    echo = load_run(GOLDEN / "Echo")
    samples = sample_thoughts(echo, k=3)
    assert len(samples) == 3
    assert all(isinstance(s, str) and len(s) > 0 for s in samples)


def test_sample_thoughts_handles_short_journal(tmp_path: Path) -> None:
    """When the journal has fewer thoughts than k, return them all."""
    rd = tmp_path / "tiny"
    rd.mkdir()
    (rd / "journal.jsonl").write_text(
        '{"timestamp": "2026-01-01T00:00:00+00:00", "type": "thought", "content": "one"}\n'
        '{"timestamp": "2026-01-01T00:01:00+00:00", "type": "thought", "content": "two"}\n'
    )
    (rd / "state.json").write_text('{"identity": {"mood": {}}, "thought_count": 2}')
    (rd / "metrics.json").write_text('{"event_counts": {"thought": 2}}')
    run = load_run(rd)
    samples = sample_thoughts(run, k=5)
    assert samples == ["one", "two"]


# ---------------------------------------------------------------------------
# render_comparison — produces the expected markdown sections
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_render_includes_all_expected_sections() -> None:
    md = compare_runs(GOLDEN / "Rafael", GOLDEN / "Echo")
    for section in (
        "# Comparison",
        "## Headline metrics",
        "## Mood per dimension",
        "## Cosmic-attractor word ranks",
        "## Sample thoughts",
    ):
        assert section in md, f"missing section: {section}"
    # Headline table has the four required rows
    for label in ("Thoughts", "Top-word density", "non-degenerate", "influence rate"):
        assert label in md, f"missing headline row: {label}"


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_render_signed_deltas_show_explicit_sign() -> None:
    md = compare_runs(GOLDEN / "Rafael", GOLDEN / "Echo")
    # Some delta should be negative (density drop) and rendered with `-`
    assert "-0." in md or "-1." in md, "expected at least one negative delta to surface"


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_render_attractor_table_shows_movement_column() -> None:
    md = compare_runs(GOLDEN / "Rafael", GOLDEN / "Echo")
    assert "escaped from" in md or "both out" in md, \
        "attractor movement narrative didn't surface in the rendered table"


# ---------------------------------------------------------------------------
# Self-comparison sanity (run against itself → all deltas should be 0)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_render_includes_top_word_table() -> None:
    """Regression for PR #85 review: the docstring promised a top-word table;
    the renderer must actually emit one."""
    md = compare_runs(GOLDEN / "Rafael", GOLDEN / "Echo")
    assert "## Top 10 content words (side-by-side)" in md
    # Header row should contain both labels
    assert "Rafael" in md and "Echo" in md
    # At least one word should render with the table markup
    assert "| 1 |" in md


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_render_omits_criteria_section_when_no_manifest_criteria(tmp_path: Path) -> None:
    """Golden runs don't ship manifest.yaml with success_criteria; renderer
    must gracefully omit the section rather than fail or render empty tables."""
    md = compare_runs(GOLDEN / "Rafael", GOLDEN / "Echo")
    # No criteria → no section header
    assert "## Success criteria status" not in md


def test_render_includes_criteria_section_when_manifest_has_them(tmp_path: Path) -> None:
    """When at least one run has success_criteria in its manifest.yaml,
    the criteria section appears with pass/fail marks."""
    # Build two minimal "runs" with metrics + manifest with criteria
    import json
    for name, dens in [("a", 1.0), ("b", 0.5)]:
        d = tmp_path / name
        d.mkdir()
        (d / "journal.jsonl").write_text(
            '{"timestamp": "2026-01-01T00:00:00+00:00", "type": "thought", "content": "x"}\n'
        )
        (d / "state.json").write_text('{"identity": {"mood": {}}, "thought_count": 1}')
        (d / "metrics.json").write_text(json.dumps({
            "event_counts": {"thought": 1},
            "vocabulary": {"top_word_density_per_thought": dens, "top_50": [], "attractor_ranks_in_top_10": {}, "attractor_ranks_in_top_50": {}},
            "mood": {"final": {}, "dimensions_non_degenerate": 0, "collapse_score": 0.0, "initial": {}},
            "perception": {"n_traces": 0, "influence_rate": 0.0, "sample_traces": []},
            "reflections": {"rate_per_thought": 0.0, "shifts_per_reflection": 0.0, "n_amendments_in_state": 0},
            "performance": {"cycle_interval_stats": {}, "cycle_rate_trajectory": [], "trajectory_window_size": 30},
        }))
        # Manifest with one passing and one failing criterion
        (d / "manifest.yaml").write_text(
            f"name: {name}\n"
            "consciousness_name: X\n"
            "duration: {thoughts: 1}\n"
            "success_criteria:\n"
            "  - kind: vocabulary.top_word_density_per_thought\n"
            "    op: \">\"\n"
            "    value: 0.0\n"
        )
        (d / "meta.yaml").write_text('branch_sha: "?"\nexit_reason: synthetic\n')
    md = compare_runs(tmp_path / "a", tmp_path / "b")
    assert "## Success criteria status" in md
    assert "✅" in md   # criterion passes for both runs


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_compare_against_self_yields_zero_deltas() -> None:
    diff = compute_diff(load_run(GOLDEN / "Echo"), load_run(GOLDEN / "Echo"))
    assert diff["top_word_density"]["delta"] == pytest.approx(0.0)
    assert diff["n_thoughts"]["delta"] == 0
    for dim, vals in diff["mood"].items():
        assert vals["delta"] == pytest.approx(0.0), f"non-zero mood delta on self-compare: {dim}"
