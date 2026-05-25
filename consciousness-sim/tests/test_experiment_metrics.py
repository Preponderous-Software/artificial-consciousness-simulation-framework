"""Tests for experiments/metrics.py.

Two complementary modes:

1. **In-memory fixtures** — small synthetic event lists with known shapes;
   verify each metric function in isolation.

2. **Golden journals** — load the four real reference runs from
   `experiments/golden/` and assert known empirical values. If a future
   refactor changes a metric's behavior, the assertion will catch it.

Pinned expected values were computed from the original manual analyses
(Rafael / Sage / Echo / Wren RUN_REPORT files). Tolerances are tight where
the metric is deterministic and looser where it depends on JSON-parsing order.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from experiments.metrics import (
    DEFAULT_ATTRACTOR_WORDS,
    attractor_words_in_top_n,
    compute_all,
    content_word_distribution,
    cycle_interval_stats,
    cycle_rate_trajectory,
    event_type_counts,
    identity_shifts_per_reflection,
    mood_collapse_score,
    mood_dimensions_non_degenerate,
    perception_influence_rate,
    perception_word_overlap,
    reflection_rate,
    top_word_density,
)


GOLDEN = Path(__file__).resolve().parents[1] / "experiments" / "golden"


# ---------------------------------------------------------------------------
# Helpers — in-memory event fixtures
# ---------------------------------------------------------------------------

def _thought(content: str, ts: str = "2026-01-01T00:00:00+00:00") -> dict:
    return {"timestamp": ts, "type": "thought", "content": content}


def _perception(content: str, ts: str = "2026-01-01T00:00:30+00:00") -> dict:
    return {"timestamp": ts, "type": "perception", "content": content}


def _reflection(content: str, ts: str = "2026-01-01T00:01:00+00:00") -> dict:
    return {"timestamp": ts, "type": "reflection", "content": content}


# ---------------------------------------------------------------------------
# In-memory: vocabulary
# ---------------------------------------------------------------------------

def test_content_word_distribution_filters_stop_and_short_words() -> None:
    events = [
        _thought("the void within me stirs gently"),
        _thought("the void hums softly"),
    ]
    c = content_word_distribution(events)
    assert c["void"] == 2
    assert c["stirs"] == 1
    assert "the" not in c          # stop word
    assert "me" not in c           # stop word + too short
    assert "i" not in c            # stop word


def test_top_word_density_per_thought() -> None:
    c = Counter({"void": 4, "ripple": 2})
    assert top_word_density(c, n_thoughts=2) == 2.0
    assert top_word_density(c, n_thoughts=0) == 0.0
    assert top_word_density(Counter(), n_thoughts=5) == 0.0


def test_attractor_words_in_top_n_returns_rank_or_none() -> None:
    c = Counter({"threads": 50, "void": 30, "ripple": 10, "tapestry": 5})
    ranks = attractor_words_in_top_n(c, ["threads", "tapestry", "cosmic"], n=3)
    assert ranks == {"threads": 1, "tapestry": None, "cosmic": None}


# ---------------------------------------------------------------------------
# In-memory: mood
# ---------------------------------------------------------------------------

def test_mood_dimensions_non_degenerate_counts_only_middle_values() -> None:
    state = {"identity": {"mood": {"a": 0.0, "b": 1.0, "c": 0.5, "d": 0.3}}}
    assert mood_dimensions_non_degenerate(state) == 2
    # Tight eps still excludes 0 and 1
    assert mood_dimensions_non_degenerate(state, eps=0.001) == 2


def test_mood_collapse_score_zero_when_at_initial() -> None:
    initial = {"curiosity": 0.7, "wonder": 0.6, "melancholy": 0.2, "contentment": 0.5}
    state = {"identity": {"mood": initial}}
    assert mood_collapse_score(state, initial=initial) == pytest.approx(0.0)


def test_mood_collapse_score_increases_with_distance() -> None:
    initial = {"curiosity": 0.5}
    state = {"identity": {"mood": {"curiosity": 1.0}}}
    assert mood_collapse_score(state, initial=initial) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# In-memory: perception influence
# ---------------------------------------------------------------------------

def test_perception_word_overlap_traces_new_words_in_subsequent_thoughts() -> None:
    events = [
        _thought("I exist in this space"),
        _perception("[wikipedia: Alpine] Alpine soldiers wear the cappello distinctive headwear"),
        _thought("I think about cappello headwear traditions"),
        _thought("the alpine ranges loom"),
        _thought("nothing related"),
    ]
    traces = perception_word_overlap(events, window=3)
    assert len(traces) == 1
    trace = traces[0]
    assert trace.perception_title == "Alpine"
    # `cappello`, `alpine` are new and from perception text
    assert "cappello" in trace.new_words_in_next_thoughts
    assert "alpine" in trace.new_words_in_next_thoughts


def test_perception_influence_rate_handles_empty() -> None:
    assert perception_influence_rate([]) == 0.0


# ---------------------------------------------------------------------------
# In-memory: event counts + reflections + shifts
# ---------------------------------------------------------------------------

def test_event_type_counts_and_reflection_rate() -> None:
    events = [_thought("a"), _thought("b"), _reflection("r"), _perception("[w: X] y")]
    assert event_type_counts(events) == {"thought": 2, "reflection": 1, "perception": 1}
    assert reflection_rate(events) == pytest.approx(0.5)


def test_identity_shifts_reads_from_state_amendments_per_75() -> None:
    """Issue #75: shifts weren't journaled, so the metric must read state.amendments."""
    events = [_thought("a"), _reflection("r1"), _reflection("r2")]
    state = {"identity": {"amendments": ["x", "y", "z"]}}
    assert identity_shifts_per_reflection(events, state) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# In-memory: performance
# ---------------------------------------------------------------------------

def test_cycle_interval_stats_uses_consecutive_thoughts() -> None:
    events = [
        _thought("a", ts="2026-01-01T00:00:00+00:00"),
        _thought("b", ts="2026-01-01T00:00:10+00:00"),
        _thought("c", ts="2026-01-01T00:00:30+00:00"),
        _thought("d", ts="2026-01-01T00:01:00+00:00"),
    ]
    s = cycle_interval_stats(events)
    assert s.n_intervals == 3
    assert s.mean_s == pytest.approx((10 + 20 + 30) / 3)


def test_cycle_rate_trajectory_returns_one_per_window() -> None:
    events = [
        _thought(f"t{i}", ts=f"2026-01-01T00:{i:02d}:00+00:00")
        for i in range(60)
    ]
    traj = cycle_rate_trajectory(events, window=30)
    # 60 thoughts, window=30 → contract says floor(60/30) = 2 windows
    assert len(traj) == 2
    assert all(t > 0 for t in traj)


def test_cycle_rate_trajectory_matches_floor_contract_exactly() -> None:
    """Regression for the off-by-one fix on Copilot review of PR #85.

    Contract: `len(out) == n_thoughts // window`. With 90 thoughts at
    window=30 we expect 3 windows, not 2.
    """
    events = [
        _thought(f"t{i}", ts=f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00")
        for i in range(90)
    ]
    traj = cycle_rate_trajectory(events, window=30)
    assert len(traj) == 3, f"expected floor(90/30)=3 windows, got {len(traj)}"


def test_cycle_rate_trajectory_empty_when_below_window() -> None:
    """Fewer thoughts than `window` should yield an empty list."""
    events = [_thought(f"t{i}", ts=f"2026-01-01T00:00:{i:02d}+00:00") for i in range(10)]
    assert cycle_rate_trajectory(events, window=30) == []


def test_cycle_rate_trajectory_divides_by_intervals_not_window() -> None:
    """Regression for PR #85 review: W consecutive thoughts have W-1 intervals
    between them. Dividing by W (the count) instead of W-1 (the interval
    count) underestimates s/thought systematically.

    Construct a journal where every interval is exactly 10s, so each window's
    avg s/thought must equal 10.0 regardless of which window.
    """
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = [
        {"type": "thought", "content": f"t{i}",
         "timestamp": (t0 + timedelta(seconds=i * 10)).isoformat()}
        for i in range(60)
    ]
    traj = cycle_rate_trajectory(events, window=30)
    assert len(traj) == 2
    for rate in traj:
        assert rate == pytest.approx(10.0, abs=0.01), \
            f"expected 10.0 s/thought (each interval is 10s), got {rate}"


def test_mood_collapse_score_reads_initial_from_state_when_present() -> None:
    """Regression for PR #85 review: mood_collapse_score should prefer the
    initial mood persisted in state.identity.initial_mood (e.g. when a
    manifest override changed it) instead of always using the project defaults.
    """
    # Override-via-state: pretend the run started with a different baseline
    state = {
        "identity": {
            "initial_mood": {"curiosity": 0.5, "wonder": 0.5, "melancholy": 0.5, "contentment": 0.5},
            "mood":         {"curiosity": 0.5, "wonder": 0.5, "melancholy": 0.5, "contentment": 0.5},
        }
    }
    # Final equals initial, so score should be 0 — but only if the metric reads
    # initial_mood from state. If it hardcodes the default {0.7, 0.6, 0.2, 0.5},
    # the score would be nonzero.
    from experiments.metrics import mood_collapse_score
    assert mood_collapse_score(state) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# sample_traces dedup — issue #106
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_perceptions,expected_sample_titles", [
    (0, []),
    (1, ["Topic 1"]),
    (2, ["Topic 1", "Topic 2"]),
    (3, ["Topic 1", "Topic 2", "Topic 3"]),
    (5, ["Topic 1", "Topic 3", "Topic 5"]),
])
def test_sample_traces_dedups_when_fewer_than_three(n_perceptions: int, expected_sample_titles: list[str]) -> None:
    """compute_all.sample_traces must not repeat the same trace when n_traces < 3."""
    import tempfile, json as _json
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        d = _Path(tmp)
        journal_lines = []
        for i in range(1, n_perceptions + 1):
            journal_lines.append(_json.dumps({
                "timestamp": f"2026-01-01T00:00:{i:02d}+00:00",
                "type": "perception",
                "content": f"[mock: Topic {i}] some unique words here {i}",
            }))
            journal_lines.append(_json.dumps({
                "timestamp": f"2026-01-01T00:00:{i:02d}+00:00",
                "type": "thought",
                "content": f"I think about thing {i}.",
            }))
        (d / "journal.jsonl").write_text("\n".join(journal_lines) + ("\n" if journal_lines else ""))
        (d / "state.json").write_text(_json.dumps({"identity": {}, "thought_count": n_perceptions}))

        metrics = compute_all(d / "journal.jsonl", d / "state.json")
        sample_titles = [s["title"] for s in metrics["perception"]["sample_traces"]]

    assert sample_titles == expected_sample_titles, (
        f"For n={n_perceptions}: got {sample_titles}, expected {expected_sample_titles}"
    )


def test_compute_all_includes_mood_initial_section() -> None:
    """metrics dict should expose `mood.initial` so report.py + compare.py can
    render deltas without re-reading state.json."""
    import tempfile, json as _json
    from pathlib import Path as _Path
    with tempfile.TemporaryDirectory() as tmp:
        d = _Path(tmp)
        (d / "journal.jsonl").write_text(
            '{"timestamp": "2026-01-01T00:00:00+00:00", "type": "thought", "content": "x"}\n'
        )
        (d / "state.json").write_text(_json.dumps({
            "identity": {
                "initial_mood": {"curiosity": 0.3, "wonder": 0.3},
                "mood": {"curiosity": 0.6, "wonder": 0.4},
            },
            "thought_count": 1,
        }))
        from experiments.metrics import compute_all
        metrics = compute_all(d / "journal.jsonl", d / "state.json")
    assert "initial" in metrics["mood"]
    assert metrics["mood"]["initial"]["curiosity"] == 0.3


# ---------------------------------------------------------------------------
# Golden journals — pin known empirical values
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
@pytest.mark.parametrize("name,expected_thoughts,expected_attractor", [
    # Rafael: cosmic attractor dominant — threads should rank ≤5 in top-50
    ("Rafael", 80, lambda r: r["threads"] is not None and r["threads"] <= 5),
    # Sage: perception in, attractor still strong but slightly broken
    ("Sage", 70, lambda r: r["threads"] is not None and r["threads"] <= 10),
    # Echo: full fix in, threads escaped top-10
    ("Echo", 195, lambda r: r["threads"] is None or r["threads"] > 10),
    # Wren: partial reversion — threads back in top-10 but cosmic still 0
    ("Wren", 195, lambda r: True),  # vocabulary is variable; just verify N
])
def test_golden_journal_basic_shape(name, expected_thoughts, expected_attractor) -> None:
    journal = GOLDEN / name / "journal.jsonl"
    state = GOLDEN / name / "state.json"
    metrics = compute_all(journal, state)

    n_thoughts = metrics["event_counts"].get("thought", 0)
    assert n_thoughts >= expected_thoughts, f"{name}: expected ≥{expected_thoughts} thoughts, got {n_thoughts}"

    attractor_ranks = metrics["vocabulary"]["attractor_ranks_in_top_10"]
    assert expected_attractor(attractor_ranks), (
        f"{name}: attractor check failed; ranks={attractor_ranks}"
    )


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_golden_rafael_mood_collapsed() -> None:
    """Rafael ran without the mood fix — final mood should show collapse."""
    metrics = compute_all(GOLDEN / "Rafael" / "journal.jsonl", GOLDEN / "Rafael" / "state.json")
    assert metrics["mood"]["dimensions_non_degenerate"] <= 1
    # wonder / melancholy / contentment all ≈ 0
    final = metrics["mood"]["final"]
    assert final["melancholy"] <= 0.05
    assert final["contentment"] <= 0.05


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_golden_echo_mood_healthy() -> None:
    """Echo ran with the mood fix — should show non-degenerate mood across multiple dims."""
    metrics = compute_all(GOLDEN / "Echo" / "journal.jsonl", GOLDEN / "Echo" / "state.json")
    # At least 1 dim strictly in (0.05, 0.95) — and the empirical value is wonder ≈ 0.98 and melancholy ≈ 0.34
    final = metrics["mood"]["final"]
    assert final["wonder"] > 0.5, f"Echo wonder should have stayed high, got {final['wonder']}"
    assert final["melancholy"] > 0.1, f"Echo melancholy should have stayed >0.1, got {final['melancholy']}"


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_golden_wren_amendments_present() -> None:
    """Wren actually triggered identity shifts (5 amendments — see #71/#75/#76 discussion)."""
    metrics = compute_all(GOLDEN / "Wren" / "journal.jsonl", GOLDEN / "Wren" / "state.json")
    assert metrics["reflections"]["n_amendments_in_state"] == 5


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden journals not yet committed")
def test_golden_top_word_density_ordering() -> None:
    """Compute-all is the contract surface — verify the headline ordering across runs.

    Empirically: Rafael 1.42 > Sage 0.97 > Echo 0.78 ≈ Wren 0.76 (Echo/Wren close).
    """
    def density(name: str) -> float:
        m = compute_all(GOLDEN / name / "journal.jsonl", GOLDEN / name / "state.json")
        return m["vocabulary"]["top_word_density_per_thought"]

    rafael = density("Rafael")
    sage = density("Sage")
    echo = density("Echo")
    wren = density("Wren")

    assert rafael > sage > 0.85, f"Rafael={rafael} should exceed Sage={sage}"
    assert echo < sage, f"Echo={echo} should be below Sage={sage}"
    assert wren < sage, f"Wren={wren} should be below Sage={sage}"
