"""Unit tests for experiments/regression.py (#87, range pins added in #173).

Pure function over two dicts. No subprocesses and no network; the only
filesystem read is the shipped golden snapshot itself, which the last four
tests assert against directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.regression import check_smoke_regression

GOLDEN_SNAPSHOT = (
    Path(__file__).resolve().parents[1] / "experiments" / "golden" / "_smoke_expected.json"
)

# Equilibrium of the continuously-triggered curiosity dimension:
# initial + drift_rate / homeostasis_rate = 0.7 + 0.05 / 0.3 (#173).
CURIOSITY_EQUILIBRIUM = 0.8666666666666667
# Its value after exactly six triggered cycles — the floor the smoke manifest's
# own `event_counts.thought >= 6` criterion guarantees.
CURIOSITY_AT_SIX_CYCLES = 0.8470585


def _metrics(**overrides: object) -> dict:
    base = {
        "vocabulary": {"top_word_density_per_thought": 2.0},
        "mood": {
            "final": {
                "curiosity": 0.8666666666666667,
                "wonder": 0.6,
                "melancholy": 0.2,
                "contentment": 0.5,
            },
            "dimensions_non_degenerate": 4,
        },
        "perception": {"influence_rate": 0.0},
    }
    base.update(overrides)
    return base


def _expected() -> dict:
    return {
        "vocabulary.top_word_density_per_thought": 2.0,
        "mood.final.curiosity": 0.8666666666666667,
        "mood.dimensions_non_degenerate": 4,
        "perception.influence_rate": 0.0,
    }


def test_matching_metrics_produce_no_failures() -> None:
    assert check_smoke_regression(_metrics(), _expected()) == []


def test_numeric_drift_beyond_epsilon_is_reported() -> None:
    metrics = _metrics()
    metrics["vocabulary"]["top_word_density_per_thought"] = 999.0

    failures = check_smoke_regression(metrics, _expected())

    assert len(failures) == 1
    assert "vocabulary.top_word_density_per_thought" in failures[0]
    assert "999.0" in failures[0]


def test_numeric_drift_within_epsilon_is_not_reported() -> None:
    metrics = _metrics()
    metrics["vocabulary"]["top_word_density_per_thought"] = 2.0 + 1e-9

    assert check_smoke_regression(metrics, _expected(), epsilon=1e-6) == []


def test_missing_metric_path_is_reported() -> None:
    metrics = _metrics()
    del metrics["perception"]

    failures = check_smoke_regression(metrics, _expected())

    assert len(failures) == 1
    assert "perception.influence_rate" in failures[0]
    assert "missing" in failures[0]


def test_integer_dimension_count_mismatch_is_reported() -> None:
    metrics = _metrics()
    metrics["mood"]["dimensions_non_degenerate"] = 3

    failures = check_smoke_regression(metrics, _expected())

    assert len(failures) == 1
    assert "mood.dimensions_non_degenerate" in failures[0]
    # An integer pin renders as `4`, not `4.0` — the comparison widens to float
    # but the message reports the pin as written.
    assert "expected 4," in failures[0]


def test_multiple_failures_are_all_reported() -> None:
    metrics = _metrics()
    metrics["vocabulary"]["top_word_density_per_thought"] = 0.0
    metrics["perception"]["influence_rate"] = 1.0

    failures = check_smoke_regression(metrics, _expected())

    assert len(failures) == 2


# ---------------------------------------------------------------------------
# Range pins — #173
# ---------------------------------------------------------------------------


def _range_expected(**spec: object) -> dict:
    return {"mood.final.curiosity": dict(spec)}


def test_value_inside_range_is_not_reported() -> None:
    metrics = _metrics()
    metrics["mood"]["final"]["curiosity"] = 0.85

    expected = _range_expected(min=CURIOSITY_AT_SIX_CYCLES, max=CURIOSITY_EQUILIBRIUM)

    assert check_smoke_regression(metrics, expected) == []


def test_value_below_range_minimum_is_reported() -> None:
    metrics = _metrics()
    metrics["mood"]["final"]["curiosity"] = 0.7

    failures = check_smoke_regression(
        metrics, _range_expected(min=CURIOSITY_AT_SIX_CYCLES, max=CURIOSITY_EQUILIBRIUM)
    )

    assert len(failures) == 1
    assert "mood.final.curiosity" in failures[0]
    assert ">=" in failures[0]


def test_value_above_range_maximum_is_reported() -> None:
    metrics = _metrics()
    metrics["mood"]["final"]["curiosity"] = 1.0

    failures = check_smoke_regression(
        metrics, _range_expected(min=CURIOSITY_AT_SIX_CYCLES, max=CURIOSITY_EQUILIBRIUM)
    )

    assert len(failures) == 1
    assert "mood.final.curiosity" in failures[0]
    assert "<=" in failures[0]


def test_range_bounds_are_inclusive_within_epsilon() -> None:
    """A run that converges to the equilibrium to the last float bit must pass;
    so must one that stops at exactly the six-cycle floor."""
    at_max = _metrics()
    at_max["mood"]["final"]["curiosity"] = CURIOSITY_EQUILIBRIUM + 1e-9
    at_min = _metrics()
    at_min["mood"]["final"]["curiosity"] = CURIOSITY_AT_SIX_CYCLES - 1e-9

    expected = _range_expected(min=CURIOSITY_AT_SIX_CYCLES, max=CURIOSITY_EQUILIBRIUM)

    assert check_smoke_regression(at_max, expected) == []
    assert check_smoke_regression(at_min, expected) == []


def test_range_with_only_a_maximum_leaves_the_lower_side_unbounded() -> None:
    metrics = _metrics()
    metrics["mood"]["final"]["curiosity"] = 0.0

    assert check_smoke_regression(metrics, _range_expected(max=CURIOSITY_EQUILIBRIUM)) == []


def test_range_with_only_a_minimum_leaves_the_upper_side_unbounded() -> None:
    metrics = _metrics()
    metrics["mood"]["final"]["curiosity"] = 1.0

    assert check_smoke_regression(metrics, _range_expected(min=CURIOSITY_AT_SIX_CYCLES)) == []


def test_note_key_is_documentation_only_and_does_not_affect_the_check() -> None:
    metrics = _metrics()
    metrics["mood"]["final"]["curiosity"] = 0.85

    expected = _range_expected(
        min=CURIOSITY_AT_SIX_CYCLES, max=CURIOSITY_EQUILIBRIUM, note="why this is a range"
    )

    assert check_smoke_regression(metrics, expected) == []


def test_non_numeric_actual_against_a_range_is_reported() -> None:
    metrics = _metrics()
    metrics["mood"]["final"]["curiosity"] = "high"

    failures = check_smoke_regression(metrics, _range_expected(min=0.0, max=1.0))

    assert len(failures) == 1
    assert "expected numeric" in failures[0]


def test_range_pin_with_an_unknown_key_is_reported_as_malformed() -> None:
    failures = check_smoke_regression(_metrics(), _range_expected(mim=0.8, max=0.9))

    assert len(failures) == 1
    assert "malformed range pin" in failures[0]
    assert "mim" in failures[0]


def test_range_pin_with_neither_bound_is_reported_as_malformed() -> None:
    failures = check_smoke_regression(_metrics(), _range_expected(note="no bounds here"))

    assert len(failures) == 1
    assert "malformed range pin" in failures[0]


def test_range_pin_with_a_non_numeric_bound_is_reported_as_malformed() -> None:
    failures = check_smoke_regression(_metrics(), _range_expected(min="0.8"))

    assert len(failures) == 1
    assert "malformed range pin" in failures[0]
    assert "'min'" in failures[0]


def test_missing_metric_path_for_a_range_pin_is_reported() -> None:
    metrics = _metrics()
    del metrics["mood"]

    failures = check_smoke_regression(metrics, _range_expected(min=0.0, max=1.0))

    assert len(failures) == 1
    assert "missing" in failures[0]


# ---------------------------------------------------------------------------
# The shipped snapshot itself — #173
# ---------------------------------------------------------------------------


def _shipped_expected() -> dict:
    with GOLDEN_SNAPSHOT.open(encoding="utf-8") as handle:
        loaded: dict = json.load(handle)
    return loaded


def test_shipped_snapshot_accepts_a_short_run_from_a_slow_host() -> None:
    """The #173 failure: a host that completed only seven cycles before the
    manifest's stop condition fired produced curiosity=0.85294, which the old
    point pin at the equilibrium rejected."""
    metrics = _metrics()
    metrics["mood"]["final"]["curiosity"] = 0.8529409500000001

    assert check_smoke_regression(metrics, _shipped_expected()) == []


def test_shipped_snapshot_accepts_a_long_converged_run() -> None:
    """The fast-CI case that the old point pin was measured against."""
    metrics = _metrics()
    metrics["mood"]["final"]["curiosity"] = CURIOSITY_EQUILIBRIUM

    assert check_smoke_regression(metrics, _shipped_expected()) == []


def test_shipped_snapshot_rejects_saturated_curiosity() -> None:
    """The band still catches the #134 failure mode — a homeostasis regression
    that lets the continuously-triggered dimension climb to the 1.0 ceiling."""
    metrics = _metrics()
    metrics["mood"]["final"]["curiosity"] = 1.0

    failures = check_smoke_regression(metrics, _shipped_expected())

    assert len(failures) == 1
    assert "mood.final.curiosity" in failures[0]


def test_shipped_snapshot_rejects_curiosity_stuck_at_its_baseline() -> None:
    """And the lower bound catches the opposite regression — triggers never
    firing, leaving the dimension pinned at its 0.7 baseline."""
    metrics = _metrics()
    metrics["mood"]["final"]["curiosity"] = 0.7

    failures = check_smoke_regression(metrics, _shipped_expected())

    assert len(failures) == 1
    assert "mood.final.curiosity" in failures[0]
