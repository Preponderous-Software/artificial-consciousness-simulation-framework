"""Unit tests for experiments/regression.py (#87).

Pure function over two dicts. No subprocesses, no network, no filesystem.
"""

from __future__ import annotations

from experiments.regression import check_smoke_regression


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


def test_multiple_failures_are_all_reported() -> None:
    metrics = _metrics()
    metrics["vocabulary"]["top_word_density_per_thought"] = 0.0
    metrics["perception"]["influence_rate"] = 1.0

    failures = check_smoke_regression(metrics, _expected())

    assert len(failures) == 2
