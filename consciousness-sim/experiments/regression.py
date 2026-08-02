"""Smoke-regression check for CI (#87, Phase 3 of #57).

Pins a curated subset of `mock-smoke-baseline`'s metrics against a golden
snapshot (`experiments/golden/_smoke_expected.json`) so a refactor that
silently changes cycle/mood/perception semantics fails CI instead of drifting
unnoticed. `scripts/experiment.py check-smoke` is the CLI wrapper this feeds.

Only fields that are actually stable run-to-run are pinned. Raw event counts
(`event_counts.thought`, etc.) are **not** pinned here: with
`thought_loop.min_interval_seconds`/`max_interval_seconds` both 0, MockProvider
cycles complete in ~3ms, so the runner's 1s poll loop can overshoot the
manifest's `duration.thoughts` target by dozens of cycles before it next
checks — verified empirically (three local runs of the same manifest produced
244/265/264 thought events). The manifest's own `event_counts.thought >= 6`
success criterion (evaluated separately via `evaluate_success_criterion`)
already floor-checks that count; this module pins ratio/equilibrium metrics
(word density, final mood, perception influence rate) that MockProvider's
deterministic output holds constant regardless of exact cycle count.
"""

from __future__ import annotations

from typing import Any

DEFAULT_EPSILON = 1e-6


def _get_dotted(metrics: dict[str, Any], dotted: str) -> Any:
    cursor: Any = metrics
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(dotted)
        cursor = cursor[part]
    return cursor


def check_smoke_regression(
    metrics: dict[str, Any],
    expected: dict[str, Any],
    epsilon: float = DEFAULT_EPSILON,
) -> list[str]:
    """Compare a freshly produced run's `metrics` against the pinned `expected`
    snapshot. Returns a list of human-readable failure descriptions; an empty
    list means every pinned field matched within tolerance.
    """
    failures: list[str] = []
    for dotted, exp_value in expected.items():
        try:
            actual = _get_dotted(metrics, dotted)
        except KeyError:
            failures.append(f"{dotted}: missing from metrics (expected {exp_value!r})")
            continue
        if isinstance(exp_value, (int, float)) and not isinstance(exp_value, bool):
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                failures.append(f"{dotted}: expected numeric {exp_value!r}, got {actual!r}")
                continue
            delta = float(actual) - float(exp_value)
            if abs(delta) > epsilon:
                failures.append(
                    f"{dotted}: expected {exp_value!r}, got {actual!r} (delta={delta:.6g})"
                )
        elif actual != exp_value:
            failures.append(f"{dotted}: expected {exp_value!r}, got {actual!r}")
    return failures
