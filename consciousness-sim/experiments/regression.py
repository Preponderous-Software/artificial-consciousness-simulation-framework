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
already floor-checks that count.

Two kinds of expectation are therefore supported (#173):

* A **point pin** — a bare number, compared within `epsilon`. Correct for
  metrics MockProvider's deterministic output holds constant regardless of
  cycle count: word density, perception influence rate, and any mood dimension
  that sits at its baseline because nothing in the mock text triggers it.
* A **range pin** — a mapping with `min` and/or `max` (plus an optional
  free-text `note`), compared inclusively with `epsilon` slack at each bound.
  Correct for metrics that *approach* a stable value asymptotically rather than
  holding it. `mood.final.curiosity` is the motivating case: it is triggered on
  every cycle, so it climbs geometrically toward
  `initial + drift_rate / homeostasis_rate` and reaches that limit only in the
  limit of infinite cycles. Pinning the limit as a point value passed on fast
  CI runners (hundreds of cycles) and failed deterministically on slower hosts
  that completed only single-digit cycles before the stop condition fired.

So "final mood" is *not* unconditionally cycle-count independent, and pinning
it needs the shape of the metric taken into account per dimension.
"""

from __future__ import annotations

from typing import Any

DEFAULT_EPSILON = 1e-6

# Keys a range pin may carry. `note` is documentation only — the golden
# snapshot is JSON and so cannot carry comments.
_RANGE_KEYS = frozenset({"min", "max", "note"})


def _get_dotted(metrics: dict[str, Any], dotted: str) -> Any:
    cursor: Any = metrics
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(dotted)
        cursor = cursor[part]
    return cursor


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_point(dotted: str, actual: Any, exp_value: Any, epsilon: float) -> str | None:
    # `exp_value` is rendered rather than the float it is compared as, so an
    # integer pin still reports as `4` and not `4.0`.
    if not _is_number(actual):
        return f"{dotted}: expected numeric {exp_value!r}, got {actual!r}"
    delta = float(actual) - float(exp_value)
    if abs(delta) > epsilon:
        return f"{dotted}: expected {exp_value!r}, got {actual!r} (delta={delta:.6g})"
    return None


def _check_range(dotted: str, actual: Any, spec: dict[str, Any], epsilon: float) -> str | None:
    unknown = sorted(set(spec) - _RANGE_KEYS)
    if unknown:
        return (
            f"{dotted}: malformed range pin — unknown key(s) {unknown} "
            f"(allowed: min, max, note)"
        )
    lower = spec.get("min")
    upper = spec.get("max")
    if lower is None and upper is None:
        return f"{dotted}: malformed range pin — needs at least one of 'min'/'max'"
    for bound_name, bound in (("min", lower), ("max", upper)):
        if bound is not None and not _is_number(bound):
            return f"{dotted}: malformed range pin — {bound_name!r} must be numeric, got {bound!r}"
    if not _is_number(actual):
        return f"{dotted}: expected numeric in range {spec!r}, got {actual!r}"
    value = float(actual)
    if lower is not None and value < float(lower) - epsilon:
        return f"{dotted}: expected >= {lower!r}, got {actual!r} (delta={value - float(lower):.6g})"
    if upper is not None and value > float(upper) + epsilon:
        return f"{dotted}: expected <= {upper!r}, got {actual!r} (delta={value - float(upper):.6g})"
    return None


def check_smoke_regression(
    metrics: dict[str, Any],
    expected: dict[str, Any],
    epsilon: float = DEFAULT_EPSILON,
) -> list[str]:
    """Compare a freshly produced run's `metrics` against the pinned `expected`
    snapshot. Returns a list of human-readable failure descriptions; an empty
    list means every pinned field matched within tolerance.

    Each entry in `expected` is either a point pin (a bare value) or a range pin
    (a mapping carrying `min` and/or `max`) — see the module docstring for when
    each applies.
    """
    failures: list[str] = []
    for dotted, exp_value in expected.items():
        try:
            actual = _get_dotted(metrics, dotted)
        except KeyError:
            failures.append(f"{dotted}: missing from metrics (expected {exp_value!r})")
            continue
        failure: str | None
        if isinstance(exp_value, dict):
            failure = _check_range(dotted, actual, exp_value, epsilon)
        elif _is_number(exp_value):
            failure = _check_point(dotted, actual, exp_value, epsilon)
        elif actual != exp_value:
            failure = f"{dotted}: expected {exp_value!r}, got {actual!r}"
        else:
            failure = None
        if failure is not None:
            failures.append(failure)
    return failures
