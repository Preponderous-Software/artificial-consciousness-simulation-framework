"""Tests for the LLM provider circuit breaker (#114)."""

from __future__ import annotations

import asyncio
import logging

import pytest

from llm.circuit_breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
    LLMUnavailableError,
    build_circuit_breaker,
)


class FakeClock:
    """Monotonic clock the tests advance explicitly, so cooldowns cost no wall time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _breaker(clock: FakeClock, **kwargs) -> CircuitBreaker:
    params = dict(
        name="test",
        failure_threshold=3,
        cooldown_seconds=60.0,
        max_cooldown_seconds=300.0,
        time_fn=clock,
    )
    params.update(kwargs)
    return CircuitBreaker(**params)


async def _boom() -> str:
    raise TimeoutError("simulated Ollama timeout")


async def _ok() -> str:
    return "response"


# ---------------------------------------------------------------------------
# Closed state
# ---------------------------------------------------------------------------


def test_closed_breaker_passes_calls_through() -> None:
    breaker = _breaker(FakeClock())
    assert asyncio.run(breaker.call(_ok)) == "response"
    assert breaker.state == CLOSED
    assert breaker.consecutive_failures == 0


def test_failures_below_threshold_keep_circuit_closed() -> None:
    breaker = _breaker(FakeClock())
    for _ in range(2):
        with pytest.raises(TimeoutError):
            asyncio.run(breaker.call(_boom))
    assert breaker.state == CLOSED
    assert breaker.consecutive_failures == 2


def test_success_resets_the_failure_counter() -> None:
    breaker = _breaker(FakeClock())
    for _ in range(2):
        with pytest.raises(TimeoutError):
            asyncio.run(breaker.call(_boom))
    asyncio.run(breaker.call(_ok))
    assert breaker.consecutive_failures == 0
    # A third failure after the reset must not open the circuit.
    with pytest.raises(TimeoutError):
        asyncio.run(breaker.call(_boom))
    assert breaker.state == CLOSED


# ---------------------------------------------------------------------------
# Opening + fast-fail
# ---------------------------------------------------------------------------


def test_three_consecutive_timeouts_open_the_circuit_and_next_call_fast_fails() -> None:
    """#114 acceptance: 3 consecutive timeouts → open → next call fast-fails."""
    clock = FakeClock()
    breaker = _breaker(clock)
    calls: list[int] = []

    async def counting_boom() -> str:
        calls.append(1)
        raise TimeoutError("simulated Ollama timeout")

    for _ in range(3):
        with pytest.raises(TimeoutError):
            asyncio.run(breaker.call(counting_boom))
    assert breaker.state == OPEN
    assert len(calls) == 3

    # Fourth call never reaches the provider.
    with pytest.raises(LLMUnavailableError):
        asyncio.run(breaker.call(counting_boom))
    assert len(calls) == 3, "open circuit must not invoke the wrapped function"


def test_open_circuit_error_names_the_remaining_cooldown() -> None:
    clock = FakeClock()
    breaker = _breaker(clock)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            asyncio.run(breaker.call(_boom))
    clock.advance(20.0)
    with pytest.raises(LLMUnavailableError, match="40s"):
        asyncio.run(breaker.call(_ok))


def test_opening_logs_one_warning_not_one_per_call(caplog) -> None:
    clock = FakeClock()
    breaker = _breaker(clock)
    with caplog.at_level(logging.WARNING, logger="llm.circuit_breaker"):
        for _ in range(3):
            with pytest.raises(TimeoutError):
                asyncio.run(breaker.call(_boom))
        for _ in range(5):
            with pytest.raises(LLMUnavailableError):
                asyncio.run(breaker.call(_ok))
    opens = [r for r in caplog.records if "OPEN" in r.getMessage()]
    assert len(opens) == 1, f"expected exactly one OPEN warning, got {len(opens)}"


# ---------------------------------------------------------------------------
# Half-open probe
# ---------------------------------------------------------------------------


def test_cooldown_expiry_admits_probe_and_success_closes_circuit() -> None:
    """#114 acceptance: cooldown expiry → half-open probe → success → closed."""
    clock = FakeClock()
    breaker = _breaker(clock)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            asyncio.run(breaker.call(_boom))
    assert breaker.state == OPEN

    clock.advance(60.0)
    assert breaker.state == HALF_OPEN, "state property must reflect an elapsed cooldown"
    assert asyncio.run(breaker.call(_ok)) == "response"
    assert breaker.state == CLOSED
    assert breaker.consecutive_failures == 0


def test_half_open_probe_failure_reopens_with_longer_cooldown() -> None:
    """#114 acceptance: half-open probe fails → re-opens with longer cooldown."""
    clock = FakeClock()
    breaker = _breaker(clock)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            asyncio.run(breaker.call(_boom))
    assert breaker.stats()["cooldown_seconds"] == 60.0

    clock.advance(60.0)
    with pytest.raises(TimeoutError):
        asyncio.run(breaker.call(_boom))
    assert breaker.state == OPEN
    assert breaker.stats()["cooldown_seconds"] == 120.0

    # The old 60s cooldown is no longer enough to admit a probe.
    clock.advance(60.0)
    assert breaker.state == OPEN
    with pytest.raises(LLMUnavailableError):
        asyncio.run(breaker.call(_ok))
    clock.advance(60.0)
    assert breaker.state == HALF_OPEN


def test_failed_probe_logs_the_reopen_message_not_the_first_open_message(caplog) -> None:
    """A re-open after a failed probe is a distinct event from the first open —
    reporting it as 'OPEN after N consecutive failures' would misattribute the
    cause to a fresh burst of failures."""
    clock = FakeClock()
    breaker = _breaker(clock)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            asyncio.run(breaker.call(_boom))

    clock.advance(60.0)
    caplog.clear()  # drop the first-open record; only the probe matters here
    with caplog.at_level(logging.WARNING, logger="llm.circuit_breaker"):
        with pytest.raises(TimeoutError):
            asyncio.run(breaker.call(_boom))
    messages = [r.getMessage() for r in caplog.records]
    assert any("re-OPEN after failed probe" in m for m in messages), messages
    assert not any("consecutive failures" in m for m in messages), messages


def test_cooldown_backoff_is_capped() -> None:
    clock = FakeClock()
    breaker = _breaker(clock, cooldown_seconds=60.0, max_cooldown_seconds=300.0)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            asyncio.run(breaker.call(_boom))
    # Realized ladder: 60 → 120 → 240 → 300 → 300 ...
    for expected in (120.0, 240.0, 300.0, 300.0):
        clock.advance(breaker.stats()["cooldown_seconds"])
        with pytest.raises(TimeoutError):
            asyncio.run(breaker.call(_boom))
        assert breaker.stats()["cooldown_seconds"] == expected


def test_second_caller_during_half_open_fast_fails_while_probe_in_flight() -> None:
    """Only one probe is admitted; a concurrent caller must not pile on."""
    clock = FakeClock()
    breaker = _breaker(clock)

    async def scenario() -> None:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                await breaker.call(_boom)
        clock.advance(60.0)

        release = asyncio.Event()

        async def slow_probe() -> str:
            await release.wait()
            return "probe-ok"

        probe = asyncio.create_task(breaker.call(slow_probe))
        await asyncio.sleep(0)  # let the probe consume the half-open slot
        with pytest.raises(LLMUnavailableError, match="probe call is already in flight"):
            await breaker.call(_ok)
        release.set()
        assert await probe == "probe-ok"
        assert breaker.state == CLOSED

    asyncio.run(scenario())


def test_reset_forces_the_circuit_closed() -> None:
    clock = FakeClock()
    breaker = _breaker(clock)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            asyncio.run(breaker.call(_boom))
    assert breaker.state == OPEN
    breaker.reset()
    assert breaker.state == CLOSED
    assert breaker.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Which failures count
# ---------------------------------------------------------------------------


def test_non_transient_errors_do_not_trip_the_circuit() -> None:
    """A deterministic error (missing package, empty response) is not evidence
    the server is saturated — retrying later would not help."""
    clock = FakeClock()
    breaker = _breaker(clock)

    async def deterministic_failure() -> str:
        raise ValueError("ollama returned empty content")

    for _ in range(5):
        with pytest.raises(ValueError):
            asyncio.run(breaker.call(deterministic_failure))
    assert breaker.state == CLOSED
    assert breaker.consecutive_failures == 0


def test_connection_errors_trip_the_circuit() -> None:
    clock = FakeClock()
    breaker = _breaker(clock)

    async def refused() -> str:
        raise ConnectionRefusedError("connection refused")

    for _ in range(3):
        with pytest.raises(ConnectionRefusedError):
            asyncio.run(breaker.call(refused))
    assert breaker.state == OPEN


def test_should_trip_excludes_its_own_fast_fail() -> None:
    assert CircuitBreaker.should_trip(LLMUnavailableError("open")) is False
    assert CircuitBreaker.should_trip(TimeoutError("t")) is True


def test_cancellation_releases_the_probe_slot_without_counting_a_failure() -> None:
    clock = FakeClock()
    breaker = _breaker(clock)

    async def scenario() -> None:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                await breaker.call(_boom)
        clock.advance(60.0)

        async def hangs() -> str:
            await asyncio.Event().wait()
            return "never"

        probe = asyncio.create_task(breaker.call(hangs))
        await asyncio.sleep(0)
        probe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await probe
        # Slot released: the next call is admitted as a fresh probe rather
        # than fast-failing forever behind an abandoned one.
        assert await breaker.call(_ok) == "response"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Construction + config
# ---------------------------------------------------------------------------


def test_invalid_thresholds_rejected() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError, match="cooldown_seconds"):
        CircuitBreaker(cooldown_seconds=0)


def test_max_cooldown_never_below_base_cooldown() -> None:
    breaker = CircuitBreaker(cooldown_seconds=120.0, max_cooldown_seconds=30.0)
    assert breaker.max_cooldown_seconds == 120.0


def test_build_circuit_breaker_returns_none_when_absent_or_disabled() -> None:
    assert build_circuit_breaker(None) is None
    assert build_circuit_breaker({}) is None
    assert build_circuit_breaker({"enabled": False}) is None


def test_build_circuit_breaker_applies_config_values() -> None:
    breaker = build_circuit_breaker(
        {
            "enabled": True,
            "failure_threshold": 5,
            "cooldown_seconds": 15,
            "max_cooldown_seconds": 90,
        },
        name="ollama:Aria",
    )
    assert breaker is not None
    assert breaker.name == "ollama:Aria"
    assert breaker.failure_threshold == 5
    assert breaker.base_cooldown_seconds == 15.0
    assert breaker.max_cooldown_seconds == 90.0


def test_build_circuit_breaker_defaults_enabled_when_key_omitted() -> None:
    breaker = build_circuit_breaker({"failure_threshold": 2})
    assert breaker is not None
    assert breaker.failure_threshold == 2
