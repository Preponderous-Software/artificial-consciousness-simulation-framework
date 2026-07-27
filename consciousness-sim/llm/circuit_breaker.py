"""Circuit breaker for LLM provider calls (#114).

Theory mapping — none. This is infrastructure, not a cognitive-architecture
component: it governs how the thought loop's access to its generative model
degrades when the substrate (a local Ollama server) is saturated. It makes no
claim about GWT/HOT/PP/AST commitments and adds no representational content.
Its only architectural consequence is temporal: a saturated provider now
fails a cycle in <1s instead of burning a full request timeout, so the
consecutive-failure shutdown in `core/consciousness.py` is reached in
bounded wall-clock time rather than after tens of minutes of silence.

Behaviour (per #114):
  closed    → calls pass through; consecutive retryable failures are counted.
  open      → calls short-circuit with LLMUnavailableError until the cooldown
              expires. One WARNING is logged on the transition, not per call.
  half-open → exactly one probe call is admitted. Success closes the circuit;
              failure re-opens it with a doubled cooldown, capped at
              `max_cooldown_seconds`.

The realized cooldown ladder at the defaults is 60s → 120s → 240s → 300s
(doubling, capped), which approximates the 60/120/300 ladder sketched in #114.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 60.0
DEFAULT_MAX_COOLDOWN_SECONDS = 300.0


class LLMUnavailableError(RuntimeError):
    """Raised instead of calling the provider while the circuit is open.

    Deliberately a plain RuntimeError subclass: `LLMProvider._is_retryable_error`
    must not classify it as transient, or `with_backoff` would sleep-and-retry
    the very call the breaker exists to short-circuit.
    """


class CircuitBreaker:
    """Fast-fails provider calls after repeated transient failures.

    One breaker instance is shared by a provider's generate and embed paths:
    the contended resource is the server, not the individual endpoint, so a
    saturated server should stop receiving both kinds of request.

    `time_fn` is injectable so tests can advance the cooldown clock without
    sleeping. It must be monotonic.
    """

    def __init__(
        self,
        *,
        name: str = "llm",
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        max_cooldown_seconds: float = DEFAULT_MAX_COOLDOWN_SECONDS,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if cooldown_seconds <= 0:
            raise ValueError(f"cooldown_seconds must be > 0, got {cooldown_seconds}")
        self.name = name
        self.failure_threshold = int(failure_threshold)
        self.base_cooldown_seconds = float(cooldown_seconds)
        self.max_cooldown_seconds = max(float(max_cooldown_seconds), float(cooldown_seconds))
        self._time = time_fn

        self._state: str = CLOSED
        self._consecutive_failures: int = 0
        self._current_cooldown: float = float(cooldown_seconds)
        self._opened_at: float = 0.0
        # Guards the half-open probe so only one call is admitted at a time.
        self._probe_in_flight: bool = False

    # ---- introspection ----

    @property
    def state(self) -> str:
        """Current state label, resolving an elapsed cooldown to 'half_open'.

        Read-only: does not consume the half-open probe slot, so observers
        (state.json health block, dashboards) can poll this freely.
        """
        if self._state == OPEN and self._cooldown_elapsed():
            return HALF_OPEN
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def stats(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "consecutive_failures": self._consecutive_failures,
            "cooldown_seconds": self._current_cooldown,
        }

    # ---- state transitions ----

    def _cooldown_elapsed(self) -> bool:
        return (self._time() - self._opened_at) >= self._current_cooldown

    def _open(self, reason: str) -> None:
        # Only a transition out of CLOSED is a first open; re-entering OPEN
        # from HALF_OPEN means a probe just failed and gets its own message.
        first_open = self._state == CLOSED
        self._state = OPEN
        self._opened_at = self._time()
        self._probe_in_flight = False
        if first_open:
            logger.warning(
                "Circuit breaker %r OPEN after %d consecutive failures (%s); "
                "short-circuiting calls for %.0fs",
                self.name, self._consecutive_failures, reason, self._current_cooldown,
            )
        else:
            logger.warning(
                "Circuit breaker %r re-OPEN after failed probe (%s); "
                "next retry in %.0fs",
                self.name, reason, self._current_cooldown,
            )

    def _close(self) -> None:
        was = self._state
        self._state = CLOSED
        self._consecutive_failures = 0
        self._current_cooldown = self.base_cooldown_seconds
        self._probe_in_flight = False
        if was != CLOSED:
            logger.info("Circuit breaker %r CLOSED — provider responded successfully", self.name)

    def reset(self) -> None:
        """Force the breaker closed. Used when an out-of-band health check
        establishes the provider is reachable again."""
        self._close()

    # ---- call path ----

    def _before_call(self) -> None:
        """Raise LLMUnavailableError unless this call should be admitted."""
        if self._state == CLOSED:
            return
        if self._state == HALF_OPEN:
            # A probe is already out; a second caller must not pile on.
            if self._probe_in_flight:
                raise LLMUnavailableError(
                    f"{self.name} circuit is half-open; a probe call is already in flight"
                )
            self._probe_in_flight = True
            return
        if not self._cooldown_elapsed():
            remaining = self._current_cooldown - (self._time() - self._opened_at)
            raise LLMUnavailableError(
                f"{self.name} circuit is open (fast-fail); retrying in {max(remaining, 0.0):.0f}s"
            )
        # State is OPEN with the cooldown elapsed, so no probe can be
        # outstanding — both _open() and _close() clear the flag, and it is
        # only ever set on a path that leaves the state HALF_OPEN.
        # Admit exactly one probe.
        self._state = HALF_OPEN
        self._probe_in_flight = True
        logger.info(
            "Circuit breaker %r HALF-OPEN — admitting one probe call after %.0fs cooldown",
            self.name, self._current_cooldown,
        )

    def _on_success(self) -> None:
        self._close()

    def _on_failure(self, exc: BaseException) -> None:
        self._consecutive_failures += 1
        reason = f"{type(exc).__name__}: {exc}"
        if self._state == HALF_OPEN:
            # Probe failed — back off further before the next probe.
            self._current_cooldown = min(self._current_cooldown * 2, self.max_cooldown_seconds)
            self._open(reason)
            return
        if self._consecutive_failures >= self.failure_threshold:
            self._open(reason)

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Invoke `func` under the breaker.

        Only failures classified as transient by `should_trip` count toward
        opening the circuit; a deterministic error (e.g. a missing package)
        propagates without tripping, because retrying later would not help.
        """
        self._before_call()
        try:
            result = await func(*args, **kwargs)
        except asyncio.CancelledError:
            self._probe_in_flight = False
            raise
        except Exception as exc:
            if self.should_trip(exc):
                self._on_failure(exc)
            else:
                self._probe_in_flight = False
            raise
        self._on_success()
        return result

    @staticmethod
    def should_trip(exc: BaseException) -> bool:
        """Whether `exc` counts as evidence the provider is unavailable.

        Timeouts and connection errors are the saturation signature from #114.
        LLMUnavailableError is excluded so a fast-fail can never itself widen
        the cooldown (the breaker raises it before `func` is ever called, but
        a nested breaker would otherwise double-count).
        """
        if isinstance(exc, LLMUnavailableError):
            return False
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return True
        name = exc.__class__.__name__.lower()
        return any(token in name for token in ("timeout", "connection", "unavailable"))


def build_circuit_breaker(
    config: dict[str, Any] | None,
    *,
    name: str = "llm",
    time_fn: Callable[[], float] = time.monotonic,
) -> CircuitBreaker | None:
    """Build a breaker from an `llm.circuit_breaker` config block.

    Returns None when the block is absent or `enabled: false`, so the breaker
    is strictly opt-out and pre-#114 behaviour remains reachable.
    """
    if not config:
        return None
    if not bool(config.get("enabled", True)):
        return None
    return CircuitBreaker(
        name=name,
        failure_threshold=int(config.get("failure_threshold", DEFAULT_FAILURE_THRESHOLD)),
        cooldown_seconds=float(config.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS)),
        max_cooldown_seconds=float(
            config.get("max_cooldown_seconds", DEFAULT_MAX_COOLDOWN_SECONDS)
        ),
        time_fn=time_fn,
    )
