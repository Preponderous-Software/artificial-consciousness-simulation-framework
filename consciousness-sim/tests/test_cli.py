"""Tests for interfaces/cli.py payload handling (#11).

The Rich dashboard consumes `dict[str, object]` event payloads emitted by
`Consciousness._emit()`. These tests pin the narrowing behavior of the
display handlers: a malformed payload must degrade to a default rather than
raise inside a handler, since `_emit()` isolates but also swallows handler
exceptions, which would silently freeze the displayed counters.
"""

from __future__ import annotations

import asyncio

from interfaces.cli import ConsciousnessCLI, _as_int


class _FakeMind:
    """Minimum surface area ConsciousnessCLI subscribes to on construction."""

    def __init__(self) -> None:
        self.on_initialized: list = []
        self.on_thought: list = []
        self.on_memory_stored: list = []
        self.on_reflection: list = []


def _cli() -> ConsciousnessCLI:
    return ConsciousnessCLI(_FakeMind())


def test_as_int_passes_through_numeric_values() -> None:
    assert _as_int(7, 0) == 7
    assert _as_int(7.9, 0) == 7


def test_as_int_falls_back_on_non_numeric_values() -> None:
    assert _as_int("12", 3) == 3
    assert _as_int(None, 3) == 3
    assert _as_int({"n": 1}, 3) == 3


def test_as_int_treats_bool_as_non_numeric() -> None:
    """`True` is an int subclass; reporting a long-term count of 1 because a
    payload carried a flag would be worse than keeping the default."""
    assert _as_int(True, 5) == 5


def test_on_initialized_reads_short_term_contents() -> None:
    cli = _cli()
    asyncio.run(
        cli._on_initialized(
            {
                "short_term": [{"content": "first"}, {"content": "second"}],
                "long_term_count": 4,
            }
        )
    )
    assert cli.thoughts == ["first", "second"]
    assert cli.long_term_count == 4


def test_on_initialized_ignores_malformed_short_term() -> None:
    cli = _cli()
    asyncio.run(cli._on_initialized({"short_term": "not-a-list", "long_term_count": 2}))
    assert cli.thoughts == []
    assert cli.long_term_count == 2


def test_on_memory_keeps_previous_count_on_malformed_payload() -> None:
    cli = _cli()
    cli.long_term_count = 9
    asyncio.run(cli._on_memory({"long_term_count": None, "content": "a memory"}))
    assert cli.long_term_count == 9
    assert cli.memories == ["a memory"]
