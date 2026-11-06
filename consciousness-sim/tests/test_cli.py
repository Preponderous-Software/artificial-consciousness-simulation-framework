"""Tests for interfaces/cli.py payload handling (#11) and TUI detach (#239).

The Rich dashboard consumes `dict[str, object]` event payloads emitted by
`Consciousness._emit()`. These tests pin the narrowing behavior of the
display handlers: a malformed payload must degrade to a default rather than
raise inside a handler, since `_emit()` isolates but also swallows handler
exceptions, which would silently freeze the displayed counters.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from interfaces.cli import INTERNAL_SPAWN_ENV, ConsciousnessCLI, _as_int


class _FakeMind:
    """Minimum surface area ConsciousnessCLI subscribes to on construction."""

    def __init__(self) -> None:
        self.on_initialized: list = []
        self.on_thought: list = []
        self.on_memory_stored: list = []
        self.on_reflection: list = []
        self.name = "Aria"
        self.config = {"llm": {"provider": "mock", "model": "mock-model"}}


def _cli() -> ConsciousnessCLI:
    return ConsciousnessCLI(_FakeMind())


def test_as_int_accepts_everything_int_already_accepted() -> None:
    assert _as_int(7, 0) == 7
    assert _as_int(7.9, 0) == 7
    assert _as_int("12", 0) == 12


def test_as_int_falls_back_where_int_would_have_raised() -> None:
    assert _as_int(None, 3) == 3
    assert _as_int({"n": 1}, 3) == 3
    assert _as_int("twelve", 3) == 3


def test_as_int_falls_back_on_non_finite_floats() -> None:
    """`json.loads` reconstructs Infinity/NaN by default, and `int()` rejects
    both — the infinities with OverflowError rather than ValueError. Either
    escaping this helper would be swallowed by `_emit()` and freeze the
    displayed counter, which is what the narrowing exists to prevent."""
    assert _as_int(float("inf"), 3) == 3
    assert _as_int(float("-inf"), 3) == 3
    assert _as_int(float("nan"), 3) == 3


def test_as_int_treats_bool_as_non_numeric() -> None:
    """`True` is an int subclass, so `int(True)` used to render as 1; reporting
    a long-term count of 1 because a payload carried a flag is worse than
    keeping the previous value."""
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


def test_on_initialized_ignores_non_iterable_short_term() -> None:
    """A non-iterable `short_term` used to raise TypeError inside the handler,
    where `_emit()` swallows it and the count silently never updates."""
    cli = _cli()
    asyncio.run(cli._on_initialized({"short_term": 5, "long_term_count": 2}))
    assert cli.thoughts == []
    assert cli.long_term_count == 2


def test_on_memory_keeps_previous_count_on_malformed_payload() -> None:
    cli = _cli()
    cli.long_term_count = 9
    asyncio.run(cli._on_memory({"long_term_count": None, "content": "a memory"}))
    assert cli.long_term_count == 9
    assert cli.memories == ["a memory"]


def test_detach_relaunches_spawn_as_an_internal_child(monkeypatch, tmp_path) -> None:
    """The child must skip spawn.py's duplicate check, or it refuses its own PID."""
    launched: list[tuple[list[str], dict]] = []

    class _FakePopen:
        pid = 4242

        def __init__(self, args: list[str], **kwargs: object) -> None:
            launched.append((args, kwargs))

    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    monkeypatch.setenv("PARENT_ENV_SENTINEL", "inherited")
    monkeypatch.delenv(INTERNAL_SPAWN_ENV, raising=False)
    monkeypatch.setattr("subprocess.Popen", _FakePopen)

    _cli()._spawn_detached_background()

    [(args, kwargs)] = launched
    assert args[0] == sys.executable
    assert Path(args[1]).name == "spawn.py"
    assert args[2:] == [
        "--name", "Aria", "--provider", "mock", "--model", "mock-model", "--headless",
    ]
    assert kwargs["env"][INTERNAL_SPAWN_ENV] == "1"
    assert kwargs["env"]["PARENT_ENV_SENTINEL"] == "inherited"
    assert (tmp_path / "Aria" / "pid").read_text() == "4242"
