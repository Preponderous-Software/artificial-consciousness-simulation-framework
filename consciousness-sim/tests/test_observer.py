"""Tests for interfaces/observer.py — the passive read-only observer stream.

Observer had zero dedicated test coverage prior to this file despite being
the read-side implementation of GWT-3 (global broadcast) referenced in its
own theory-mapping docstring. Tests exercise subscription filtering,
handler registration against a Consciousness-shaped object, and both
output formats — no LLM/network calls are needed since Observer only
consumes already-built event payloads.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from interfaces.observer import Observer, _ALL_EVENTS


class _FakeMind:
    """Minimum surface area Observer.attach() reads from Consciousness."""

    def __init__(self) -> None:
        self.on_thought: list = []
        self.on_reflection: list = []
        self.on_memory_stored: list = []
        self.on_identity_shift: list = []


def test_default_subscribe_is_all_event_types() -> None:
    observer = Observer()
    assert observer.subscribe == _ALL_EVENTS


def test_custom_subscribe_is_stored_as_given() -> None:
    observer = Observer(subscribe=("thought", "memory"))
    assert observer.subscribe == ("thought", "memory")


def test_attach_registers_handle_on_all_subscribed_channels() -> None:
    mind = _FakeMind()
    observer = Observer()
    observer.attach(mind)
    assert mind.on_thought == [observer.handle]
    assert mind.on_reflection == [observer.handle]
    assert mind.on_memory_stored == [observer.handle]
    assert mind.on_identity_shift == [observer.handle]


def test_attach_only_registers_subscribed_channels() -> None:
    mind = _FakeMind()
    observer = Observer(subscribe=("thought", "memory"))
    observer.attach(mind)
    assert mind.on_thought == [observer.handle]
    assert mind.on_memory_stored == [observer.handle]
    assert mind.on_reflection == []
    assert mind.on_identity_shift == []


def test_attach_skips_channel_missing_from_consciousness_object() -> None:
    """A Consciousness-like object lacking an on_* attribute must not raise."""
    mind = SimpleNamespace(on_thought=[])
    observer = Observer()
    observer.attach(mind)
    assert mind.on_thought == [observer.handle]


def test_handle_json_format_prints_raw_json(capsys: pytest.CaptureFixture[str]) -> None:
    observer = Observer(output_format="json")
    payload = {"type": "thought", "content": "hello"}
    asyncio.run(observer.handle(payload))
    out = capsys.readouterr().out
    assert json.loads(out.strip()) == payload


def test_handle_rich_format_prints_type_and_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    observer = Observer(output_format="rich")
    asyncio.run(observer.handle({"type": "reflection", "content": "considering"}))
    out = capsys.readouterr().out
    assert "reflection" in out
    assert "considering" in out


def test_handle_ignores_event_type_not_in_subscribe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    observer = Observer(subscribe=("thought",), output_format="json")
    asyncio.run(observer.handle({"type": "memory", "content": "unsubscribed"}))
    out = capsys.readouterr().out
    assert out == ""


def test_handle_passes_subscribed_event_type(
    capsys: pytest.CaptureFixture[str],
) -> None:
    observer = Observer(subscribe=("thought",), output_format="json")
    payload = {"type": "thought", "content": "subscribed"}
    asyncio.run(observer.handle(payload))
    out = capsys.readouterr().out
    assert json.loads(out.strip()) == payload


def test_handle_with_empty_subscribe_tuple_passes_everything(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Characterizes current behavior: an empty subscribe tuple is falsy, so
    the `if self.subscribe and event_type not in self.subscribe` guard never
    filters — every event type is handled rather than none."""
    observer = Observer(subscribe=(), output_format="json")
    payload = {"type": "memory", "content": "anything"}
    asyncio.run(observer.handle(payload))
    out = capsys.readouterr().out
    assert json.loads(out.strip()) == payload
