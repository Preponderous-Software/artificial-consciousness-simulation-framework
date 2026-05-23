"""Tests for the FastAPI web dashboard backend (interfaces/web/server.py).

Covers Copilot review comment #5 on PR #52: previously the endpoints had no
test coverage. These tests pin the /instances response schema, the
/stream/{name} SSE framing (history + history_end + heartbeat), and the
security fix from comment #1 (404 on unknown instance, no directory
creation as a side effect of a GET).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def consciousness_home(monkeypatch):
    """Isolated CONSCIOUSNESS_HOME for each test."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("CONSCIOUSNESS_HOME", tmp)
        yield Path(tmp)


@pytest.fixture
def server(consciousness_home):
    """Fresh server module per test — module-level _registry/_sse_queues are global."""
    import importlib
    import interfaces.web.server as srv
    importlib.reload(srv)
    return srv


@pytest.fixture
def client(server):
    return TestClient(server.app), server


def _seed_instance(home: Path, dir_name: str, identity_name: str | None = None, thought_count: int = 7):
    """Write a state.json + journal.jsonl that /instances expects."""
    d = home / dir_name
    d.mkdir(parents=True)
    state = {
        "identity": {
            "name": identity_name or dir_name,
            "self_concept": "I am a test fixture.",
            "values": ["curiosity"],
            "mood": {"wonder": 0.5, "contentment": 0.3},
        },
        "short_term": [],
        "thought_count": thought_count,
    }
    (d / "state.json").write_text(json.dumps(state))
    (d / "journal.jsonl").write_text(
        json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "type": "thought", "content": "first"}) + "\n"
        + json.dumps({"timestamp": "2026-01-01T00:00:30+00:00", "type": "reflection", "content": "second"}) + "\n"
    )
    return d


# ---------------------------------------------------------------------------
# /instances schema
# ---------------------------------------------------------------------------

def test_instances_empty_when_no_home(client):
    c, _ = client
    r = c.get("/instances")
    assert r.status_code == 200
    assert r.json() == []


def test_instances_lists_seeded_instance_with_full_schema(client, consciousness_home):
    c, _ = client
    _seed_instance(consciousness_home, "Aria")

    r = c.get("/instances")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    inst = body[0]
    # Required schema fields — must remain stable for the frontend
    assert inst["id"] == "Aria"
    assert inst["name"] == "Aria"
    assert inst["thought_count"] == 7
    assert inst["mood"] == {"wonder": 0.5, "contentment": 0.3}
    assert inst["self_concept"] == "I am a test fixture."
    assert inst["values"] == ["curiosity"]
    # Online/streamable distinction (Copilot comment #3)
    assert inst["running"] is False        # no live pid + not in registry
    assert inst["streamable"] is False
    assert inst["online"] == inst["running"]  # legacy alias


def test_instances_separates_id_from_display_name(client, consciousness_home):
    """Copilot comment #2: directory name (id) and identity.name can differ."""
    c, _ = client
    # The directory is "alpha-7" but the agent has renamed itself "Alpha".
    _seed_instance(consciousness_home, "alpha-7", identity_name="Alpha")

    body = c.get("/instances").json()
    assert len(body) == 1
    assert body[0]["id"] == "alpha-7"     # stable, used for /stream/{id}
    assert body[0]["name"] == "Alpha"     # display label


def test_instances_skips_directories_without_state_json(client, consciousness_home):
    c, _ = client
    (consciousness_home / "no_state_here").mkdir()
    (consciousness_home / "no_state_here" / "journal.jsonl").write_text("")
    assert c.get("/instances").json() == []


def test_instances_streamable_true_when_registered(client, consciousness_home):
    """An in-process registered instance is streamable even without on-disk state yet."""
    c, srv = client
    _seed_instance(consciousness_home, "Beta")

    mind = type("M", (), {
        "name": "Beta",
        "on_thought": [], "on_reflection": [],
        "on_memory_stored": [], "on_identity_shift": [],
    })()
    srv.register(mind)

    inst = next(i for i in c.get("/instances").json() if i["id"] == "Beta")
    assert inst["streamable"] is True
    assert inst["running"] is True


# ---------------------------------------------------------------------------
# /stream/{name} — Copilot comments #1 (no dir creation) and #5 (SSE framing)
# ---------------------------------------------------------------------------

def test_stream_404s_on_unknown_instance_and_does_not_create_directory(client, consciousness_home):
    """Copilot comment #1: a GET on an unknown name must not create a directory."""
    c, _ = client
    assert not (consciousness_home / "Nobody").exists()

    r = c.get("/stream/Nobody")
    assert r.status_code == 404
    assert not (consciousness_home / "Nobody").exists(), \
        "stream_events created a directory for an unknown instance — comment #1 regression"


async def _collect_sse(server_module, name: str, max_events: int):
    """Drive the SSE handler's generator directly, capture N events, then close.

    Going through httpx/ASGI hangs on shutdown because the generator's
    `await wait_for(queue.get(), 25)` never sees the disconnect event.
    Invoking the route handler directly and iterating body_iterator gives
    deterministic cancellation: closing the async generator (via `aclose`)
    raises CancelledError inside wait_for, which the server's `finally:`
    catches and cleans up the queue subscription.
    """
    response = await server_module.stream_events(name)
    events: list[dict] = []
    buffer = b""
    gen = response.body_iterator
    try:
        async for chunk in gen:
            if isinstance(chunk, str):
                chunk = chunk.encode()
            buffer += chunk
            while b"\n\n" in buffer:
                frame, buffer = buffer.split(b"\n\n", 1)
                for line in frame.splitlines():
                    if line.startswith(b"data: "):
                        events.append(json.loads(line[6:].decode()))
                        if len(events) >= max_events:
                            return events
    finally:
        await gen.aclose()
    return events


def test_stream_emits_history_then_history_end(server, consciousness_home):
    """Copilot comment #5: /stream must replay history then emit history_end."""
    _seed_instance(consciousness_home, "Gamma")
    events = asyncio.run(_collect_sse(server, "Gamma", max_events=3))

    assert len(events) == 3
    assert events[0]["type"] == "thought"   and events[0]["_history"] is True
    assert events[1]["type"] == "reflection" and events[1]["_history"] is True
    assert events[2] == {"type": "history_end", "live": False}


def test_stream_marks_live_when_instance_registered(server, consciousness_home):
    _seed_instance(consciousness_home, "Delta")
    mind = type("M", (), {
        "name": "Delta",
        "on_thought": [], "on_reflection": [],
        "on_memory_stored": [], "on_identity_shift": [],
    })()
    server.register(mind)

    events = asyncio.run(_collect_sse(server, "Delta", max_events=3))
    assert events[-1] == {"type": "history_end", "live": True}


def test_stream_handles_fresh_registered_instance_with_no_journal(server):
    """Registered in-process but no journal.jsonl yet — must still stream cleanly."""
    mind = type("M", (), {
        "name": "Echo",
        "on_thought": [], "on_reflection": [],
        "on_memory_stored": [], "on_identity_shift": [],
    })()
    server.register(mind)

    events = asyncio.run(_collect_sse(server, "Echo", max_events=1))
    assert events == [{"type": "history_end", "live": True}]


def test_register_hooks_on_perception_when_present(server):
    """Issue #53: perception events must broadcast through the SSE pipeline."""
    perception_handlers: list = []
    mind = type("M", (), {
        "name": "Foxtrot",
        "on_thought": [], "on_reflection": [],
        "on_memory_stored": [], "on_identity_shift": [],
        "on_perception": perception_handlers,
    })()
    server.register(mind)

    # The registered _broadcast was appended to on_perception
    assert len(perception_handlers) == 1


def test_register_tolerates_minds_without_on_perception(server):
    """Older Consciousness builds without on_perception still register fine."""
    mind = type("M", (), {
        "name": "Golf",
        "on_thought": [], "on_reflection": [],
        "on_memory_stored": [], "on_identity_shift": [],
        # no on_perception
    })()
    # Must not raise AttributeError
    server.register(mind)
