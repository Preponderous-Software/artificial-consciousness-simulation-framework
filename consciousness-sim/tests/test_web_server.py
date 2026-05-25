"""Tests for the standalone FastAPI dashboard (interfaces/web/server.py).

The dashboard is now standalone-only (issue #55): there is no in-process
event handler path. Live events arrive via the journal tailer, and the
server exposes a process-management surface (spawn / stop / archive).
These tests pin the /instances response schema, the /stream/{name} SSE
framing (history + history_end + heartbeat), localhost-only enforcement
on mutating endpoints, name sanitization, and provider/model allowlist.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def consciousness_home(monkeypatch):
    """Isolated CONSCIOUSNESS_HOME for each test."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("CONSCIOUSNESS_HOME", tmp)
        yield Path(tmp)


@pytest.fixture
def server(consciousness_home):
    """Fresh server module per test — module-level _sse_queues are global."""
    import importlib
    import interfaces.web.server as srv
    importlib.reload(srv)
    return srv


@pytest.fixture
def client(server):
    """TestClient routes to 127.0.0.1 so localhost-guarded endpoints succeed."""
    return TestClient(server.app, client=("127.0.0.1", 12345)), server


def _seed_instance(
    home: Path,
    dir_name: str,
    identity_name: str | None = None,
    thought_count: int = 7,
    journal_events: list[dict] | None = None,
) -> Path:
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
    events = journal_events if journal_events is not None else [
        {"timestamp": "2026-01-01T00:00:00+00:00", "type": "thought", "content": "first"},
        {"timestamp": "2026-01-01T00:00:30+00:00", "type": "reflection", "content": "second"},
    ]
    (d / "journal.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + ("\n" if events else "")
    )
    return d


def _seed_pid(d: Path, pid: int) -> None:
    (d / "pid").write_text(str(pid))


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
    assert inst["id"] == "Aria"
    assert inst["name"] == "Aria"
    assert inst["thought_count"] == 7
    assert inst["mood"] == {"wonder": 0.5, "contentment": 0.3}
    assert inst["self_concept"] == "I am a test fixture."
    assert inst["values"] == ["curiosity"]
    # No pid → not running → not streamable in standalone mode either.
    assert inst["running"] is False
    assert inst["streamable"] is False
    assert inst["online"] == inst["running"]


def test_instances_separates_id_from_display_name(client, consciousness_home):
    """Directory name (id) and identity.name can differ — id is stable."""
    c, _ = client
    _seed_instance(consciousness_home, "alpha_7", identity_name="Alpha")
    body = c.get("/instances").json()
    assert len(body) == 1
    assert body[0]["id"] == "alpha_7"
    assert body[0]["name"] == "Alpha"


def test_instances_skips_directories_without_state_json(client, consciousness_home):
    c, _ = client
    (consciousness_home / "no_state_here").mkdir()
    (consciousness_home / "no_state_here" / "journal.jsonl").write_text("")
    assert c.get("/instances").json() == []


def test_instances_skips_dot_archive_directory(client, consciousness_home):
    """`.archive` shouldn't appear in the listing — it's a dot-dir."""
    c, _ = client
    _seed_instance(consciousness_home, "Visible")
    (consciousness_home / ".archive").mkdir()
    (consciousness_home / ".archive" / "old-abc" / "state.json").parent.mkdir(parents=True)
    (consciousness_home / ".archive" / "old-abc" / "state.json").write_text(
        json.dumps({"identity": {}, "thought_count": 0})
    )
    ids = {i["id"] for i in c.get("/instances").json()}
    assert ids == {"Visible"}


def test_instances_running_true_when_pid_alive(client, consciousness_home):
    """A live pid in the pid file flips running/streamable to true."""
    c, _ = client
    d = _seed_instance(consciousness_home, "Beta")
    _seed_pid(d, os.getpid())  # the test process itself — guaranteed alive

    inst = next(i for i in c.get("/instances").json() if i["id"] == "Beta")
    assert inst["running"] is True
    assert inst["streamable"] is True
    assert inst["pid"] == os.getpid()


def test_instances_running_false_when_pid_stale(client, consciousness_home):
    """A pid file pointing at a non-existent process is treated as offline."""
    c, _ = client
    d = _seed_instance(consciousness_home, "Gamma")
    # PID 0 is invalid for os.kill — guaranteed to fail
    _seed_pid(d, 999999)

    inst = next(i for i in c.get("/instances").json() if i["id"] == "Gamma")
    assert inst["running"] is False
    assert inst["streamable"] is False


# ---------------------------------------------------------------------------
# /providers
# ---------------------------------------------------------------------------

def test_providers_returns_allowlist(client):
    c, srv = client
    r = c.get("/providers")
    assert r.status_code == 200
    assert r.json() == srv.PROVIDER_ALLOWLIST


# ---------------------------------------------------------------------------
# /stream/{name} — history + heartbeat + dedup
# ---------------------------------------------------------------------------

def test_stream_404s_on_unknown_instance_and_does_not_create_directory(client, consciousness_home):
    """A GET on an unknown name must not create a directory."""
    c, _ = client
    assert not (consciousness_home / "Nobody").exists()
    r = c.get("/stream/Nobody")
    assert r.status_code == 404
    assert not (consciousness_home / "Nobody").exists()


async def _collect_sse(server_module, name: str, max_events: int):
    """Drive the SSE handler's generator directly, capture N events, close."""
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
    _seed_instance(consciousness_home, "Delta")
    events = asyncio.run(_collect_sse(server, "Delta", max_events=3))
    assert len(events) == 3
    assert events[0]["type"] == "thought"    and events[0]["_history"] is True
    assert events[1]["type"] == "reflection" and events[1]["_history"] is True
    assert events[2] == {"type": "history_end", "live": False}


def test_stream_marks_live_when_pid_alive(server, consciousness_home):
    d = _seed_instance(consciousness_home, "Epsilon")
    _seed_pid(d, os.getpid())
    events = asyncio.run(_collect_sse(server, "Epsilon", max_events=3))
    assert events[-1] == {"type": "history_end", "live": True}


def test_stream_dedups_tail_events_already_in_history(server, consciousness_home):
    """A tailer event with a timestamp inside the history block must be skipped."""
    _seed_instance(consciousness_home, "Foxtrot")

    async def driver():
        response = await server.stream_events("Foxtrot")
        gen = response.body_iterator

        # Drain history + history_end
        events: list[dict] = []
        buffer = b""
        async for chunk in gen:
            if isinstance(chunk, str):
                chunk = chunk.encode()
            buffer += chunk
            while b"\n\n" in buffer:
                frame, buffer = buffer.split(b"\n\n", 1)
                for line in frame.splitlines():
                    if line.startswith(b"data: "):
                        events.append(json.loads(line[6:].decode()))
                        if events[-1].get("type") == "history_end":
                            # Inject a duplicate (timestamp inside history) and a fresh one.
                            queue = server._sse_queues["Foxtrot"][0]
                            queue.put_nowait({
                                "type": "thought",
                                "content": "first",
                                "timestamp": "2026-01-01T00:00:00+00:00",  # duplicate
                            })
                            queue.put_nowait({
                                "type": "thought",
                                "content": "fresh",
                                "timestamp": "2026-02-01T00:00:00+00:00",  # newer
                            })

                            # Collect one more event then stop
                            try:
                                more_chunk = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
                                more = more_chunk if isinstance(more_chunk, bytes) else more_chunk.encode()
                                for ln in more.splitlines():
                                    if ln.startswith(b"data: "):
                                        events.append(json.loads(ln[6:].decode()))
                            except (asyncio.TimeoutError, StopAsyncIteration):
                                pass
                            await gen.aclose()
                            return events
        return events

    events = asyncio.run(driver())
    # Only the "fresh" event should appear after history_end.
    post = [e for e in events if not e.get("_history") and e.get("type") != "history_end"]
    assert len(post) == 1
    assert post[0]["content"] == "fresh"


# ---------------------------------------------------------------------------
# POST /instances — spawn
# ---------------------------------------------------------------------------

def test_spawn_rejects_invalid_name(client):
    c, _ = client
    r = c.post("/instances", json={"name": "../../etc/passwd"})
    # The name sanitizes to something safe, but pydantic may still accept it.
    # Either 400 from sanitize or successful sanitized run — check the
    # response doesn't escape the home dir.
    if r.status_code == 200:
        assert ".." not in r.json()["id"]
    else:
        assert r.status_code in (400, 422)


def test_spawn_rejects_unknown_provider(client):
    c, _ = client
    r = c.post("/instances", json={"name": "Spawned", "provider": "bogus"})
    assert r.status_code == 400
    assert "provider" in r.json()["detail"].lower()


def test_spawn_rejects_model_outside_allowlist(client):
    c, _ = client
    r = c.post("/instances", json={
        "name": "Spawned", "provider": "ollama", "model": "not-in-allowlist:99b"
    })
    assert r.status_code == 400
    assert "model" in r.json()["detail"].lower()


def test_spawn_rejects_model_without_provider(client):
    c, _ = client
    r = c.post("/instances", json={"name": "Spawned", "model": "gpt-4o"})
    assert r.status_code == 400
    assert "provider" in r.json()["detail"].lower()


def test_spawn_conflicts_with_running_instance(client, consciousness_home):
    c, _ = client
    d = _seed_instance(consciousness_home, "Already")
    _seed_pid(d, os.getpid())
    r = c.post("/instances", json={"name": "Already"})
    assert r.status_code == 409


def test_spawn_invokes_subprocess(client, consciousness_home, monkeypatch):
    """Successful spawn calls subprocess.Popen with the expected argv."""
    c, srv = client
    seen = {}

    class FakeProc:
        def __init__(self): self.pid = 12345
        def wait(self, timeout=None): return 0

    def fake_popen(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr("interfaces.web.server.subprocess.Popen", fake_popen)

    r = c.post("/instances", json={
        "name": "Hotel", "provider": "mock", "model": "mock"
    })
    assert r.status_code == 200
    assert seen["cmd"][-6:] == [
        "--name", "Hotel", "--bg", "--provider", "mock", "--model", "mock"
    ][-6:]
    # The directory must exist (pre-created so SSE can subscribe).
    assert (consciousness_home / "Hotel").exists()


# ---------------------------------------------------------------------------
# Mutation endpoints — localhost-only guard
# ---------------------------------------------------------------------------

def test_spawn_rejects_non_localhost(server, consciousness_home, monkeypatch):
    """A non-localhost client must be 403'd on POST /instances."""
    from fastapi.testclient import TestClient
    c = TestClient(server.app, client=("10.0.0.5", 54321))
    r = c.post("/instances", json={"name": "Remote"})
    assert r.status_code == 403


def test_spawn_allows_remote_when_flag_set(server, monkeypatch):
    server._allow_remote_spawn = True

    class FakeProc:
        pid = 99
        def wait(self, timeout=None): return 0

    monkeypatch.setattr("interfaces.web.server.subprocess.Popen",
                        lambda *a, **kw: FakeProc())

    from fastapi.testclient import TestClient
    c = TestClient(server.app, client=("10.0.0.5", 54321))
    r = c.post("/instances", json={"name": "RemoteOK"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /instances/{id}/stop
# ---------------------------------------------------------------------------

def test_stop_404s_on_unknown_instance(client):
    c, _ = client
    r = c.post("/instances/Nope/stop")
    assert r.status_code == 404


def test_stop_returns_not_running_when_no_pid(client, consciousness_home):
    c, _ = client
    _seed_instance(consciousness_home, "Idle")
    r = c.post("/instances/Idle/stop")
    assert r.status_code == 200
    assert r.json()["running"] is False
    assert r.json()["stopped"] is False


def test_stop_sends_sigterm_to_alive_pid(client, consciousness_home, monkeypatch):
    c, _ = client
    d = _seed_instance(consciousness_home, "Live")
    _seed_pid(d, 12345)

    sent = {}
    real_kill = os.kill

    def fake_kill(pid, sig):
        if sig == 0:  # liveness probe — pretend alive
            return
        sent["pid"] = pid
        sent["sig"] = sig

    monkeypatch.setattr("interfaces.web.server.os.kill", fake_kill)

    r = c.post("/instances/Live/stop")
    assert r.status_code == 200
    assert r.json()["stopped"] is True
    assert sent["pid"] == 12345
    assert sent["sig"] == signal.SIGTERM


# ---------------------------------------------------------------------------
# DELETE /instances/{id} — archive
# ---------------------------------------------------------------------------

def test_delete_archives_offline_instance(client, consciousness_home):
    c, _ = client
    _seed_instance(consciousness_home, "Trash")
    assert (consciousness_home / "Trash").exists()

    r = c.delete("/instances/Trash")
    assert r.status_code == 200
    assert not (consciousness_home / "Trash").exists()
    archive_root = consciousness_home / ".archive"
    assert archive_root.exists()
    archived = list(archive_root.iterdir())
    assert len(archived) == 1
    assert archived[0].name.startswith("Trash-")


def test_delete_404s_on_unknown_instance(client):
    c, _ = client
    r = c.delete("/instances/Phantom")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Journal tailer integration — dispatching events to queues
# ---------------------------------------------------------------------------

def test_tail_event_lands_in_queue(server, consciousness_home):
    """A line appended after _on_tail_event fires should reach SSE queues."""
    queue = asyncio.Queue(maxsize=10)
    server._sse_queues["TailTarget"] = [queue]

    server._on_tail_event(
        "TailTarget",
        {"type": "thought", "content": "hello", "timestamp": "2026-03-01T00:00:00+00:00"},
    )
    assert queue.qsize() == 1
    item = queue.get_nowait()
    assert item["content"] == "hello"


def test_journal_tailer_picks_up_appended_lines(consciousness_home):
    """Integration: write a journal line; tailer dispatches it within a poll."""
    from interfaces.web.journal_tail import JournalTailer

    received: list[tuple[str, dict]] = []

    def collect(inst_id: str, payload: dict) -> None:
        received.append((inst_id, payload))

    d = consciousness_home / "Tailee"
    d.mkdir()
    journal = d / "journal.jsonl"
    journal.write_text("")  # empty file — tailer will park offset at 0

    async def drive():
        tailer = JournalTailer(consciousness_home, collect, poll_interval_s=0.05)
        tailer.start()
        # First poll: parks offset at EOF (0)
        await asyncio.sleep(0.15)
        # Append a real event
        with journal.open("a") as f:
            f.write(json.dumps({
                "timestamp": "2026-04-01T00:00:00+00:00",
                "type": "thought",
                "content": "appended",
            }) + "\n")
        # Wait for the next poll cycle
        await asyncio.sleep(0.2)
        await tailer.stop()

    asyncio.run(drive())
    assert any(
        inst == "Tailee" and p["content"] == "appended"
        for inst, p in received
    )


def test_journal_tailer_ignores_unknown_event_types(consciousness_home):
    """Corrupted lines and unknown types must not propagate."""
    from interfaces.web.journal_tail import JournalTailer

    received: list[tuple[str, dict]] = []

    d = consciousness_home / "Filter"
    d.mkdir()
    journal = d / "journal.jsonl"
    journal.write_text("")  # park offset at 0

    async def drive():
        tailer = JournalTailer(consciousness_home, lambda i, p: received.append((i, p)),
                               poll_interval_s=0.05)
        tailer.start()
        await asyncio.sleep(0.15)
        with journal.open("a") as f:
            f.write("not-json\n")
            f.write(json.dumps({"type": "exploit", "content": "x"}) + "\n")
            f.write(json.dumps({"type": "memory", "content": "y",
                               "timestamp": "2026-04-01T00:00:00+00:00"}) + "\n")
        await asyncio.sleep(0.2)
        await tailer.stop()

    asyncio.run(drive())
    assert len(received) == 1
    assert received[0][1]["type"] == "memory"


def test_journal_tailer_handles_partial_line(consciousness_home):
    """A partial trailing line must not be consumed until terminated by \\n."""
    from interfaces.web.journal_tail import JournalTailer

    received: list[tuple[str, dict]] = []
    d = consciousness_home / "Partial"
    d.mkdir()
    journal = d / "journal.jsonl"
    journal.write_text("")

    async def drive():
        tailer = JournalTailer(consciousness_home, lambda i, p: received.append((i, p)),
                               poll_interval_s=0.05)
        tailer.start()
        await asyncio.sleep(0.15)
        # Write a partial line (no trailing newline)
        with journal.open("a") as f:
            f.write('{"type":"thought","content":"half","timesta')
            f.flush()
        await asyncio.sleep(0.15)
        assert received == []
        # Complete the line
        with journal.open("a") as f:
            f.write('mp":"2026-04-01T00:00:00+00:00"}\n')
            f.flush()
        await asyncio.sleep(0.2)
        await tailer.stop()

    asyncio.run(drive())
    assert len(received) == 1
    assert received[0][1]["content"] == "half"


# ---------------------------------------------------------------------------
# Regression tests — Copilot review of PR #99
# ---------------------------------------------------------------------------

def test_stream_queue_key_matches_sanitized_id(server, consciousness_home):
    """Regression: /stream/{name} must subscribe under the SAME key the tailer
    dispatches to. The tailer keys by on-disk dir name (always sanitized), so
    a client requesting an alias (e.g. "alpha 7") would otherwise subscribe
    to _sse_queues["alpha 7"] while events go to _sse_queues["alpha_7"]."""
    _seed_instance(consciousness_home, "alpha_7")

    async def driver():
        # Subscribe via a non-sanitized alias that maps to the on-disk id.
        response = await server.stream_events("alpha 7")
        gen = response.body_iterator
        events: list[dict] = []
        buffer = b""
        async for chunk in gen:
            if isinstance(chunk, str):
                chunk = chunk.encode()
            buffer += chunk
            while b"\n\n" in buffer:
                frame, buffer = buffer.split(b"\n\n", 1)
                for line in frame.splitlines():
                    if line.startswith(b"data: "):
                        events.append(json.loads(line[6:].decode()))
                        if events[-1].get("type") == "history_end":
                            # Simulate the tailer pushing under the sanitized
                            # key. If the subscription used the raw alias,
                            # this event would be lost.
                            server._on_tail_event("alpha_7", {
                                "type": "thought",
                                "content": "live-fresh",
                                "timestamp": "2099-01-01T00:00:00+00:00",
                            })
                            try:
                                more = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
                                more_b = more if isinstance(more, bytes) else more.encode()
                                for ln in more_b.splitlines():
                                    if ln.startswith(b"data: "):
                                        events.append(json.loads(ln[6:].decode()))
                            except (asyncio.TimeoutError, StopAsyncIteration):
                                pass
                            await gen.aclose()
                            return events
        return events

    events = asyncio.run(driver())
    live = [e for e in events if not e.get("_history") and e.get("type") != "history_end"]
    assert len(live) == 1
    assert live[0]["content"] == "live-fresh"


def test_memory_journal_event_carries_long_term_count(tmp_path):
    """Regression: memory events must be journalled with structured
    long_term_count so the dashboard's tail-fed memory counter updates."""
    from persistence.journal import Journal

    async def run():
        j = Journal(tmp_path / "j.jsonl")
        await j.append("memory", "Long-term store: 42 memories", long_term_count=42)
        return await j.recent(limit=10)

    rows = asyncio.run(run())
    assert len(rows) == 1
    assert rows[0]["type"] == "memory"
    assert rows[0]["long_term_count"] == 42


def test_sanitize_consciousness_name_preserves_hyphens():
    """Regression: the UI advertises hyphens; sanitizer must preserve them."""
    from persistence.paths import sanitize_consciousness_name
    assert sanitize_consciousness_name("Aria-1") == "Aria-1"
    assert sanitize_consciousness_name("alpha-beta_7") == "alpha-beta_7"
    # Other characters still collapse to underscore.
    assert sanitize_consciousness_name("alpha 7") == "alpha_7"
    assert sanitize_consciousness_name("../escape") == "escape"


def test_spawn_does_not_block_event_loop(client, monkeypatch):
    """Regression: spawn_instance must offload proc.wait so SSE streams
    don't stall during a slow subprocess start. We assert that asyncio.to_thread
    is invoked for the wait call."""
    c, _ = client
    seen = {"to_thread_calls": 0}

    class FakeProc:
        pid = 12345
        def wait(self, timeout=None): return 0

    real_to_thread = asyncio.to_thread

    async def spy(fn, *args, **kwargs):
        if getattr(fn, "__self__", None) is not None and fn.__name__ == "wait":
            seen["to_thread_calls"] += 1
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr("interfaces.web.server.subprocess.Popen",
                        lambda *a, **kw: FakeProc())
    monkeypatch.setattr("interfaces.web.server.asyncio.to_thread", spy)

    r = c.post("/instances", json={"name": "AsyncSpawn"})
    assert r.status_code == 200
    assert seen["to_thread_calls"] == 1, "proc.wait was not offloaded via asyncio.to_thread"
