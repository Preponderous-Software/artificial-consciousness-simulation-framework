"""Web dashboard backend for real-time consciousness observation.

Theory mapping — GWT-3 (global broadcast): the web server makes the
workspace broadcast externally legible to remote observers, extending
the broadcast to any HTTP client without requiring a co-located terminal.
Gap: observers are read-only; they cannot inject into the workspace
(write endpoints are out of scope for this issue).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from persistence.journal import Journal
from persistence.paths import consciousness_dir, consciousness_root

logger = logging.getLogger(__name__)

app = FastAPI(title="Consciousness Dashboard", docs_url=None, redoc_url=None)

# In-process registry: name -> Consciousness (populated by register())
_registry: dict[str, Any] = {}
# Per-instance SSE queues: name -> list of Queue (one per connected client)
_sse_queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
# Start times for in-process instances
_started_at: dict[str, str] = {}


def register(mind: Any) -> None:
    """Hook a running Consciousness instance into the live event stream."""
    name = mind.name
    _registry[name] = mind
    _started_at[name] = datetime.now(timezone.utc).isoformat()
    _sse_queues.setdefault(name, [])

    async def _broadcast(payload: dict[str, Any]) -> None:
        for q in list(_sse_queues.get(name, [])):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    mind.on_thought.append(_broadcast)
    mind.on_reflection.append(_broadcast)
    mind.on_memory_stored.append(_broadcast)
    mind.on_identity_shift.append(_broadcast)
    if hasattr(mind, "on_perception"):
        mind.on_perception.append(_broadcast)
    logger.info("Web dashboard: registered '%s' for live streaming", name)


@app.get("/instances")
async def list_instances() -> list[dict[str, Any]]:
    """Return all known consciousness instances with their current state."""
    result: list[dict[str, Any]] = []
    root = consciousness_root()
    if not root.exists():
        return result

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        state_path = d / "state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        identity = state.get("identity", {})
        # Directory name is the stable id used by /stream/{id}; identity.name
        # is just a display label that can drift (e.g. amended identity).
        dir_name = d.name
        display_name = identity.get("name", dir_name)

        pid_path = d / "pid"
        # streamable: this dashboard process can broadcast live events for it.
        # running:    a process is alive on disk (pid exists + responding).
        # online (legacy alias): true if either is true — kept for the frontend.
        streamable = dir_name in _registry
        running = streamable
        started_at: str | None = _started_at.get(dir_name)

        if not running and pid_path.exists():
            try:
                pid = int(pid_path.read_text().strip())
                os.kill(pid, 0)
                running = True
                if started_at is None:
                    started_at = datetime.fromtimestamp(
                        pid_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat()
            except (ProcessLookupError, OSError, ValueError):
                pass

        if started_at is None and state_path.exists():
            try:
                started_at = datetime.fromtimestamp(
                    state_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                pass

        result.append({
            "id": dir_name,
            "name": display_name,
            "online": running,
            "running": running,
            "streamable": streamable,
            "thought_count": state.get("thought_count", 0),
            "mood": identity.get("mood", {}),
            "self_concept": identity.get("self_concept", ""),
            "values": identity.get("values", []),
            "started_at": started_at,
        })

    return result


@app.get("/stream/{name}")
async def stream_events(name: str) -> StreamingResponse:
    """SSE stream of live thought/reflection/memory events for a named instance."""
    # Refuse unknown names so a GET cannot create a fresh directory under
    # CONSCIOUSNESS_HOME via Journal's mkdir-on-construct. Only an in-process
    # registered instance (no on-disk state yet) or a pre-existing directory
    # is valid.
    instance_dir = consciousness_dir(name)
    if name not in _registry and not instance_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown instance: {name!r}")

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
    queues = _sse_queues.setdefault(name, [])
    queues.append(queue)

    journal_path = instance_dir / "journal.jsonl"
    history = await Journal(journal_path).recent(limit=50) if journal_path.exists() else []
    # Whether this instance can deliver live events (i.e. is co-located)
    is_live = name in _registry

    async def generate() -> AsyncGenerator[str, None]:
        # Tag history events so the client can style them differently
        for entry in history:
            yield f"data: {json.dumps({**entry, '_history': True})}\n\n"

        # Signal end of history and whether live events will follow
        yield f"data: {json.dumps({'type': 'history_end', 'live': is_live})}\n\n"

        # Stream live events; heartbeat every 25 s to keep connection alive
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=25.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            try:
                queues.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the frontend SPA from the static/ subdirectory
_static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


async def start(port: int, host: str = "127.0.0.1") -> None:
    """Start the uvicorn server as a background asyncio task.

    Defaults to localhost-only — the dashboard exposes raw thought/journal
    content with no authentication, so binding 0.0.0.0 should be an opt-in.
    """
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        loop="none",   # reuse the running event loop
        access_log=False,
    )
    server = uvicorn.Server(config)
    # Prevent uvicorn from hijacking signal handlers already installed
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    asyncio.create_task(server.serve())
    logger.info("Web dashboard available at http://%s:%d", host, port)
