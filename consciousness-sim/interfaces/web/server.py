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
from pathlib import Path
from typing import Any

from fastapi import FastAPI
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


def register(mind: Any) -> None:
    """Hook a running Consciousness instance into the live event stream."""
    name = mind.name
    _registry[name] = mind
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

        pid_path = d / "pid"
        online = False
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text().strip())
                os.kill(pid, 0)
                online = True
            except (ProcessLookupError, OSError, ValueError):
                pass
        # Instances registered in this process are always online
        identity = state.get("identity", {})
        name = identity.get("name", d.name)
        if name in _registry:
            online = True

        result.append({
            "name": name,
            "online": online,
            "thought_count": state.get("thought_count", 0),
            "mood": identity.get("mood", {}),
            "self_concept": identity.get("self_concept", ""),
            "values": identity.get("values", []),
        })

    return result


@app.get("/stream/{name}")
async def stream_events(name: str) -> StreamingResponse:
    """SSE stream of live thought/reflection/memory events for a named instance."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
    queues = _sse_queues.setdefault(name, [])
    queues.append(queue)

    journal = Journal(consciousness_dir(name) / "journal.jsonl")
    history = await journal.recent(limit=50)

    async def generate() -> AsyncGenerator[str, None]:
        # Replay journal history so the client has context immediately
        for entry in history:
            yield f"data: {json.dumps(entry)}\n\n"

        # Stream live events; heartbeat every 25 s to keep the connection alive
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


async def start(port: int) -> None:
    """Start the uvicorn server as a background asyncio task."""
    import uvicorn

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        loop="none",   # reuse the running event loop
        access_log=False,
    )
    server = uvicorn.Server(config)
    # install=False so uvicorn doesn't hijack signal handlers
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    asyncio.create_task(server.serve())
    logger.info("Web dashboard available at http://localhost:%d", port)
