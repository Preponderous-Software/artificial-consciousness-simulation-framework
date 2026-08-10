"""Standalone web dashboard for the consciousness simulation framework.

Theory mapping — GWT-3 (global broadcast): the dashboard is the external
broadcast surface. It owns no consciousness itself; it discovers running
instances by scanning CONSCIOUSNESS_HOME, tails their append-only journals,
and acts as a process manager (spawn / stop / archive). Decoupling the
dashboard from any single agent makes the broadcast horizontal — many
instances feed one observer (issue #55).

Architecture: this server is always standalone (launched via
``scripts/web.py``). Live events arrive exclusively through the journal
tailer; there is no in-process event handler path. POST and DELETE
endpoints (spawn / stop / archive) are localhost-only by default to
prevent remote process control of the host.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from interfaces.web.journal_tail import JournalTailer
from persistence.journal import Journal
from persistence.paths import (
    consciousness_dir,
    consciousness_root,
    sanitize_consciousness_name,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_tailer().start()
    logger.info("Web dashboard ready (standalone mode, journal tail active)")
    try:
        yield
    finally:
        if _tailer is not None:
            await _tailer.stop()


app = FastAPI(
    title="Consciousness Dashboard",
    docs_url=None,
    redoc_url=None,
    lifespan=_lifespan,
)

# Per-instance SSE queues — populated by the journal tailer, drained by SSE
# clients in stream_events().
_sse_queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

# Per-instance SSE delivery counters (issue #94). A slow browser tab can
# silently miss events for an entire run with no operator-visible signal;
# these make the divergence between "what happened" and "what the dashboard
# showed" observable via /instances and the throttled WARNING below.
_sse_events_total: dict[str, int] = {}
_sse_drops_total: dict[str, int] = {}
_drops_since_last_warn: dict[str, int] = {}
# Sentinel: -inf guarantees the first drop's `now - last >= 60.0` check fires
# regardless of `time.monotonic()`'s origin (mirrors interfaces/discord/webhook.py).
_last_drop_warning_at: dict[str, float] = {}

# Provider/model allowlist surfaced to the UI and enforced on POST /instances.
# Keep this tight — any value here becomes a runnable provider/model on the
# host. New options should be added intentionally.
PROVIDER_ALLOWLIST: dict[str, list[str]] = {
    "ollama": ["llama3.2:3b", "llama3.1:8b", "mistral:7b"],
    "anthropic": [
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ],
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "mock": ["mock"],
}

# Allow non-localhost mutations only when explicitly enabled at startup.
_allow_remote_spawn = False

_tailer: JournalTailer | None = None


def _on_tail_event(instance_id: str, payload: dict[str, Any]) -> None:
    """Dispatch a tailed journal event to every SSE subscriber for this instance."""
    queues = _sse_queues.get(instance_id)
    if not queues:
        return
    for q in list(queues):
        try:
            q.put_nowait(payload)
            _sse_events_total[instance_id] = _sse_events_total.get(instance_id, 0) + 1
        except asyncio.QueueFull:
            # Slow consumer — drop the event rather than back-pressure the tailer.
            _record_drop(instance_id)


def _record_drop(instance_id: str) -> None:
    """Count a dropped SSE event and throttle the warning to once per minute per instance."""
    _sse_drops_total[instance_id] = _sse_drops_total.get(instance_id, 0) + 1
    _drops_since_last_warn[instance_id] = _drops_since_last_warn.get(instance_id, 0) + 1
    now = time.monotonic()
    last = _last_drop_warning_at.get(instance_id, -math.inf)
    if now - last >= 60.0:
        logger.warning(
            "SSE queue full for %s — dropped %d event(s) in the last minute",
            instance_id, _drops_since_last_warn[instance_id],
        )
        _drops_since_last_warn[instance_id] = 0
        _last_drop_warning_at[instance_id] = now


def _is_localhost(request: Request) -> bool:
    """Approximate localhost-only guard for mutating endpoints.

    Trusts request.client.host directly — uvicorn populates it from the TCP
    peer, not from forwarded headers, so this isn't spoofable via X-Forwarded-For.
    """
    client = request.client
    if client is None:
        return False
    host = client.host
    return host in ("127.0.0.1", "::1", "localhost")


def _require_local(request: Request) -> None:
    if _allow_remote_spawn or _is_localhost(request):
        return
    raise HTTPException(
        status_code=403,
        detail="Process-management endpoints are localhost-only. "
               "Re-run scripts/web.py with --allow-remote-spawn to override (no auth — opt-in).",
    )


def _pid_alive(pid_path: Path) -> int | None:
    """Return the live PID from a pid file, or None if missing/stale."""
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ProcessLookupError, OSError, ValueError):
        return None


@app.get("/instances")
async def list_instances() -> list[dict[str, Any]]:
    """Return all known consciousness instances with their current state."""
    result: list[dict[str, Any]] = []
    root = consciousness_root()
    if not root.exists():
        return result

    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        state_path = d / "state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        identity = state.get("identity", {})
        dir_name = d.name
        display_name = identity.get("name", dir_name)
        pid = _pid_alive(d / "pid")
        running = pid is not None
        started_at: str | None = None
        if running:
            try:
                started_at = datetime.fromtimestamp(
                    (d / "pid").stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                pass
        if started_at is None:
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
            # Streamable === running in standalone mode (the tailer can deliver
            # live events for any running instance with a journal). Retained as
            # a separate field so the existing frontend keeps working.
            "streamable": running,
            "pid": pid,
            "thought_count": state.get("thought_count", 0),
            "mood": identity.get("mood", {}),
            "self_concept": identity.get("self_concept", ""),
            "values": identity.get("values", []),
            "started_at": started_at,
            "sse_events_total": _sse_events_total.get(dir_name, 0),
            "sse_drops_total": _sse_drops_total.get(dir_name, 0),
            "sse_clients": len(_sse_queues.get(dir_name, [])),
        })

    return result


@app.get("/providers")
async def list_providers() -> dict[str, list[str]]:
    """Return the provider/model allowlist used by the spawn UI."""
    return PROVIDER_ALLOWLIST


@app.get("/stream/{name}")
async def stream_events(name: str) -> StreamingResponse:
    """SSE stream of live thought/reflection/memory events for a named instance."""
    # Sanitize once and use the safe id consistently: the tailer dispatches
    # events keyed by the on-disk directory name (always sanitized), so a
    # client requesting an alias like "alpha 7" would otherwise subscribe to
    # _sse_queues["alpha 7"] while the tailer pushes to _sse_queues["alpha_7"]
    # and live events would never reach the client (Copilot review on #99).
    try:
        safe_id = sanitize_consciousness_name(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown instance: {name!r}")
    instance_dir = consciousness_root() / safe_id
    if not instance_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown instance: {name!r}")

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
    queues = _sse_queues.setdefault(safe_id, [])
    queues.append(queue)

    journal_path = instance_dir / "journal.jsonl"
    history = await Journal(journal_path).recent(limit=50) if journal_path.exists() else []
    # If the pid file is alive, the tailer will deliver future events through
    # the queue — flag the connection as live.
    is_live = _pid_alive(instance_dir / "pid") is not None

    # Highest journal timestamp included in the history block. Any tailer
    # event that arrives with timestamp <= this value has already been
    # replayed and must be suppressed to avoid duplicate bubbles in the UI.
    history_ts_high = max(
        (h.get("timestamp", "") for h in history),
        default="",
    )

    async def generate() -> AsyncGenerator[str, None]:
        for entry in history:
            yield f"data: {json.dumps({**entry, '_history': True})}\n\n"

        yield f"data: {json.dumps({'type': 'history_end', 'live': is_live})}\n\n"

        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=25.0)
                    ts = payload.get("timestamp", "")
                    if ts and history_ts_high and ts <= history_ts_high:
                        # Already in the history block — skip.
                        continue
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


class SpawnRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    provider: str | None = None
    model: str | None = None


@app.post("/instances")
async def spawn_instance(req: SpawnRequest, request: Request) -> dict[str, Any]:
    _require_local(request)
    try:
        safe_id = sanitize_consciousness_name(req.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    target_dir = consciousness_root() / safe_id
    if _pid_alive(target_dir / "pid") is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Instance {safe_id!r} is already running.",
        )

    if req.provider is not None:
        if req.provider not in PROVIDER_ALLOWLIST:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider: {req.provider!r}. "
                       f"Allowed: {sorted(PROVIDER_ALLOWLIST)}",
            )
        if req.model is not None and req.model not in PROVIDER_ALLOWLIST[req.provider]:
            raise HTTPException(
                status_code=400,
                detail=f"Model {req.model!r} not in allowlist for provider "
                       f"{req.provider!r}: {PROVIDER_ALLOWLIST[req.provider]}",
            )
    elif req.model is not None:
        raise HTTPException(
            status_code=400,
            detail="provider must be specified when model is given.",
        )

    # Pre-create the directory so the SSE endpoint can subscribe before the
    # subprocess has written anything yet.
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "spawn.py"),
        "--name", safe_id,
        "--bg",
    ]
    if req.provider:
        cmd.extend(["--provider", req.provider])
    if req.model:
        cmd.extend(["--model", req.model])

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Wait for spawn.py --bg to fork its daemon and write the pid file.
        # Off-thread so the event loop (and other SSE streams) stay responsive.
        await asyncio.to_thread(proc.wait, 10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to spawn {safe_id!r}: {exc}",
        )

    pid = _pid_alive(target_dir / "pid")
    return {
        "id": safe_id,
        "name": safe_id,
        "pid": pid,
        "running": pid is not None,
    }


@app.post("/instances/{name}/stop")
async def stop_instance(name: str, request: Request) -> dict[str, Any]:
    _require_local(request)
    try:
        safe_id = sanitize_consciousness_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    target_dir = consciousness_root() / safe_id
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown instance: {safe_id!r}")

    pid_path = target_dir / "pid"
    pid = _pid_alive(pid_path)
    if pid is None:
        # Stale pid file → clean it up so the UI reflects offline state.
        pid_path.unlink(missing_ok=True)
        return {"id": safe_id, "running": False, "stopped": False, "detail": "not running"}

    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"SIGTERM failed: {exc}")

    return {"id": safe_id, "pid": pid, "running": True, "stopped": True, "detail": "SIGTERM sent"}


@app.delete("/instances/{name}")
async def archive_instance(name: str, request: Request) -> dict[str, Any]:
    """Stop (if running) and archive the instance directory.

    Archives to ``<home>/.archive/<id>-<utc-ts>/`` — destructive HTTP needs
    to be reversible by default per the issue's open-question lean.
    """
    _require_local(request)
    try:
        safe_id = sanitize_consciousness_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    target_dir = consciousness_root() / safe_id
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown instance: {safe_id!r}")

    pid = _pid_alive(target_dir / "pid")
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        # Brief grace window — match scripts/stop.py's 5s.
        for _ in range(20):
            await asyncio.sleep(0.25)
            if _pid_alive(target_dir / "pid") is None:
                break
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    archive_root = consciousness_root() / ".archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = archive_root / f"{safe_id}-{ts}"
    shutil.move(str(target_dir), str(archive_path))
    return {"id": safe_id, "archived_to": str(archive_path)}


# Serve the frontend SPA from the static/ subdirectory
_static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


def configure_tailer() -> JournalTailer:
    """Return the singleton tailer for this process, creating it if needed."""
    global _tailer
    if _tailer is None:
        _tailer = JournalTailer(consciousness_root(), _on_tail_event)
    return _tailer


async def start(port: int, host: str = "127.0.0.1", allow_remote_spawn: bool = False) -> None:
    """Run the dashboard on the given host/port until the process exits.

    Defaults to localhost-only — the dashboard exposes raw thought/journal
    content AND the spawn/stop control plane with no authentication.
    Binding to 0.0.0.0 should be an explicit opt-in.
    """
    import uvicorn

    global _allow_remote_spawn
    _allow_remote_spawn = bool(allow_remote_spawn)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    logger.info("Web dashboard listening at http://%s:%d", host, port)
    await server.serve()
