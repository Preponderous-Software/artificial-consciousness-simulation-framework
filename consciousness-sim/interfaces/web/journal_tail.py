"""Polling-based journal tailer for the standalone web dashboard.

Theory mapping — GWT-3 (global broadcast): when the agent process and the
dashboard process are different (the standalone-manager case introduced in
issue #55), in-process event handlers cannot bridge them. The append-only
journal is a substrate-independent broadcast channel that preserves event
ordering and loses no events on dashboard restart.

Mechanism: 500ms polling. ~500ms worst-case latency in exchange for
cross-platform simplicity and zero new runtime dependencies. The tailer
discovers instance directories dynamically by scanning CONSCIOUSNESS_HOME
each tick — instances spawned after the dashboard starts are picked up
automatically.

Dedup: when an instance is first observed, the tailer seeks to end-of-file
so events already journalled are not replayed through the live queue.
The SSE handler runs its own history replay (last 50 lines) for late
joiners; tailer events that overlap that window are deduplicated by
timestamp in the SSE generator (see server.stream_events).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Event types journalled by core.consciousness — used to filter unknown line
# kinds so a corrupted journal can't inject arbitrary payloads into the SSE
# stream.
_KNOWN_EVENT_TYPES = frozenset({
    "thought", "reflection", "perception", "identity_shift", "memory",
    # MemoryConsolidator passes emit a "consolidation" event (#89) so
    # the dashboard can surface per-pass stored counts and error flags.
    "consolidation",
})


class JournalTailer:
    """Watches every <home>/<id>/journal.jsonl for new lines.

    Each new line is parsed as JSON and dispatched to ``on_event(id, payload)``.
    The tailer owns the polling task lifecycle; callers start/stop it.
    """

    def __init__(
        self,
        home: Path,
        on_event: Callable[[str, dict[str, Any]], None],
        poll_interval_s: float = 0.5,
    ) -> None:
        self._home = home
        self._on_event = on_event
        self._poll_interval = poll_interval_s
        # instance id → byte offset in journal.jsonl
        self._offsets: dict[str, int] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run())
            logger.info("Journal tailer started (poll interval %.1fs, home=%s)",
                        self._poll_interval, self._home)

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.to_thread(self._poll_once)
            except Exception:
                # Never let a transient filesystem error kill the tailer.
                logger.exception("Journal tail poll failed; continuing")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval)
            except asyncio.TimeoutError:
                pass

    def _poll_once(self) -> None:
        if not self._home.exists():
            return
        for entry in self._home.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            journal_path = entry / "journal.jsonl"
            if not journal_path.exists():
                continue
            self._read_new_lines(entry.name, journal_path)

    def _read_new_lines(self, instance_id: str, path: Path) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            return

        offset = self._offsets.get(instance_id)
        if offset is None:
            # First time we've seen this instance — start at EOF so we don't
            # replay pre-existing history through the live queue. The SSE
            # handler still serves history independently via Journal.recent().
            self._offsets[instance_id] = size
            return

        if size < offset:
            # File was truncated (rotated, manually cleared). Reset to start.
            offset = 0

        if size == offset:
            return

        try:
            with path.open("rb") as f:
                f.seek(offset)
                chunk = f.read(size - offset)
        except OSError:
            return

        # Only consume complete lines; defer any partial trailing line until
        # the next poll picks it up (avoids JSONDecodeError on a half-flush).
        if not chunk.endswith(b"\n"):
            last_nl = chunk.rfind(b"\n")
            if last_nl == -1:
                return
            consumed = last_nl + 1
            chunk = chunk[:consumed]
        else:
            consumed = len(chunk)

        for raw_line in chunk.splitlines():
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Journal tail: skipping corrupted line in %s", path)
                continue
            if not isinstance(payload, dict):
                continue
            event_type = payload.get("type")
            if event_type not in _KNOWN_EVENT_TYPES:
                continue
            try:
                self._on_event(instance_id, payload)
            except Exception:
                logger.exception("Journal tail on_event handler raised; continuing")

        self._offsets[instance_id] = offset + consumed
