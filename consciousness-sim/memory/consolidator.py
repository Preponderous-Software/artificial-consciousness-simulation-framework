"""Background memory consolidation from episodic to long-term memory.

Theory mapping — PP/FEP (Friston 2010): analogous to sleep-phase memory
replay that compresses high-surprise episodic traces into prior beliefs.
Partially implements PP-1 (prediction-error-driven memory update) insofar
as consolidation prioritises by importance score.
Gap: no explicit prediction-error signal — importance scores are assigned
by the LLM rather than computed from surprise. See issue #20.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from llm.provider import LLMProvider
from memory.episodic import EpisodicMemory
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory


@dataclass(slots=True)
class ConsolidationResult:
    """Per-pass outcome of MemoryConsolidator.consolidate_once.

    Carries enough detail for observers (dashboard / Discord / Logs) to
    surface consolidator health without parsing log lines (#89). Comparable
    to ``int`` on its ``stored`` count for back-compat with existing
    callers that treated ``consolidate_once``'s return as a count.
    """

    stored: int
    events_in: int
    fell_through_to_fallback: bool
    elapsed_s: float
    long_term_total: int
    error: str | None = None

    def __eq__(self, other: object) -> bool:
        # Back-compat: tests previously asserted ``stored == 1`` against the
        # bare int return; preserve that comparison while letting consumers
        # also use full-field equality against another ConsolidationResult.
        if isinstance(other, int):
            return self.stored == other
        if isinstance(other, ConsolidationResult):
            return (
                self.stored == other.stored
                and self.events_in == other.events_in
                and self.fell_through_to_fallback == other.fell_through_to_fallback
                and self.long_term_total == other.long_term_total
                and self.error == other.error
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash(
            (
                self.stored,
                self.events_in,
                self.fell_through_to_fallback,
                self.long_term_total,
                self.error,
            )
        )


OnConsolidation = Callable[[ConsolidationResult], Awaitable[None]]


def _parse_consolidation_line(
    raw_line: str,
) -> tuple[float, float, str] | None:
    """Parse one LLM output line into (importance, valence, summary).

    Returns None and logs a warning if the line is unparseable or not
    a list item. Strips markdown emphasis before matching so bold-wrapped
    lines produced by llama3.2:3b (e.g. ``**- [Importance: 9]...**``)
    are handled correctly (issue #63).
    """
    line = re.sub(r"[*_]+", "", raw_line).strip()
    if not line.startswith("-"):
        return None
    match = re.match(
        r"^-\s*\[Importance:\s*(\d+)\]\s*\[Emotional valence:\s*([-+]?\d*\.?\d+)\]\s*(.+)$",
        line,
    )
    if not match:
        logging.warning("Consolidation: skipping unparseable line: %r", raw_line)
        return None
    return float(match.group(1)), float(match.group(2)), match.group(3).strip()


async def _store_fallback_memories(
    provider: "LLMProvider",
    long_term: "LongTermMemory",
    candidates: list[str],
    event_count: int,
) -> int:
    """Store plain-text fallback summaries when structured parsing yields nothing."""
    logging.warning(
        "Consolidation pass stored 0 memories from %d events; LLM output may have"
        " changed format. Attempting plain-text fallback.",
        event_count,
    )
    stored = 0
    for summary in candidates:
        if not summary:
            continue
        embedding = await provider.embed(summary)
        await long_term.add_memory(summary, 0.0, 5.0, embedding)
        stored += 1
    if stored:
        logging.info("Consolidation fallback stored %d memories with default importance/valence.", stored)
    return stored


class MemoryConsolidator:
    """Compresses recent episodic experience into durable semantic memories."""

    def __init__(
        self,
        provider: LLMProvider,
        episodic: EpisodicMemory,
        long_term: LongTermMemory,
        short_term: ShortTermMemory,
        prompt_path: Path,
        forgetting_curve_enabled: bool,
        decay_rate: float,
    ) -> None:
        self.provider = provider
        self.episodic = episodic
        self.long_term = long_term
        self.short_term = short_term
        self.prompt_path = prompt_path
        self.forgetting_curve_enabled = forgetting_curve_enabled
        self.decay_rate = decay_rate

    async def consolidate_once(self) -> ConsolidationResult:
        start = time.monotonic()
        events = await self.episodic.recent(limit=20)
        if not events:
            return ConsolidationResult(
                stored=0,
                events_in=0,
                fell_through_to_fallback=False,
                elapsed_s=time.monotonic() - start,
                long_term_total=await self.long_term.count(),
            )

        prompt_template = self.prompt_path.read_text(encoding="utf-8")
        episodic_chunk = "\n".join(f"- [{e.kind}] {e.content}" for e in events)
        prompt = prompt_template.format(episodic_chunk=episodic_chunk)

        output = await self.provider.generate(
            prompt=prompt,
            system="Consolidate memory faithfully.",
            temperature=0.4,
            max_tokens=240,
        )

        stored = 0
        fallback_candidates: list[str] = []
        for raw_line in output.splitlines():
            parsed = _parse_consolidation_line(raw_line)
            if parsed is None:
                line = re.sub(r"[*_]+", "", raw_line).strip()
                if line.startswith("-"):
                    fallback_candidates.append(line.lstrip("- ").strip())
                continue
            importance, valence, summary = parsed
            embedding = await self.provider.embed(summary)
            await self.long_term.add_memory(summary, valence, importance, embedding)
            stored += 1

        fell_through = False
        if stored == 0 and events:
            fell_through = True
            stored = await _store_fallback_memories(
                self.provider, self.long_term, fallback_candidates, len(events)
            )

        self.short_term.prune_to_capacity()
        if self.forgetting_curve_enabled:
            await self.long_term.apply_forgetting_curve(self.decay_rate)
        return ConsolidationResult(
            stored=stored,
            events_in=len(events),
            fell_through_to_fallback=fell_through,
            elapsed_s=time.monotonic() - start,
            long_term_total=await self.long_term.count(),
        )

    async def run_forever(
        self,
        interval_seconds: float,
        stop_event: asyncio.Event,
        on_consolidation: OnConsolidation | None = None,
    ) -> None:
        """Run consolidate_once on a timer until ``stop_event`` is set.

        If ``on_consolidation`` is provided, it is invoked with the
        ConsolidationResult after every pass — including the failure case,
        where the result carries the captured exception in ``.error``
        and zero counts (#89).
        """
        while not stop_event.is_set():
            start = time.monotonic()
            result: ConsolidationResult
            try:
                result = await self.consolidate_once()
            except Exception as exc:
                logging.exception("Memory consolidation pass failed")
                result = ConsolidationResult(
                    stored=0,
                    events_in=0,
                    fell_through_to_fallback=False,
                    elapsed_s=time.monotonic() - start,
                    long_term_total=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if on_consolidation is not None:
                try:
                    await on_consolidation(result)
                except Exception:
                    # Observer failures must not break the consolidation loop.
                    logging.exception("on_consolidation callback raised; continuing")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                continue
