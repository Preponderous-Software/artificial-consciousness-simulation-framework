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
from pathlib import Path

from llm.provider import LLMProvider
from memory.episodic import EpisodicMemory
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory


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

    async def consolidate_once(self) -> int:
        events = await self.episodic.recent(limit=20)
        if not events:
            return 0

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
            # Strip markdown emphasis (* and _) BEFORE the prefix check so lines
            # like `- **[Importance: 9] [Emotional valence: 0.8]** summary` (the
            # dominant llama3.2:3b output, per issue #63) parse correctly. Also
            # handles `**- [Importance...]**` if the model wraps the whole line.
            line = re.sub(r"[*_]+", "", raw_line).strip()
            if not line.startswith("-"):
                continue
            match = re.match(
                r"^-\s*\[Importance:\s*(\d+)\]\s*\[Emotional valence:\s*([-+]?\d*\.?\d+)\]\s*(.+)$",
                line,
            )
            if not match:
                logging.warning("Consolidation: skipping unparseable line: %r", raw_line)
                fallback_candidates.append(line.lstrip("- ").strip())
                continue
            importance = float(match.group(1))
            valence = float(match.group(2))
            summary = match.group(3).strip()
            embedding = await self.provider.embed(summary)
            await self.long_term.add_memory(summary, valence, importance, embedding)
            stored += 1

        if stored == 0 and events:
            logging.warning(
                "Consolidation pass stored 0 memories from %d events; LLM output may have changed format."
                " Attempting plain-text fallback.",
                len(events),
            )
            for summary in fallback_candidates:
                if not summary:
                    continue
                embedding = await self.provider.embed(summary)
                await self.long_term.add_memory(summary, 0.0, 5.0, embedding)
                stored += 1
            if stored:
                logging.info("Consolidation fallback stored %d memories with default importance/valence.", stored)

        self.short_term.prune_to_capacity()
        if self.forgetting_curve_enabled:
            await self.long_term.apply_forgetting_curve(self.decay_rate)
        return stored

    async def run_forever(self, interval_seconds: float, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self.consolidate_once()
            except Exception:
                logging.exception("Memory consolidation pass failed")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                continue
