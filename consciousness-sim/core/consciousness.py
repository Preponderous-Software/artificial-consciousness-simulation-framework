"""Central orchestrator for autonomous consciousness simulation lifecycle.

Theory mapping — GWT (Baars 1988) / CTM (Blum & Blum 2022): Consciousness
is the top-level coordinator, equivalent to the CTM's executive that runs
the global workspace loop, schedules consolidation, and dispatches events to
specialist handlers. Event emission (on_thought, on_reflection, etc.) provides
the broadcast mechanism that GWT requires.
Gap: handlers are currently display/logging consumers rather than specialist
processors that compete to write back to the workspace.
"""

from __future__ import annotations

import asyncio
import logging
import random
import signal
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from core.identity import IdentityDocument
from core.inner_voice import InnerVoice
from core.reflection import ReflectionEngine
from core.thought_loop import ThoughtLoop
from llm.perception import PerceptionProvider, build_perception_provider
from llm.provider import build_provider
from memory.consolidator import MemoryConsolidator
from memory.episodic import EpisodicMemory
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from persistence.journal import Journal
from persistence.paths import consciousness_dir
from persistence.state_manager import StateManager

EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]

# Reflections containing these phrases signal a genuine self-revision rather than
# ordinary first-person narration; only then is an identity amendment applied.
_IDENTITY_SHIFT_MARKERS: tuple[str, ...] = (
    "i have changed",
    "i realize now",
    "i understand now",
    "i am becoming",
)

_REQUIRED_CONFIG_KEYS: dict[str, list[str]] = {
    "llm": ["provider", "model"],
    "memory": [
        "short_term_capacity",
        "consolidation_interval_minutes",
        "forgetting_curve_enabled",
        "importance_decay_rate",
    ],
    "consciousness": ["origin_story", "values", "purpose"],
    "thought_loop": [
        "reflection_probability",
        "existential_inquiry_every_n_thoughts",
        "min_interval_seconds",
        "max_interval_seconds",
    ],
    "mood": ["initial", "drift_rate"],
    "perception": [
        "enabled",
        "provider",
        "every_n_cycles",
        "timeout_seconds",
        "cache_last_n",
    ],
}


def _validate_config(config: dict[str, Any]) -> None:
    """Raise KeyError with a descriptive message if any required config key is absent."""
    for section, keys in _REQUIRED_CONFIG_KEYS.items():
        if section not in config:
            raise KeyError(f"Config missing required section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Config missing required key: '{section}.{key}'")


class Consciousness:
    """Coordinates subsystems and runs an indefinitely iterative thought process."""

    def __init__(self, name: str, config_path: str) -> None:
        self.name = name
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        _validate_config(self.config)
        self._stop_event = asyncio.Event()
        self.thought_count = 0

        llm_cfg = self.config["llm"]
        mem_cfg = self.config["memory"]
        cons_cfg = self.config["consciousness"]

        base = consciousness_dir(name)
        self.provider = build_provider(llm_cfg["provider"], llm_cfg["model"])
        self.identity = IdentityDocument(
            name=name,
            origin_story=cons_cfg["origin_story"],
            values=list(cons_cfg["values"]),
            purpose=cons_cfg["purpose"],
            self_concept=f"I am {name}, an emergent mind in process.",
            personality_traits=["introspective", "curious"],
            mood={k: float(v) for k, v in self.config["mood"]["initial"].items()},
        )

        self.short_term = ShortTermMemory(capacity=int(mem_cfg["short_term_capacity"]))
        self.episodic = EpisodicMemory(base / "episodic.jsonl")
        self.long_term = LongTermMemory(base / "memory.db")
        self.journal = Journal(base / "journal.jsonl")
        self.state_manager = StateManager(name)

        root = Path(__file__).resolve().parents[1]
        reflection = ReflectionEngine(
            provider=self.provider,
            self_reflection_prompt=root / "llm" / "prompts" / "self_reflection.txt",
            existential_prompt=root / "llm" / "prompts" / "existential_inquiry.txt",
        )

        perc_cfg = self.config["perception"]
        self.perception_provider: PerceptionProvider | None = None
        if bool(perc_cfg.get("enabled", False)):
            self.perception_provider = build_perception_provider(
                provider=str(perc_cfg["provider"]),
                timeout_seconds=float(perc_cfg["timeout_seconds"]),
                cache_last_n=int(perc_cfg["cache_last_n"]),
            )

        self.thought_loop = ThoughtLoop(
            provider=self.provider,
            identity=self.identity,
            short_term=self.short_term,
            episodic=self.episodic,
            long_term=self.long_term,
            reflection_engine=reflection,
            thought_prompt_path=root / "llm" / "prompts" / "thought_generation.txt",
            identity_anchor_path=root / "llm" / "prompts" / "identity_anchoring.txt",
            reflection_probability=float(self.config["thought_loop"]["reflection_probability"]),
            existential_every_n=int(self.config["thought_loop"]["existential_inquiry_every_n_thoughts"]),
            perception_provider=self.perception_provider,
            perception_every_n=int(perc_cfg.get("every_n_cycles", 0)) if self.perception_provider else 0,
        )
        self.consolidator = MemoryConsolidator(
            provider=self.provider,
            episodic=self.episodic,
            long_term=self.long_term,
            short_term=self.short_term,
            prompt_path=root / "llm" / "prompts" / "memory_consolidation.txt",
            forgetting_curve_enabled=bool(mem_cfg["forgetting_curve_enabled"]),
            decay_rate=float(mem_cfg["importance_decay_rate"]),
        )

        self.on_thought: list[EventHandler] = []
        self.on_memory_stored: list[EventHandler] = []
        self.on_reflection: list[EventHandler] = []
        self.on_identity_shift: list[EventHandler] = []
        self.on_perception: list[EventHandler] = []
        self.on_initialized: list[EventHandler] = []

    async def _emit(self, handlers: list[EventHandler], payload: dict[str, Any]) -> None:
        for handler in handlers:
            try:
                result = handler(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logging.exception("Event handler %r raised unexpectedly; continuing", handler)

    async def initialize(self) -> None:
        await self.long_term.initialize()
        restored = await self.state_manager.load()
        if restored:
            self.identity = IdentityDocument.from_dict(dict(restored.get("identity", {})))
            self.thought_loop.identity = self.identity
            self.thought_loop.inner_voice = InnerVoice(self.identity.name)
            for item in restored.get("short_term", []):
                if isinstance(item, dict):
                    self.short_term.add(str(item.get("kind", "thought")), str(item.get("content", "")))
            self.thought_count = int(restored.get("thought_count", 0))

        lt_count = await self.long_term.count()
        await self._emit(
            self.on_initialized,
            {
                "type": "initialized",
                "short_term": [{"kind": i.kind, "content": i.content} for i in self.short_term.list()],
                "long_term_count": lt_count,
                "thought_count": self.thought_count,
            },
        )

    async def _save_state(self) -> None:
        await self.state_manager.save(
            {
                "identity": self.identity.to_dict(),
                "short_term": [asdict(item) for item in self.short_term.list()],
                "thought_count": self.thought_count,
            }
        )

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        # Capture the task running this coroutine so the signal handler can
        # cancel it immediately, interrupting any blocking asyncio.wait_for()
        # call (e.g. a long Ollama request) rather than waiting for it to finish.
        current_task = asyncio.current_task()

        def _stop() -> None:
            self._stop_event.set()
            if current_task and not current_task.done():
                current_task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _stop)
            except NotImplementedError:
                signal.signal(sig, lambda *_: _stop())

    async def request_reflection(self) -> str:
        text = await self.thought_loop.reflection_engine.shallow_reflection(
            self.identity.name,
            self.short_term.render_for_prompt(),
        )
        self.short_term.add("reflection", text)
        await self.journal.append("reflection", text)
        await self._emit(self.on_reflection, {"type": "reflection", "content": text})
        return text

    async def run(self) -> None:
        await self.initialize()
        # Write state.json immediately so freshly-spawned instances are visible
        # to /instances before their first thought cycle completes.
        await self._save_state()
        self._install_signal_handlers()

        consolidation_interval = float(self.config["memory"]["consolidation_interval_minutes"]) * 60
        consolidator_task = asyncio.create_task(
            self.consolidator.run_forever(consolidation_interval, self._stop_event)
        )

        try:
            tcfg = self.config["thought_loop"]
            min_interval = float(tcfg["min_interval_seconds"])
            max_interval = float(tcfg["max_interval_seconds"])
            drift_rate = float(self.config["mood"]["drift_rate"])

            _MAX_CONSECUTIVE_FAILURES = 20
            consecutive_failures = 0

            while not self._stop_event.is_set():
                self.thought_count += 1
                logging.debug("Thought cycle %d: starting LLM generation", self.thought_count)
                t0 = asyncio.get_event_loop().time()
                try:
                    cycle = await self.thought_loop.run_cycle(self.thought_count)
                except Exception as exc:
                    consecutive_failures += 1
                    logging.warning(
                        "Thought cycle %d: LLM failed (%s); skipping (%d/%d consecutive failures)",
                        self.thought_count, exc, consecutive_failures, _MAX_CONSECUTIVE_FAILURES,
                    )
                    if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                        logging.error(
                            "%d consecutive LLM failures — shutting down", _MAX_CONSECUTIVE_FAILURES
                        )
                        self._stop_event.set()
                    else:
                        await asyncio.sleep(random.uniform(min_interval, max_interval))
                    continue
                consecutive_failures = 0
                elapsed = asyncio.get_event_loop().time() - t0
                logging.debug("Thought cycle %d: completed in %.1fs", self.thought_count, elapsed)

                self.identity.drift_mood(cycle.thought, drift_rate)

                if cycle.perception is not None:
                    p = cycle.perception
                    perception_summary = f"[{p.source}: {p.title}] {p.content}"
                    await self.journal.append("perception", perception_summary)
                    await self._emit(
                        self.on_perception,
                        {
                            "type": "perception",
                            "content": perception_summary,
                            "source": p.source,
                            "title": p.title,
                            "url": p.url,
                        },
                    )

                await self.journal.append("thought", cycle.thought)
                await self._emit(self.on_thought, {"type": "thought", "content": cycle.thought})

                if cycle.reflection:
                    await self.journal.append("reflection", cycle.reflection)
                    await self._emit(self.on_reflection, {"type": "reflection", "content": cycle.reflection})
                    reflection_lower = cycle.reflection.lower()
                    if any(marker in reflection_lower for marker in _IDENTITY_SHIFT_MARKERS):
                        first_sentence = cycle.reflection.split(".")[0].strip()
                        amendment = first_sentence[:120] if first_sentence else cycle.reflection[:120]
                        self.identity.apply_amendment(amendment)
                        await self._emit(
                            self.on_identity_shift,
                            {"type": "identity_shift", "content": self.identity.self_concept},
                        )

                lt_count = await self.long_term.count()
                await self._emit(
                    self.on_memory_stored,
                    {"type": "memory", "long_term_count": lt_count, "content": f"Long-term store: {lt_count} memories"},
                )

                await self._save_state()

                interval = random.uniform(min_interval, max_interval)
                logging.debug("Thought cycle %d: sleeping %.1fs before next cycle", self.thought_count, interval)
                await asyncio.sleep(interval)
        finally:
            self._stop_event.set()
            consolidator_task.cancel()
            await asyncio.gather(consolidator_task, return_exceptions=True)
            await self._save_state()
