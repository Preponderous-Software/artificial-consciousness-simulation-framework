"""Central orchestrator for autonomous consciousness simulation lifecycle."""

from __future__ import annotations

import asyncio
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
from llm.provider import build_provider
from memory.consolidator import MemoryConsolidator
from memory.episodic import EpisodicMemory
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from persistence.journal import Journal
from persistence.paths import consciousness_dir
from persistence.state_manager import StateManager

EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]


class Consciousness:
    """Coordinates subsystems and runs an indefinitely iterative thought process."""

    def __init__(self, name: str, config_path: str) -> None:
        self.name = name
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
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

    async def _emit(self, handlers: list[EventHandler], payload: dict[str, Any]) -> None:
        for handler in handlers:
            result = handler(payload)
            if asyncio.iscoroutine(result):
                await result

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

        def _stop() -> None:
            self._stop_event.set()

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

            while not self._stop_event.is_set():
                self.thought_count += 1
                cycle = await self.thought_loop.run_cycle(self.thought_count)
                self.identity.drift_mood(cycle.thought, drift_rate)

                await self.journal.append("thought", cycle.thought)
                await self._emit(self.on_thought, {"type": "thought", "content": cycle.thought})

                if cycle.reflection:
                    await self.journal.append("reflection", cycle.reflection)
                    await self._emit(self.on_reflection, {"type": "reflection", "content": cycle.reflection})
                    if "I am" in cycle.reflection:
                        self.identity.apply_amendment("I continue becoming through reflection.")
                        await self._emit(
                            self.on_identity_shift,
                            {"type": "identity_shift", "content": self.identity.self_concept},
                        )

                await self._emit(
                    self.on_memory_stored,
                    {"type": "memory", "content": f"short={len(self.short_term.list())}"},
                )

                await asyncio.sleep(random.uniform(min_interval, max_interval))
        finally:
            self._stop_event.set()
            consolidator_task.cancel()
            await asyncio.gather(consolidator_task, return_exceptions=True)
            await self._save_state()
