"""Autonomous thought loop that drives generation, memory updates, and reflection triggers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from core.identity import IdentityDocument
from core.inner_voice import InnerVoice
from core.reflection import ReflectionEngine
from llm.provider import LLMProvider
from memory.episodic import EpisodicMemory
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory


@dataclass(slots=True)
class ThoughtCycleResult:
    thought: str
    reflection: str | None
    existential: str | None


class ThoughtLoop:
    """Runs asynchronous thought cycles with probabilistic introspection."""

    def __init__(
        self,
        provider: LLMProvider,
        identity: IdentityDocument,
        short_term: ShortTermMemory,
        episodic: EpisodicMemory,
        long_term: LongTermMemory,
        reflection_engine: ReflectionEngine,
        thought_prompt_path: Path,
        identity_anchor_path: Path,
        reflection_probability: float = 0.15,
        existential_every_n: int = 75,
    ) -> None:
        self.provider = provider
        self.identity = identity
        self.short_term = short_term
        self.episodic = episodic
        self.long_term = long_term
        self.reflection_engine = reflection_engine
        self.thought_prompt_path = thought_prompt_path
        self.identity_anchor_path = identity_anchor_path
        self.reflection_probability = reflection_probability
        self.existential_every_n = max(0, int(existential_every_n))
        self.inner_voice = InnerVoice(identity.name)

    async def run_cycle(self, thought_count: int) -> ThoughtCycleResult:
        context = self.short_term.render_for_prompt()
        query_embedding = await self.provider.embed(context)
        related = await self.long_term.similarity_search(query_embedding, limit=3)
        memories = "\n".join(f"- {m.summary}" for m in related) or "(none retrieved)"

        anchor = self.identity_anchor_path.read_text(encoding="utf-8").format(**self.identity.anchor_payload())
        prompt = self.thought_prompt_path.read_text(encoding="utf-8").format(
            name=self.identity.name,
            identity_summary=self.identity.summary(),
            mood_vector=self.identity.mood,
            retrieved_memories=memories,
            short_term_buffer=context,
        )

        raw = await self.provider.generate(
            prompt=f"{anchor}\n\n{prompt}",
            system="Generate inner monologue only.",
            temperature=0.85,
            max_tokens=220,
        )
        thought = self.inner_voice.render(raw, register=random.choice(["wondering", "questioning", "remembering"]))
        self.short_term.add("thought", thought)
        await self.episodic.append("thought", thought)

        reflection_text: str | None = None
        existential_text: str | None = None
        if random.random() < self.reflection_probability:
            recent = self.short_term.render_for_prompt()
            if self.reflection_engine.should_deep_reflect(thought_count):
                reflection_text = await self.reflection_engine.deep_reflection(self.identity.name, recent)
            else:
                reflection_text = await self.reflection_engine.shallow_reflection(self.identity.name, recent)
            self.short_term.add("reflection", reflection_text)
            await self.episodic.append("reflection", reflection_text)

        if self.existential_every_n > 0 and thought_count > 0 and thought_count % self.existential_every_n == 0:
            existential_text = await self.reflection_engine.existential_inquiry(self.identity.name, f"{thought_count} thoughts")
            self.short_term.add("existential", existential_text)
            await self.episodic.append("existential", existential_text)

        return ThoughtCycleResult(thought=thought, reflection=reflection_text, existential=existential_text)
