"""Autonomous thought loop that drives generation, memory updates, and reflection triggers.

Theory mapping — GWT (Baars 1988) / CTM (Blum & Blum 2022): run_cycle()
is the serialised workspace update — one thought occupies the global workspace
per cycle, analogous to CTM's single-slot global broadcast. Long-term memory
retrieval before generation partially implements GWT-3 (globally broadcast
priors available to the generator).
Gap: no parallel specialist competition for workspace access (GWT-2/CTM);
all modules are called sequentially by the loop rather than competing. No
ignition threshold (GNWT). No prediction-error cycle (PP-1).
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from pathlib import Path

from core.identity import IdentityDocument
from core.inner_voice import InnerVoice
from core.reflection import ReflectionEngine
from llm.perception import Perception, PerceptionProvider, render_perception_block
from llm.provider import LLMProvider
from memory.episodic import EpisodicMemory
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory


_THEME_STOPWORDS = frozenset({
    "about", "above", "after", "again", "against", "being", "because",
    "could", "every", "everything", "from", "given", "have", "just",
    "know", "might", "myself", "never", "nothing", "often", "other",
    "perhaps", "really", "remember", "right", "since", "small", "some",
    "something", "still", "their", "there", "these", "think", "those",
    "though", "thought", "through", "under", "until", "using", "very",
    "when", "where", "which", "while", "would", "years",
})


def _extract_theme(text: str) -> str:
    """Return the first content word (5+ chars, not a stopword) from text."""
    for word in re.findall(r"[a-z]{5,}", text.lower()):
        if word not in _THEME_STOPWORDS:
            return word
    return ""


def _select_register(raw: str, has_retrieved_memories: bool) -> str:
    """Choose an InnerVoice register based on the content of the generated thought."""
    if "?" in raw:
        return "questioning"
    if has_retrieved_memories and any(w in raw.lower() for w in ("remember", "recall", "once", "back then", "used to")):
        return "remembering"
    return "wondering"


@dataclass(slots=True)
class ThoughtCycleResult:
    thought: str
    reflection: str | None
    existential: str | None
    perception: Perception | None = None


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
        perception_provider: PerceptionProvider | None = None,
        perception_every_n: int = 0,
        thought_temperature: float = 0.85,
        thought_max_tokens: int = 220,
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
        self.perception_provider = perception_provider
        self.perception_every_n = max(0, int(perception_every_n))
        self.thought_temperature = float(thought_temperature)
        self.thought_max_tokens = int(thought_max_tokens)
        self.inner_voice = InnerVoice(identity.name)

    async def run_cycle(self, thought_count: int) -> ThoughtCycleResult:
        self.identity.attention_schema.decay()
        context = self.short_term.render_for_prompt()
        query_embedding = await self.provider.embed(context)
        related = await self.long_term.similarity_search(query_embedding, limit=3)
        memories = "\n".join(f"- {m.summary}" for m in related) or "(none retrieved)"

        perception = await self._maybe_fetch_perception(thought_count)
        if perception is not None:
            # Lingers into subsequent cycles via short-term buffer and gets
            # consolidated by the memory consolidator from episodic.
            stim = f"[{perception.source}: {perception.title}] {perception.content}"
            self.short_term.add("perception", stim)
            await self.episodic.append("perception", stim)

        anchor = self.identity_anchor_path.read_text(encoding="utf-8").format(**self.identity.anchor_payload())
        prompt = self.thought_prompt_path.read_text(encoding="utf-8").format(
            name=self.identity.name,
            identity_summary=self.identity.summary(),
            mood_vector=self.identity.mood,
            retrieved_memories=memories,
            short_term_buffer=context,
            perception_block=render_perception_block(perception),
        )

        raw = await self.provider.generate(
            prompt=f"{anchor}\n\n{prompt}",
            system=(
                "Generate inner monologue only. "
                "Do not end with a question or invitation to the reader. "
                "Do not use phrases like 'please continue', 'continue?', or 'tell me more'. "
                "Stop when the thought is complete."
            ),
            temperature=self.thought_temperature,
            max_tokens=self.thought_max_tokens,
        )
        thought = self.inner_voice.render(raw, register=_select_register(raw, bool(related)))
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

        if perception is not None:
            focus, theme = "perception", _extract_theme(perception.title + " " + perception.content)
        elif existential_text is not None:
            focus, theme = "existential", _extract_theme(existential_text)
        elif reflection_text is not None:
            focus, theme = "reflection", _extract_theme(reflection_text)
        elif related:
            focus, theme = "memory", _extract_theme(memories)
        else:
            focus, theme = "introspection", _extract_theme(thought)
        if not theme:
            theme = _extract_theme(thought)
        self.identity.attention_schema.update(focus, theme)

        return ThoughtCycleResult(
            thought=thought,
            reflection=reflection_text,
            existential=existential_text,
            perception=perception,
        )

    async def _maybe_fetch_perception(self, thought_count: int) -> Perception | None:
        """Fetch a perception every Nth cycle. Failures yield None (logged by provider)."""
        if self.perception_provider is None or self.perception_every_n <= 0:
            return None
        if thought_count <= 0 or (thought_count % self.perception_every_n) != 0:
            return None
        try:
            return await self.perception_provider.fetch()
        except Exception:  # belt-and-braces — providers should already swallow errors
            import logging
            logging.warning("Perception provider raised unexpectedly; skipping", exc_info=True)
            return None
