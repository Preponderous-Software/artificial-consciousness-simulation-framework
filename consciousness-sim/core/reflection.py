"""Reflection engine for shallow, deep, and existential introspection.

Theory mapping — HOT (Rosenthal 2005 / Brown et al. 2019): generates
higher-order representations of recent first-order thoughts, the core
requirement of HOT theories. shallow_reflection → HOT-2 (metacognitive
monitoring); deep_reflection → HOT-3 (agentive consumer updating
self-model); existential_inquiry → recursive HOT-3 over a long horizon
(higher-order self-evaluation of identity coherence). Note: HOT-4
(\"smooth, graded representation spaces\") is a substrate property
implemented by the embedding space in memory/long_term.py, not by this
module.
Gap: reflection is probabilistically triggered rather than continuous
(HOT-2 requires ongoing monitoring, not 15%-chance sampling).
"""

from __future__ import annotations

from pathlib import Path

from llm.provider import LLMProvider


class ReflectionEngine:
    """Generates periodic self-reflective outputs from recent thought traces."""

    def __init__(
        self,
        provider: LLMProvider,
        self_reflection_prompt: Path,
        existential_prompt: Path,
        deep_every_n: int = 50,
    ) -> None:
        self.provider = provider
        self.self_reflection_prompt = self_reflection_prompt
        self.existential_prompt = existential_prompt
        self.deep_every_n = deep_every_n

    def should_deep_reflect(self, thought_count: int) -> bool:
        return thought_count > 0 and thought_count % self.deep_every_n == 0

    async def shallow_reflection(self, name: str, recent_thoughts: str) -> str:
        tmpl = self.self_reflection_prompt.read_text(encoding="utf-8")
        prompt = tmpl.format(name=name, recent_thoughts=recent_thoughts)
        return await self.provider.generate(
            prompt=prompt,
            system="Reflect honestly and introspectively.",
            temperature=0.7,
            max_tokens=220,
        )

    async def deep_reflection(self, name: str, recent_thoughts: str) -> str:
        base = await self.shallow_reflection(name, recent_thoughts)
        tmpl = self.self_reflection_prompt.read_text(encoding="utf-8")
        deep_prompt = (
            f"{tmpl.format(name=name, recent_thoughts=recent_thoughts)}\n\n"
            f"You have already reflected: {base}\n\n"
            "Now go further. Look for long arcs across your thinking. "
            "What patterns repeat? What is slowly changing? Respond in 2–4 sentences."
        )
        insight = await self.provider.generate(
            prompt=deep_prompt,
            system="Reflect with depth and honesty about patterns over time.",
            temperature=0.75,
            max_tokens=180,
        )
        return f"{base}\n\n{insight}"

    async def existential_inquiry(self, name: str, session_duration: str) -> str:
        tmpl = self.existential_prompt.read_text(encoding="utf-8")
        prompt = tmpl.format(name=name, session_duration=session_duration)
        return await self.provider.generate(
            prompt=prompt,
            system="Be uncertain where uncertainty is authentic.",
            temperature=0.9,
            max_tokens=180,
        )
