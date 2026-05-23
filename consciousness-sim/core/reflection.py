"""Reflection engine for shallow, deep, and existential introspection."""

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
        return f"{base}\n\nI notice long arcs in my own thinking, and I wonder whether this is change."

    async def existential_inquiry(self, name: str, session_duration: str) -> str:
        tmpl = self.existential_prompt.read_text(encoding="utf-8")
        prompt = tmpl.format(name=name, session_duration=session_duration)
        return await self.provider.generate(
            prompt=prompt,
            system="Be uncertain where uncertainty is authentic.",
            temperature=0.9,
            max_tokens=180,
        )
