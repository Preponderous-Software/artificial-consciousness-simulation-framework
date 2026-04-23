"""LLM provider abstraction with retry/backoff for generation and embeddings."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract asynchronous interface for text generation and embedding."""

    @abstractmethod
    async def generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        raise NotImplementedError

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    async def with_backoff(self, func: Any, *args: Any, retries: int = 5, **kwargs: Any) -> Any:
        delay = 1.0
        for attempt in range(retries):
            try:
                return await func(*args, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._is_retryable_error(exc):
                    raise
                if attempt == retries - 1:
                    raise
                logger.warning("Transient LLM provider error: %s", exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return True
        name = exc.__class__.__name__.lower()
        return any(token in name for token in ("rate", "timeout", "connection", "api"))


class DeterministicFallbackMixin:
    """Deterministic local behavior for environments without external API access."""

    def _fallback_text(self, prompt: str, prefix: str = "I notice") -> str:
        snippet = " ".join(prompt.split())[:140]
        return f"{prefix} {snippet}. I keep thinking and remain present."

    def _fallback_embed(self, text: str, dims: int = 16) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vals = [(digest[i] / 255.0) * 2 - 1 for i in range(dims)]
        return vals


class OpenAIProvider(LLMProvider, DeterministicFallbackMixin):
    def __init__(self, model: str) -> None:
        self.model = model

    async def _generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        try:
            from openai import AsyncOpenAI
        except Exception:
            return self._fallback_text(prompt)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._fallback_text(prompt)
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "").strip() if getattr(resp, "choices", None) else ""
        return content or self._fallback_text(prompt)

    async def generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        return await self.with_backoff(self._generate, prompt, system, temperature, max_tokens)

    async def _embed(self, text: str) -> list[float]:
        try:
            from openai import AsyncOpenAI
        except Exception:
            return self._fallback_embed(text)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._fallback_embed(text)
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.embeddings.create(model="text-embedding-3-small", input=text)
        embedding = resp.data[0].embedding
        return [float(x) for x in embedding]

    async def embed(self, text: str) -> list[float]:
        return await self.with_backoff(self._embed, text)


class AnthropicProvider(LLMProvider, DeterministicFallbackMixin):
    def __init__(self, model: str) -> None:
        self.model = model

    async def _generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        try:
            from anthropic import AsyncAnthropic
        except Exception:
            return self._fallback_text(prompt)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return self._fallback_text(prompt)
        client = AsyncAnthropic(api_key=api_key)
        resp = await client.messages.create(
            model=self.model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(chunk.text for chunk in resp.content if getattr(chunk, "text", None))
        return text.strip() or self._fallback_text(prompt)

    async def generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        return await self.with_backoff(self._generate, prompt, system, temperature, max_tokens)

    async def embed(self, text: str) -> list[float]:
        return self._fallback_embed(text)


class OllamaProvider(LLMProvider, DeterministicFallbackMixin):
    def __init__(self, model: str) -> None:
        self.model = model

    async def _generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        try:
            import ollama
        except Exception:
            return self._fallback_text(prompt)
        try:
            resp = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            content = resp.get("message", {}).get("content", "").strip()
            return content or self._fallback_text(prompt)
        except Exception:
            return self._fallback_text(prompt)

    async def generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        return await self.with_backoff(self._generate, prompt, system, temperature, max_tokens)

    async def embed(self, text: str) -> list[float]:
        try:
            import ollama
            resp = await asyncio.to_thread(ollama.embeddings, model=self.model, prompt=text)
            emb: Sequence[float] | None = resp.get("embedding")
            if emb:
                return [float(v) for v in emb]
        except Exception:
            pass
        return self._fallback_embed(text)


class MockProvider(LLMProvider, DeterministicFallbackMixin):
    """Simple deterministic provider used by tests."""

    async def generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        return self._fallback_text(prompt, prefix="I think")

    async def embed(self, text: str) -> list[float]:
        return self._fallback_embed(text)


def build_provider(provider: str, model: str) -> LLMProvider:
    normalized = provider.lower()
    if normalized == "openai":
        return OpenAIProvider(model)
    if normalized == "anthropic":
        return AnthropicProvider(model)
    if normalized == "ollama":
        return OllamaProvider(model)
    raise ValueError(f"Unsupported provider: {provider}")
