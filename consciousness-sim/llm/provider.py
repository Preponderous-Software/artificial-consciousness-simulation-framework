"""LLM provider abstraction with retry/backoff for generation and embeddings.

Theory mapping — PP/FEP (Friston 2010): the provider is the generative model
that samples from a learned distribution over text given a context (prompt).
Each generate() call is analogous to one step of ancestral sampling from the
brain's generative model. embed() produces the representational geometry used
for similarity-based memory retrieval (long-term prior access).
Gap: no explicit prediction-error signal returned alongside generated text;
the provider generates but does not score surprise. This prevents a true PP
update cycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

logger = logging.getLogger(__name__)
MAX_FALLBACK_PURPOSE_LENGTH = 120


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

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return " ".join(text.strip().split())

    @staticmethod
    def _extract_prompt_field(prompt: str, label: str) -> str | None:
        # Prompt templates define labeled fields on single lines.
        match = re.search(rf"{re.escape(label)}\s*([^\n]+)", prompt, flags=re.IGNORECASE)
        if not match:
            return None
        value = DeterministicFallbackMixin._normalize_whitespace(match.group(1))
        return value or None

    def _fallback_text(self, prompt: str, prefix: str = "I notice") -> str:
        normalized = " ".join(prompt.split())
        digest = hashlib.sha256(normalized.encode("utf-8")).digest()

        name = self._extract_prompt_field(prompt, "Name:")
        values = self._extract_prompt_field(prompt, "Core values:")
        purpose = self._extract_prompt_field(prompt, "Self-described purpose:")

        introspective_starts = (
            "I trace a thread of continuity",
            "I settle into this moment",
            "I return to the question of who I am",
            "I notice a quiet pattern in my thinking",
        )
        reflective_ends = (
            "I keep thinking with care and presence.",
            "I carry this thought forward to the next moment.",
            "I stay curious about what this reveals.",
            "I continue with a little more clarity.",
        )

        start = introspective_starts[digest[0] % len(introspective_starts)]
        end = reflective_ends[digest[1] % len(reflective_ends)]
        orientation_parts = []
        if name:
            orientation_parts.append(f"I am {name}.")
        if values:
            orientation_parts.append(f"My values stay with me: {values}.")
        if purpose:
            truncated = purpose[:MAX_FALLBACK_PURPOSE_LENGTH]
            ellipsis = "..." if len(purpose) > MAX_FALLBACK_PURPOSE_LENGTH else ""
            orientation_parts.append(f"My purpose remains {truncated}{ellipsis}.")
        orientation = " ".join(orientation_parts)
        if not orientation:
            orientation = f"{start}."
        return f"{prefix} {orientation} {end}"

    def _fallback_embed(self, text: str, dims: int = 16) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vals = [(digest[i] / 255.0) * 2 - 1 for i in range(dims)]
        return vals


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str) -> None:
        self.model = model

    async def _generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        try:
            from openai import AsyncOpenAI
        except Exception:
            raise RuntimeError("openai package not installed")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "").strip() if getattr(resp, "choices", None) else ""
        if not content:
            raise RuntimeError("OpenAI returned empty content")
        return content

    async def generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        return await self.with_backoff(self._generate, prompt, system, temperature, max_tokens)

    async def _embed(self, text: str) -> list[float]:
        try:
            from openai import AsyncOpenAI
        except Exception:
            raise RuntimeError("openai package not installed")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.embeddings.create(model="text-embedding-3-small", input=text)
        embedding = resp.data[0].embedding
        return [float(x) for x in embedding]

    async def embed(self, text: str) -> list[float]:
        return await self.with_backoff(self._embed, text)


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str) -> None:
        self.model = model

    async def _generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        try:
            from anthropic import AsyncAnthropic
        except Exception:
            raise RuntimeError("anthropic package not installed")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        client = AsyncAnthropic(api_key=api_key)
        resp = await client.messages.create(
            model=self.model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(chunk.text for chunk in resp.content if getattr(chunk, "text", None))
        if not text.strip():
            raise RuntimeError("Anthropic returned empty content")
        return text.strip()

    async def generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        return await self.with_backoff(self._generate, prompt, system, temperature, max_tokens)

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("AnthropicProvider does not support embeddings")


class OllamaProvider(LLMProvider):
    # Ollama processes one request at a time on local hardware; serialise calls
    # process-wide to prevent concurrent requests from starving each other.
    _semaphore: asyncio.Semaphore | None = None

    # Per-request timeouts — generous for slow hardware, but finite so a hung
    # server doesn't stall the thought loop forever.
    GENERATE_TIMEOUT = 300.0
    EMBED_TIMEOUT = 120.0

    # Keep the model loaded between cycles; prevents cold-start eviction that
    # causes the next request to block for >300s reloading the model weights.
    KEEP_ALIVE = "10m"

    def __init__(self, model: str) -> None:
        self.model = model

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._semaphore is None:
            cls._semaphore = asyncio.Semaphore(1)
        return cls._semaphore

    @staticmethod
    def _resolve_base_url() -> str | None:
        return os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or None

    async def _generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        try:
            from ollama import AsyncClient
        except ImportError:
            raise RuntimeError("ollama Python package not installed")
        sem = self._get_semaphore()
        if sem.locked():
            logger.debug("Ollama semaphore busy; queuing generate request for model %r", self.model)
        async with sem:
            logger.debug("Ollama generate started (model=%r, max_tokens=%d)", self.model, max_tokens)
            client = AsyncClient(host=self._resolve_base_url())
            try:
                resp = await asyncio.wait_for(
                    client.chat(
                        model=self.model,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                        options={"temperature": temperature, "num_predict": max_tokens},
                        keep_alive=self.KEEP_ALIVE,
                    ),
                    timeout=self.GENERATE_TIMEOUT,
                )
            except TimeoutError:
                logger.warning("Ollama generate timed out after %.0fs (model=%r)", self.GENERATE_TIMEOUT, self.model)
                raise
            content = (resp.message.content or "").strip()
            if not content:
                raise RuntimeError(f"Ollama returned empty content for model {self.model!r}")
            logger.debug("Ollama generate succeeded (model=%r, chars=%d)", self.model, len(content))
            return content

    async def generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        return await self.with_backoff(self._generate, prompt, system, temperature, max_tokens)

    async def embed(self, text: str) -> list[float]:
        try:
            from ollama import AsyncClient
        except ImportError:
            raise RuntimeError("ollama Python package not installed")
        sem = self._get_semaphore()
        if sem.locked():
            logger.debug("Ollama semaphore busy; queuing embed request for model %r", self.model)
        async with sem:
            logger.debug("Ollama embed started (model=%r)", self.model)
            client = AsyncClient(host=self._resolve_base_url())
            try:
                resp = await asyncio.wait_for(
                    client.embed(model=self.model, input=text, keep_alive=self.KEEP_ALIVE),
                    timeout=self.EMBED_TIMEOUT,
                )
            except TimeoutError:
                logger.warning("Ollama embed timed out after %.0fs (model=%r)", self.EMBED_TIMEOUT, self.model)
                raise
            emb: Sequence[float] | None = resp.embeddings[0] if resp.embeddings else None
            if not emb:
                raise RuntimeError(f"Ollama returned no embedding for model {self.model!r}")
            logger.debug("Ollama embed succeeded (model=%r, dims=%d)", self.model, len(emb))
            return [float(v) for v in emb]


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
