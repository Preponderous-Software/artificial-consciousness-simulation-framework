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
import collections
import hashlib
import logging
import os
import re
import subprocess
import urllib.parse
from abc import ABC, abstractmethod
from typing import Any, Sequence

from llm.circuit_breaker import CircuitBreaker, LLMUnavailableError

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

    async def try_ensure_running(self) -> bool:
        """Check provider health and attempt recovery if needed. Returns True if healthy."""
        return True

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
        # A fast-fail from an open circuit is the opposite of retryable: the
        # breaker raised it precisely to stop us from waiting on the provider
        # again (#114). Sleeping and retrying here would reintroduce the delay.
        if isinstance(exc, LLMUnavailableError):
            return False
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

    # Default LRU embed cache size (#113). At 768-dim float32 vectors (~3 KB
    # each), 256 entries cap memory at <1 MB. Set to 0 to disable.
    DEFAULT_EMBED_CACHE_SIZE: int = 256

    def __init__(
        self,
        model: str,
        embed_cache_size: int = DEFAULT_EMBED_CACHE_SIZE,
        *,
        embed_model: str | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self.model = model
        # Optional dedicated embedding model (#112). Embeds and generations
        # otherwise contend for the same model slot in Ollama's queue, and a
        # generation-sized model pays generation-sized context-load cost for
        # what is only a vector lookup. None → use the generation model.
        self.embed_model = embed_model or model
        # Per-instance LRU cache keyed by sha256 of the input text (#113).
        # Most embed calls during a long run repeat near-identical context
        # strings cycle-to-cycle, so caching cuts Ollama load substantially
        # under contention. Keys include the model name so two providers
        # sharing one Python process don't cross-contaminate.
        self._embed_cache_size: int = max(0, int(embed_cache_size))
        self._embed_cache: "collections.OrderedDict[str, list[float]]" = collections.OrderedDict()
        self.embed_cache_hits: int = 0
        self.embed_cache_misses: int = 0
        # Optional circuit breaker (#114). None → pre-#114 behaviour: every
        # call waits out the full request timeout, however saturated the
        # server is. Generate and embed share one breaker because they
        # contend for the same server, not the same endpoint.
        self.circuit_breaker = circuit_breaker

    def _embed_cache_key(self, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Keyed on the embedding model, not the generation model: two
        # providers differing only in embed_model produce different vectors
        # (and different dimensionalities) for identical text.
        return f"{self.embed_model}:{h}"

    @property
    def circuit_state(self) -> str | None:
        """Breaker state label, or None when no breaker is configured."""
        return self.circuit_breaker.state if self.circuit_breaker is not None else None

    async def _guarded(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run `func` under the circuit breaker when one is configured."""
        if self.circuit_breaker is None:
            return await func(*args, **kwargs)
        return await self.circuit_breaker.call(func, *args, **kwargs)

    @property
    def embed_cache_stats(self) -> dict[str, int]:
        """Return a snapshot of cache occupancy + hit/miss counters."""
        return {
            "size": len(self._embed_cache),
            "capacity": self._embed_cache_size,
            "hits": self.embed_cache_hits,
            "misses": self.embed_cache_misses,
        }

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
        # Breaker sits outside with_backoff so an open circuit short-circuits
        # the whole retry ladder, not just one attempt of it.
        return await self._guarded(
            self.with_backoff, self._generate, prompt, system, temperature, max_tokens
        )

    async def embed(self, text: str) -> list[float]:
        if self._embed_cache_size > 0:
            key = self._embed_cache_key(text)
            cached = self._embed_cache.get(key)
            if cached is not None:
                self._embed_cache.move_to_end(key)
                self.embed_cache_hits += 1
                logger.debug(
                    "Ollama embed cache hit (model=%r, hits=%d/misses=%d)",
                    self.embed_model, self.embed_cache_hits, self.embed_cache_misses,
                )
                return list(cached)
            self.embed_cache_misses += 1

        # Cache hits are served above without reaching this point, so an open
        # circuit never blocks a lookup that would not have touched the server.
        vec = await self._guarded(self._do_embed, text)

        if self._embed_cache_size > 0:
            self._embed_cache[key] = list(vec)
            while len(self._embed_cache) > self._embed_cache_size:
                self._embed_cache.popitem(last=False)
        return vec

    async def _do_embed(self, text: str) -> list[float]:
        try:
            from ollama import AsyncClient
        except ImportError:
            raise RuntimeError("ollama Python package not installed")
        sem = self._get_semaphore()
        if sem.locked():
            logger.debug(
                "Ollama semaphore busy; queuing embed request for model %r", self.embed_model
            )
        async with sem:
            logger.debug("Ollama embed started (model=%r)", self.embed_model)
            client = AsyncClient(host=self._resolve_base_url())
            try:
                resp = await asyncio.wait_for(
                    client.embed(model=self.embed_model, input=text, keep_alive=self.KEEP_ALIVE),
                    timeout=self.EMBED_TIMEOUT,
                )
            except TimeoutError:
                logger.warning(
                    "Ollama embed timed out after %.0fs (model=%r)",
                    self.EMBED_TIMEOUT, self.embed_model,
                )
                raise
            emb: Sequence[float] | None = resp.embeddings[0] if resp.embeddings else None
            if not emb:
                raise RuntimeError(f"Ollama returned no embedding for model {self.embed_model!r}")
            logger.debug("Ollama embed succeeded (model=%r, dims=%d)", self.embed_model, len(emb))
            return [float(v) for v in emb]

    async def _is_ollama_healthy(self) -> bool:
        # HTTP check against /api/tags is more reliable than a raw TCP connect:
        # a hung Ollama process keeps the port open but stops answering HTTP.
        raw = self._resolve_base_url() or "http://localhost:11434"
        base = raw if "://" in raw else f"http://{raw}"
        url = base.rstrip("/") + "/api/tags"
        loop = asyncio.get_event_loop()
        try:
            import urllib.request as _req
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: _req.urlopen(url, timeout=5)),
                timeout=6.0,
            )
            return True
        except Exception:
            return False

    async def try_ensure_running(self) -> bool:
        """Start 'ollama serve' if unhealthy; poll up to 30s for it to come up.

        Deliberately does not reset the circuit breaker (#114): a saturated
        server keeps answering /api/tags promptly while generate/embed time
        out, so treating this probe as recovery would reopen the floodgates
        on exactly the failure mode the breaker exists to damp. The breaker
        recovers on its own via the half-open probe once its cooldown lapses.
        """
        if await self._is_ollama_healthy():
            return True
        logger.warning("Ollama unreachable; attempting auto-start via 'ollama serve'")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            logger.error("'ollama' binary not found — cannot auto-start")
            return False
        except Exception as exc:
            logger.error("Failed to launch 'ollama serve': %s", exc)
            return False
        for _ in range(15):
            await asyncio.sleep(2.0)
            if await self._is_ollama_healthy():
                logger.info("Ollama came up after auto-start")
                return True
        logger.error("Ollama did not become healthy within 30s after auto-start")
        return False


class MockProvider(LLMProvider, DeterministicFallbackMixin):
    """Simple deterministic provider used by tests."""

    async def generate(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        return self._fallback_text(prompt, prefix="I think")

    async def embed(self, text: str) -> list[float]:
        return self._fallback_embed(text)


def build_provider(
    provider: str,
    model: str,
    *,
    embed_cache_size: int | None = None,
    embed_model: str | None = None,
    circuit_breaker: CircuitBreaker | None = None,
) -> LLMProvider:
    """Build a concrete LLMProvider.

    ``embed_cache_size`` (#113), ``embed_model`` (#112) and ``circuit_breaker``
    (#114) are all optional and forwarded to OllamaProvider only; other
    providers ignore them. Omit / pass None to use the provider's own default.
    """
    normalized = provider.lower()
    if normalized == "openai":
        return OpenAIProvider(model)
    if normalized == "anthropic":
        return AnthropicProvider(model)
    if normalized == "ollama":
        kwargs: dict[str, Any] = {}
        if embed_cache_size is not None:
            kwargs["embed_cache_size"] = embed_cache_size
        return OllamaProvider(
            model, embed_model=embed_model, circuit_breaker=circuit_breaker, **kwargs
        )
    if normalized == "mock":
        # Deterministic fallback — used by the experiment harness (#57) and any
        # configuration that wants to run offline without an LLM round-trip.
        return MockProvider()
    raise ValueError(f"Unsupported provider: {provider}")
