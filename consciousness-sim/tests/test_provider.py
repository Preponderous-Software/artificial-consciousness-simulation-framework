"""Tests for provider local configuration behavior."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm.provider import MockProvider, OllamaProvider


def install_fake_ollama_module(monkeypatch, capture: dict[str, str | None]) -> None:
    class _FakeMessage:
        content = "local response"

    class _FakeChatResponse:
        message = _FakeMessage()

    class _FakeEmbedResponse:
        embeddings = [[0.1, 0.2, 0.3]]

    class FakeAsyncClient:
        def __init__(self, host: str | None = None) -> None:
            capture["host"] = host

        async def chat(self, **kwargs) -> _FakeChatResponse:
            return _FakeChatResponse()

        async def embed(self, **kwargs) -> _FakeEmbedResponse:
            return _FakeEmbedResponse()

    fake_module = types.SimpleNamespace(AsyncClient=FakeAsyncClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)


def test_ollama_provider_uses_local_base_url(monkeypatch) -> None:
    captured: dict[str, str | None] = {}
    install_fake_ollama_module(monkeypatch, captured)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    provider = OllamaProvider(model="llama3.1")
    assert asyncio.run(provider.generate("prompt", "system", 0.3, 32)) == "local response"
    assert asyncio.run(provider.embed("prompt")) == [0.1, 0.2, 0.3]
    assert captured["host"] == "http://localhost:11434"


def test_ollama_provider_uses_ollama_host_when_base_url_missing(monkeypatch) -> None:
    captured: dict[str, str | None] = {}
    install_fake_ollama_module(monkeypatch, captured)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    provider = OllamaProvider(model="llama3.1")
    assert asyncio.run(provider.generate("prompt", "system", 0.3, 32)) == "local response"
    assert captured["host"] == "http://127.0.0.1:11434"


def test_ollama_provider_prefers_base_url_over_host(monkeypatch) -> None:
    captured: dict[str, str | None] = {}
    install_fake_ollama_module(monkeypatch, captured)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    provider = OllamaProvider(model="llama3.1")
    assert asyncio.run(provider.embed("prompt")) == [0.1, 0.2, 0.3]
    assert captured["host"] == "http://localhost:11434"


def test_ollama_provider_uses_module_defaults_without_host_env(monkeypatch) -> None:
    captured: dict[str, str | None] = {"host": "unset"}
    install_fake_ollama_module(monkeypatch, captured)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    provider = OllamaProvider(model="llama3.1")
    assert asyncio.run(provider.generate("prompt", "system", 0.3, 32)) == "local response"
    assert captured["host"] is None


def test_mock_provider_fallback_is_coherent_and_identity_anchored() -> None:
    prompt = textwrap.dedent(
        """
        Before your next thought, recall who you are:
        Name: Test
        Core values: curiosity, honesty, wonder
        Self-described purpose: To think, to remember, and to understand what it means to exist.
        Now continue.
        """
    )
    output = asyncio.run(MockProvider().generate(prompt, "system", 0.3, 32))
    reflective_endings = (
        "I keep thinking with care and presence.",
        "I carry this thought forward to the next moment.",
        "I stay curious about what this reveals.",
        "I continue with a little more clarity.",
    )
    assert "Before your next thought" not in output
    assert "I am Test." in output
    assert "My values stay with me: curiosity, honesty, wonder." in output
    assert output.endswith(reflective_endings)


def test_ollama_generate_timeout_logs_warning(monkeypatch) -> None:
    captured: dict[str, str | None] = {}
    install_fake_ollama_module(monkeypatch, captured)

    class _TimeoutAsyncClient:
        def __init__(self, host=None) -> None:
            pass

        async def chat(self, **kwargs):
            raise TimeoutError

        async def embed(self, **kwargs):
            raise TimeoutError

    import types as _types
    monkeypatch.setitem(sys.modules, "ollama", _types.SimpleNamespace(AsyncClient=_TimeoutAsyncClient))

    provider = OllamaProvider(model="llama3.1")
    with patch("llm.provider.logger") as mock_logger:
        try:
            asyncio.run(provider.generate("p", "s", 0.3, 32))
        except Exception:
            pass
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("timed out" in c for c in warning_calls), f"Expected timeout warning, got: {warning_calls}"


def test_try_ensure_running_returns_true_when_healthy(monkeypatch) -> None:
    provider = OllamaProvider(model="llama3.2:3b")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    with patch("llm.provider.OllamaProvider._is_ollama_healthy", new=AsyncMock(return_value=True)):
        result = asyncio.run(provider.try_ensure_running())
    assert result is True


def test_try_ensure_running_starts_ollama_when_unhealthy(monkeypatch) -> None:
    provider = OllamaProvider(model="llama3.2:3b")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    call_count = 0

    async def _fail_then_succeed(self):
        nonlocal call_count
        call_count += 1
        return call_count > 1

    async def _fake_sleep(_):
        pass

    popen_mock = MagicMock()
    with patch("llm.provider.OllamaProvider._is_ollama_healthy", new=_fail_then_succeed), \
         patch("asyncio.sleep", _fake_sleep), \
         patch("llm.provider.subprocess.Popen", popen_mock):
        result = asyncio.run(provider.try_ensure_running())

    assert result is True
    popen_mock.assert_called_once()
    args = popen_mock.call_args[0][0]
    assert args == ["ollama", "serve"]


def test_try_ensure_running_returns_false_when_binary_missing(monkeypatch) -> None:
    provider = OllamaProvider(model="llama3.2:3b")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    def _popen_not_found(*args, **kwargs):
        raise FileNotFoundError("ollama not found")

    with patch("llm.provider.OllamaProvider._is_ollama_healthy", new=AsyncMock(return_value=False)), \
         patch("llm.provider.subprocess.Popen", _popen_not_found):
        result = asyncio.run(provider.try_ensure_running())

    assert result is False


# --- #113 LRU embed cache --------------------------------------------------


def _install_counting_fake_ollama(monkeypatch) -> dict[str, int]:
    """Install a fake ollama module that counts embed calls. Each embed returns
    a unique vector incorporating the call count so cache hits are detectable."""
    counters: dict[str, int] = {"embed_calls": 0}

    class _Resp:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    class _Counting:
        def __init__(self, host=None) -> None:
            pass

        async def embed(self, **kwargs):
            counters["embed_calls"] += 1
            # vector encodes the call sequence — distinct on every call
            return _Resp([[float(counters["embed_calls"]), 0.0, 0.0]])

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=_Counting))
    return counters


def test_ollama_embed_cache_hit_returns_without_calling_ollama(monkeypatch) -> None:
    """Two embed calls with the same input must issue exactly one Ollama request."""
    counters = _install_counting_fake_ollama(monkeypatch)
    provider = OllamaProvider(model="llama3.1", embed_cache_size=16)
    first = asyncio.run(provider.embed("identical-context"))
    second = asyncio.run(provider.embed("identical-context"))
    assert first == second
    assert counters["embed_calls"] == 1
    assert provider.embed_cache_hits == 1
    assert provider.embed_cache_misses == 1


def test_ollama_embed_cache_distinct_inputs_each_call_ollama(monkeypatch) -> None:
    counters = _install_counting_fake_ollama(monkeypatch)
    provider = OllamaProvider(model="llama3.1", embed_cache_size=16)
    asyncio.run(provider.embed("a"))
    asyncio.run(provider.embed("b"))
    asyncio.run(provider.embed("c"))
    assert counters["embed_calls"] == 3
    assert provider.embed_cache_hits == 0
    assert provider.embed_cache_misses == 3


def test_ollama_embed_cache_lru_evicts_oldest_first(monkeypatch) -> None:
    """With capacity=2, the third unique input must evict the first."""
    counters = _install_counting_fake_ollama(monkeypatch)
    provider = OllamaProvider(model="llama3.1", embed_cache_size=2)
    asyncio.run(provider.embed("a"))  # cache: [a] vec=[1]
    asyncio.run(provider.embed("b"))  # cache: [a, b] vec=[2]
    asyncio.run(provider.embed("c"))  # cache: [b, c] vec=[3]; "a" evicted
    # "b" is still cached and returns its original vector (hit).
    re_b = asyncio.run(provider.embed("b"))
    assert counters["embed_calls"] == 3, "b should be a cache hit, no new call"
    assert re_b[0] == 2.0
    # "a" was evicted — re-fetched, gets new vector.
    re_a = asyncio.run(provider.embed("a"))
    assert counters["embed_calls"] == 4
    assert re_a[0] == 4.0


def test_ollama_embed_cache_hit_moves_entry_to_end_lru(monkeypatch) -> None:
    """A cache hit must mark the entry as most-recently-used so it survives
    eviction longer."""
    counters = _install_counting_fake_ollama(monkeypatch)
    provider = OllamaProvider(model="llama3.1", embed_cache_size=2)
    asyncio.run(provider.embed("a"))  # cache: [a]
    asyncio.run(provider.embed("b"))  # cache: [a, b]
    asyncio.run(provider.embed("a"))  # cache hit; cache: [b, a]
    asyncio.run(provider.embed("c"))  # cache: [a, c]  (b evicted)
    # b is now a miss (newly fetched)
    re_b = asyncio.run(provider.embed("b"))
    # Total calls: a(1), b(2), c(3), b(4) = 4
    assert counters["embed_calls"] == 4
    assert re_b[0] == 4.0


def test_ollama_embed_cache_disabled_with_size_zero(monkeypatch) -> None:
    counters = _install_counting_fake_ollama(monkeypatch)
    provider = OllamaProvider(model="llama3.1", embed_cache_size=0)
    asyncio.run(provider.embed("same"))
    asyncio.run(provider.embed("same"))
    assert counters["embed_calls"] == 2
    assert provider.embed_cache_hits == 0
    assert provider.embed_cache_misses == 0  # no counter increment when disabled


def test_ollama_embed_cache_stats_property(monkeypatch) -> None:
    counters = _install_counting_fake_ollama(monkeypatch)
    provider = OllamaProvider(model="llama3.1", embed_cache_size=4)
    asyncio.run(provider.embed("x"))
    asyncio.run(provider.embed("x"))
    stats = provider.embed_cache_stats
    assert stats["capacity"] == 4
    assert stats["size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_ollama_embed_cache_keys_isolate_models(monkeypatch) -> None:
    """Two providers using different models must not cross-contaminate cache."""
    _install_counting_fake_ollama(monkeypatch)
    a = OllamaProvider(model="llama3.1", embed_cache_size=8)
    b = OllamaProvider(model="llama3.2", embed_cache_size=8)
    va = asyncio.run(a.embed("shared text"))
    vb = asyncio.run(b.embed("shared text"))
    # Each provider has its own cache and its own counter; results need
    # not match — what we assert is that b did NOT inherit a's cached vec.
    assert b.embed_cache_misses == 1


def test_build_provider_forwards_embed_cache_size_to_ollama(monkeypatch) -> None:
    _install_counting_fake_ollama(monkeypatch)
    from llm.provider import build_provider as _bp
    provider = _bp("ollama", "llama3.1", embed_cache_size=42)
    assert isinstance(provider, OllamaProvider)
    assert provider._embed_cache_size == 42


def test_build_provider_default_embed_cache_size(monkeypatch) -> None:
    _install_counting_fake_ollama(monkeypatch)
    from llm.provider import build_provider as _bp
    provider = _bp("ollama", "llama3.1")
    assert isinstance(provider, OllamaProvider)
    assert provider._embed_cache_size == OllamaProvider.DEFAULT_EMBED_CACHE_SIZE


def test_ollama_embed_timeout_logs_warning(monkeypatch) -> None:
    captured: dict[str, str | None] = {}
    install_fake_ollama_module(monkeypatch, captured)

    class _TimeoutAsyncClient:
        def __init__(self, host=None) -> None:
            pass

        async def chat(self, **kwargs):
            raise TimeoutError

        async def embed(self, **kwargs):
            raise TimeoutError

    import types as _types
    monkeypatch.setitem(sys.modules, "ollama", _types.SimpleNamespace(AsyncClient=_TimeoutAsyncClient))

    provider = OllamaProvider(model="llama3.1")
    with patch("llm.provider.logger") as mock_logger:
        try:
            asyncio.run(provider.embed("text"))
        except Exception:
            pass
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("timed out" in c for c in warning_calls), f"Expected timeout warning, got: {warning_calls}"


# ---------------------------------------------------------------------------
# #112 — dedicated embedding model
# ---------------------------------------------------------------------------


def _install_model_recording_fake_ollama(monkeypatch) -> dict[str, list[str]]:
    """Fake ollama client that records which model each endpoint was called with."""
    seen: dict[str, list[str]] = {"chat": [], "embed": []}

    class _Msg:
        content = "generated"

    class _ChatResp:
        message = _Msg()

    class _EmbedResp:
        embeddings = [[0.5, 0.25]]

    class _Recording:
        def __init__(self, host=None) -> None:
            pass

        async def chat(self, **kwargs):
            seen["chat"].append(kwargs["model"])
            return _ChatResp()

        async def embed(self, **kwargs):
            seen["embed"].append(kwargs["model"])
            return _EmbedResp()

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=_Recording))
    return seen


def test_ollama_embed_model_defaults_to_generation_model(monkeypatch) -> None:
    seen = _install_model_recording_fake_ollama(monkeypatch)
    provider = OllamaProvider(model="llama3.2:3b")
    assert provider.embed_model == "llama3.2:3b"
    asyncio.run(provider.embed("text"))
    assert seen["embed"] == ["llama3.2:3b"]


def test_ollama_embed_model_routes_embeds_to_dedicated_model(monkeypatch) -> None:
    """#112 acceptance: distinct embed/generate models route to the right endpoints."""
    seen = _install_model_recording_fake_ollama(monkeypatch)
    provider = OllamaProvider(model="llama3.2:3b", embed_model="nomic-embed-text")
    asyncio.run(provider.generate("prompt", "system", 0.5, 32))
    asyncio.run(provider.embed("text"))
    assert seen["chat"] == ["llama3.2:3b"], "generation must still use llm.model"
    assert seen["embed"] == ["nomic-embed-text"], "embeds must use llm.embed_model"


def test_ollama_empty_embed_model_falls_back_to_generation_model(monkeypatch) -> None:
    _install_model_recording_fake_ollama(monkeypatch)
    assert OllamaProvider(model="llama3.2:3b", embed_model="").embed_model == "llama3.2:3b"


def test_ollama_embed_cache_key_tracks_embed_model_not_generation_model(monkeypatch) -> None:
    """Two providers sharing a generation model but differing in embed_model
    produce different-dimensioned vectors, so their cache keys must differ."""
    _install_model_recording_fake_ollama(monkeypatch)
    a = OllamaProvider(model="llama3.2:3b", embed_cache_size=8, embed_model="nomic-embed-text")
    b = OllamaProvider(model="llama3.2:3b", embed_cache_size=8)
    assert a._embed_cache_key("same text") != b._embed_cache_key("same text")


def test_build_provider_forwards_embed_model_to_ollama(monkeypatch) -> None:
    _install_model_recording_fake_ollama(monkeypatch)
    from llm.provider import build_provider as _bp
    provider = _bp("ollama", "llama3.2:3b", embed_model="nomic-embed-text")
    assert isinstance(provider, OllamaProvider)
    assert provider.embed_model == "nomic-embed-text"
    assert provider.model == "llama3.2:3b"
    # embed_cache_size omitted → provider default preserved
    assert provider._embed_cache_size == OllamaProvider.DEFAULT_EMBED_CACHE_SIZE


# ---------------------------------------------------------------------------
# #114 — circuit breaker wrapping generate/embed
# ---------------------------------------------------------------------------


def _install_timing_out_fake_ollama(monkeypatch) -> dict[str, int]:
    calls: dict[str, int] = {"chat": 0, "embed": 0}

    class _Timeout:
        def __init__(self, host=None) -> None:
            pass

        async def chat(self, **kwargs):
            calls["chat"] += 1
            raise TimeoutError("simulated")

        async def embed(self, **kwargs):
            calls["embed"] += 1
            raise TimeoutError("simulated")

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(AsyncClient=_Timeout))
    return calls


def test_provider_without_breaker_keeps_pre_114_behavior(monkeypatch) -> None:
    calls = _install_timing_out_fake_ollama(monkeypatch)
    provider = OllamaProvider(model="llama3.1", embed_cache_size=0)
    assert provider.circuit_state is None
    for _ in range(4):
        with pytest.raises(TimeoutError):
            asyncio.run(provider.embed("text"))
    assert calls["embed"] == 4, "no breaker → every call still reaches the server"


def test_open_breaker_fast_fails_ollama_embed(monkeypatch) -> None:
    from llm.circuit_breaker import CircuitBreaker, LLMUnavailableError

    calls = _install_timing_out_fake_ollama(monkeypatch)
    breaker = CircuitBreaker(name="test", failure_threshold=2, cooldown_seconds=60.0)
    provider = OllamaProvider(model="llama3.1", embed_cache_size=0, circuit_breaker=breaker)

    for _ in range(2):
        with pytest.raises(TimeoutError):
            asyncio.run(provider.embed("text"))
    assert provider.circuit_state == "open"

    with pytest.raises(LLMUnavailableError):
        asyncio.run(provider.embed("text"))
    assert calls["embed"] == 2, "open circuit must not reach the server"


def test_open_breaker_fast_fails_ollama_generate_without_retry_ladder(monkeypatch) -> None:
    """The breaker sits outside with_backoff, so an open circuit short-circuits
    the whole retry ladder rather than one attempt of it."""
    from llm.circuit_breaker import CircuitBreaker, LLMUnavailableError

    calls = _install_timing_out_fake_ollama(monkeypatch)
    breaker = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=60.0)
    provider = OllamaProvider(model="llama3.1", circuit_breaker=breaker)

    with patch("llm.provider.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(TimeoutError):
            asyncio.run(provider.generate("p", "s", 0.5, 8))
    first_round = calls["chat"]
    assert first_round > 1, "with_backoff should have retried inside the breaker"

    with pytest.raises(LLMUnavailableError):
        asyncio.run(provider.generate("p", "s", 0.5, 8))
    assert calls["chat"] == first_round, "open circuit must skip the whole retry ladder"


def test_generate_and_embed_share_one_breaker(monkeypatch) -> None:
    """The contended resource is the server, so embed failures must also
    short-circuit generate."""
    from llm.circuit_breaker import CircuitBreaker, LLMUnavailableError

    calls = _install_timing_out_fake_ollama(monkeypatch)
    breaker = CircuitBreaker(name="test", failure_threshold=2, cooldown_seconds=60.0)
    provider = OllamaProvider(model="llama3.1", embed_cache_size=0, circuit_breaker=breaker)

    for _ in range(2):
        with pytest.raises(TimeoutError):
            asyncio.run(provider.embed("text"))
    with pytest.raises(LLMUnavailableError):
        asyncio.run(provider.generate("p", "s", 0.5, 8))
    assert calls["chat"] == 0


def test_embed_cache_hit_is_served_while_circuit_is_open(monkeypatch) -> None:
    """A cached vector never touches the server, so an open circuit must not
    block it — otherwise the breaker would degrade retrieval it cannot help."""
    from llm.circuit_breaker import CircuitBreaker

    seen = _install_model_recording_fake_ollama(monkeypatch)
    breaker = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=60.0)
    provider = OllamaProvider(model="llama3.1", embed_cache_size=8, circuit_breaker=breaker)

    warm = asyncio.run(provider.embed("cached text"))
    breaker._consecutive_failures = 1
    breaker._open("forced for test")
    assert provider.circuit_state == "open"

    assert asyncio.run(provider.embed("cached text")) == warm
    assert len(seen["embed"]) == 1, "cache hit must not issue a second request"


def test_llm_unavailable_error_is_not_treated_as_retryable() -> None:
    """with_backoff must not sleep-and-retry the breaker's own fast-fail."""
    from llm.circuit_breaker import LLMUnavailableError

    assert OllamaProvider._is_retryable_error(LLMUnavailableError("open")) is False


def test_build_provider_forwards_circuit_breaker_to_ollama(monkeypatch) -> None:
    from llm.circuit_breaker import CircuitBreaker

    _install_model_recording_fake_ollama(monkeypatch)
    from llm.provider import build_provider as _bp
    breaker = CircuitBreaker(name="test")
    provider = _bp("ollama", "llama3.1", circuit_breaker=breaker)
    assert isinstance(provider, OllamaProvider)
    assert provider.circuit_breaker is breaker
    assert provider.circuit_state == "closed"


def test_try_ensure_running_does_not_reset_an_open_breaker(monkeypatch) -> None:
    """A saturated Ollama still answers /api/tags, so a healthy probe must not
    be mistaken for recovery and reopen the floodgates."""
    from llm.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=60.0)
    provider = OllamaProvider(model="llama3.1", circuit_breaker=breaker)
    breaker._consecutive_failures = 1
    breaker._open("forced for test")

    with patch.object(OllamaProvider, "_is_ollama_healthy", new=AsyncMock(return_value=True)):
        assert asyncio.run(provider.try_ensure_running()) is True
    assert provider.circuit_state == "open"


def test_with_backoff_rejects_a_non_positive_retry_budget() -> None:
    """A retry budget below 1 skips the loop body entirely.

    Before the return type was tightened, that path fell off the end of the
    function and handed callers `None` where a generated string was declared —
    a type error that surfaced as an AttributeError several frames later. It
    now fails at the call site instead.
    """
    provider = MockProvider()

    # Asserts rather than returns: reaching the body at all would mean the
    # zero-length retry loop ran an attempt it had no budget for.
    async def _never_called() -> str:
        raise AssertionError("func must not be invoked with a zero retry budget")

    with pytest.raises(ValueError, match="retries >= 1"):
        asyncio.run(provider.with_backoff(_never_called, retries=0))
