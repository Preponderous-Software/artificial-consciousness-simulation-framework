"""Tests for provider local configuration behavior."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import types
from unittest.mock import AsyncMock, MagicMock, patch

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


def test_try_ensure_running_returns_true_when_reachable(monkeypatch) -> None:
    provider = OllamaProvider(model="llama3.2:3b")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    async def _fake_open_connection(host, port):
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    with patch("asyncio.open_connection", _fake_open_connection):
        result = asyncio.run(provider.try_ensure_running())
    assert result is True


def test_try_ensure_running_starts_ollama_when_unreachable(monkeypatch) -> None:
    provider = OllamaProvider(model="llama3.2:3b")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    call_count = 0

    async def _fail_then_succeed(host, port):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionRefusedError("not up yet")
        reader = MagicMock()
        writer = MagicMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()
        return reader, writer

    async def _fake_sleep(_):
        pass

    popen_mock = MagicMock()
    with patch("asyncio.open_connection", _fail_then_succeed), \
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

    async def _always_fail(host, port):
        raise ConnectionRefusedError("not up")

    def _popen_not_found(*args, **kwargs):
        raise FileNotFoundError("ollama not found")

    with patch("asyncio.open_connection", _always_fail), \
         patch("llm.provider.subprocess.Popen", _popen_not_found):
        result = asyncio.run(provider.try_ensure_running())

    assert result is False


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
