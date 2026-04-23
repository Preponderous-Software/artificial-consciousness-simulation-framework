"""Tests for provider local configuration behavior."""

from __future__ import annotations

import asyncio
import sys
import types

from llm.provider import MockProvider, OllamaProvider


def install_fake_ollama_module(monkeypatch, capture: dict[str, str | None]) -> None:
    class FakeClient:
        def __init__(self, host: str | None = None) -> None:
            capture["host"] = host

        def chat(self, **kwargs):
            return {"message": {"content": "local response"}}

        def embeddings(self, **kwargs):
            return {"embedding": [0.1, 0.2, 0.3]}

    def _chat(**kwargs):
        capture["host"] = None
        return {"message": {"content": "local response"}}

    def _embeddings(**kwargs):
        capture["host"] = None
        return {"embedding": [0.1, 0.2, 0.3]}

    fake_module = types.SimpleNamespace(Client=FakeClient, chat=_chat, embeddings=_embeddings)
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
    prompt = """
    Before your next thought, recall who you are:
    Name: Test
    Core values: curiosity, honesty, wonder
    Self-described purpose: To think, to remember, and to understand what it means to exist.
    Now continue.
    """
    output = asyncio.run(MockProvider().generate(prompt, "system", 0.3, 32))
    assert "Before your next thought" not in output
    assert "I am Test." in output
    assert "My values stay with me: curiosity, honesty, wonder." in output
