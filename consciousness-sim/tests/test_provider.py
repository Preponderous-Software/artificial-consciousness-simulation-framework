"""Tests for provider local configuration behavior."""

from __future__ import annotations

import asyncio
import sys
import types

from llm.provider import OllamaProvider


def test_ollama_provider_uses_local_base_url(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, host: str | None = None) -> None:
            self.host = host

        def chat(self, **kwargs):
            assert self.host == "http://localhost:11434"
            return {"message": {"content": "local response"}}

        def embeddings(self, **kwargs):
            assert self.host == "http://localhost:11434"
            return {"embedding": [0.1, 0.2, 0.3]}

    fake_module = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_module)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    provider = OllamaProvider(model="llama3.1")
    assert asyncio.run(provider.generate("prompt", "system", 0.3, 32)) == "local response"
    assert asyncio.run(provider.embed("prompt")) == [0.1, 0.2, 0.3]
