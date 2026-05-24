"""Tests for the Discord webhook sink (issue #56).

Covers:
- URL allowlist + masking (URL never appears in logs)
- Embed body format per event type (color matches dashboard palette)
- Token-bucket rate limiting + throttled drop warnings
- Truncation
- Never-raises contract (HTTP errors / timeouts / exceptions all swallowed + logged)
- Env-var substitution in config + factory rejection of unresolved ${VAR}
- Consciousness wiring with discord.enabled=true / false / section absent
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from interfaces.discord.webhook import (
    DiscordWebhookSink,
    _mask_url,
    build_sink_from_config,
)


VALID_URL = "https://discord.com/api/webhooks/123456789012345678/abcdefSECRETtoken_xyz"
VALID_URL_MASKED = "https://discord.com/api/webhooks/***/***"


# ---------------------------------------------------------------------------
# Helpers — fake httpx (mirrors test_provider.py / test_perception.py pattern)
# ---------------------------------------------------------------------------

def install_fake_httpx(monkeypatch, *, status_code: int = 204, raise_on_post: Exception | None = None):
    """Install a fake httpx module on sys.modules. Returns a captured-state dict."""
    captured: dict[str, object] = {"posts": []}

    class _Response:
        def __init__(self) -> None:
            self.status_code = status_code

    class _FakeClient:
        def __init__(self, timeout=None, **kwargs) -> None:
            captured["client_kwargs"] = {"timeout": timeout, **kwargs}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json=None):
            captured["posts"].append({"url": url, "json": json})
            if raise_on_post is not None:
                raise raise_on_post
            return _Response()

    fake = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake)
    return captured


# ---------------------------------------------------------------------------
# URL handling — allowlist + masking
# ---------------------------------------------------------------------------

def test_mask_url_replaces_id_and_token() -> None:
    assert _mask_url(VALID_URL) == VALID_URL_MASKED
    assert _mask_url("https://discordapp.com/api/webhooks/x/y") == "https://discordapp.com/api/webhooks/***/***"
    assert _mask_url("") == ""


def test_init_rejects_non_allowlisted_host() -> None:
    with pytest.raises(ValueError, match="not in allowlist"):
        DiscordWebhookSink("https://evil.example.com/api/webhooks/1/2", events={"thought"})


def test_init_rejects_unresolved_env_var_reference() -> None:
    with pytest.raises(ValueError, match="unresolved"):
        DiscordWebhookSink("${MISSING_WEBHOOK}", events={"thought"})


def test_init_rejects_empty_url() -> None:
    with pytest.raises(ValueError):
        DiscordWebhookSink("", events={"thought"})


def test_repr_masks_url() -> None:
    sink = DiscordWebhookSink(VALID_URL, events={"thought"})
    assert "SECRETtoken_xyz" not in repr(sink)
    assert "***" in repr(sink)


def test_secret_never_appears_in_log_records(monkeypatch, caplog) -> None:
    """Belt-and-braces masking: even a code path that logs the raw URL is scrubbed."""
    install_fake_httpx(monkeypatch, status_code=500)
    sink = DiscordWebhookSink(VALID_URL, events={"thought"})

    with caplog.at_level(logging.WARNING, logger="interfaces.discord.webhook"):
        asyncio.run(sink._post({"type": "thought", "content": "hi"}))

    assert "SECRETtoken_xyz" not in caplog.text, \
        "raw webhook token appeared in a log record — secret leak"
    assert "***" in caplog.text


# ---------------------------------------------------------------------------
# Embed body format
# ---------------------------------------------------------------------------

def test_post_builds_thought_embed_with_correct_color(monkeypatch) -> None:
    captured = install_fake_httpx(monkeypatch)
    sink = DiscordWebhookSink(VALID_URL, events={"thought"}, username="Sage")
    asyncio.run(sink._post({"type": "thought", "content": "I observe."}))

    body = captured["posts"][0]["json"]
    assert body["username"] == "Sage"
    embed = body["embeds"][0]
    assert embed["description"] == "I observe."
    assert embed["color"] == 0x7c83ff   # accent (matches dashboard --accent)
    assert embed["footer"]["text"] == "thought"


def test_post_uses_perception_color_and_attaches_url(monkeypatch) -> None:
    captured = install_fake_httpx(monkeypatch)
    sink = DiscordWebhookSink(VALID_URL, events={"perception"}, include_perception_url=True)
    asyncio.run(sink._post({
        "type": "perception",
        "content": "[wikipedia: Sodium chlorate] An inorganic compound…",
        "source": "wikipedia",
        "title": "Sodium chlorate",
        "url": "https://en.wikipedia.org/wiki/Sodium_chlorate",
    }))
    embed = captured["posts"][0]["json"]["embeds"][0]
    assert embed["color"] == 0xfbbf24                                # perceive amber
    assert embed["url"] == "https://en.wikipedia.org/wiki/Sodium_chlorate"
    assert embed["title"] == "Sodium chlorate"
    assert "wikipedia: Sodium chlorate" in embed["footer"]["text"]


def test_post_truncates_long_content_with_suffix(monkeypatch) -> None:
    captured = install_fake_httpx(monkeypatch)
    sink = DiscordWebhookSink(VALID_URL, events={"thought"}, truncate_chars=100)
    long_text = "x" * 500
    asyncio.run(sink._post({"type": "thought", "content": long_text}))
    desc = captured["posts"][0]["json"]["embeds"][0]["description"]
    assert len(desc) <= 100
    assert desc.endswith("… [truncated]")


def test_post_falls_back_to_thought_color_for_unknown_type(monkeypatch) -> None:
    captured = install_fake_httpx(monkeypatch)
    sink = DiscordWebhookSink(VALID_URL, events={"thought"})
    asyncio.run(sink._post({"type": "made_up_event", "content": "hi"}))
    assert captured["posts"][0]["json"]["embeds"][0]["color"] == 0x7c83ff


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limit_drops_after_capacity_exhausted(monkeypatch) -> None:
    captured = install_fake_httpx(monkeypatch)
    sink = DiscordWebhookSink(VALID_URL, events={"thought"}, rate_limit_per_min=3)

    async def burst() -> None:
        for _ in range(10):
            await sink._post({"type": "thought", "content": "spam"})

    asyncio.run(burst())
    # Bucket starts full at 3; rapid bursts get only ~3 through
    assert 1 <= len(captured["posts"]) <= 4


def test_rate_limit_drop_warning_throttled_to_once_per_minute(monkeypatch, caplog) -> None:
    install_fake_httpx(monkeypatch)
    sink = DiscordWebhookSink(VALID_URL, events={"thought"}, rate_limit_per_min=1)

    async def burst() -> None:
        for _ in range(20):
            await sink._post({"type": "thought", "content": "x"})

    with caplog.at_level(logging.WARNING, logger="interfaces.discord.webhook"):
        asyncio.run(burst())

    # Many drops, but the warning fires at most once in a single test span
    drop_warnings = [r for r in caplog.records if "rate-limit" in r.getMessage().lower()]
    assert 1 <= len(drop_warnings) <= 2  # 1 normally; second only if minute-boundary crossed


# ---------------------------------------------------------------------------
# Never-raises contract
# ---------------------------------------------------------------------------

def test_post_swallows_http_500(monkeypatch, caplog) -> None:
    install_fake_httpx(monkeypatch, status_code=500)
    sink = DiscordWebhookSink(VALID_URL, events={"thought"})
    with caplog.at_level(logging.WARNING, logger="interfaces.discord.webhook"):
        asyncio.run(sink._post({"type": "thought", "content": "x"}))
    assert any("HTTP 500" in r.getMessage() for r in caplog.records)


def test_post_swallows_429_rate_limit(monkeypatch, caplog) -> None:
    install_fake_httpx(monkeypatch, status_code=429)
    sink = DiscordWebhookSink(VALID_URL, events={"thought"})
    with caplog.at_level(logging.WARNING, logger="interfaces.discord.webhook"):
        asyncio.run(sink._post({"type": "thought", "content": "x"}))
    assert any("429" in r.getMessage() for r in caplog.records)


def test_post_swallows_connection_error(monkeypatch, caplog) -> None:
    install_fake_httpx(monkeypatch, raise_on_post=ConnectionError("boom"))
    sink = DiscordWebhookSink(VALID_URL, events={"thought"})
    with caplog.at_level(logging.WARNING, logger="interfaces.discord.webhook"):
        # Must not raise — thought cycles cannot be broken by Discord outages
        asyncio.run(sink._post({"type": "thought", "content": "x"}))
    assert any("POST failed" in r.getMessage() for r in caplog.records)


def test_post_swallows_timeout(monkeypatch) -> None:
    install_fake_httpx(monkeypatch, raise_on_post=TimeoutError("slow"))
    sink = DiscordWebhookSink(VALID_URL, events={"thought"})
    # Should not raise
    asyncio.run(sink._post({"type": "thought", "content": "x"}))


# ---------------------------------------------------------------------------
# register() — hooks the requested event channels
# ---------------------------------------------------------------------------

def test_register_subscribes_to_each_requested_event(monkeypatch) -> None:
    install_fake_httpx(monkeypatch)
    mind = type("M", (), {
        "name": "Test",
        "on_thought": [], "on_reflection": [], "on_perception": [],
        "on_identity_shift": [], "on_memory_stored": [],
    })()

    sink = DiscordWebhookSink(VALID_URL, events={"thought", "perception"})
    sink.register(mind)

    assert len(mind.on_thought) == 1
    assert len(mind.on_perception) == 1
    assert mind.on_reflection == []        # not requested
    assert mind.on_identity_shift == []    # not requested


def test_register_picks_up_mind_name_when_username_unset(monkeypatch) -> None:
    install_fake_httpx(monkeypatch)
    mind = type("M", (), {
        "name": "Rafael",
        "on_thought": [], "on_reflection": [], "on_perception": [],
        "on_identity_shift": [], "on_memory_stored": [],
    })()
    sink = DiscordWebhookSink(VALID_URL, events={"thought"}, username=None)
    sink.register(mind)
    assert sink.username == "Rafael"


def test_register_warns_when_event_channel_missing(monkeypatch, caplog) -> None:
    install_fake_httpx(monkeypatch)
    # Mind without on_perception (older Consciousness)
    mind = type("M", (), {
        "name": "Old", "on_thought": [], "on_reflection": [],
        "on_identity_shift": [], "on_memory_stored": [],
    })()
    sink = DiscordWebhookSink(VALID_URL, events={"perception"})
    with caplog.at_level(logging.WARNING, logger="interfaces.discord.webhook"):
        sink.register(mind)
    assert any("no on_perception channel" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Factory + config-level integration
# ---------------------------------------------------------------------------

def test_factory_returns_none_when_disabled() -> None:
    assert build_sink_from_config({"enabled": False}) is None
    assert build_sink_from_config({}) is None
    assert build_sink_from_config(None) is None  # type: ignore[arg-type]


def test_factory_builds_sink_when_enabled(monkeypatch) -> None:
    install_fake_httpx(monkeypatch)
    sink = build_sink_from_config({
        "enabled": True,
        "webhook_url": VALID_URL,
        "events": ["thought"],
        "rate_limit": {"max_per_minute": 10},
        "truncate_chars": 500,
    })
    assert isinstance(sink, DiscordWebhookSink)
    assert sink.events == {"thought"}
    assert sink.truncate_chars == 500


def test_factory_rejects_unresolved_env_var() -> None:
    """If discord.enabled=true but the env var isn't set, fail loudly at startup."""
    with pytest.raises(ValueError, match="unresolved"):
        build_sink_from_config({
            "enabled": True,
            "webhook_url": "${UNSET_DISCORD_WEBHOOK_FOR_TEST}",
        })


# ---------------------------------------------------------------------------
# Consciousness wiring — env substitution + sink construction
# ---------------------------------------------------------------------------

def _base_cfg(**discord_overrides) -> dict:
    return {
        "consciousness": {"origin_story": "o", "values": ["v"], "purpose": "p"},
        "llm": {"provider": "ollama", "model": "llama3"},
        "thought_loop": {
            "min_interval_seconds": 0, "max_interval_seconds": 0,
            "reflection_probability": 0.0, "existential_inquiry_every_n_thoughts": 0,
        },
        "memory": {
            "short_term_capacity": 5, "consolidation_interval_minutes": 5,
            "forgetting_curve_enabled": False, "importance_decay_rate": 0.01,
        },
        "mood": {"initial": {"curiosity": 0.5}, "drift_rate": 0.01},
        "perception": {
            "enabled": False, "provider": "mock",
            "every_n_cycles": 0, "timeout_seconds": 1.0, "cache_last_n": 0,
        },
        "discord": discord_overrides,
    }


def test_consciousness_with_no_discord_section_works(tmp_path, monkeypatch) -> None:
    """Discord section is optional — older configs without it must continue to load."""
    from core.consciousness import Consciousness
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    cfg = _base_cfg()
    del cfg["discord"]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    mind = Consciousness(name="A", config_path=str(cfg_path))
    assert mind.discord_sink is None


def test_consciousness_with_discord_disabled_yields_no_sink(tmp_path, monkeypatch) -> None:
    from core.consciousness import Consciousness
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    cfg = _base_cfg(enabled=False, webhook_url=VALID_URL)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    mind = Consciousness(name="A", config_path=str(cfg_path))
    assert mind.discord_sink is None


def test_consciousness_expands_env_var_in_discord_url(tmp_path, monkeypatch) -> None:
    """Env-var substitution happens at config load, before sink construction."""
    from core.consciousness import Consciousness
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_WH_URL", VALID_URL)

    cfg = _base_cfg(
        enabled=True,
        webhook_url="${TEST_WH_URL}",
        events=["thought", "perception"],
        rate_limit={"max_per_minute": 25},
        truncate_chars=1800,
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    mind = Consciousness(name="A", config_path=str(cfg_path))
    assert mind.discord_sink is not None
    assert mind.discord_sink.webhook_url == VALID_URL    # env var resolved
    assert mind.discord_sink.events == {"thought", "perception"}
    # Sink registered on the requested channels
    assert len(mind.on_thought) == 1
    assert len(mind.on_perception) == 1


def test_consciousness_raises_clearly_when_discord_enabled_but_env_missing(tmp_path, monkeypatch) -> None:
    from core.consciousness import Consciousness
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    monkeypatch.delenv("NEVER_SET_WEBHOOK", raising=False)

    cfg = _base_cfg(enabled=True, webhook_url="${NEVER_SET_WEBHOOK}")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="unresolved|empty"):
        Consciousness(name="A", config_path=str(cfg_path))
