"""Tests for WikipediaPerception's optional ``topic`` (perception.topic).

Covers:
- a topic routes each fetch through one CirrusSearch hit, then that article's
  summary; no topic keeps the random-summary endpoint and never searches
- random offsets stay inside CirrusSearch's 10,000-offset cap, and inside
  totalhits once it is known
- a first guess past the end of a small result set is redrawn inside it
- zero hits and search-API errors yield None + a WARNING (no raise)
- build_perception_provider passes topic through
- _validate_config accepts null / a query, rejects blank and non-wikipedia
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import types
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from core.consciousness import _validate_config
from llm.perception import WikipediaPerception, build_perception_provider


def _summary(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "extract": f"{title} is a concrete thing.",
        "content_urls": {"desktop": {"page": f"https://en.wikipedia.org/wiki/{title}"}},
    }


def _search(titles: list[str], totalhits: int) -> dict[str, Any]:
    return {
        "query": {
            "searchinfo": {"totalhits": totalhits},
            "search": [{"title": t} for t in titles],
        }
    }


def install_routing_httpx(monkeypatch, search_payloads: list[dict[str, Any]]) -> list[str]:
    """Fake httpx: search URLs pop the next payload; summary URLs echo their title."""
    calls: list[str] = []
    queue = list(search_payloads)

    class _Response:
        def __init__(self, payload: Any) -> None:
            self._payload = payload

        def json(self) -> Any:
            return self._payload

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def __init__(self, timeout=None, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers=None):
            calls.append(url)
            if url.startswith(WikipediaPerception.SEARCH_URL):
                return _Response(queue.pop(0))
            title = url.rsplit("/", 1)[-1]
            return _Response(_summary(title))

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(AsyncClient=_FakeClient))
    return calls


def _query(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def test_topic_fetches_one_search_hit_then_its_summary(monkeypatch) -> None:
    calls = install_routing_httpx(monkeypatch, [_search(["Concho River"], 313)])
    wp = WikipediaPerception(topic='incategory:"Rivers of Texas"', rng=random.Random(0))

    result = asyncio.run(wp.fetch())

    assert result is not None
    assert result.title == "Concho_River"
    assert result.source == "wikipedia"
    search = _query(calls[0])
    assert search["srsearch"] == 'incategory:"Rivers of Texas"'
    assert search["srlimit"] == "1"
    assert search["srnamespace"] == "0"
    assert calls[1] == WikipediaPerception.SUMMARY_URL + "Concho_River"


def test_topic_title_is_url_quoted(monkeypatch) -> None:
    calls = install_routing_httpx(monkeypatch, [_search(["Red Oak Creek (Trinity River tributary)"], 5)])
    wp = WikipediaPerception(topic="creek", rng=random.Random(0))

    asyncio.run(wp.fetch())

    assert calls[1] == WikipediaPerception.SUMMARY_URL + "Red_Oak_Creek_%28Trinity_River_tributary%29"


def test_no_topic_uses_random_endpoint_and_never_searches(monkeypatch) -> None:
    calls = install_routing_httpx(monkeypatch, [])
    wp = WikipediaPerception()

    asyncio.run(wp.fetch())

    assert calls == [WikipediaPerception.API_URL]


def test_offsets_respect_cap_then_totalhits(monkeypatch) -> None:
    calls = install_routing_httpx(
        monkeypatch,
        [_search([f"Article {i}"], 7) for i in range(20)],
    )
    wp = WikipediaPerception(topic="q", cache_last_n=0, rng=random.Random(1))

    for _ in range(20):
        asyncio.run(wp.fetch())

    offsets = [int(_query(u)["sroffset"]) for u in calls if u.startswith(WikipediaPerception.SEARCH_URL)]
    assert offsets[0] <= WikipediaPerception.MAX_SEARCH_OFFSET
    assert all(0 <= o < 7 for o in offsets[1:])


def test_large_topics_are_capped_at_search_offset_limit(monkeypatch) -> None:
    calls = install_routing_httpx(
        monkeypatch,
        [_search([f"Article {i}"], 1_678_113) for i in range(30)],
    )
    wp = WikipediaPerception(topic='"United States"', cache_last_n=0, rng=random.Random(2))

    for _ in range(30):
        asyncio.run(wp.fetch())

    offsets = [int(_query(u)["sroffset"]) for u in calls if u.startswith(WikipediaPerception.SEARCH_URL)]
    assert all(0 <= o <= WikipediaPerception.MAX_SEARCH_OFFSET for o in offsets)


def test_first_guess_past_small_result_set_is_redrawn(monkeypatch) -> None:
    calls = install_routing_httpx(
        monkeypatch,
        [_search([], 2), _search(["Wichita River"], 2)],
    )
    wp = WikipediaPerception(topic="q", rng=random.Random(0))

    result = asyncio.run(wp.fetch())

    assert result is not None and result.title == "Wichita_River"
    searches = [u for u in calls if u.startswith(WikipediaPerception.SEARCH_URL)]
    assert len(searches) == 2
    assert int(_query(searches[1])["sroffset"]) < 2


def test_topic_with_no_hits_returns_none_with_warning(monkeypatch, caplog) -> None:
    calls = install_routing_httpx(monkeypatch, [_search([], 0) for _ in range(3)])
    wp = WikipediaPerception(topic="zzqqxx-no-such-thing", rng=random.Random(0))

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(wp.fetch())

    assert result is None
    assert all(u.startswith(WikipediaPerception.SEARCH_URL) for u in calls)
    assert "returned no articles" in caplog.text


def test_search_api_error_returns_none_with_warning(monkeypatch, caplog) -> None:
    install_routing_httpx(
        monkeypatch,
        [{"error": {"code": "badquery", "info": "bad"}} for _ in range(3)],
    )
    wp = WikipediaPerception(topic="q", rng=random.Random(0))

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(wp.fetch())

    assert result is None
    assert "search error" in caplog.text


def test_blank_topic_is_treated_as_no_topic() -> None:
    assert WikipediaPerception(topic="   ")._topic is None


def test_factory_passes_topic_through() -> None:
    wp = build_perception_provider("wikipedia", topic='"United States"')
    assert isinstance(wp, WikipediaPerception)
    assert wp._topic == '"United States"'


def _config(**perception: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "llm": {"provider": "ollama", "model": "llama3"},
        "memory": {
            "short_term_capacity": 5,
            "consolidation_interval_minutes": 5,
            "forgetting_curve_enabled": False,
            "importance_decay_rate": 0.01,
        },
        "consciousness": {"origin_story": "o", "values": ["curiosity"], "purpose": "p"},
        "thought_loop": {
            "reflection_probability": 0.0,
            "existential_inquiry_every_n_thoughts": 5,
            "min_interval_seconds": 0,
            "max_interval_seconds": 0,
        },
        "mood": {"initial": {"curiosity": 0.5}, "drift_rate": 0.01},
        "perception": {
            "enabled": True,
            "provider": "wikipedia",
            "every_n_cycles": 1,
            "timeout_seconds": 1.0,
            "cache_last_n": 0,
        },
    }
    cfg["perception"].update(perception)
    return cfg


@pytest.mark.parametrize("topic", [None, '"United States"'])
def test_validate_config_accepts_null_or_query_topic(topic) -> None:
    _validate_config(_config(topic=topic))


@pytest.mark.parametrize("topic", ["", "   ", 5, ["United States"]])
def test_validate_config_rejects_malformed_topic(topic) -> None:
    with pytest.raises(ValueError, match="perception.topic"):
        _validate_config(_config(topic=topic))


def test_validate_config_rejects_topic_for_non_wikipedia_provider() -> None:
    with pytest.raises(ValueError, match="only supported with provider 'wikipedia'"):
        _validate_config(_config(provider="mock", topic="rivers"))
