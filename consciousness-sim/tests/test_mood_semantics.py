"""Tests for embedding-based mood trigger scoring (#21)."""

from __future__ import annotations

import asyncio
import logging
import math

import pytest

from core.mood_semantics import DEFAULT_ANCHORS, DEFAULT_THRESHOLD, SemanticMoodScorer


class _StubEmbedder:
    """Returns a scripted vector per exact text, counting every call.

    Duck-typed against the one method ``SemanticMoodScorer`` uses; unknown
    texts get a fixed vector so a test only has to script what it asserts on.
    """

    def __init__(self, vectors: dict[str, list[float]], default: list[float] | None = None) -> None:
        self._vectors = dict(vectors)
        self._default = [0.0, 0.0] if default is None else list(default)
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self._vectors.get(text, self._default))


class _FailingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        raise RuntimeError("embed backend unavailable")


# ---------------------------------------------------------------------------
# Strength ramp
# ---------------------------------------------------------------------------


def test_strength_is_zero_at_and_below_threshold() -> None:
    scorer = SemanticMoodScorer(_StubEmbedder({}), anchors={"a": ["x"]}, threshold=0.5)
    assert scorer._strength(0.5) == 0.0
    assert scorer._strength(0.1) == 0.0
    assert scorer._strength(-0.9) == 0.0


def test_strength_ramps_linearly_to_one_at_a_perfect_match() -> None:
    scorer = SemanticMoodScorer(_StubEmbedder({}), anchors={"a": ["x"]}, threshold=0.5)
    assert scorer._strength(0.75) == pytest.approx(0.5)
    assert scorer._strength(1.0) == pytest.approx(1.0)
    # Numerical overshoot past 1.0 is clamped rather than amplifying drift.
    assert scorer._strength(1.2) == pytest.approx(1.0)


def test_threshold_outside_the_valid_range_is_rejected() -> None:
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="threshold"):
            SemanticMoodScorer(_StubEmbedder({}), anchors={"a": ["x"]}, threshold=bad)


def test_default_threshold_is_used_when_unspecified() -> None:
    scorer = SemanticMoodScorer(_StubEmbedder({}))
    assert scorer.threshold == DEFAULT_THRESHOLD
    assert set(scorer.anchors) == set(DEFAULT_ANCHORS)


def test_anchor_dimensions_with_no_phrases_are_dropped() -> None:
    scorer = SemanticMoodScorer(
        _StubEmbedder({}), anchors={"melancholy": [], "contentment": ["quiet ease"]}
    )
    assert set(scorer.anchors) == {"contentment"}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_attributes_strength_to_the_semantically_nearest_dimension() -> None:
    """The point of #21: a text with no lexical overlap still moves a dimension."""
    embedder = _StubEmbedder(
        {
            "a sense of loss": [1.0, 0.0],
            "quiet ease": [0.0, 1.0],
            "everything she built is gone now": [1.0, 0.0],
        }
    )
    scorer = SemanticMoodScorer(
        embedder,
        anchors={"melancholy": ["a sense of loss"], "contentment": ["quiet ease"]},
        threshold=0.4,
    )
    scores = asyncio.run(scorer.score("everything she built is gone now"))
    assert scores["melancholy"] == pytest.approx(1.0)
    assert scores["contentment"] == 0.0


def test_score_takes_the_best_matching_anchor_phrase_per_dimension() -> None:
    embedder = _StubEmbedder(
        {
            "far anchor": [0.0, 1.0],
            "near anchor": [1.0, 0.0],
            "text": [1.0, 0.0],
        }
    )
    scorer = SemanticMoodScorer(
        embedder, anchors={"melancholy": ["far anchor", "near anchor"]}, threshold=0.0
    )
    scores = asyncio.run(scorer.score("text"))
    assert scores["melancholy"] == pytest.approx(1.0)


def test_score_scales_partial_similarity_through_the_ramp() -> None:
    embedder = _StubEmbedder({"anchor": [1.0, 0.0], "text": [1.0, 1.0]})
    scorer = SemanticMoodScorer(embedder, anchors={"wonder": ["anchor"]}, threshold=0.0)
    scores = asyncio.run(scorer.score("text"))
    assert scores["wonder"] == pytest.approx(1.0 / math.sqrt(2.0))


def test_anchor_phrases_are_embedded_once_across_calls() -> None:
    """Recurring per-cycle cost must be one embed, not one per anchor phrase."""
    embedder = _StubEmbedder({"anchor one": [1.0, 0.0], "anchor two": [0.0, 1.0]})
    scorer = SemanticMoodScorer(
        embedder, anchors={"melancholy": ["anchor one", "anchor two"]}, threshold=0.0
    )
    asyncio.run(scorer.score("first"))
    assert embedder.calls == ["anchor one", "anchor two", "first"]
    asyncio.run(scorer.score("second"))
    assert embedder.calls[-1] == "second"
    assert embedder.calls.count("anchor one") == 1
    assert embedder.calls.count("anchor two") == 1


def test_blank_text_scores_zero_without_embedding_anything() -> None:
    embedder = _StubEmbedder({})
    scorer = SemanticMoodScorer(embedder, anchors={"wonder": ["anchor"]}, threshold=0.0)
    assert asyncio.run(scorer.score("   ")) == {"wonder": 0.0}
    assert embedder.calls == []


def test_zero_length_query_embedding_scores_zero() -> None:
    embedder = _StubEmbedder({"anchor": [1.0, 0.0], "text": []})
    scorer = SemanticMoodScorer(embedder, anchors={"wonder": ["anchor"]}, threshold=0.0)
    assert asyncio.run(scorer.score("text")) == {"wonder": 0.0}


def test_anchor_embedding_of_the_wrong_dimensionality_is_skipped() -> None:
    """Vectors from different embedding models are not compared (cf. #112)."""
    embedder = _StubEmbedder({"anchor": [1.0, 0.0], "text": [1.0, 0.0, 0.0]})
    scorer = SemanticMoodScorer(embedder, anchors={"wonder": ["anchor"]}, threshold=0.0)
    assert asyncio.run(scorer.score("text")) == {"wonder": 0.0}


def test_zero_vector_anchor_is_dropped_with_a_warning(caplog) -> None:
    embedder = _StubEmbedder({"dead anchor": [0.0, 0.0], "live anchor": [1.0, 0.0], "t": [1.0, 0.0]})
    scorer = SemanticMoodScorer(
        embedder, anchors={"wonder": ["dead anchor", "live anchor"]}, threshold=0.0
    )
    with caplog.at_level(logging.WARNING):
        scores = asyncio.run(scorer.score("t"))
    assert "dead anchor" in caplog.text
    assert scores["wonder"] == pytest.approx(1.0)


def test_embed_failure_propagates_to_the_caller() -> None:
    """No fabricated vector is substituted here — the caller decides (#46)."""
    embedder = _FailingEmbedder()
    scorer = SemanticMoodScorer(embedder, anchors={"wonder": ["anchor"]}, threshold=0.0)
    with pytest.raises(RuntimeError, match="embed backend unavailable"):
        asyncio.run(scorer.score("text"))


def test_anchor_embedding_is_retried_after_a_failure() -> None:
    """A failed anchor pass must not memoise an empty anchor set."""

    class _FlakyEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, text: str) -> list[float]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return [1.0, 0.0]

    embedder = _FlakyEmbedder()
    scorer = SemanticMoodScorer(embedder, anchors={"wonder": ["anchor"]}, threshold=0.0)
    with pytest.raises(RuntimeError):
        asyncio.run(scorer.score("text"))
    assert asyncio.run(scorer.score("text"))["wonder"] == pytest.approx(1.0)


def test_concurrent_scores_embed_the_anchors_only_once() -> None:
    """The memoisation guard holds under overlapping callers."""

    class _SlowEmbedder:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def embed(self, text: str) -> list[float]:
            self.calls.append(text)
            await asyncio.sleep(0)
            return [1.0, 0.0]

    embedder = _SlowEmbedder()
    scorer = SemanticMoodScorer(embedder, anchors={"wonder": ["anchor"]}, threshold=0.0)

    async def _run() -> None:
        await asyncio.gather(scorer.score("a"), scorer.score("b"))

    asyncio.run(_run())
    assert embedder.calls.count("anchor") == 1


def test_default_anchors_cover_the_shipped_mood_dimensions() -> None:
    """A dimension in the shipped config with no anchors could never drift."""
    from core.identity import IdentityDocument

    assert set(DEFAULT_ANCHORS) == set(IdentityDocument._MOOD_TRIGGERS)
