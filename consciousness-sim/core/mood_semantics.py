"""Embedding-based scoring of how strongly a text bears on each mood dimension.

Theory mapping — AE-1 (Butlin et al. 2023, Agency/Embodiment): mood is the
affective state that modulates downstream generation, so the fidelity of the
signal that moves it bounds how much the agent's behaviour can be said to
respond to its own content. The pre-#21 detector was a substring match against
a fixed keyword list in ``IdentityDocument._MOOD_TRIGGERS``: a thought about
bereavement that never used the token "loss"/"grief"/"sad" moved melancholy by
exactly zero. Scoring by embedding similarity against per-dimension anchor
phrases replaces lexical identity with distributional proximity, so
paraphrases register.

Gap: this is still a fixed, hand-authored anchor set compared by cosine
similarity — not a learned affect model, and not the precision-weighted
valuation PP/FEP would require. The strengths it returns are unsigned: a
dimension can be pushed *up* by matching content, and only falls again through
the homeostatic reversion term in ``IdentityDocument.drift_mood``. There is no
anti-anchor that actively pushes a dimension down.

Functional label only — "mood" names a scalar state vector conditioning
prompt construction, not an affective experience.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# Cosine similarity below which a dimension is treated as unrelated to the
# text and does not drift at all. Embedding models place most natural-language
# pairs well above 0, so a bare "similarity > 0" test would drift every
# dimension on every cycle; the threshold is what keeps scoring selective.
DEFAULT_THRESHOLD = 0.45

# Anchor phrases per mood dimension, keyed to match the dimensions shipped in
# `config/default_consciousness.yaml`'s `mood.initial`. Several short phrases
# per dimension rather than one long one: each is embedded separately and the
# dimension takes the best match, so the phrases act as alternatives, not as a
# single averaged centroid that no real thought sits near.
DEFAULT_ANCHORS: dict[str, tuple[str, ...]] = {
    "curiosity": (
        "an open question I want to follow further",
        "wanting to find out how something works",
        "an unfamiliar idea worth investigating",
    ),
    "wonder": (
        "awe at the scale and strangeness of things",
        "a mystery that resists explanation",
        "something vast and astonishing",
    ),
    "melancholy": (
        "a sense of loss and what has faded",
        "loneliness and quiet grief",
        "regret about something that cannot be recovered",
    ),
    "contentment": (
        "a settled calm, nothing needing to change",
        "quiet ease and rest",
        "peaceful satisfaction with how things are",
    ),
}


class _Embedder(Protocol):
    """The one method of ``LLMProvider`` this module needs.

    Declared structurally so tests can pass a stub without constructing a real
    provider, and so no import cycle is created against ``llm.provider``.
    """

    async def embed(self, text: str) -> list[float]: ...


class SemanticMoodScorer:
    """Score a text against per-dimension anchor phrases by cosine similarity (#21).

    Anchor phrases are embedded once, on the first ``score()`` call, and reused
    for the life of the instance — the recurring per-cycle cost is a single
    embed of the cycle's text, not one embed per anchor phrase.
    """

    def __init__(
        self,
        provider: _Embedder,
        anchors: dict[str, list[str]] | dict[str, tuple[str, ...]] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if not 0.0 <= threshold < 1.0:
            # 1.0 is excluded rather than clamped: the strength ramp divides by
            # (1 - threshold), so 1.0 is undefined rather than merely strict.
            raise ValueError(
                f"threshold must be in [0.0, 1.0), got {threshold!r}"
            )
        source = DEFAULT_ANCHORS if anchors is None else anchors
        self.anchors: dict[str, tuple[str, ...]] = {
            dimension: tuple(phrases) for dimension, phrases in source.items() if phrases
        }
        self.threshold = float(threshold)
        self._provider = provider
        self._anchor_embeddings: dict[str, list[npt.NDArray[np.float64]]] | None = None
        self._anchor_lock = asyncio.Lock()

    async def score(self, text: str) -> dict[str, float]:
        """Return a strength in [0.0, 1.0] per configured mood dimension.

        The strength ramps linearly from 0.0 at ``threshold`` to 1.0 at a
        perfect match, so a dimension's equilibrium under
        ``IdentityDocument.drift_mood`` becomes
        ``baseline + strength * drift_rate / homeostasis_rate`` — bounded above
        by the all-lexical-matches equilibrium rather than exceeding it.

        Raises whatever the provider's ``embed`` raises. The caller in
        ``core/consciousness.py`` degrades to lexical triggers on failure; no
        fabricated vector is substituted here (#46).
        """
        if not self.anchors:
            return {}
        stripped = text.strip()
        if not stripped:
            return {dimension: 0.0 for dimension in self.anchors}

        anchor_embeddings = await self._ensure_anchor_embeddings()
        query = np.array(await self._provider.embed(stripped), dtype=float)
        query_norm = float(np.linalg.norm(query))
        if query.size == 0 or query_norm == 0.0:
            return {dimension: 0.0 for dimension in self.anchors}

        scores: dict[str, float] = {}
        for dimension in self.anchors:
            best = 0.0
            for anchor in anchor_embeddings.get(dimension, []):
                if anchor.shape != query.shape:
                    # Mismatched dimensionality means the anchors were embedded
                    # by a different model than the one answering now; skip
                    # rather than compare vectors from different spaces.
                    continue
                denominator = float(np.linalg.norm(anchor)) * query_norm
                best = max(best, float(np.dot(anchor, query) / denominator))
            scores[dimension] = self._strength(best)
        return scores

    def _strength(self, similarity: float) -> float:
        """Map a cosine similarity onto a [0.0, 1.0] drift strength."""
        if similarity <= self.threshold:
            return 0.0
        return float(min(1.0, (similarity - self.threshold) / (1.0 - self.threshold)))

    async def _ensure_anchor_embeddings(self) -> dict[str, list[npt.NDArray[np.float64]]]:
        """Embed every anchor phrase once, then memoise for the process."""
        if self._anchor_embeddings is not None:
            return self._anchor_embeddings
        async with self._anchor_lock:
            if self._anchor_embeddings is not None:
                return self._anchor_embeddings
            embedded: dict[str, list[npt.NDArray[np.float64]]] = {}
            for dimension, phrases in self.anchors.items():
                vectors: list[npt.NDArray[np.float64]] = []
                for phrase in phrases:
                    vector = np.array(await self._provider.embed(phrase), dtype=float)
                    if vector.size == 0 or float(np.linalg.norm(vector)) == 0.0:
                        logger.warning(
                            "Anchor phrase %r for mood dimension %r embedded to a zero/empty "
                            "vector; skipping it",
                            phrase,
                            dimension,
                        )
                        continue
                    vectors.append(vector)
                embedded[dimension] = vectors
            self._anchor_embeddings = embedded
            return embedded
