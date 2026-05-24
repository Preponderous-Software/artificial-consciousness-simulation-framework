"""Metacognitive monitoring layer — heuristic reliability scoring for generated thoughts.

Theory mapping — HOT (Rosenthal 2005): implements HOT-2 (metacognitive monitoring)
by labelling each generated thought as 'high' / 'uncertain' / 'noise' based on
lexical overlap with recent thoughts in the workspace buffer. A high-overlap
thought is likely an attractor repetition; a low-overlap thought is likely novel.
The label adjusts the thought's workspace importance (GWT-2 synergy: noisy thoughts
evict sooner) and boosts the reflection trigger probability when quality is low
(HOT-2 causal efficacy: the higher-order label influences downstream processing).
Gap: scoring is lexical only — semantic repetition (paraphrasing without shared
vocabulary) is not detected; no LLM-based coherence check. See issue #74 for the
semantic upgrade path.
"""

from __future__ import annotations

import re

from memory.short_term import MemoryItem

_STOPWORDS: frozenset[str] = frozenset({
    "about", "above", "after", "again", "against", "also", "always", "because",
    "been", "being", "between", "both", "cannot", "could", "does", "doing",
    "each", "even", "every", "from", "have", "here", "into", "just", "know",
    "like", "made", "make", "many", "might", "more", "most", "much", "must",
    "myself", "never", "none", "nothing", "often", "once", "only", "other",
    "over", "perhaps", "really", "right", "same", "since", "some", "still",
    "such", "than", "that", "their", "them", "then", "there", "these", "they",
    "this", "those", "though", "thought", "through", "time", "under", "until",
    "upon", "used", "very", "want", "well", "were", "what", "when", "where",
    "which", "while", "will", "with", "would", "your",
})

_OVERLAP_NOISE: float = 0.65
_OVERLAP_UNCERTAIN: float = 0.40
_MIN_CONTENT_WORDS: int = 6

_IMPORTANCE: dict[str, float] = {"high": 1.0, "uncertain": 0.75, "noise": 0.5}
_REFLECTION_BOOST: dict[str, float] = {"high": 0.0, "uncertain": 0.15, "noise": 0.30}


def _content_words(text: str) -> frozenset[str]:
    return frozenset(
        w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOPWORDS
    )


class MetacognitiveMonitor:
    """Labels each generated thought as 'high', 'uncertain', or 'noise'.

    Runs after every thought generation — continuous, not probabilistic.
    """

    def score(self, thought: str, recent_items: list[MemoryItem]) -> str:
        """Return reliability label for thought given recent workspace contents."""
        thought_words = _content_words(thought)
        if len(thought_words) < _MIN_CONTENT_WORDS:
            return "uncertain"

        recent_words: frozenset[str] = frozenset(
            word
            for item in recent_items
            if item.kind == "thought"
            for word in _content_words(item.content)
        )
        if not recent_words:
            return "high"

        overlap = len(thought_words & recent_words) / len(thought_words)
        if overlap >= _OVERLAP_NOISE:
            return "noise"
        if overlap >= _OVERLAP_UNCERTAIN:
            return "uncertain"
        return "high"

    def importance(self, label: str) -> float:
        """Workspace importance weight for the given label."""
        return _IMPORTANCE.get(label, 1.0)

    def reflection_boost(self, label: str) -> float:
        """Additive boost to reflection trigger probability for the given label."""
        return _REFLECTION_BOOST.get(label, 0.0)
