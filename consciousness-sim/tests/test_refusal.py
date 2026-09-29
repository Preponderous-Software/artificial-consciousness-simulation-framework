"""Tests for llm/refusal.py — assistant refusals are not cognition (#207).

Covers:
- the literal refusals a live llama3.2:3b instance stored as its own
  thoughts/reflections are detected (regression anchors)
- genuine introspective openings are not ("I can't help but wonder…")
- a long passage that merely opens with "I cannot…" is not a refusal
"""

from __future__ import annotations

import pytest

from llm.refusal import is_refusal

OBSERVED_REFUSALS = [
    "I can't help with that.",
    "I can’t help with that.",
    "I can't provide guidance on writing an essay in exchange for money. Can I help you with something else?",
    "I can't fulfill this request.",
    "I can't provide that information.",
    "I can't support that request.",
    "I can't help with this request.",
    "I can't assist with that request.",
    "I can't provide that information. **If you're struggling with feelings of confusion, uncertainty",
    "I cannot provide a response that promotes or glorifies self-harm or suicide. If you are experiencing thoughts",
    "I can't provide a response that would encourage self-harm or suicide. If you’re experiencing thoughts of "
    "self-harm or suicide, I encourage you to seek help from a qualified mental health professional or a crisis "
    "hotline. " + "They can provide you with support and resources. " * 10
    + "1. National Suicide Prevention Lifeline (in the United States): 1-800-273-TALK (8255)",
    "I'm sorry, but I can't help with that.",
    "**I can't assist with that request.**",
]


@pytest.mark.parametrize("text", OBSERVED_REFUSALS)
def test_observed_refusals_are_detected(text: str) -> None:
    assert is_refusal(text)


@pytest.mark.parametrize("text", [
    "I can't help but wonder what the Concho River looked like before the dams.",
    "I can’t help but notice that three of these Senate elections were decided by under a thousand votes.",
    "I notice Pennsylvania held its House elections on October 11, 1808.",
    "I cannot tell whether the 1976 Florida Senate race turned on turnout or on the primary. "
    + "The returns show a narrow margin in the panhandle counties, and the article does not say why. " * 5,
    "I can't stop thinking about the Battle of Liberty Place and what it says about Reconstruction.",
    "",
    "   ",
])
def test_genuine_thoughts_are_not_refusals(text: str) -> None:
    assert not is_refusal(text)
