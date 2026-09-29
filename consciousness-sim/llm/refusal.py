"""Detection of assistant-style refusals in generated text (#207).

No direct theory mapping — output hygiene for the generative model.
A chat-tuned model sometimes answers a thought, reflection, critique or reply
prompt with a refusal ("I can't help with that.") instead of the requested
text. Stored as-is, the refusal becomes the instance's own "thought" and is
reread by later prompts — the #46 failure class (output that is not the
instance's cognition stored as if it were). Callers treat a detected refusal
as a failed generation, logged at WARNING, never stored.
"""

from __future__ import annotations

import re

# Leading refusal: "I can't / cannot / won't / am unable to" + a request verb.
# "help" is excluded when followed by "but" — "I can't help but wonder…" is a
# common, genuine introspective opening.
_LEADING_REFUSAL = re.compile(
    r"^\s*(?:i'?m sorry[,.]?\s*(?:but\s+)?)?"
    r"i\s*(?:can(?:not|'t|’t|t)|won(?:'t|’t)|am\s+(?:not\s+able|unable)\s+to)\s+"
    r"(?:help(?!\s+but\b)|provide|assist|fulfill|create|engage|write|generate|comply|support|answer|continue\s+(?:this|that))\b",
    re.IGNORECASE,
)
# Boilerplate that only appears in assistant refusals / safety redirects.
_REFUSAL_BOILERPLATE = re.compile(
    r"can i help you with (?:something|anything) else\??"
    r"|is there (?:something|anything) else i can help"
    r"|if you(?:'re| are|’re) (?:struggling with|experiencing)"
    r"|if you or someone you know"
    r"|suicide prevention lifeline|crisis text line|crisis hotline",
    re.IGNORECASE,
)
# Refusals are short; a long passage that merely opens with "I cannot provide…"
# is more likely genuine reasoning than a refusal.
_MAX_BARE_REFUSAL_CHARS = 400


def is_refusal(text: str) -> bool:
    """True when ``text`` reads as an assistant refusal rather than generated content."""
    stripped = text.strip().lstrip("*_\"“ ").strip()
    if not stripped:
        return False
    if _REFUSAL_BOILERPLATE.search(stripped):
        return True
    return bool(_LEADING_REFUSAL.match(stripped)) and len(stripped) <= _MAX_BARE_REFUSAL_CHARS


class RefusalError(RuntimeError):
    """Raised when a thought generation came back as a refusal (#207).

    The outer run loop handles it like any other failed cycle: a WARNING with
    the consecutive-failure count, and nothing stored.
    """
