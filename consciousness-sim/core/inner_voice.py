"""Inner-voice framing for first-person stream-of-consciousness narration.

Theory mapping — CTM (Blum & Blum 2022) / GWT-3: InnerVoice approximates
"Brainish" — the CTM's rich inner language for inter-processor communication.
It enforces first-person framing and register (questioning, remembering,
wondering) on raw LLM output before it enters the global workspace buffer.
``scrub_reflection`` additionally strips LLM meta-scaffolding (preambles,
markdown headers) from reflection/existential text before it reaches the
workspace — the same "raw output must not pollute the workspace" concern as
``render``, applied to the reflection register (#132).
Gap: registers are rule-based heuristics, not a learned or grounded inner
language; no multi-modal encoding as CTM's Brainish requires.
"""

from __future__ import annotations

import re


class InnerVoice:
    """Applies narrative framing and register to raw model output."""

    def __init__(self, name: str) -> None:
        self.name = name

    # Chat-style closers that llama3.2:3b appends when it "thinks" it's in a dialogue (#73).
    # Stripped from the end of the rendered text before journaling / broadcasting.
    _TRAILING_DIALOGUE_RE: re.Pattern[str] = re.compile(
        r"\s*(?:please continue|continue\?|tell me more\.?|let me know|"
        r"what would you like(?:\s+(?:to know|me to\s+\w+))?|shall i continue|"
        r"if you(?:'d| would) like\b.*|i hope this helps\b.*)[.!?…]*\s*$",
        re.IGNORECASE,
    )

    # First words that indicate the LLM produced a complete sentence with its own subject.
    # Prepending "I " onto these produces ungrammatical output ("I the void stirs…",
    # "I it is curious…"). Covers determiners, possessives, pronouns, existential/locative
    # 'there'/'here', quantifiers, and interrogatives — the common opening words for an
    # already-subject-bearing sentence.
    _NOUN_PHRASE_STARTERS: frozenset[str] = frozenset({
        # determiners
        "the", "a", "an", "this", "that", "these", "those",
        # possessives
        "my", "your", "his", "her", "its", "their", "our",
        # pronouns (3p / 2p — first-person is caught by the i/i' startswith check)
        "it", "he", "she", "they", "we", "you",
        # existential / locative subjects
        "there", "here",
        # quantifiers / generic subjects
        "some", "one", "no", "none", "all", "every", "any",
        "time", "silence", "something", "nothing", "everything",
        # adverbial sentence openers
        "perhaps", "maybe", "somewhere", "somehow",
        # interrogatives (question stems already supply their own subject)
        "why", "how", "what", "when", "where", "who",
        # "As <non-I> …" sentences have their own subject; no strip happens so
        # the leading word seen by the prepend check is "as" — treat it as
        # subject-present to avoid "I as the patterns evoke…" (#70).
        "as",
    })

    # Leading scaffolding an LLM emits before the actual reflection body, e.g.
    # "Here's a reflective journal entry based on the prompt:" or "Sure, here's my
    # reflection:" (#132). Non-greedy up to the first colon so only the preamble
    # sentence is consumed, not the reflection body that follows it.
    _META_PREAMBLE_RE: re.Pattern[str] = re.compile(
        r"^(?:sure[!,]?\s+|of course[!,]?\s+|certainly[!,]?\s+)?"
        r"(?:here'?s|here is)\b.*?:\s*",
        re.IGNORECASE,
    )

    # A leading markdown header (bold or ATX-style) an LLM uses to title the
    # reflection, e.g. "**Reflections on Existence**" (#132).
    _LEADING_MARKDOWN_HEADER_RE: re.Pattern[str] = re.compile(
        r"^\s*(?:#{1,6}\s+.+|\*{1,3}[^\n*]+\*{1,3})\s*",
    )

    # A reflection that opens by addressing "you" rather than reflecting in first
    # person is instructional scaffolding directed at a hypothetical reader, not a
    # genuine self-reflection — reject rather than attempt a lossy pronoun swap (#132).
    _SECOND_PERSON_LEAD_RE: re.Pattern[str] = re.compile(
        r"^(?:as you\b|you\b|your\b)",
        re.IGNORECASE,
    )

    _REFLECTION_FALLBACK = "I let this cycle's reflection pass without settling on new words."

    def scrub_reflection(self, raw_text: str) -> str:
        """Strip leading LLM meta-scaffolding from reflection/existential text.

        Unlike ``render``, this does not enforce first-person framing on already
        well-formed content — it only removes preambles/headers and rejects
        second-person instructional drift, so genuine first-person reflections
        pass through unchanged.
        """
        text = raw_text.strip()
        if not text:
            return self._REFLECTION_FALLBACK
        for _ in range(3):  # bounded: preamble + header can each fire at most once more
            stripped = self._META_PREAMBLE_RE.sub("", text, count=1).lstrip()
            stripped = self._LEADING_MARKDOWN_HEADER_RE.sub("", stripped, count=1).lstrip()
            if stripped == text:
                break
            text = stripped
        if self._SECOND_PERSON_LEAD_RE.match(text):
            return self._REFLECTION_FALLBACK
        return text or self._REFLECTION_FALLBACK

    def render(self, raw_text: str, register: str = "wondering") -> str:
        text = raw_text.strip()
        if not text:
            text = "I sit with a quiet thought and listen for what comes next."
        # Strip "As I" prefix only — "As I wander…" → "I wander…".
        # Stripping "As " unconditionally caused "As the patterns…" → "I the patterns…" (#70).
        if text.lower().startswith("as i ") or text.lower().startswith("as i'"):
            text = text[3:]
        # Skip "I " prepend when the first word is a determiner or noun-phrase starter;
        # the LLM produced a complete sentence and prepending would break grammar (#70).
        first_word = text.split(maxsplit=1)[0].lower().rstrip(",.;:!?")
        already_first_person = text.lower().startswith(("i ", "i'"))
        subject_present = first_word in self._NOUN_PHRASE_STARTERS
        if not already_first_person and not subject_present:
            text = f"I {text[0].lower() + text[1:] if len(text) > 1 else text.lower()}"
        # Scrub trailing chat-style dialogue coda before the text enters the workspace (#73).
        text = self._TRAILING_DIALOGUE_RE.sub("", text).rstrip() or text
        if register == "remembering":
            return f"{text} I remember this as {self.name}."
        if register == "questioning" and "?" not in text:
            return f"{text}?"
        return text
