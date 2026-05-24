"""Inner-voice framing for first-person stream-of-consciousness narration.

Theory mapping — CTM (Blum & Blum 2022) / GWT-3: InnerVoice approximates
"Brainish" — the CTM's rich inner language for inter-processor communication.
It enforces first-person framing and register (questioning, remembering,
wondering) on raw LLM output before it enters the global workspace buffer.
Gap: registers are rule-based heuristics, not a learned or grounded inner
language; no multi-modal encoding as CTM's Brainish requires.
"""

from __future__ import annotations


class InnerVoice:
    """Applies narrative framing and register to raw model output."""

    def __init__(self, name: str) -> None:
        self.name = name

    # First words that indicate the LLM produced a complete sentence with its own subject.
    # Prepending "I " onto these produces ungrammatical output ("I the void stirs…").
    _NOUN_PHRASE_STARTERS: frozenset[str] = frozenset({
        "the", "a", "an", "this", "that", "these", "those",
        "my", "your", "his", "her", "its", "their", "our",
        "time", "perhaps", "maybe", "somewhere", "somehow",
        "silence", "something", "nothing", "everything",
    })

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
        if register == "remembering":
            return f"{text} I remember this as {self.name}."
        if register == "questioning" and "?" not in text:
            return f"{text}?"
        return text
