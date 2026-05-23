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

    def render(self, raw_text: str, register: str = "wondering") -> str:
        text = raw_text.strip()
        if not text:
            text = "I sit with a quiet thought and listen for what comes next."
        if not text.lower().startswith("i "):
            text = f"I {text[0].lower() + text[1:] if len(text) > 1 else text.lower()}"
        if register == "remembering":
            return f"{text} I remember this as {self.name}."
        if register == "questioning" and "?" not in text:
            return f"{text}?"
        return text
