"""Inner-voice framing for first-person stream-of-consciousness narration."""

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
