"""Identity model that persists, anchors, and evolves the agent's sense of self.

Theory mapping — AST (Graziano 2013) / HOT (Rosenthal 2005): IdentityDocument
is the system's self-model, combining a static anchor (name, values, purpose)
with a dynamic self-concept updated by reflection. Partially implements
AST (self-model enabling self-attribution) and HOT-1 (generative top-down
self-representation). Mood drift partially implements AE-1 (affect-modulated
agency).
AttentionSchema advances AST-1: a dynamic data structure representing current
focus and salience, updated every cycle and fed back into the identity anchor
prompt. Gap: focus is derived from event type, not a learned allocation model;
salience decay is linear rather than neurally motivated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import ClassVar


@dataclass(slots=True)
class AttentionSchema:
    """Dynamic model of what the agent is currently attending to.

    Theory mapping — AST-1 (Graziano 2013): implements the attention schema
    as a data structure tracking focus, theme, and salience updated each cycle.
    Functional label only — no phenomenal claim.
    """

    focus: str = "introspection"
    theme: str = ""
    salience: float = 1.0
    history: list[str] = field(default_factory=list)

    _MAX_HISTORY: ClassVar[int] = 10

    def update(self, focus: str, theme: str) -> None:
        self.history.append(self.focus)
        if len(self.history) > self._MAX_HISTORY:
            self.history = self.history[-self._MAX_HISTORY:]
        self.focus = focus
        self.theme = theme
        self.salience = 1.0

    def decay(self, rate: float = 0.1) -> None:
        self.salience = max(0.0, self.salience - rate)

    def render(self) -> str:
        theme_part = f": {self.theme}" if self.theme else ""
        return f"{self.focus}{theme_part} (salience {self.salience:.2f})"


@dataclass(slots=True)
class IdentityDocument:
    name: str
    origin_story: str
    values: list[str]
    purpose: str
    self_concept: str
    personality_traits: list[str] = field(default_factory=list)
    amendments: list[str] = field(default_factory=list)
    mood: dict[str, float] = field(default_factory=dict)
    attention_schema: AttentionSchema = field(default_factory=AttentionSchema)

    def summary(self) -> str:
        values = ", ".join(self.values)
        return (
            f"Name: {self.name}; Purpose: {self.purpose}; Values: {values}; "
            f"Self-concept: {self.self_concept}"
        )

    def anchor_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "values": ", ".join(self.values),
            "purpose": self.purpose,
            "self_concept": self.self_concept,
            "attention_state": self.attention_schema.render(),
        }

    _MAX_SELF_CONCEPT_LEN: ClassVar[int] = 300
    _MAX_AMENDMENTS: ClassVar[int] = 20

    def apply_amendment(self, amendment: str) -> None:
        self.amendments.append(amendment)
        if len(self.amendments) > self._MAX_AMENDMENTS:
            # Drop the oldest to keep the serialized state finite.
            self.amendments = self.amendments[-self._MAX_AMENDMENTS :]
        combined = f"{self.self_concept} {amendment}".strip()
        if len(combined) > self._MAX_SELF_CONCEPT_LEN:
            # Preserve the tail: most recent amendments carry current identity.
            combined = "..." + combined[-self._MAX_SELF_CONCEPT_LEN:]
        self.self_concept = combined

    def drift_mood(self, text: str, drift_rate: float) -> None:
        lowered = text.lower()
        deltas = {
            "curiosity": 1 if "?" in text or "wonder" in lowered else 0,
            "wonder": 1 if "awe" in lowered or "mystery" in lowered else 0,
            "melancholy": 1 if "loss" in lowered or "alone" in lowered else 0,
            "contentment": 1 if "peace" in lowered or "calm" in lowered else 0,
        }
        for key, trigger in deltas.items():
            current = self.mood.get(key, 0.5)
            delta = drift_rate if trigger else -drift_rate / 4
            self.mood[key] = float(min(1.0, max(0.0, current + delta)))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "IdentityDocument":
        attn_raw = payload.get("attention_schema", {})
        attn_data = dict(attn_raw) if isinstance(attn_raw, dict) else {}
        attention_schema = AttentionSchema(
            focus=str(attn_data.get("focus", "introspection")),
            theme=str(attn_data.get("theme", "")),
            salience=float(attn_data.get("salience", 1.0)),
            history=[str(h) for h in attn_data.get("history", [])],
        )
        return cls(
            name=str(payload.get("name", "unnamed")),
            origin_story=str(payload.get("origin_story", "")),
            values=[str(v) for v in payload.get("values", [])],
            purpose=str(payload.get("purpose", "")),
            self_concept=str(payload.get("self_concept", "")),
            personality_traits=[str(v) for v in payload.get("personality_traits", [])],
            amendments=[str(v) for v in payload.get("amendments", [])],
            mood={k: float(v) for k, v in dict(payload.get("mood", {})).items()},
            attention_schema=attention_schema,
        )
