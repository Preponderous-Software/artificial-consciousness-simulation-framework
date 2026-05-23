"""Identity model that persists, anchors, and evolves the agent's sense of self.

Theory mapping — AST (Graziano 2013) / HOT (Rosenthal 2005): IdentityDocument
is the system's self-model, combining a static anchor (name, values, purpose)
with a dynamic self-concept updated by reflection. Partially implements
AST (self-model enabling self-attribution) and HOT-1 (generative top-down
self-representation). Mood drift partially implements AE-1 (affect-modulated
agency).
Gap: no attention state tracked (AST-1 requires modelling current attention,
not just stable identity). See issue #22.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import ClassVar


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
        return cls(
            name=str(payload.get("name", "unnamed")),
            origin_story=str(payload.get("origin_story", "")),
            values=[str(v) for v in payload.get("values", [])],
            purpose=str(payload.get("purpose", "")),
            self_concept=str(payload.get("self_concept", "")),
            personality_traits=[str(v) for v in payload.get("personality_traits", [])],
            amendments=[str(v) for v in payload.get("amendments", [])],
            mood={k: float(v) for k, v in dict(payload.get("mood", {})).items()},
        )
