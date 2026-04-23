"""Identity model that persists, anchors, and evolves the agent's sense of self."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


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

    def apply_amendment(self, amendment: str) -> None:
        self.amendments.append(amendment)
        self.self_concept = f"{self.self_concept} {amendment}".strip()

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
