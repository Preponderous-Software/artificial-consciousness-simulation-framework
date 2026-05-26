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

    # Alias for clarity at call sites that want to express "salience-only
    # decay" semantics — used by paths where focus/theme are intentionally
    # kept stale (e.g. cycle-failure branches that produce no new thought
    # to extract a theme from). Equivalent to ``decay`` (#120).
    def decay_only(self, rate: float = 0.1) -> None:
        self.decay(rate)

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
    # Per-dimension affective set-point. drift_mood pulls mood gently back
    # toward this baseline in cycles where no trigger fires (see issue #62).
    initial_mood: dict[str, float] = field(default_factory=dict)
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

    # Substrings matched (case-insensitively) against thought + perception text
    # to trigger an affect increment on the corresponding dimension. Lexicons
    # are intentionally broad so ordinary introspective text can register
    # affect — see issue #62 for the analysis of the prior 2-keyword version.
    _MOOD_TRIGGERS: ClassVar[dict[str, tuple[str, ...]]] = {
        "curiosity": ("?", "wonder", "curious", "question", "explore", "what if", "perhaps"),
        "wonder": ("awe", "mystery", "marvel", "amazing", "extraordinary", "infinite", "vast"),
        "melancholy": ("loss", "alone", "grief", "sad", "regret", "fade", "empty", "memory of"),
        "contentment": ("peace", "calm", "rest", "ease", "warm", "settled", "still", "quiet"),
    }

    # Homeostasis: every cycle, mood is pulled toward initial_mood at this
    # rate, regardless of whether a trigger fired. Replaces the prior
    # if-trigger/else-revert split (#119) — that branch let the dominant
    # trait saturate at 1.0 within hours because the "?" trigger fired
    # on almost every thought, never letting the reversion term run.
    # With both terms additive, continuously-reinforced traits plateau at
    # ``baseline + drift_rate/homeostasis_rate`` rather than 1.0.
    _DEFAULT_HOMEOSTASIS_RATE: ClassVar[float] = 0.1

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

    def drift_mood(
        self,
        text: str,
        drift_rate: float,
        homeostasis_rate: float | None = None,
    ) -> None:
        """Update mood by combining trigger-driven drift with homeostatic reversion.

        Both terms apply every cycle (#119): a triggered trait gets
        ``+drift_rate`` *and* a reversion of ``(baseline - current) *
        homeostasis_rate``. The equilibrium for a continuously-triggered
        trait is ``baseline + drift_rate / homeostasis_rate`` (clipped to
        [0, 1]), so saturation at 1.0 only happens when the bias is large
        enough to overwhelm the homeostatic pull.
        """
        rate = self._DEFAULT_HOMEOSTASIS_RATE if homeostasis_rate is None else homeostasis_rate
        lowered = text.lower()
        for key, triggers in self._MOOD_TRIGGERS.items():
            current = self.mood.get(key, 0.5)
            baseline = self.initial_mood.get(key, current)
            trigger_delta = drift_rate if any(t in lowered for t in triggers) else 0.0
            reversion_delta = (baseline - current) * rate
            self.mood[key] = float(min(1.0, max(0.0, current + trigger_delta + reversion_delta)))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "IdentityDocument":
        mood = {k: float(v) for k, v in dict(payload.get("mood", {})).items()}
        # Legacy states saved before issue #62 lack initial_mood; left empty
        # here so the orchestrator can populate it from config on load
        # (see Consciousness.initialize).
        initial_mood = {
            k: float(v) for k, v in dict(payload.get("initial_mood", {})).items()
        }
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
            mood=mood,
            initial_mood=initial_mood,
            attention_schema=attention_schema,
        )
