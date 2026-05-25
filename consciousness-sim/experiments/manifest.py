"""Experiment manifest — declarative spec for a reproducible run.

A manifest fully defines an experiment: the consciousness name, the config
overrides relative to `config/default_consciousness.yaml`, how long to run,
and optional success criteria. Saved alongside the recorded run so any future
re-execution against the same `branch_sha` is reproducible (modulo LLM
nondeterminism — captured via `temperature` / `seed` where supported).

YAML format example (see `experiments/manifests/` for shippable specs):

    name: sage-perception-baseline
    description: Fresh agent with perception on, mock provider, 30-thought window.
    consciousness_name: SageTest
    config_overrides:
      llm:
        provider: mock
      perception:
        enabled: true
        provider: mock
    duration:
      thoughts: 30
    success_criteria:
      # `kind` is a dotted path into the metrics dict produced by
      # experiments.metrics.compute_all (see metrics.json artifact)
      - kind: mood.dimensions_non_degenerate
        op: ">="
        value: 2
    tags: [perception-on, mood-fix-on]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Duration(BaseModel):
    """Either wall-clock minutes OR thought count; exactly one must be set."""

    minutes: float | None = None
    thoughts: int | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "Duration":
        if (self.minutes is None) == (self.thoughts is None):
            raise ValueError("Duration must set exactly one of `minutes` or `thoughts`")
        if self.minutes is not None and self.minutes <= 0:
            raise ValueError("Duration.minutes must be positive")
        if self.thoughts is not None and self.thoughts <= 0:
            raise ValueError("Duration.thoughts must be positive")
        return self


class SuccessCriterion(BaseModel):
    """A single pass/fail check evaluated against `metrics.json` after the run.

    `kind` is a dotted path into the metrics dict (e.g. `mood.dimensions_non_degenerate`,
    `vocabulary.top_word_density_per_thought`). `op` is one of the standard
    comparison operators.
    """

    kind: str
    op: Literal[">", ">=", "<", "<=", "==", "!="]
    value: float


class ExperimentManifest(BaseModel):
    """Top-level manifest. Parsed from YAML; validated by Pydantic."""

    name: str = Field(..., description="Stable identifier; used as the directory under experiments/")
    description: str = ""
    consciousness_name: str = Field(..., description="Used as spawn.py --name")
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    duration: Duration
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    compare_against: str | None = None      # name of a prior run to diff against (Phase 2)
    branch_sha: str | None = None           # captured at run time if absent here

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentManifest":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Manifest {path} must be a YAML mapping at the top level")
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialize back to YAML — used by the runner to freeze the spec next to the run."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


def evaluate_success_criterion(
    criterion: SuccessCriterion, metrics: dict[str, Any]
) -> tuple[bool, float | None]:
    """Walk metrics by dotted path, apply `op`, return (passed, actual_value).

    Missing keys return (False, None) — the criterion fails closed.
    """
    cursor: Any = metrics
    for part in criterion.kind.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return False, None
        cursor = cursor[part]
    if not isinstance(cursor, (int, float)):
        return False, None
    actual = float(cursor)
    ops = {
        ">":  lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<":  lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    return ops[criterion.op](actual, criterion.value), actual
