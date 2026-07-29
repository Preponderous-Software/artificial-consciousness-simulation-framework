"""Experiment manifest — declarative spec for a reproducible run.

A manifest fully defines an experiment: the consciousness name, the config
overrides relative to `config/default_consciousness.yaml`, how long to run,
and optional success criteria. Saved alongside the recorded run so any future
re-execution against the same `branch_sha` is reproducible (modulo LLM
nondeterminism — captured via `temperature` / `seed` where supported).

Schema versioning: every manifest carries `schema_version: int` (default 1).
Old manifests without the field load as v1. Bump when adding required fields
or changing field semantics. The loader's responsibility to upgrade.

YAML format example (see `experiments/manifests/` for shippable specs):

    schema_version: 1
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
      thoughts: 30        # cumulative thought_count target (state.json["thought_count"])
      # or: minutes: 20    # wall-clock minutes target
      # or: add_thoughts: 30  # produce N MORE thoughts (pairs with resume_from)
    success_criteria:
      # `kind` is a dotted path into the metrics dict produced by
      # experiments.metrics.compute_all (see metrics.json artifact)
      - kind: mood.dimensions_non_degenerate
        op: ">="
        value: 2
    resume_from: null        # name (or run-dir path) of a prior session to continue from
    replicates: null         # if int, run the manifest N times sequentially
    tags: [perception-on, mood-fix-on]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# Current manifest schema. Bump when adding required fields or changing
# semantics of an existing field; bump procedure: the loader's
# upgrade-on-read path.
CURRENT_SCHEMA_VERSION = 1


class Duration(BaseModel):
    """How long the experiment runs. Exactly one field must be set.

    - `minutes`: wall-clock minutes. The runner SIGTERMs the agent when the
      timer elapses regardless of how many thoughts were produced.
    - `thoughts`: cumulative `state.json["thought_count"]` target. For a
      fresh run, this is equivalent to "produce N thoughts". For a resumed
      run (see `resume_from`), the agent already has prior thoughts, so
      `thoughts: 250` against a resumed session at 200 means "produce 50 more".
    - `add_thoughts`: produce this many MORE thoughts THIS run, regardless
      of prior count. Pairs naturally with `resume_from` because the operator
      usually knows "I want 50 more cycles," not "I want a cumulative target".
    """

    minutes: float | None = None
    thoughts: int | None = None
    add_thoughts: int | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "Duration":
        set_fields = sum(
            1 for v in (self.minutes, self.thoughts, self.add_thoughts) if v is not None
        )
        if set_fields != 1:
            raise ValueError(
                "Duration must set exactly one of `minutes`, `thoughts`, or `add_thoughts`"
            )
        if self.minutes is not None and self.minutes <= 0:
            raise ValueError("Duration.minutes must be positive")
        if self.thoughts is not None and self.thoughts <= 0:
            raise ValueError("Duration.thoughts must be positive")
        if self.add_thoughts is not None and self.add_thoughts <= 0:
            raise ValueError("Duration.add_thoughts must be positive")
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

    schema_version: int = Field(
        default=CURRENT_SCHEMA_VERSION,
        description="Manifest schema version; bumped when fields change semantics. "
                    "Old manifests without this field are read as v1.",
    )
    name: str = Field(..., description="Stable identifier; used as the directory under experiments/")
    description: str = ""
    consciousness_name: str = Field(..., description="Used as spawn.py --name")
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    duration: Duration
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    branch_sha: str | None = None           # captured at run time if absent here
    resume_from: str | None = Field(
        default=None,
        description="Name of a consciousness in CONSCIOUSNESS_HOME (e.g. 'Echo') OR a "
                    "path to a recorded run directory. When set, the runner copies the "
                    "source's journal.jsonl + state.json into the new instance's dir "
                    "before spawning, instead of wiping. The new instance picks up where "
                    "the source left off, including identity name (which overrides "
                    "consciousness_name semantically — the agent's self-concept survives).",
    )
    replicates: int | None = Field(
        default=None,
        description="If set, run the manifest N times sequentially. Each replicate writes "
                    "to <run_dir>/replicate-<i>/. A replicates_index.md at the top of the "
                    "run dir lists all child reports. Aggregate metrics (mean/stddev) "
                    "across replicates are a Phase-2 follow-up.",
    )

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
