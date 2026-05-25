"""Tests for experiments/manifest.py — Pydantic schema validation + YAML round-trip."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from experiments.manifest import (
    Duration,
    ExperimentManifest,
    SuccessCriterion,
    evaluate_success_criterion,
)


def test_duration_requires_exactly_one_field() -> None:
    Duration(thoughts=10)        # fine
    Duration(minutes=5.0)        # fine
    with pytest.raises(ValueError, match="exactly one"):
        Duration()
    with pytest.raises(ValueError, match="exactly one"):
        Duration(thoughts=10, minutes=5.0)


def test_duration_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        Duration(thoughts=0)
    with pytest.raises(ValueError, match="positive"):
        Duration(minutes=-1.0)


def test_manifest_from_yaml_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "test-manifest.yaml"
    path.write_text(textwrap.dedent("""
        name: sample
        description: a tiny test manifest
        consciousness_name: Sample
        config_overrides:
          llm:
            provider: mock
        duration:
          thoughts: 5
        success_criteria:
          - kind: mood.dimensions_non_degenerate
            op: ">="
            value: 2
        tags: [mock, smoke-test]
    """).strip())

    manifest = ExperimentManifest.from_yaml(path)
    assert manifest.name == "sample"
    assert manifest.consciousness_name == "Sample"
    assert manifest.duration.thoughts == 5
    assert len(manifest.success_criteria) == 1
    assert manifest.tags == ["mock", "smoke-test"]

    # Round-trip through to_yaml must parse back cleanly
    re_loaded = ExperimentManifest.model_validate(yaml.safe_load(manifest.to_yaml()))
    assert re_loaded.name == manifest.name
    assert re_loaded.duration.thoughts == manifest.duration.thoughts


def test_manifest_rejects_missing_required_fields(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: only-this-field\n")
    with pytest.raises(ValueError):
        ExperimentManifest.from_yaml(bad)


def test_manifest_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        ExperimentManifest.from_yaml(bad)


def test_evaluate_success_criterion_walks_dotted_path() -> None:
    metrics = {"mood": {"dimensions_non_degenerate": 3}, "vocabulary": {"top_word_density_per_thought": 0.8}}
    c1 = SuccessCriterion(kind="mood.dimensions_non_degenerate", op=">=", value=2)
    passed, actual = evaluate_success_criterion(c1, metrics)
    assert passed is True
    assert actual == 3.0


def test_evaluate_success_criterion_fails_closed_on_missing_path() -> None:
    metrics = {"mood": {}}
    c = SuccessCriterion(kind="mood.nonexistent_key", op=">", value=0)
    passed, actual = evaluate_success_criterion(c, metrics)
    assert passed is False
    assert actual is None


def test_evaluate_supports_all_ops() -> None:
    metrics = {"x": 1.0}
    for op, value, expected in [
        (">",  0.5, True),
        (">=", 1.0, True),
        ("<",  2.0, True),
        ("<=", 1.0, True),
        ("==", 1.0, True),
        ("!=", 2.0, True),
        (">",  2.0, False),
    ]:
        c = SuccessCriterion(kind="x", op=op, value=value)
        passed, _ = evaluate_success_criterion(c, metrics)
        assert passed is expected, f"op={op}, value={value} expected {expected}"


# ---------------------------------------------------------------------------
# Schema versioning + new fields (resume_from, add_thoughts, replicates)
# ---------------------------------------------------------------------------

def test_manifest_default_schema_version_is_1() -> None:
    m = ExperimentManifest.model_validate({
        "name": "x", "consciousness_name": "X",
        "duration": {"thoughts": 1},
    })
    assert m.schema_version == 1


def test_manifest_accepts_explicit_schema_version(tmp_path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text(
        "schema_version: 1\nname: x\nconsciousness_name: X\nduration: {thoughts: 1}\n"
    )
    m = ExperimentManifest.from_yaml(p)
    assert m.schema_version == 1
    # to_yaml emits the field so future loaders can detect format
    assert "schema_version: 1" in m.to_yaml()


def test_duration_accepts_add_thoughts_xor_other_modes() -> None:
    Duration(add_thoughts=5)             # fine on its own
    with pytest.raises(ValueError, match="exactly one"):
        Duration(add_thoughts=5, thoughts=10)
    with pytest.raises(ValueError, match="exactly one"):
        Duration(add_thoughts=5, minutes=2.0)
    with pytest.raises(ValueError, match="positive"):
        Duration(add_thoughts=0)
    with pytest.raises(ValueError, match="positive"):
        Duration(add_thoughts=-3)


def test_manifest_accepts_resume_from(tmp_path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text(
        "name: x\nconsciousness_name: X\nresume_from: Echo\n"
        "duration: {add_thoughts: 5}\n"
    )
    m = ExperimentManifest.from_yaml(p)
    assert m.resume_from == "Echo"
    assert m.duration.add_thoughts == 5


def test_manifest_accepts_replicates(tmp_path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text(
        "name: x\nconsciousness_name: X\nreplicates: 3\n"
        "duration: {thoughts: 5}\n"
    )
    m = ExperimentManifest.from_yaml(p)
    assert m.replicates == 3


def test_manifest_backward_compatible_with_no_new_fields(tmp_path) -> None:
    """Pre-existing manifests without schema_version / resume_from / replicates
    must still parse without error — defaults fill in the gaps."""
    p = tmp_path / "old.yaml"
    p.write_text("name: x\nconsciousness_name: X\nduration: {thoughts: 1}\n")
    m = ExperimentManifest.from_yaml(p)
    assert m.schema_version == 1
    assert m.resume_from is None
    assert m.replicates is None
    assert m.duration.add_thoughts is None
