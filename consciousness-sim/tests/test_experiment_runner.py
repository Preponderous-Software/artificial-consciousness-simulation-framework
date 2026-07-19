"""Integration tests for experiments/runner.py.

Spawns a real subprocess with the MockProvider + MockPerception so no network
or Ollama is required. The whole test should complete in well under a minute.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.manifest import ExperimentManifest
from experiments.runner import (
    run_experiment,
    run_experiment_replicated,
    start_detached,
    status,
)


def _build_manifest(name: str, n_thoughts: int) -> ExperimentManifest:
    """Manifest using mock LLM + mock perception so no network or Ollama needed."""
    return ExperimentManifest.model_validate({
        "name": name,
        "description": "Integration test — mock provider + mock perception",
        "consciousness_name": name + "TestAgent",
        "config_overrides": {
            "llm": {"provider": "mock"},
            "perception": {
                "enabled": True,
                "provider": "mock",
                "every_n_cycles": 2,
            },
            "thought_loop": {
                # Speed up: zero sleep between cycles
                "min_interval_seconds": 0,
                "max_interval_seconds": 0,
            },
        },
        "duration": {"thoughts": n_thoughts},
        "tags": ["smoke-test"],
    })


# ---------------------------------------------------------------------------
# Pure unit tests — no subprocess, no I/O
# ---------------------------------------------------------------------------

def test_merge_config_deep_merges_nested_dicts() -> None:
    from experiments.runner import _merge_config

    base = {"llm": {"provider": "ollama", "model": "llama3.2:3b"}}
    overrides = {"llm": {"provider": "mock"}}
    merged = _merge_config(base, overrides)
    assert merged == {"llm": {"provider": "mock", "model": "llama3.2:3b"}}


def test_merge_config_replaces_lists_wholesale_not_concatenated() -> None:
    from experiments.runner import _merge_config

    base = {"tags": ["a", "b"]}
    overrides = {"tags": ["c"]}
    merged = _merge_config(base, overrides)
    assert merged["tags"] == ["c"]


def test_merge_config_preserves_base_only_keys() -> None:
    from experiments.runner import _merge_config

    base = {"a": 1, "b": {"nested": 2}}
    overrides = {"c": 3}
    merged = _merge_config(base, overrides)
    assert merged == {"a": 1, "b": {"nested": 2}, "c": 3}


def test_runner_produces_all_artifacts(tmp_path: Path, monkeypatch) -> None:
    """Happy path: target reached, all six artifacts present, report renders."""
    # Isolate ~/.consciousness so we don't trash real instances
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))

    manifest = _build_manifest("smoke-happy-path", n_thoughts=3)
    run_dir = run_experiment(
        manifest,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=2.0,
        poll_interval_s=0.2,
    )

    assert run_dir.exists()
    for artifact in ("manifest.yaml", "meta.yaml", "journal.jsonl",
                      "state.json", "metrics.json", "report.md"):
        path = run_dir / artifact
        assert path.exists() and path.stat().st_size > 0, f"missing or empty: {artifact}"

    # meta.yaml structure
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text())
    assert "exit_reason" in meta
    assert "started_at" in meta
    assert "ended_at" in meta
    assert "wall_clock_minutes" in meta

    # metrics.json contains the expected top-level shape
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for key in ("event_counts", "vocabulary", "mood", "perception", "reflections", "performance"):
        assert key in metrics, f"missing metrics section: {key}"
    assert metrics["event_counts"].get("thought", 0) >= 3

    # report.md has the expected sections
    report = (run_dir / "report.md").read_text()
    for section in ("# Experiment report", "## Summary", "## Mood", "## Vocabulary", "## Performance"):
        assert section in report


def test_runner_target_reason_is_recorded(tmp_path: Path, monkeypatch) -> None:
    """When the thought target is hit cleanly, exit_reason should say so."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))

    manifest = _build_manifest("smoke-target-hit", n_thoughts=2)
    run_dir = run_experiment(
        manifest,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=2.0,
        poll_interval_s=0.2,
    )
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text())
    assert "reached target" in meta["exit_reason"]


def test_runner_wall_clock_cap_terminates_long_run(tmp_path: Path, monkeypatch) -> None:
    """A tiny wall-clock cap should SIGTERM the process and record that reason."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))

    # Target many thoughts so we definitely hit the wall-clock cap first.
    # Mock provider is fast; use a very tight cap to be sure we hit it.
    manifest = _build_manifest("smoke-walltime-cap", n_thoughts=10_000)
    run_dir = run_experiment(
        manifest,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=0.05,    # 3 seconds — well below typical mock-run rate
        poll_interval_s=0.2,
    )
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text())
    assert "max_wall_clock_minutes" in meta["exit_reason"]


def test_runner_starts_from_clean_consciousness_dir(tmp_path: Path, monkeypatch) -> None:
    """If a previous instance's data exists, the runner deletes it before spawning."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))
    # The runtime sanitizes the consciousness name (`-` → `_`), so pre-create
    # the stale dir at the sanitized location the runner will actually wipe.
    from persistence.paths import sanitize_consciousness_name
    raw_name = "smoke-clean-startTestAgent"
    prior_dir = tmp_path / "home" / sanitize_consciousness_name(raw_name)
    prior_dir.mkdir(parents=True)
    (prior_dir / "journal.jsonl").write_text('{"timestamp": "2000-01-01T00:00:00Z", "type": "thought", "content": "stale"}\n')
    (prior_dir / "stale-marker").write_text("if this survives the runner didn't reset")

    manifest = _build_manifest("smoke-clean-start", n_thoughts=2)
    run_dir = run_experiment(
        manifest,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=2.0,
        poll_interval_s=0.2,
    )
    # The stale-marker file should not exist anymore (dir was wiped before spawn)
    assert not (prior_dir / "stale-marker").exists()
    # The recorded journal should be from THIS run, not the stale one
    journal_text = (run_dir / "journal.jsonl").read_text()
    assert "stale" not in journal_text


# ---------------------------------------------------------------------------
# Resume mode
# ---------------------------------------------------------------------------

def test_runner_resume_from_seeds_state_and_journal(tmp_path: Path, monkeypatch) -> None:
    """resume_from copies journal.jsonl + state.json from the source into the
    new instance's dir BEFORE spawning, so the agent picks up where it left off."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))

    # 1. Run a fresh experiment to produce some prior state
    source_manifest = _build_manifest("resume-source", n_thoughts=2)
    source_run_dir = run_experiment(
        source_manifest,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=2.0,
        poll_interval_s=0.2,
    )
    starting_thoughts = json.loads((source_run_dir / "state.json").read_text())["thought_count"]
    assert starting_thoughts >= 2

    # 2. Run a second experiment with resume_from pointing at the source run dir
    resumed = ExperimentManifest.model_validate({
        "name": "resume-target",
        "consciousness_name": "resumeTargetAgent",
        "resume_from": str(source_run_dir),
        "config_overrides": _build_manifest("ignored", 1).config_overrides,
        # add_thoughts: 1 means "produce 1 more thought beyond starting count"
        "duration": {"add_thoughts": 1},
    })
    resumed_run = run_experiment(
        resumed,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=2.0,
        poll_interval_s=0.2,
    )

    meta = yaml.safe_load((resumed_run / "meta.yaml").read_text())
    assert meta.get("resumed_from") == str(source_run_dir)
    assert meta.get("starting_thought_count") == starting_thoughts


def test_runner_resume_from_unknown_source_raises(tmp_path: Path, monkeypatch) -> None:
    """A resume_from that points at nothing should fail loudly at runtime."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))
    from experiments.runner import RunnerError
    manifest = ExperimentManifest.model_validate({
        "name": "resume-bad",
        "consciousness_name": "BadResume",
        "resume_from": "DoesNotExistAnywhere",
        "duration": {"thoughts": 1},
    })
    with pytest.raises(RunnerError, match="resume_from"):
        run_experiment(
            manifest,
            experiments_root=tmp_path / "experiments",
            max_wall_clock_minutes=1.0,
            poll_interval_s=0.1,
        )


# ---------------------------------------------------------------------------
# add_thoughts (delta) duration mode
# ---------------------------------------------------------------------------

def test_add_thoughts_target_uses_starting_count(tmp_path: Path, monkeypatch) -> None:
    """For a fresh run, add_thoughts: N is equivalent to producing N thoughts.
    The point of add_thoughts is to be unambiguous when resuming."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))
    base = _build_manifest("add-thoughts-fresh", 2)
    # Replace thoughts target with add_thoughts
    spec = base.model_dump(mode="json")
    spec["duration"] = {"add_thoughts": 2}
    manifest = ExperimentManifest.model_validate(spec)
    run_dir = run_experiment(
        manifest,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=2.0,
        poll_interval_s=0.2,
    )
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text())
    assert meta["starting_thought_count"] == 0
    assert "added" in meta["exit_reason"] or "reached" in meta["exit_reason"]


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------

def test_replicates_produces_n_subdirs_plus_index(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))
    base = _build_manifest("replicates-2x", 2)
    spec = base.model_dump(mode="json")
    spec["replicates"] = 2
    manifest = ExperimentManifest.model_validate(spec)
    parent = run_experiment_replicated(
        manifest,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=2.0,
        poll_interval_s=0.2,
    )
    assert parent.is_dir()
    assert (parent / "replicates_index.md").exists()
    children = sorted(parent.glob("replicate-*"))
    assert len(children) == 2
    for c in children:
        assert (c / "report.md").exists()
        assert (c / "metrics.json").exists()


def test_replicates_one_or_none_falls_through_to_single_run(tmp_path: Path, monkeypatch) -> None:
    """replicates: 1 (or unset) should behave exactly like a normal run."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))
    base = _build_manifest("replicates-fallthrough", 2)
    spec = base.model_dump(mode="json")
    spec["replicates"] = 1
    manifest = ExperimentManifest.model_validate(spec)
    run_dir = run_experiment_replicated(
        manifest,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=2.0,
        poll_interval_s=0.2,
    )
    # Single run dir: report.md sits directly under run_dir, not under replicate-0
    assert (run_dir / "report.md").exists()
    assert not (run_dir / "replicate-0").exists()


# ---------------------------------------------------------------------------
# Status function
# ---------------------------------------------------------------------------

def test_status_reports_done_after_completed_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))
    manifest = _build_manifest("status-done", 2)
    run_dir = run_experiment(
        manifest,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=2.0,
        poll_interval_s=0.2,
    )
    info = status(run_dir)
    assert info["state"] == "done"
    assert "exit_reason" in info


def test_status_reports_unknown_for_nonexistent_dir(tmp_path: Path) -> None:
    info = status(tmp_path / "nonexistent")
    assert info["state"] == "unknown"


def test_status_reports_running_when_marker_present(tmp_path: Path) -> None:
    """A run dir with .STARTED but no report.md is in-progress."""
    fake_run = tmp_path / "fake"
    fake_run.mkdir()
    (fake_run / ".STARTED").touch()
    info = status(fake_run)
    assert info["state"] == "running"


def test_status_reports_done_for_replicates_parent_dir(tmp_path: Path) -> None:
    """Regression for PR #85 review: a replicated run's parent dir doesn't
    have a `report.md`; it has `replicates_index.md`. status() must detect that."""
    parent = tmp_path / "replicated"
    parent.mkdir()
    (parent / "replicates_index.md").write_text("# 2 replicates\n")
    (parent / "replicate-0").mkdir()
    (parent / "replicate-1").mkdir()
    info = status(parent)
    assert info["state"] == "done"
    assert info["kind"] == "replicated"
    assert info["n_replicates"] == 2


def test_replicates_loop_accepts_prebuilt_parent_dir(tmp_path: Path, monkeypatch) -> None:
    """Regression for PR #85 review: when `start_detached` pre-creates the
    parent dir, `run_experiment_replicated` must populate THAT dir rather
    than creating its own timestamped sibling."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))
    base = _build_manifest("replicates-prebuilt", 2)
    spec = base.model_dump(mode="json")
    spec["replicates"] = 2
    manifest = ExperimentManifest.model_validate(spec)

    pre_built = tmp_path / "experiments" / "replicates-prebuilt" / "MY-FIXED-NAME"
    pre_built.mkdir(parents=True)

    returned = run_experiment_replicated(
        manifest,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=2.0,
        poll_interval_s=0.2,
        parent_dir=pre_built,
    )
    assert returned == pre_built, f"expected populate-in-place, got fresh dir {returned}"
    assert (pre_built / "replicates_index.md").exists()
    assert (pre_built / "replicate-0" / "report.md").exists()
    assert (pre_built / "replicate-1" / "report.md").exists()


# ---------------------------------------------------------------------------
# Schema versioning in metrics output
# ---------------------------------------------------------------------------

def test_metrics_json_carries_schema_version(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))
    manifest = _build_manifest("schema-version", 2)
    run_dir = run_experiment(
        manifest,
        experiments_root=tmp_path / "experiments",
        max_wall_clock_minutes=2.0,
        poll_interval_s=0.2,
    )
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics.get("_schema_version") == 1
    # And meta.yaml carries the manifest schema version
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text())
    assert meta.get("manifest_schema_version") == 1
