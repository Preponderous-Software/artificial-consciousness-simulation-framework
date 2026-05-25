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
from experiments.runner import run_experiment


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
