"""Tests for scripts/experiment.py — the experiment harness CLI (#57).

Covers the click wiring of every subcommand, not the library code behind it
(that lives in test_experiment_runner / _compare / _prune / _regression /
_report / _metrics):
- run: the manifest summary echo, foreground vs --detach dispatch, and the
  single vs replicated artifact hints.
- status: the state line printed first, then every other key.
- list: the missing / empty experiments root, skipping golden/, manifests/,
  meta-less and unparseable run dirs, and newest-first ordering.
- replay-analysis: the missing-artifact refusal and the metrics.json +
  report.md rewrite from a golden journal.
- compare: stdout vs --output.
- check-smoke: the missing-metrics refusal, the pass message, and exit 1 on
  drift.
- prune: the default --keep-last 10 policy, dry-run by default, and the two
  flags that drop the dry-run safety.

Nothing here spawns a simulation: run / prune / compare are exercised through
stubs of the names scripts.experiment imported, and list reads a temporary
EXPERIMENTS_ROOT.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment as exp_mod  # noqa: E402
from scripts.experiment import main as experiment_main  # noqa: E402

_MANIFEST = _REPO_ROOT / "experiments" / "manifests" / "mock-smoke-baseline.yaml"
_GOLDEN = _REPO_ROOT / "experiments" / "golden"
_EXPECTED = _GOLDEN / "_smoke_expected.json"


def _invoke(args: list[str]) -> Any:
    return CliRunner().invoke(experiment_main, args)


def _stub_usage_reporting(monkeypatch) -> None:
    monkeypatch.setattr(exp_mod, "start_usage_reporting", lambda log: None)
    monkeypatch.setattr(exp_mod, "report_experiment_started", lambda client: None)


def _write_run(root: Path, manifest: str, ts: str, meta: str | None) -> Path:
    run_dir = root / manifest / ts
    run_dir.mkdir(parents=True)
    if meta is not None:
        (run_dir / "meta.yaml").write_text(meta, encoding="utf-8")
    return run_dir


def _passing_metrics() -> dict[str, Any]:
    """A metrics.json body satisfying every pin in _smoke_expected.json."""
    return {
        "vocabulary": {"top_word_density_per_thought": 2.0},
        "mood": {
            "final": {
                "curiosity": 0.86,
                "wonder": 0.6,
                "melancholy": 0.2,
                "contentment": 0.5,
            },
            "dimensions_non_degenerate": 4,
        },
        "perception": {"influence_rate": 0.0},
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_foreground_echoes_manifest_and_single_run_artifacts(monkeypatch, tmp_path) -> None:
    _stub_usage_reporting(monkeypatch)
    calls: list[tuple[str, float]] = []

    def _fake_replicated(manifest, max_wall_clock_minutes):
        calls.append((manifest.name, max_wall_clock_minutes))
        return tmp_path / "run-sentinel"

    monkeypatch.setattr(exp_mod, "run_experiment_replicated", _fake_replicated)
    monkeypatch.setattr(
        exp_mod, "start_detached",
        lambda *a, **k: pytest.fail("start_detached must not run without --detach"),
    )

    result = _invoke(["run", str(_MANIFEST), "--max-wall-clock-minutes", "2.5"])

    assert result.exit_code == 0, result.output
    assert calls == [("mock-smoke-baseline", 2.5)]
    assert "Running manifest: mock-smoke-baseline" in result.output
    assert "consciousness: MockSmoke" in result.output
    assert "tags:          ['mock', 'smoke-test', 'harness-self-test']" in result.output
    assert "replicates:" not in result.output
    assert f"Run complete: {tmp_path / 'run-sentinel'}" in result.output
    assert "report.md" in result.output
    assert "metrics.json" in result.output
    assert "replicates_index.md" not in result.output


def test_run_replicated_manifest_points_at_replicates_index(monkeypatch, tmp_path) -> None:
    _stub_usage_reporting(monkeypatch)
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    manifest["replicates"] = 3
    manifest_path = tmp_path / "replicated.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    run_dir = tmp_path / "replicated-run"
    monkeypatch.setattr(exp_mod, "run_experiment_replicated", lambda m, **k: run_dir)

    result = _invoke(["run", str(manifest_path)])

    assert result.exit_code == 0, result.output
    assert "replicates:    3" in result.output
    assert str(run_dir / "replicates_index.md") in result.output
    assert "(3 replicates under" in result.output
    assert "metrics.json" not in result.output


def test_run_detach_forks_and_returns_without_running(monkeypatch, tmp_path) -> None:
    _stub_usage_reporting(monkeypatch)
    detached: list[tuple[Path, float]] = []

    def _fake_start_detached(manifest_path, max_wall_clock_minutes):
        detached.append((manifest_path, max_wall_clock_minutes))
        return tmp_path / "detached-run"

    monkeypatch.setattr(exp_mod, "start_detached", _fake_start_detached)
    monkeypatch.setattr(
        exp_mod, "run_experiment_replicated",
        lambda *a, **k: pytest.fail("--detach must not run in the foreground"),
    )

    result = _invoke(["run", str(_MANIFEST), "--detach"])

    assert result.exit_code == 0, result.output
    assert detached == [(_MANIFEST, 30.0)]
    assert f"Detached: {tmp_path / 'detached-run'}" in result.output
    assert f"status {tmp_path / 'detached-run'}" in result.output
    assert "_detached.log" in result.output
    assert "Run complete" not in result.output


def test_run_rejects_missing_manifest(tmp_path) -> None:
    result = _invoke(["run", str(tmp_path / "absent.yaml")])

    assert result.exit_code == 2
    assert "does not exist" in result.output


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_prints_state_line_then_remaining_keys(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        exp_mod, "runner_status",
        lambda run_dir: {"kind": "single", "state": "done", "exit_reason": "thoughts_reached"},
    )

    result = _invoke(["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == f"{tmp_path}: done"
    assert lines[1:] == ["  kind: single", "  exit_reason: thoughts_reached"]


def test_status_reads_a_real_run_dir(tmp_path) -> None:
    (tmp_path / "report.md").write_text("# report", encoding="utf-8")

    result = _invoke(["status", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == f"{tmp_path}: done"
    assert "  kind: single" in result.output


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_reports_missing_experiments_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_ROOT", tmp_path / "absent")

    result = _invoke(["list"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "No experiments directory yet."


def test_list_reports_no_runs_when_only_protected_dirs_exist(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_ROOT", tmp_path)
    _write_run(tmp_path, "golden", "Echo", "exit_reason: golden\nwall_clock_minutes: 1\n")
    _write_run(tmp_path, "manifests", "x", "exit_reason: manifest\nwall_clock_minutes: 1\n")
    (tmp_path / "loose-file.py").write_text("", encoding="utf-8")

    result = _invoke(["list"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "No recorded runs found."


def test_list_rows_are_newest_first_and_skip_unreadable_runs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(exp_mod, "EXPERIMENTS_ROOT", tmp_path)
    _write_run(tmp_path, "smoke", "20260101T000000Z", "exit_reason: thoughts_reached\nwall_clock_minutes: 1.25\n")
    _write_run(tmp_path, "smoke", "20260102T000000Z", "exit_reason: wall_clock\nwall_clock_minutes: 30\n")
    _write_run(tmp_path, "smoke", "20260103T000000Z", None)
    _write_run(tmp_path, "smoke", "20260104T000000Z", "exit_reason: [unclosed\n")
    _write_run(tmp_path, "a-longer-manifest", "20260105T000000Z", "{}\n")

    result = _invoke(["list"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "a-longer-manifest  20260105T000000Z  0.0m      ?",
        "smoke              20260102T000000Z  30.0m     wall_clock",
        "smoke              20260101T000000Z  1.2m      thoughts_reached",
    ]


# ---------------------------------------------------------------------------
# replay-analysis
# ---------------------------------------------------------------------------


def test_replay_analysis_refuses_when_an_artifact_is_missing(tmp_path) -> None:
    (tmp_path / "journal.jsonl").write_text("", encoding="utf-8")

    result = _invoke(["replay-analysis", str(tmp_path)])

    assert result.exit_code == 1
    assert f"Missing artifact: {tmp_path / 'state.json'}" in result.output
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.skipif(not (_GOLDEN / "Echo").exists(), reason="golden journals not yet committed")
def test_replay_analysis_rewrites_metrics_and_report(tmp_path) -> None:
    shutil.copy(_GOLDEN / "Echo" / "journal.jsonl", tmp_path / "journal.jsonl")
    shutil.copy(_GOLDEN / "Echo" / "state.json", tmp_path / "state.json")
    shutil.copy(_MANIFEST, tmp_path / "manifest.yaml")
    (tmp_path / "meta.yaml").write_text(
        "branch_sha: sentinel-sha\nexit_reason: thoughts_reached\nwall_clock_minutes: 2.0\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text('{"stale": true}', encoding="utf-8")

    result = _invoke(["replay-analysis", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert f"Re-analyzed {tmp_path}" in result.output
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "stale" not in metrics
    assert metrics["event_counts"]["thought"] > 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "mock-smoke-baseline" in report
    assert "sentinel-sha" in report


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_echoes_markdown_and_forwards_samples(monkeypatch, tmp_path) -> None:
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    seen: list[tuple[Path, Path, int]] = []

    def _fake_compare(a, b, k_samples):
        seen.append((a, b, k_samples))
        return "# comparison-sentinel"

    monkeypatch.setattr(exp_mod, "compare_runs", _fake_compare)

    result = _invoke(["compare", str(run_a), str(run_b), "--samples", "5"])

    assert result.exit_code == 0, result.output
    assert seen == [(run_a, run_b, 5)]
    assert result.output.strip() == "# comparison-sentinel"


def test_compare_writes_output_file_instead_of_stdout(monkeypatch, tmp_path) -> None:
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    out = tmp_path / "cmp.md"
    monkeypatch.setattr(exp_mod, "compare_runs", lambda a, b, k_samples: "# comparison-sentinel")

    result = _invoke(["compare", str(run_a), str(run_b), "--output", str(out)])

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == "# comparison-sentinel"
    assert "# comparison-sentinel" not in result.output
    assert f"Wrote comparison to {out}" in result.output


# ---------------------------------------------------------------------------
# check-smoke
# ---------------------------------------------------------------------------


def test_check_smoke_refuses_without_metrics_json(tmp_path) -> None:
    result = _invoke(["check-smoke", str(tmp_path)])

    assert result.exit_code == 1
    assert f"Missing artifact: {tmp_path / 'metrics.json'}" in result.output


def test_check_smoke_passes_against_default_golden_snapshot(tmp_path) -> None:
    (tmp_path / "metrics.json").write_text(json.dumps(_passing_metrics()), encoding="utf-8")
    pinned = len(json.loads(_EXPECTED.read_text(encoding="utf-8")))

    result = _invoke(["check-smoke", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert f"Smoke regression check passed against {_EXPECTED} ({pinned} field(s) pinned)." in result.output


def test_check_smoke_exits_1_and_lists_drift(tmp_path) -> None:
    metrics = _passing_metrics()
    metrics["mood"]["final"]["wonder"] = 0.95
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({"mood.final.wonder": 0.6}), encoding="utf-8")

    result = _invoke(["check-smoke", str(tmp_path), "--expected", str(expected)])

    assert result.exit_code == 1
    assert f"Smoke regression check FAILED against {expected}:" in result.output
    assert "mood.final.wonder" in result.output


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def _capture_prune(monkeypatch, targets: list[Path]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _fake_prune(root, **kwargs):
        calls.append({"root": root, **kwargs})
        return targets

    monkeypatch.setattr(exp_mod, "prune_runs", _fake_prune)
    return calls


def test_prune_defaults_to_keep_last_10_dry_run(monkeypatch, tmp_path) -> None:
    calls = _capture_prune(monkeypatch, [tmp_path / "old-run"])

    result = _invoke(["prune"])

    assert result.exit_code == 0, result.output
    assert calls == [{
        "root": exp_mod.EXPERIMENTS_ROOT,
        "older_than_days": None,
        "keep_last": 10,
        "manifest": None,
        "dry_run": True,
    }]
    assert "defaulting to --keep-last 10" in result.output
    assert "Would delete 1 run dir(s):" in result.output
    assert "Pass --yes to actually delete." in result.output


def test_prune_explicit_flag_suppresses_default_policy(monkeypatch) -> None:
    calls = _capture_prune(monkeypatch, [])

    result = _invoke(["prune", "--older-than", "30", "--manifest", "smoke"])

    assert result.exit_code == 0, result.output
    assert calls[0]["older_than_days"] == 30
    assert calls[0]["keep_last"] is None
    assert calls[0]["manifest"] == "smoke"
    assert "defaulting" not in result.output
    assert result.output.strip() == "Nothing to prune."


def test_prune_yes_deletes(monkeypatch, tmp_path) -> None:
    calls = _capture_prune(monkeypatch, [tmp_path / "old-run"])

    result = _invoke(["prune", "--keep-last", "5", "--yes"])

    assert result.exit_code == 0, result.output
    assert calls[0]["dry_run"] is False
    assert "Deleted 1 run dir(s):" in result.output
    assert "Pass --yes" not in result.output


def test_prune_no_dry_run_alone_also_deletes(monkeypatch, tmp_path) -> None:
    """Characterizes current behaviour: --no-dry-run drops the safety on its
    own, as the inline comment in cmd_prune says, although the option's help
    text claims it "requires --yes"."""
    calls = _capture_prune(monkeypatch, [tmp_path / "old-run"])

    result = _invoke(["prune", "--keep-last", "5", "--no-dry-run"])

    assert result.exit_code == 0, result.output
    assert calls[0]["dry_run"] is False
    assert "Deleted 1 run dir(s):" in result.output
