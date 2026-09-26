"""Tests for scripts/resume.py — restart a persisted instance by name.

Covers:
- the orchestrator is built for the requested name against the checkout's
  ``config/default_consciousness.yaml``, and the CLI dashboard is run once
  against that orchestrator.
- ``--log-level`` (default WARNING) and ``--name`` reach
  ``configure_logging`` unchanged.
- the startup usage event is tagged ``resume`` and is not a service.
- ``--name`` is required.

``Consciousness``, ``ConsciousnessCLI``, ``configure_logging``, and the usage
reporting calls are replaced with recorders, so no provider, run loop, log
file, or network is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from click.testing import CliRunner

# Make scripts/ importable the same way spawn.py's tests do.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.resume as resume  # noqa: E402


class _Recorder:
    """Collects every call the patched collaborators receive."""

    def __init__(self) -> None:
        self.minds: list[dict[str, Any]] = []
        self.cli_runs: list[Any] = []
        self.logging_calls: list[tuple[str, str]] = []
        self.startups: list[tuple[str, bool]] = []


def _install(monkeypatch, tmp_path: Path) -> _Recorder:
    rec = _Recorder()

    class _FakeConsciousness:
        def __init__(self, name: str, config_path: str) -> None:
            rec.minds.append({"name": name, "config_path": config_path, "mind": self})

    class _FakeCLI:
        def __init__(self, mind: Any) -> None:
            self.mind = mind

        async def run(self) -> None:
            rec.cli_runs.append(self.mind)

    def _configure_logging(name: str, level: str) -> Path:
        rec.logging_calls.append((name, level))
        return tmp_path / name / "run.log"

    def _report_startup(client: Any, command: str, service: bool = False) -> None:
        rec.startups.append((command, service))

    monkeypatch.setattr(resume, "Consciousness", _FakeConsciousness)
    monkeypatch.setattr(resume, "ConsciousnessCLI", _FakeCLI)
    monkeypatch.setattr(resume, "configure_logging", _configure_logging)
    monkeypatch.setattr(resume, "start_usage_reporting", lambda log: object())
    monkeypatch.setattr(resume, "report_startup", _report_startup)
    # A developer's consciousness-sim/.env must not leak into the test run.
    monkeypatch.setattr(resume, "load_dotenv", lambda: None)
    return rec


def test_resume_builds_named_instance_against_default_config(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)

    result = CliRunner().invoke(resume.main, ["--name", "Aria"])

    assert result.exit_code == 0, result.output
    assert len(rec.minds) == 1
    assert rec.minds[0]["name"] == "Aria"
    config_path = Path(rec.minds[0]["config_path"])
    assert config_path == _REPO_ROOT / "config" / "default_consciousness.yaml"
    assert config_path.is_file()


def test_resume_runs_cli_once_against_the_built_instance(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)

    result = CliRunner().invoke(resume.main, ["--name", "Aria"])

    assert result.exit_code == 0, result.output
    assert rec.cli_runs == [rec.minds[0]["mind"]]


def test_resume_default_log_level_is_warning(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)

    result = CliRunner().invoke(resume.main, ["--name", "Aria"])

    assert result.exit_code == 0, result.output
    assert rec.logging_calls == [("Aria", "WARNING")]


def test_resume_passes_explicit_log_level_through(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)

    result = CliRunner().invoke(resume.main, ["--name", "Aria", "--log-level", "DEBUG"])

    assert result.exit_code == 0, result.output
    assert rec.logging_calls == [("Aria", "DEBUG")]


def test_resume_reports_startup_as_resume_not_service(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)

    result = CliRunner().invoke(resume.main, ["--name", "Aria"])

    assert result.exit_code == 0, result.output
    assert rec.startups == [("resume", False)]


def test_resume_requires_name(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)

    result = CliRunner().invoke(resume.main, [])

    assert result.exit_code == 2
    assert "--name" in result.output
    assert rec.minds == []
    assert rec.cli_runs == []
