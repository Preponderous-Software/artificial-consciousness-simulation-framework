"""Tests for scripts/resume.py — restart a persisted instance by name.

Covers:
- the orchestrator is built for the requested name against the checkout's
  ``config/default_consciousness.yaml``, and the CLI dashboard is run once
  against that orchestrator.
- ``--log-level`` (default WARNING) and ``--name`` reach
  ``configure_logging`` unchanged.
- the startup usage event is tagged ``resume`` and is not a service.
- ``--name`` is required.
- an instance's persisted ``config.yaml`` (#187) is preferred over the
  default, and ``--config`` replaces it.
- a resumed instance records a pid file for its lifetime and refuses a live
  duplicate unless ``--force`` is passed (#237).

``Consciousness``, ``ConsciousnessCLI``, ``configure_logging``, and the usage
reporting calls are replaced with recorders, so no provider, run loop, log
file, or network is touched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

from click.testing import CliRunner

# Make scripts/ importable the same way spawn.py's tests do.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.resume as resume  # noqa: E402
import scripts.spawn as spawn  # noqa: E402


class _Recorder:
    """Collects every call the patched collaborators receive."""

    def __init__(self) -> None:
        self.minds: list[dict[str, Any]] = []
        self.cli_runs: list[Any] = []
        self.logging_calls: list[tuple[str, str]] = []
        self.startups: list[tuple[str, bool]] = []
        self.exit_handlers: list[Callable[[], None]] = []
        self.pid_during_run: list[str | None] = []


def _install(monkeypatch, tmp_path: Path) -> _Recorder:
    rec = _Recorder()
    # An instance's persisted config.yaml (#187) is looked up under
    # CONSCIOUSNESS_HOME; the developer's real ~/.consciousness must not leak in.
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    class _FakeConsciousness:
        def __init__(self, name: str, config_path: str) -> None:
            rec.minds.append({"name": name, "config_path": config_path, "mind": self})

    class _FakeCLI:
        def __init__(self, mind: Any) -> None:
            self.mind = mind

        async def run(self) -> None:
            rec.cli_runs.append(self.mind)
            pid_path = tmp_path / "Aria" / "pid"
            rec.pid_during_run.append(pid_path.read_text() if pid_path.exists() else None)

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
    monkeypatch.setattr(spawn.atexit, "register", rec.exit_handlers.append)
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


def _write_persisted(tmp_path: Path, name: str, rpt_critique: bool) -> Path:
    import yaml

    cfg = yaml.safe_load((_REPO_ROOT / "config" / "default_consciousness.yaml").read_text(encoding="utf-8"))
    cfg["thought_loop"]["rpt_critique"] = rpt_critique
    path = tmp_path / name / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_resume_prefers_persisted_instance_config(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)
    persisted = _write_persisted(tmp_path, "Aria", rpt_critique=True)

    result = CliRunner().invoke(resume.main, ["--name", "Aria"])

    assert result.exit_code == 0, result.output
    assert Path(rec.minds[0]["config_path"]) == persisted


def test_resume_config_flag_replaces_persisted_config(monkeypatch, tmp_path) -> None:
    import yaml

    rec = _install(monkeypatch, tmp_path)
    persisted = _write_persisted(tmp_path, "Aria", rpt_critique=False)
    override = _write_persisted(tmp_path, "elsewhere", rpt_critique=True)

    result = CliRunner().invoke(resume.main, ["--name", "Aria", "--config", str(override)])

    assert result.exit_code == 0, result.output
    assert Path(rec.minds[0]["config_path"]) == persisted
    assert yaml.safe_load(persisted.read_text(encoding="utf-8"))["thought_loop"]["rpt_critique"] is True


def test_resume_invalid_config_exits_without_building(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm: [unclosed\n", encoding="utf-8")

    result = CliRunner().invoke(resume.main, ["--name", "Aria", "--config", str(bad)])

    assert result.exit_code == 1
    assert "Cannot read config" in result.output
    assert rec.minds == []


# --- pid file (#237) ------------------------------------------------------


def _write_live_pid(tmp_path: Path, name: str) -> Path:
    # The test process itself is the one PID guaranteed to be alive.
    pid_path = tmp_path / name / "pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    return pid_path


def test_resume_records_pid_file_while_running(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)

    result = CliRunner().invoke(resume.main, ["--name", "Aria"])

    assert result.exit_code == 0, result.output
    assert rec.pid_during_run == [str(os.getpid())]


def test_resume_removes_its_pid_file_at_exit(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)

    result = CliRunner().invoke(resume.main, ["--name", "Aria"])

    assert result.exit_code == 0, result.output
    assert len(rec.exit_handlers) == 1
    rec.exit_handlers[0]()
    assert not (tmp_path / "Aria" / "pid").exists()


def test_resumed_instance_is_refused_by_a_later_spawn(monkeypatch, tmp_path) -> None:
    _install(monkeypatch, tmp_path)
    monkeypatch.setattr(spawn, "load_dotenv", lambda: None)
    # The exit handler is captured, not run, so the resume above still holds
    # the pid file the way a live resumed process does.
    assert CliRunner().invoke(resume.main, ["--name", "Aria"]).exit_code == 0

    result = CliRunner().invoke(spawn.main, ["--name", "Aria", "--headless"])

    assert result.exit_code == 1
    assert "already running" in result.output


def test_resume_refuses_a_live_instance(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)
    _write_live_pid(tmp_path, "Aria")

    result = CliRunner().invoke(resume.main, ["--name", "Aria"])

    assert result.exit_code == 1
    assert "already running" in result.output
    assert rec.minds == []
    assert rec.startups == [], "a refused resume is not a startup"


def test_resume_force_proceeds_past_a_live_instance(monkeypatch, tmp_path) -> None:
    rec = _install(monkeypatch, tmp_path)
    _write_live_pid(tmp_path, "Aria")

    result = CliRunner().invoke(resume.main, ["--name", "Aria", "--force"])

    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output
    assert len(rec.minds) == 1
