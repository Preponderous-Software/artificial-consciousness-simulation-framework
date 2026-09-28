"""Tests for scripts/spawn.py helpers.

Covers:
- the --config flag (#187) reaches Consciousness as the persisted
  <instance>/config.yaml, and a bad file fails before anything starts.
  Resolution rules themselves are covered in test_instance_config.py.
- _check_duplicate_pid's refusal/cleanup behaviour (#115) — duplicate spawns
  must abort, stale pid files must be cleaned up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

# Make scripts/ importable the same way spawn.py does at runtime.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.spawn as spawn  # noqa: E402
from scripts.spawn import (  # noqa: E402
    _check_duplicate_pid,
    _is_alive,
    _read_pid,
)


# --- _is_alive / _read_pid helpers ---------------------------------------


def test_is_alive_returns_true_for_current_process() -> None:
    assert _is_alive(os.getpid()) is True


def test_is_alive_returns_false_for_dead_pid() -> None:
    # Find a PID guaranteed not to exist.
    pid = 1
    for candidate in range(2**15, 2**22):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            pid = candidate
            break
        except (PermissionError, OSError):
            continue
    assert pid != 1, "could not locate a dead PID for this test"
    assert _is_alive(pid) is False


def test_is_alive_returns_false_for_zero_or_negative() -> None:
    assert _is_alive(0) is False
    assert _is_alive(-1) is False


def test_read_pid_returns_none_when_missing(tmp_path) -> None:
    assert _read_pid(tmp_path / "no_such_pid") is None


def test_read_pid_returns_none_when_malformed(tmp_path) -> None:
    p = tmp_path / "pid"
    p.write_text("not-a-number")
    assert _read_pid(p) is None


def test_read_pid_returns_int(tmp_path) -> None:
    p = tmp_path / "pid"
    p.write_text("12345")
    assert _read_pid(p) == 12345


# --- _check_duplicate_pid ---------------------------------------------------


def test_check_duplicate_pid_passes_when_no_pid_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    # No pid file exists — should return cleanly.
    _check_duplicate_pid("Aria", force=False)


def test_check_duplicate_pid_refuses_when_live(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    agent_dir = tmp_path / "Aria"
    agent_dir.mkdir()
    pid_path = agent_dir / "pid"
    # Use the current process's PID — guaranteed alive.
    pid_path.write_text(str(os.getpid()))

    with pytest.raises(SystemExit) as exc_info:
        _check_duplicate_pid("Aria", force=False)
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "already running" in err
    assert str(os.getpid()) in err
    # pid file must NOT be removed on refusal.
    assert pid_path.exists()


def test_check_duplicate_pid_force_proceeds_with_warning(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    agent_dir = tmp_path / "Aria"
    agent_dir.mkdir()
    pid_path = agent_dir / "pid"
    pid_path.write_text(str(os.getpid()))

    # Should not raise.
    _check_duplicate_pid("Aria", force=True)
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "--force" in err
    # pid file still present — caller will overwrite it.
    assert pid_path.exists()


def test_check_duplicate_pid_cleans_stale_pid_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    agent_dir = tmp_path / "Aria"
    agent_dir.mkdir()
    pid_path = agent_dir / "pid"

    # Find a PID guaranteed not to exist.
    dead_pid = None
    for candidate in range(2**15, 2**22):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            dead_pid = candidate
            break
        except (PermissionError, OSError):
            continue
    assert dead_pid is not None
    pid_path.write_text(str(dead_pid))

    _check_duplicate_pid("Aria", force=False)
    assert not pid_path.exists(), "stale pid file should have been cleaned up"


def test_check_duplicate_pid_treats_malformed_file_as_absent(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    agent_dir = tmp_path / "Aria"
    agent_dir.mkdir()
    pid_path = agent_dir / "pid"
    pid_path.write_text("garbage")
    # Malformed pid -> treated as no live conflict; proceed without raising.
    _check_duplicate_pid("Aria", force=False)


# --- --config flag (#187) ------------------------------------------------


def _install_spawn(monkeypatch, tmp_path: Path) -> list[dict]:
    """Stub every collaborator of spawn.main so no mind, log or network runs."""
    minds: list[dict] = []

    class _FakeConsciousness:
        def __init__(self, name: str, config_path: str) -> None:
            minds.append({"name": name, "config_path": config_path})
            self.name = name

    async def _fake_run(mind, headless: bool) -> None:
        return None

    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    monkeypatch.setattr(spawn, "Consciousness", _FakeConsciousness)
    monkeypatch.setattr(spawn, "_run", _fake_run)
    monkeypatch.setattr(spawn, "configure_logging", lambda name, level: tmp_path / name / "run.log")
    monkeypatch.setattr(spawn, "start_usage_reporting", lambda log: object())
    monkeypatch.setattr(spawn, "report_startup", lambda *a, **k: None)
    monkeypatch.setattr(spawn, "load_dotenv", lambda: None)
    monkeypatch.setattr(spawn.atexit, "register", lambda fn: None)
    return minds


def _custom_config(tmp_path: Path) -> Path:
    cfg = yaml.safe_load((_REPO_ROOT / "config" / "default_consciousness.yaml").read_text(encoding="utf-8"))
    cfg["thought_loop"]["rpt_critique"] = True
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_spawn_config_flag_runs_instance_against_persisted_copy(monkeypatch, tmp_path) -> None:
    from click.testing import CliRunner

    minds = _install_spawn(monkeypatch, tmp_path)
    custom = _custom_config(tmp_path)

    result = CliRunner().invoke(spawn.main, ["--name", "Aria", "--headless", "--config", str(custom)])

    assert result.exit_code == 0, result.output
    persisted = tmp_path / "Aria" / "config.yaml"
    assert minds == [{"name": "Aria", "config_path": str(persisted)}]
    assert yaml.safe_load(persisted.read_text(encoding="utf-8"))["thought_loop"]["rpt_critique"] is True


def test_spawn_invalid_config_exits_before_building_the_instance(monkeypatch, tmp_path) -> None:
    from click.testing import CliRunner

    minds = _install_spawn(monkeypatch, tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm:\n  provider: ollama\n", encoding="utf-8")

    result = CliRunner().invoke(spawn.main, ["--name", "Aria", "--headless", "--config", str(bad)])

    assert result.exit_code == 1
    assert "Invalid config" in result.output
    assert minds == []
    assert not (tmp_path / "Aria" / "config.yaml").exists()


def test_spawn_bg_rejects_invalid_config_before_detaching(monkeypatch, tmp_path) -> None:
    from click.testing import CliRunner

    _install_spawn(monkeypatch, tmp_path)
    launched: list = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: launched.append(a))
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not a mapping\n", encoding="utf-8")

    result = CliRunner().invoke(spawn.main, ["--name", "Aria", "--bg", "--config", str(bad)])

    assert result.exit_code == 1
    assert "must be a YAML mapping" in result.output
    assert launched == []
