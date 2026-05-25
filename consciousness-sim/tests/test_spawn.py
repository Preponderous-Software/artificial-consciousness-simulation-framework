"""Tests for scripts/spawn.py helpers.

Covers:
- _build_config_path's tempfile-leak regression (#105) — overrides must land
  in the consciousness dir, not a /tmp file that nobody ever cleans up.
- _check_duplicate_pid's refusal/cleanup behaviour (#115) — duplicate spawns
  must abort, stale pid files must be cleaned up.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pytest
import yaml

# Make scripts/ importable the same way spawn.py does at runtime.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.spawn import (  # noqa: E402
    _build_config_path,
    _check_duplicate_pid,
    _is_alive,
    _read_pid,
)


def test_build_config_path_returns_default_when_no_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    result = _build_config_path("Aria", provider=None, model=None)
    # Default config — same path returned without overrides
    assert result.name == "default_consciousness.yaml"
    assert result.exists()


def test_build_config_path_with_overrides_writes_to_consciousness_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    result = _build_config_path("Aria", provider="mock", model="mock")

    expected = tmp_path / "Aria" / "_spawn_config.yaml"
    assert result == expected, f"override config should land in {expected}, got {result}"
    assert result.exists()

    cfg = yaml.safe_load(result.read_text(encoding="utf-8"))
    assert cfg["llm"]["provider"] == "mock"
    assert cfg["llm"]["model"] == "mock"


def test_build_config_path_does_not_leak_tmp_files(monkeypatch, tmp_path) -> None:
    """Regression for #105 — the prior implementation left a /tmp file per spawn."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    before = set(glob.glob("/tmp/consciousness_override_*.yaml"))
    for _ in range(3):
        _build_config_path("Aria", provider="mock", model="mock")
    after = set(glob.glob("/tmp/consciousness_override_*.yaml"))
    new_files = after - before
    assert not new_files, f"_build_config_path leaked tmp files: {new_files}"


def test_build_config_path_overwrites_on_repeat_spawn(monkeypatch, tmp_path) -> None:
    """Two spawns of the same name must reuse the same file, not accumulate."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    p1 = _build_config_path("Aria", provider="mock", model="mock")
    p2 = _build_config_path("Aria", provider="anthropic", model="claude-opus-4-7")
    assert p1 == p2
    cfg = yaml.safe_load(p2.read_text(encoding="utf-8"))
    assert cfg["llm"]["provider"] == "anthropic"
    assert cfg["llm"]["model"] == "claude-opus-4-7"


def test_build_config_path_provider_only_keeps_default_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    result = _build_config_path("Aria", provider="anthropic", model=None)
    cfg = yaml.safe_load(result.read_text(encoding="utf-8"))
    assert cfg["llm"]["provider"] == "anthropic"
    # Default model from default_consciousness.yaml preserved (not None)
    assert cfg["llm"]["model"]


def test_build_config_path_model_only_keeps_default_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    result = _build_config_path("Aria", provider=None, model="llama3.1:8b")
    cfg = yaml.safe_load(result.read_text(encoding="utf-8"))
    assert cfg["llm"]["model"] == "llama3.1:8b"
    assert cfg["llm"]["provider"]


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
