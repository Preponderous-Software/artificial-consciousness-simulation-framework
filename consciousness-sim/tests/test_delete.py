"""Tests for scripts/delete.py — the delete / reset command (#10).

Covers:
- delete removing the whole instance directory and nothing beside it.
- --reset removing every entry (nested directories, a stale pid file) except
  config.yaml, and working when no config.yaml exists.
- symlinks never being followed: a symlinked entry is unlinked, and a
  symlinked instance directory is treated as absent.
- the confirmation prompt: declining removes nothing, accepting proceeds.
- the refusals that exit 1: unknown instance, live instance.

Liveness is decided by ``scripts.delete._is_alive``, which these tests either
leave real (the test process's own pid is live) or replace with a stub, so no
signal other than the harmless signal-0 probe is ever sent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.delete as delete_mod  # noqa: E402
from scripts.delete import main as delete_main  # noqa: E402


def _make_instance(home: Path, name: str = "Aria", with_config: bool = True) -> Path:
    agent_dir = home / name
    (agent_dir / "nested").mkdir(parents=True)
    (agent_dir / "state.json").write_text('{"thought_count": 7}', encoding="utf-8")
    (agent_dir / "memory.db").write_bytes(b"sqlite-sentinel")
    (agent_dir / "nested" / "extra.txt").write_text("nested-sentinel", encoding="utf-8")
    if with_config:
        (agent_dir / "config.yaml").write_text("llm:\n  model: sentinel\n", encoding="utf-8")
    return agent_dir


def test_delete_removes_instance_directory_only(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    agent_dir = _make_instance(home)
    sibling = _make_instance(home, "Echo")

    result = CliRunner().invoke(delete_main, ["--name", "Aria", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Deleted 'Aria'" in result.output
    assert not agent_dir.exists()
    assert (sibling / "state.json").exists()


def test_reset_keeps_only_config(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    agent_dir = _make_instance(home)
    (agent_dir / "pid").write_text("4242", encoding="utf-8")
    monkeypatch.setattr(delete_mod, "_is_alive", lambda pid: False)

    result = CliRunner().invoke(delete_main, ["--name", "Aria", "--reset", "--yes"])

    assert result.exit_code == 0, result.output
    assert "removed 4 entries" in result.output
    assert sorted(p.name for p in agent_dir.iterdir()) == ["config.yaml"]
    assert (agent_dir / "config.yaml").read_text(encoding="utf-8") == "llm:\n  model: sentinel\n"


def test_reset_without_config_leaves_empty_directory(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    agent_dir = _make_instance(home, with_config=False)

    result = CliRunner().invoke(delete_main, ["--name", "Aria", "--reset", "--yes"])

    assert result.exit_code == 0, result.output
    assert agent_dir.is_dir()
    assert list(agent_dir.iterdir()) == []


def test_reset_unlinks_symlinks_without_following_them(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    agent_dir = _make_instance(home)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("outside-sentinel", encoding="utf-8")
    (agent_dir / "linked").symlink_to(outside, target_is_directory=True)

    result = CliRunner().invoke(delete_main, ["--name", "Aria", "--reset", "--yes"])

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in agent_dir.iterdir()) == ["config.yaml"]
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "outside-sentinel"


def test_symlinked_instance_directory_is_treated_as_absent(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    real_dir = _make_instance(tmp_path / "elsewhere")
    home.mkdir()
    (home / "Aria").symlink_to(real_dir, target_is_directory=True)

    result = CliRunner().invoke(delete_main, ["--name", "Aria", "--yes"])

    assert result.exit_code == 1
    assert "No persisted instance named 'Aria'" in result.output
    assert (real_dir / "state.json").exists()


def test_delete_resolves_through_sanitized_name(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    agent_dir = _make_instance(home, "Aria_1")

    result = CliRunner().invoke(delete_main, ["--name", "Aria 1", "--yes"])

    assert result.exit_code == 0, result.output
    assert not agent_dir.exists()


def test_declined_confirmation_removes_nothing(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    agent_dir = _make_instance(home)

    result = CliRunner().invoke(delete_main, ["--name", "Aria"], input="n\n")

    assert result.exit_code == 1
    assert "nothing was removed" in result.output
    assert (agent_dir / "state.json").exists()
    assert (agent_dir / "nested" / "extra.txt").exists()


def test_accepted_confirmation_proceeds(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    agent_dir = _make_instance(home)

    result = CliRunner().invoke(delete_main, ["--name", "Aria", "--reset"], input="y\n")

    assert result.exit_code == 0, result.output
    assert "keeping only config.yaml" in result.output
    assert sorted(p.name for p in agent_dir.iterdir()) == ["config.yaml"]


def test_delete_refuses_live_instance(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    agent_dir = _make_instance(home)
    (agent_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")

    for args in (["--name", "Aria", "--yes"], ["--name", "Aria", "--reset", "--yes"]):
        result = CliRunner().invoke(delete_main, args)

        assert result.exit_code == 1
        assert "is running" in result.output
        assert "scripts/stop.py --name Aria" in result.output
        assert (agent_dir / "state.json").exists()


def test_delete_exits_1_for_unknown_instance(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))

    result = CliRunner().invoke(delete_main, ["--name", "Ghost", "--yes"])

    assert result.exit_code == 1
    assert "No persisted instance named 'Ghost'" in result.output
