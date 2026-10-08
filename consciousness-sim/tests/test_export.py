"""Tests for scripts/export.py — the portable-archive backup command (#10).

Covers:
- the archive layout: one top-level directory named after the sanitized
  instance, containing every regular state file (nested ones included).
- the ``pid`` file being left out, so a restored copy never reads as running.
- the refusals that exit 1: unknown instance, live instance, an output path
  that already exists.
- a stale pid file not blocking the export.
- the default output name.

Liveness is decided by ``scripts.export._is_alive``, which these tests either
leave real (the test process's own pid is live) or replace with a stub, so no
signal other than the harmless signal-0 probe is ever sent.
"""

from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.export as export_mod  # noqa: E402
from scripts.export import main as export_main  # noqa: E402


def _make_instance(home: Path, name: str = "Aria") -> Path:
    agent_dir = home / name
    (agent_dir / "nested").mkdir(parents=True)
    (agent_dir / "state.json").write_text('{"thought_count": 7}', encoding="utf-8")
    (agent_dir / "journal.jsonl").write_text('{"type": "thought"}\n', encoding="utf-8")
    (agent_dir / "memory.db").write_bytes(b"sqlite-sentinel")
    (agent_dir / "nested" / "extra.txt").write_text("nested-sentinel", encoding="utf-8")
    return agent_dir


def _members(archive: Path) -> dict[str, bytes]:
    with tarfile.open(archive, "r:gz") as tar:
        out: dict[str, bytes] = {}
        for member in tar.getmembers():
            f = tar.extractfile(member)
            assert f is not None, f"{member.name} is not a regular file"
            out[member.name] = f.read()
        return out


def test_export_archives_every_state_file_under_instance_name(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    _make_instance(home)
    archive = tmp_path / "aria.tar.gz"

    result = CliRunner().invoke(export_main, ["--name", "Aria", "--output", str(archive)])

    assert result.exit_code == 0, result.output
    assert "4 files" in result.output
    assert _members(archive) == {
        "Aria/journal.jsonl": b'{"type": "thought"}\n',
        "Aria/memory.db": b"sqlite-sentinel",
        "Aria/nested/extra.txt": b"nested-sentinel",
        "Aria/state.json": b'{"thought_count": 7}',
    }


def test_export_resolves_through_sanitized_name(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    _make_instance(home, "Aria_1")
    archive = tmp_path / "out.tar.gz"

    result = CliRunner().invoke(export_main, ["--name", "Aria 1", "-o", str(archive)])

    assert result.exit_code == 0, result.output
    assert all(name.startswith("Aria_1/") for name in _members(archive))


def test_export_omits_stale_pid_file(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    agent_dir = _make_instance(home)
    (agent_dir / "pid").write_text("4242", encoding="utf-8")
    monkeypatch.setattr(export_mod, "_is_alive", lambda pid: False)
    archive = tmp_path / "out.tar.gz"

    result = CliRunner().invoke(export_main, ["--name", "Aria", "-o", str(archive)])

    assert result.exit_code == 0, result.output
    assert "Aria/pid" not in _members(archive)
    assert (agent_dir / "pid").exists(), "export must not touch the instance directory"


def test_export_refuses_live_instance(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    agent_dir = _make_instance(home)
    (agent_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
    archive = tmp_path / "out.tar.gz"

    result = CliRunner().invoke(export_main, ["--name", "Aria", "-o", str(archive)])

    assert result.exit_code == 1
    assert "is running" in result.output
    assert "scripts/stop.py --name Aria" in result.output
    assert not archive.exists()


def test_export_exits_1_for_unknown_instance(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "home"))
    archive = tmp_path / "out.tar.gz"

    result = CliRunner().invoke(export_main, ["--name", "Ghost", "-o", str(archive)])

    assert result.exit_code == 1
    assert "No persisted instance named 'Ghost'" in result.output
    assert not archive.exists()


def test_export_refuses_to_overwrite_existing_output(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    _make_instance(home)
    archive = tmp_path / "out.tar.gz"
    archive.write_bytes(b"precious")

    result = CliRunner().invoke(export_main, ["--name", "Aria", "-o", str(archive)])

    assert result.exit_code == 1
    assert "Refusing to overwrite" in result.output
    assert archive.read_bytes() == b"precious"


def test_export_default_output_is_timestamped_in_cwd(monkeypatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(home))
    _make_instance(home)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    result = CliRunner().invoke(export_main, ["--name", "Aria"])

    assert result.exit_code == 0, result.output
    produced = list(workdir.glob("Aria-*Z.tar.gz"))
    assert len(produced) == 1
    assert "Aria/state.json" in _members(produced[0])
