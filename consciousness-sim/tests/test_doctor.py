"""Tests for scripts/doctor.py.

Covers:
- collect_instances() classification (alive / stopped / orphan) against a
  synthetic CONSCIOUSNESS_HOME (#116 acceptance criterion).
- health label derivation from state.json's health block (#117).
- duplicate live-name detection.
- --prune's stale-pid-file removal.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make scripts/ importable the same way spawn.py's tests do.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.doctor import (  # noqa: E402
    _flag_duplicate_live_names,
    _health_label,
    _prune,
    collect_instances,
)


def _dead_pid() -> int:
    """Find a PID guaranteed not to exist, mirroring tests/test_spawn.py."""
    for candidate in range(2**15, 2**22):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except (PermissionError, OSError):
            continue
    raise AssertionError("could not locate a dead PID for this test")


def _write_state(agent_dir: Path, *, name: str = "Aria", thought_count: int = 5, health: dict | None = None) -> None:
    state = {
        "identity": {"name": name},
        "thought_count": thought_count,
        "short_term": [],
    }
    if health is not None:
        state["health"] = health
    (agent_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _write_journal(agent_dir: Path, timestamp: str) -> None:
    line = json.dumps({"timestamp": timestamp, "type": "thought", "content": "hello"})
    (agent_dir / "journal.jsonl").write_text(line + "\n", encoding="utf-8")


# --- collect_instances: alive / stopped / orphan classification -----------


def test_collect_instances_classifies_alive_stopped_orphan(tmp_path) -> None:
    root = tmp_path

    alive_dir = root / "Alive"
    alive_dir.mkdir()
    (alive_dir / "pid").write_text(str(os.getpid()))
    _write_state(alive_dir, name="Alive", thought_count=10, health={"status": "ok", "consecutive_failures": 0})
    _write_journal(alive_dir, "2026-07-27T00:00:00+00:00")

    stopped_dir = root / "Stopped"
    stopped_dir.mkdir()
    _write_state(stopped_dir, name="Stopped", thought_count=3)
    _write_journal(stopped_dir, "2026-07-20T00:00:00+00:00")

    orphan_dir = root / "Orphan"
    orphan_dir.mkdir()
    (orphan_dir / "pid").write_text(str(_dead_pid()))
    _write_state(orphan_dir, name="Orphan", thought_count=1)

    instances = {i.name: i for i in collect_instances(root)}

    assert instances["Alive"].status == "alive"
    assert instances["Alive"].pid == os.getpid()
    assert instances["Alive"].thought_count == 10
    assert instances["Alive"].last_cycle == "2026-07-27T00:00:00+00:00"

    assert instances["Stopped"].status == "stopped"
    assert instances["Stopped"].pid is None
    assert instances["Stopped"].thought_count == 3

    assert instances["Orphan"].status == "orphan"
    assert instances["Orphan"].note == "pid file stale"


def test_collect_instances_empty_dir_with_no_state_is_orphan(tmp_path) -> None:
    d = tmp_path / "NeverInitialized"
    d.mkdir()

    instances = collect_instances(tmp_path)
    assert len(instances) == 1
    assert instances[0].status == "orphan"
    assert instances[0].note == "no state.json"


def test_collect_instances_returns_empty_list_for_missing_root(tmp_path) -> None:
    missing = tmp_path / "does_not_exist"
    assert collect_instances(missing) == []


def test_collect_instances_skips_dotfiles(tmp_path) -> None:
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    _write_state(hidden, name="Hidden")

    assert collect_instances(tmp_path) == []


def test_collect_instances_ignores_non_directory_entries(tmp_path) -> None:
    (tmp_path / "stray_file.txt").write_text("not a directory")
    assert collect_instances(tmp_path) == []


# --- health label -----------------------------------------------------------


def test_health_label_ok() -> None:
    state = {"health": {"status": "ok", "consecutive_failures": 0}}
    assert _health_label(state) == "ok"


def test_health_label_degraded_includes_failure_count() -> None:
    state = {"health": {"status": "degraded", "consecutive_failures": 4}}
    assert _health_label(state) == "degraded (4 failures)"


def test_health_label_missing_state_is_dash() -> None:
    assert _health_label(None) == "—"


def test_health_label_missing_health_block_is_dash() -> None:
    assert _health_label({"thought_count": 1}) == "—"


def test_collect_instances_health_only_shown_for_alive(tmp_path) -> None:
    """A stopped instance's stale health block (from its last run) should not
    be surfaced as current — the process isn't running to have a live status."""
    stopped_dir = tmp_path / "Stopped"
    stopped_dir.mkdir()
    _write_state(stopped_dir, name="Stopped", health={"status": "failing", "consecutive_failures": 19})

    instances = collect_instances(tmp_path)
    assert instances[0].status == "stopped"
    assert instances[0].health == "—"


# --- duplicate live-name detection ------------------------------------------


def test_flag_duplicate_live_names_warns_on_shared_display_name(tmp_path) -> None:
    root = tmp_path
    for dir_name in ("Aria", "Aria_2"):
        d = root / dir_name
        d.mkdir()
        (d / "pid").write_text(str(os.getpid()))
        _write_state(d, name="Aria")

    instances = collect_instances(root)
    notes = {i.name: i.note for i in instances}
    assert "duplicate live name 'Aria'" in notes["Aria"]
    assert "duplicate live name 'Aria'" in notes["Aria_2"]


def test_flag_duplicate_live_names_no_warning_when_names_differ() -> None:
    from scripts.doctor import InstanceStatus

    instances = [
        InstanceStatus("Aria", "Aria", 1, "alive", "0h00m", 1, "—", "—"),
        InstanceStatus("Wren", "Wren", 2, "alive", "0h00m", 1, "—", "—"),
    ]
    _flag_duplicate_live_names(instances)
    assert instances[0].note == ""
    assert instances[1].note == ""


def test_flag_duplicate_live_names_ignores_non_alive_instances() -> None:
    from scripts.doctor import InstanceStatus

    instances = [
        InstanceStatus("Aria", "Aria", 1, "alive", "0h00m", 1, "—", "—"),
        InstanceStatus("Aria_2", "Aria", None, "stopped", "—", 1, "—", "—"),
    ]
    _flag_duplicate_live_names(instances)
    assert instances[0].note == ""
    assert instances[1].note == ""


# --- --prune ------------------------------------------------------------


def test_prune_removes_stale_pid_files_without_prompt_when_assume_yes(tmp_path) -> None:
    root = tmp_path
    orphan_dir = root / "Orphan"
    orphan_dir.mkdir()
    (orphan_dir / "pid").write_text(str(_dead_pid()))
    _write_state(orphan_dir, name="Orphan")

    instances = collect_instances(root)
    pruned = _prune(instances, root, assume_yes=True)

    assert pruned == ["Orphan"]
    assert not (orphan_dir / "pid").exists()


def test_prune_skips_alive_instances(tmp_path) -> None:
    root = tmp_path
    alive_dir = root / "Alive"
    alive_dir.mkdir()
    (alive_dir / "pid").write_text(str(os.getpid()))
    _write_state(alive_dir, name="Alive")

    instances = collect_instances(root)
    pruned = _prune(instances, root, assume_yes=True)

    assert pruned == []
    assert (alive_dir / "pid").exists()


def test_prune_skips_stopped_instances_with_no_pid_file(tmp_path) -> None:
    root = tmp_path
    stopped_dir = root / "Stopped"
    stopped_dir.mkdir()
    _write_state(stopped_dir, name="Stopped")

    instances = collect_instances(root)
    pruned = _prune(instances, root, assume_yes=True)

    assert pruned == []


def test_prune_respects_declined_confirmation(tmp_path, monkeypatch) -> None:
    import click

    root = tmp_path
    orphan_dir = root / "Orphan"
    orphan_dir.mkdir()
    (orphan_dir / "pid").write_text(str(_dead_pid()))
    _write_state(orphan_dir, name="Orphan")

    monkeypatch.setattr(click, "confirm", lambda *a, **k: False)

    instances = collect_instances(root)
    pruned = _prune(instances, root, assume_yes=False)

    assert pruned == []
    assert (orphan_dir / "pid").exists()
