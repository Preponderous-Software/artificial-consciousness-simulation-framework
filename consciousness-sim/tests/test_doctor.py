"""Tests for scripts/doctor.py.

Covers:
- collect_instances() classification (alive / stopped / orphan) against a
  synthetic CONSCIOUSNESS_HOME (#116 acceptance criterion).
- health label derivation from state.json's health block (#117).
- duplicate live-name detection.
- --prune's stale-pid-file removal.
- the per-file readers (_is_alive / _read_pid / _last_journal_timestamp),
  _format_uptime, and _display_name's fallbacks.
- the click entry point's --json, table, and --prune output.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

# Make scripts/ importable the same way spawn.py's tests do.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.doctor import (  # noqa: E402
    InstanceStatus,
    _display_name,
    _flag_duplicate_live_names,
    _format_uptime,
    _health_label,
    _is_alive,
    _last_journal_timestamp,
    _prune,
    _read_pid,
    collect_instances,
    main as doctor_main,
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


# --- journal scan: non-object lines are corruption (#175) ------------------


def test_collect_instances_survives_non_object_journal_lines(tmp_path) -> None:
    """A line that parses but is not an object has no .get(). Doctor's whole
    purpose is reporting on unhealthy instances, so it must skip such a line
    rather than crash on the instance most likely to have one."""
    root = tmp_path
    agent_dir = root / "Corrupt"
    agent_dir.mkdir()
    _write_state(agent_dir, name="Corrupt", thought_count=7)
    good = json.dumps({"timestamp": "2026-07-27T00:00:00+00:00", "type": "thought", "content": "hello"})
    (agent_dir / "journal.jsonl").write_text(
        f"{good}\n123\n[1, 2]\nnull\n\"a bare string\"\nNOT JSON\n",
        encoding="utf-8",
    )

    instances = {i.name: i for i in collect_instances(root)}

    assert instances["Corrupt"].thought_count == 7
    assert instances["Corrupt"].last_cycle == "2026-07-27T00:00:00+00:00"


# --- state.json scan: non-object documents are corruption (#176) -----------


@pytest.mark.parametrize("document", ["[1, 2]", "123", '"a bare string"', "null", "true"])
def test_collect_instances_survives_non_object_state_json(tmp_path, document) -> None:
    """dict() raises on a top-level array/scalar/null, and neither TypeError nor
    ValueError is in _read_state's guarded tuple — so one such file aborted the
    whole survey instead of reporting the instance with an unreadable snapshot."""
    root = tmp_path
    healthy_dir = root / "Healthy"
    healthy_dir.mkdir()
    _write_state(healthy_dir, name="Healthy", thought_count=4)

    bad_dir = root / "BadState"
    bad_dir.mkdir()
    (bad_dir / "state.json").write_text(document, encoding="utf-8")

    instances = {i.name: i for i in collect_instances(root)}

    assert instances["Healthy"].thought_count == 4
    # The unreadable instance is still reported, with the same placeholders a
    # missing state.json produces.
    assert instances["BadState"].thought_count == "—"
    assert instances["BadState"].display_name == "BadState"
    assert instances["BadState"].health == "—"


# --- per-file readers ------------------------------------------------------


@pytest.mark.parametrize("pid", [0, -1])
def test_is_alive_rejects_non_positive_pid(pid) -> None:
    # os.kill(0, 0) would signal the whole process group and succeed, so the
    # guard must short-circuit before the syscall.
    assert _is_alive(pid) is False


def test_is_alive_treats_permission_error_as_alive(monkeypatch) -> None:
    def _deny(pid: int, sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(os, "kill", _deny)
    assert _is_alive(12345) is True


def test_is_alive_treats_other_os_error_as_dead(monkeypatch) -> None:
    def _fail(pid: int, sig: int) -> None:
        raise OSError("unexpected")

    monkeypatch.setattr(os, "kill", _fail)
    assert _is_alive(12345) is False


def test_read_pid_strips_whitespace(tmp_path) -> None:
    pid_path = tmp_path / "pid"
    pid_path.write_text("  4242\n", encoding="utf-8")
    assert _read_pid(pid_path) == 4242


def test_read_pid_missing_or_malformed_is_none(tmp_path) -> None:
    assert _read_pid(tmp_path / "absent") is None
    malformed = tmp_path / "pid"
    malformed.write_text("not-a-pid", encoding="utf-8")
    assert _read_pid(malformed) is None


def test_last_journal_timestamp_missing_or_empty_is_none(tmp_path) -> None:
    assert _last_journal_timestamp(tmp_path / "absent.jsonl") is None
    empty = tmp_path / "journal.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    assert _last_journal_timestamp(empty) is None


def test_last_journal_timestamp_keeps_last_string_timestamp(tmp_path) -> None:
    """Rows after the last string timestamp — a numeric timestamp, a row with
    none at all, a blank line — leave the earlier value in place."""
    journal = tmp_path / "journal.jsonl"
    rows = [
        json.dumps({"timestamp": "2026-07-01T00:00:00+00:00"}),
        json.dumps({"timestamp": "2026-07-02T00:00:00+00:00"}),
        "",
        json.dumps({"timestamp": 1751414400}),
        json.dumps({"type": "thought"}),
    ]
    journal.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert _last_journal_timestamp(journal) == "2026-07-02T00:00:00+00:00"


# --- uptime --------------------------------------------------------------


def test_format_uptime_renders_hours_and_zero_padded_minutes(tmp_path) -> None:
    pid_path = tmp_path / "pid"
    pid_path.write_text("1", encoding="utf-8")
    started = time.time() - (2 * 3600 + 5 * 60 + 30)
    os.utime(pid_path, (started, started))
    assert _format_uptime(pid_path) == "2h05m"


def test_format_uptime_missing_pid_file_is_dash(tmp_path) -> None:
    assert _format_uptime(tmp_path / "absent") == "—"


def test_format_uptime_future_mtime_is_dash(tmp_path) -> None:
    """A clock that moved backwards since spawn gives a negative delta, which
    is rendered as unknown rather than as a negative duration."""
    pid_path = tmp_path / "pid"
    pid_path.write_text("1", encoding="utf-8")
    future = time.time() + 3600
    os.utime(pid_path, (future, future))
    assert _format_uptime(pid_path) == "—"


# --- display name ----------------------------------------------------------


def test_display_name_prefers_identity_name() -> None:
    assert _display_name({"identity": {"name": "Aria"}}, "aria_dir") == "Aria"


@pytest.mark.parametrize(
    "state",
    [
        None,
        {},
        {"identity": "Aria"},
        {"identity": {}},
        {"identity": {"name": ""}},
        {"identity": {"name": 7}},
    ],
)
def test_display_name_falls_back_to_dir_name(state) -> None:
    assert _display_name(state, "aria_dir") == "aria_dir"


# --- --prune failure path ----------------------------------------------------


def test_prune_warns_and_skips_when_pid_file_cannot_be_removed(tmp_path, capsys) -> None:
    # The instance claims a stale pid file, but the file is already gone, so
    # unlink() raises FileNotFoundError (an OSError).
    (tmp_path / "Ghost").mkdir()
    instances = [InstanceStatus("Ghost", "Ghost", 99, "orphan", "—", "—", "—", "—", note="pid file stale")]

    pruned = _prune(instances, tmp_path, assume_yes=True)

    assert pruned == []
    assert "WARNING: failed to remove pid file for 'Ghost'" in capsys.readouterr().err


def test_prune_ignores_orphans_without_a_stale_pid_file(tmp_path) -> None:
    """An orphan that never wrote state.json has no pid file to remove."""
    (tmp_path / "NeverInitialized").mkdir()
    instances = collect_instances(tmp_path)
    assert instances[0].note == "no state.json"
    assert _prune(instances, tmp_path, assume_yes=True) == []


# --- click entry point -------------------------------------------------------


def test_main_json_emits_one_object_per_instance(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    stopped_dir = tmp_path / "Stopped"
    stopped_dir.mkdir()
    _write_state(stopped_dir, name="Stopped", thought_count=3)
    _write_journal(stopped_dir, "2026-07-20T00:00:00+00:00")

    result = CliRunner().invoke(doctor_main, ["--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        {
            "name": "Stopped",
            "display_name": "Stopped",
            "pid": None,
            "status": "stopped",
            "uptime": "—",
            "thought_count": 3,
            "last_cycle": "2026-07-20T00:00:00+00:00",
            "health": "—",
            "note": "",
        }
    ]


def test_main_json_with_no_instances_is_empty_list(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "missing"))
    result = CliRunner().invoke(doctor_main, ["--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


def test_main_table_reports_no_instances(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "400")
    result = CliRunner().invoke(doctor_main, [])
    assert result.exit_code == 0, result.output
    assert "No consciousness instances found under" in result.output


def test_main_table_folds_note_into_status_cell(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "400")
    orphan_dir = tmp_path / "Orphan"
    orphan_dir.mkdir()
    dead = _dead_pid()
    (orphan_dir / "pid").write_text(str(dead))
    _write_state(orphan_dir, name="Orphan", thought_count=8)

    result = CliRunner().invoke(doctor_main, [])

    assert result.exit_code == 0, result.output
    assert "orphan (pid file stale)" in result.output
    assert str(dead) in result.output
    for header in ("NAME", "PID", "STATUS", "UPTIME", "THOUGHTS", "LAST CYCLE", "HEALTH"):
        assert header in result.output


def test_main_prune_yes_reports_removal_and_recollects(tmp_path, monkeypatch) -> None:
    """After a successful prune the survey is re-run, so the instance whose
    stale pid file was removed is reported as stopped, not orphan."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    orphan_dir = tmp_path / "Orphan"
    orphan_dir.mkdir()
    (orphan_dir / "pid").write_text(str(_dead_pid()))
    _write_state(orphan_dir, name="Orphan")

    result = CliRunner().invoke(doctor_main, ["--prune", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    first_line, rest = result.output.split("\n", 1)
    assert first_line == "Removed stale pid file: Orphan"
    rows = json.loads(rest)
    assert [(r["name"], r["status"], r["pid"]) for r in rows] == [("Orphan", "stopped", None)]
    assert not (orphan_dir / "pid").exists()
