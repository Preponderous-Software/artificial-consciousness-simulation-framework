"""Enumerate consciousness instances under CONSCIOUSNESS_HOME and report status.

No direct theory mapping — read-only diagnostic script (companion to
scripts/inspect.py, which inspects a single named instance's journal/state
in depth; this script instead surveys every instance at once).

Classifies each subdirectory of CONSCIOUSNESS_HOME as alive / stopped / orphan
by cross-referencing its pid file against the live process table, and surfaces
thought count, last-cycle timestamp, and health status (#117) straight from
state.json without starting the run loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from persistence.paths import consciousness_root


def _is_alive(pid: int) -> bool:
    """Return True iff a process with this PID is currently alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user — still a conflict.
        return True
    except OSError:
        return False
    return True


def _read_pid(pid_path: Path) -> int | None:
    """Return the PID stored at pid_path or None if missing/malformed."""
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_state(state_path: Path) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    try:
        return dict(json.loads(state_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None


def _last_journal_timestamp(journal_path: Path) -> str | None:
    """Return the timestamp of the last valid line in the journal, or None."""
    if not journal_path.exists():
        return None
    last: str | None = None
    try:
        with journal_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = row.get("timestamp")
                if isinstance(ts, str):
                    last = ts
    except OSError:
        return None
    return last


def _format_uptime(pid_path: Path) -> str:
    """Approximate uptime from the pid file's mtime.

    The pid file is written once at spawn and left untouched until the
    instance stops (scripts/spawn.py), so its mtime is a reasonable proxy
    for process start time without depending on a platform-specific
    process-table API (e.g. psutil, which isn't a project dependency).
    """
    try:
        started = pid_path.stat().st_mtime
    except OSError:
        return "—"
    delta = datetime.now().timestamp() - started
    if delta < 0:
        return "—"
    hours, rem = divmod(int(delta), 3600)
    minutes, _ = divmod(rem, 60)
    return f"{hours}h{minutes:02d}m"


def _display_name(state: dict[str, Any] | None, dir_name: str) -> str:
    """Return the identity's self-reported name, falling back to the
    sanitized on-disk directory name."""
    if state:
        identity = state.get("identity")
        if isinstance(identity, dict):
            name = identity.get("name")
            if isinstance(name, str) and name:
                return name
    return dir_name


def _health_label(state: dict[str, Any] | None) -> str:
    if not state:
        return "—"
    health = state.get("health")
    if not isinstance(health, dict):
        return "—"
    status = health.get("status", "ok")
    if status == "ok":
        return "ok"
    failures = health.get("consecutive_failures", 0)
    return f"{status} ({failures} failures)"


@dataclass
class InstanceStatus:
    """One row of `doctor.py`'s report for a single CONSCIOUSNESS_HOME subdirectory."""

    name: str
    display_name: str
    pid: int | None
    status: str  # alive | stopped | orphan
    uptime: str
    thought_count: int | str
    last_cycle: str
    health: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "pid": self.pid,
            "status": self.status,
            "uptime": self.uptime,
            "thought_count": self.thought_count,
            "last_cycle": self.last_cycle,
            "health": self.health,
            "note": self.note,
        }


def collect_instances(root: Path) -> list[InstanceStatus]:
    """Classify every subdirectory of root as alive / stopped / orphan.

    - alive: pid file present and the recorded PID is currently running.
    - orphan: pid file present but stale (process gone), or a directory
      with no pid file and no state.json (never completed a save).
    - stopped: no pid file, but a prior state.json snapshot exists —
      the instance shut down cleanly (scripts/stop.py removes the pid file).
    """
    instances: list[InstanceStatus] = []
    if not root.exists():
        return instances

    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue

        pid_path = d / "pid"
        state_path = d / "state.json"
        journal_path = d / "journal.jsonl"

        pid = _read_pid(pid_path)
        state = _read_state(state_path)
        last_cycle = _last_journal_timestamp(journal_path) or "—"

        if pid is not None and _is_alive(pid):
            status = "alive"
            uptime = _format_uptime(pid_path)
            note = ""
        elif pid is not None:
            status = "orphan"
            uptime = "—"
            note = "pid file stale"
        elif state is not None:
            status = "stopped"
            uptime = "—"
            note = ""
        else:
            status = "orphan"
            uptime = "—"
            note = "no state.json"

        thought_count = state.get("thought_count", "—") if state else "—"
        health = _health_label(state) if status == "alive" else "—"

        instances.append(
            InstanceStatus(
                name=d.name,
                display_name=_display_name(state, d.name),
                pid=pid,
                status=status,
                uptime=uptime,
                thought_count=thought_count,
                last_cycle=last_cycle,
                health=health,
                note=note,
            )
        )

    _flag_duplicate_live_names(instances)
    return instances


def _flag_duplicate_live_names(instances: list[InstanceStatus]) -> None:
    """Append a note to every alive instance whose self-reported display
    name is shared with another alive instance (e.g. two on-disk directories
    spawned with names that collide once sanitized differently, or a
    misconfigured re-spawn under a new directory)."""
    live_names = Counter(i.display_name for i in instances if i.status == "alive")
    duplicates = {name for name, count in live_names.items() if count > 1}
    if not duplicates:
        return
    for inst in instances:
        if inst.status == "alive" and inst.display_name in duplicates:
            warning = f"duplicate live name '{inst.display_name}'"
            inst.note = f"{inst.note}; {warning}" if inst.note else warning


def _prune(instances: list[InstanceStatus], root: Path, assume_yes: bool) -> list[str]:
    """Remove stale pid files for orphaned instances. Returns names pruned."""
    pruned: list[str] = []
    for inst in instances:
        if inst.status != "orphan" or "pid file stale" not in inst.note:
            continue
        pid_path = root / inst.name / "pid"
        if not assume_yes and not click.confirm(
            f"Remove stale pid file for '{inst.name}' (PID {inst.pid})?"
        ):
            continue
        try:
            pid_path.unlink()
            pruned.append(inst.name)
        except OSError:
            click.echo(f"WARNING: failed to remove pid file for '{inst.name}'", err=True)
    return pruned


def _render_table(instances: list[InstanceStatus], root: Path) -> None:
    console = Console()
    if not instances:
        console.print(f"No consciousness instances found under {root}")
        return

    table = Table(title=f"Consciousness instances ({root})")
    table.add_column("NAME")
    table.add_column("PID")
    table.add_column("STATUS")
    table.add_column("UPTIME")
    table.add_column("THOUGHTS")
    table.add_column("LAST CYCLE")
    table.add_column("HEALTH")

    for inst in instances:
        status_cell = f"{inst.status} ({inst.note})" if inst.note else inst.status
        table.add_row(
            inst.name,
            str(inst.pid) if inst.pid is not None else "—",
            status_cell,
            inst.uptime,
            str(inst.thought_count),
            inst.last_cycle,
            inst.health,
        )

    console.print(table)


@click.command()
@click.option(
    "--prune", "do_prune", is_flag=True,
    help="Remove stale pid files for orphaned instances.",
)
@click.option(
    "--yes", "assume_yes", is_flag=True,
    help="Skip confirmation prompts when pruning.",
)
@click.option(
    "--json", "as_json", is_flag=True,
    help="Emit machine-readable JSON instead of a table.",
)
def main(do_prune: bool, assume_yes: bool, as_json: bool) -> None:
    root = consciousness_root()
    instances = collect_instances(root)

    if do_prune:
        pruned = _prune(instances, root, assume_yes)
        for name in pruned:
            click.echo(f"Removed stale pid file: {name}")
        if pruned:
            instances = collect_instances(root)

    if as_json:
        click.echo(json.dumps([i.to_dict() for i in instances], indent=2))
        return

    _render_table(instances, root)


if __name__ == "__main__":
    main()
