"""Stop a background consciousness instance launched with spawn.py --bg.

No direct theory mapping — infrastructure script.
Reads the PID file written by spawn.py and sends SIGTERM, then waits
briefly before escalating to SIGKILL if the process is still alive.

Malformed pid files and pids that cannot be signalled from this account
exit 1 with a message naming the file, instead of escaping as tracebacks
(#179). The pid-file parsing and liveness semantics mirror spawn.py's
``_read_pid`` / ``_is_alive``.
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

import os
import signal
import time

import click
from dotenv import load_dotenv

from persistence.paths import consciousness_dir


def _read_pid(pid_path: Path) -> int | None:
    """Return the PID stored at pid_path, or None if it is unreadable,
    non-numeric, or not a positive integer.

    Mirrors spawn.py's ``_read_pid`` plus its ``_is_alive`` ``pid <= 0``
    guard: ``os.kill(0, sig)`` signals the caller's whole process group and
    negative pids target groups too, so neither may ever reach ``os.kill``.
    """
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


@click.command()
@click.option("--name", required=True, type=str, help="Consciousness name (must match the spawned instance)")
@click.option("--force", is_flag=True, default=False, help="Send SIGKILL immediately instead of SIGTERM")
def main(name: str, force: bool) -> None:
    load_dotenv()
    pid_path = consciousness_dir(name) / "pid"

    if not pid_path.exists():
        click.echo(f"No PID file found for '{name}' ({pid_path}). Is it running in the background?", err=True)
        sys.exit(1)

    pid = _read_pid(pid_path)
    if pid is None:
        # Leave the file in place: unlike a stale pid it says nothing about
        # whether the instance is running, so removal is the user's call.
        click.echo(
            f"PID file for '{name}' is malformed ({pid_path}): expected a positive integer. "
            "Stop the instance by hand if it is running, then remove the file.",
            err=True,
        )
        sys.exit(1)

    try:
        os.kill(pid, 0)  # check process exists
    except ProcessLookupError:
        click.echo(f"Process {pid} is not running. Removing stale PID file.")
        pid_path.unlink(missing_ok=True)
        sys.exit(0)
    except PermissionError:
        # Same reading as spawn.py's _is_alive: the process exists but is owned
        # by another account, so no signal from here can reach it.
        click.echo(
            f"Process {pid} for '{name}' is running but cannot be signalled from this account "
            f"(owned by another user?). PID file left in place: {pid_path}",
            err=True,
        )
        sys.exit(1)

    sig = signal.SIGKILL if force else signal.SIGTERM
    os.kill(pid, sig)
    click.echo(f"Sent {'SIGKILL' if force else 'SIGTERM'} to '{name}' (PID {pid})")

    if not force:
        for _ in range(20):
            time.sleep(0.25)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                click.echo("Process stopped.")
                pid_path.unlink(missing_ok=True)
                return

        click.echo("Process still alive after 5 s — escalating to SIGKILL.")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
