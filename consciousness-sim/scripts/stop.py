"""Stop a background consciousness instance launched with spawn.py --bg.

No direct theory mapping — infrastructure script.
Reads the PID file written by spawn.py and sends SIGTERM, then waits
briefly before escalating to SIGKILL if the process is still alive.
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


@click.command()
@click.option("--name", required=True, type=str, help="Consciousness name (must match the spawned instance)")
@click.option("--force", is_flag=True, default=False, help="Send SIGKILL immediately instead of SIGTERM")
def main(name: str, force: bool) -> None:
    load_dotenv()
    pid_path = consciousness_dir(name) / "pid"

    if not pid_path.exists():
        click.echo(f"No PID file found for '{name}' ({pid_path}). Is it running in the background?", err=True)
        sys.exit(1)

    pid = int(pid_path.read_text().strip())

    try:
        os.kill(pid, 0)  # check process exists
    except ProcessLookupError:
        click.echo(f"Process {pid} is not running. Removing stale PID file.")
        pid_path.unlink(missing_ok=True)
        sys.exit(0)

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
