"""Delete or reset a consciousness instance's persisted state.

No direct theory mapping — infrastructure script (the ``delete`` / ``reset``
commands requested in #10).

Without ``--reset`` the whole instance directory is removed. With
``--reset`` every entry in it is removed except the per-instance
``config.yaml`` (#187), so the next spawn under the same name starts with a
fresh identity, empty memories, and the same configuration.

Both are irreversible, so the command asks for confirmation unless ``--yes``
is passed, and refuses a live instance outright — deleting files out from
under a running process would leave it writing to unlinked inodes. Stop it
first with scripts/stop.py, and back it up with scripts/export.py if the
state may be wanted later.
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

import shutil

import click
from dotenv import load_dotenv

from persistence.paths import consciousness_dir
from scripts._config import INSTANCE_CONFIG_NAME
from scripts.doctor import _is_alive, _read_pid


def _remove_entry(path: Path) -> None:
    """Remove one directory entry without following symlinks."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _reset(instance_dir: Path) -> int:
    """Remove everything in instance_dir except its config; return the count removed."""
    removed = 0
    for entry in sorted(instance_dir.iterdir()):
        if entry.name == INSTANCE_CONFIG_NAME and entry.is_file() and not entry.is_symlink():
            continue
        _remove_entry(entry)
        removed += 1
    return removed


@click.command()
@click.option("--name", required=True, type=str, help="Consciousness name to delete")
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    help=f"Remove all state but keep {INSTANCE_CONFIG_NAME}, so the next spawn starts fresh",
)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip the confirmation prompt")
def main(name: str, reset: bool, yes: bool) -> None:
    load_dotenv()
    try:
        instance_dir = consciousness_dir(name)
    except ValueError as exc:
        click.echo(f"Invalid name '{name}': {exc}", err=True)
        sys.exit(1)

    if instance_dir.is_symlink() or not instance_dir.is_dir():
        click.echo(f"No persisted instance named '{name}' ({instance_dir}).", err=True)
        sys.exit(1)

    pid = _read_pid(instance_dir / "pid")
    if pid is not None and _is_alive(pid):
        click.echo(
            f"'{name}' is running (PID {pid}). "
            f"Stop it first: python scripts/stop.py --name {name}",
            err=True,
        )
        sys.exit(1)

    if reset:
        prompt = (
            f"Remove all persisted state for '{name}' in {instance_dir}, "
            f"keeping only {INSTANCE_CONFIG_NAME}? This cannot be undone."
        )
    else:
        prompt = f"Permanently delete '{name}' and everything in {instance_dir}? This cannot be undone."
    if not yes and not click.confirm(prompt, default=False):
        click.echo("Aborted; nothing was removed.", err=True)
        sys.exit(1)

    try:
        if reset:
            removed = _reset(instance_dir)
        else:
            shutil.rmtree(instance_dir)
    except OSError as exc:
        click.echo(f"Could not remove {instance_dir}: {exc}", err=True)
        sys.exit(1)

    if reset:
        click.echo(f"Reset '{name}': removed {removed} entries, kept {INSTANCE_CONFIG_NAME} if present.")
    else:
        click.echo(f"Deleted '{name}' ({instance_dir}).")


if __name__ == "__main__":
    main()
