"""Back up a consciousness instance's persisted state to a portable archive.

No direct theory mapping — infrastructure script (the ``export`` command
requested in #10).

Writes every regular file in the instance directory (identity snapshot,
episodic/journal logs, long-term memory database, inbox, per-instance
config, logs) into a gzipped tarball whose single top-level directory is the
sanitized instance name, so extracting it into CONSCIOUSNESS_HOME restores
the instance under the same name.

Runtime artefacts are left out: the ``pid`` file (a restored copy would
otherwise read as a live or stale process) and anything that is not a
regular file, such as the ``events.sock`` relay socket. A live instance is
refused rather than exported, because ``memory.db`` and the JSONL logs may be
mid-write — stop it first with scripts/stop.py.
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

import tarfile
from datetime import datetime, timezone

import click
from dotenv import load_dotenv

from persistence.paths import consciousness_dir
from scripts.doctor import _is_alive, _read_pid

# Files that describe the running process rather than the instance's state.
_EXCLUDED_NAMES = frozenset({"pid"})


def _exported_files(instance_dir: Path) -> list[Path]:
    """Return the regular files under instance_dir to archive, sorted."""
    return sorted(
        p
        for p in instance_dir.rglob("*")
        if p.is_file() and not p.is_symlink() and p.name not in _EXCLUDED_NAMES
    )


def _default_output(dir_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(f"{dir_name}-{stamp}.tar.gz")


@click.command()
@click.option("--name", required=True, type=str, help="Consciousness name to export")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Archive path (default: ./<name>-<UTC timestamp>.tar.gz)",
)
def main(name: str, output: Path | None) -> None:
    load_dotenv()
    try:
        instance_dir = consciousness_dir(name)
    except ValueError as exc:
        click.echo(f"Invalid name '{name}': {exc}", err=True)
        sys.exit(1)

    if not instance_dir.is_dir():
        click.echo(f"No persisted instance named '{name}' ({instance_dir}).", err=True)
        sys.exit(1)

    pid = _read_pid(instance_dir / "pid")
    if pid is not None and _is_alive(pid):
        click.echo(
            f"'{name}' is running (PID {pid}); its memory database and logs may be mid-write. "
            f"Stop it first: python scripts/stop.py --name {name}",
            err=True,
        )
        sys.exit(1)

    archive = output if output is not None else _default_output(instance_dir.name)
    if archive.exists():
        click.echo(f"Refusing to overwrite existing file: {archive}", err=True)
        sys.exit(1)

    files = _exported_files(instance_dir)
    try:
        # Mode "x:gz" fails rather than truncating if the path appeared
        # between the existence check above and this open.
        with tarfile.open(archive, "x:gz") as tar:
            for path in files:
                arcname = Path(instance_dir.name) / path.relative_to(instance_dir)
                tar.add(path, arcname=str(arcname), recursive=False)
    except OSError as exc:
        click.echo(f"Could not write {archive}: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Exported '{name}' ({len(files)} files) to {archive}")


if __name__ == "__main__":
    main()
