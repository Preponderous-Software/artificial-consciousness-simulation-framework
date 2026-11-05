"""Resume a previously persisted consciousness instance by name.

No direct theory mapping — entry-point script.
Restores state from disk via StateManager and resumes the run loop
against the instance's persisted config.yaml when it has one (#187).
Equivalent to spawn.py but skips fresh identity initialization; the
persisted identity, mood, and short-term buffer are loaded instead.
Like spawn.py it refuses a live duplicate and records a pid file, so a
resumed instance is visible to spawn, stop, doctor, delete and export (#237).
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

import asyncio
import logging
import click
from dotenv import load_dotenv

from core.consciousness import Consciousness
from interfaces.cli import ConsciousnessCLI
from interfaces.usage_reporting import report_startup, start_usage_reporting
from scripts._config import ConfigError, resolve_config_path
from scripts._logging import configure_logging
from scripts.spawn import _check_duplicate_pid, _claim_pid_file


@click.command()
@click.option("--name", required=True, type=str, help="Consciousness name")
@click.option(
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Replace this instance's persisted config (<instance>/config.yaml) and resume with it (#187).",
)
@click.option("--log-level", default="WARNING", show_default=True, help="Log level (DEBUG/INFO/WARNING/ERROR)")
@click.option("--force", is_flag=True, default=False, help="Resume even if an instance with this name is already alive")
def main(name: str, config_file: Path | None, log_level: str, force: bool) -> None:
    load_dotenv()
    _check_duplicate_pid(name, force=force)
    report_startup(start_usage_reporting(log=click.echo), "resume")
    try:
        config_path = resolve_config_path(name, config_file)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    log_path = configure_logging(name, log_level)
    logging.info("Resume started — logs: %s, config: %s", log_path, config_path)
    _claim_pid_file(name)
    mind = Consciousness(name=name, config_path=str(config_path))
    cli = ConsciousnessCLI(mind)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
