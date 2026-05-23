"""Resume a previously persisted consciousness instance by name.

No direct theory mapping — entry-point script.
Restores state from disk via StateManager and resumes the run loop.
Equivalent to spawn.py but skips fresh identity initialization; the
persisted identity, mood, and short-term buffer are loaded instead.
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
from scripts._logging import configure_logging


@click.command()
@click.option("--name", required=True, type=str, help="Consciousness name")
@click.option("--log-level", default="WARNING", show_default=True, help="Log level (DEBUG/INFO/WARNING/ERROR)")
def main(name: str, log_level: str) -> None:
    load_dotenv()
    log_path = configure_logging(name, log_level)
    logging.info("Resume started — logs: %s", log_path)
    config_path = Path(__file__).resolve().parents[1] / "config" / "default_consciousness.yaml"
    mind = Consciousness(name=name, config_path=str(config_path))
    cli = ConsciousnessCLI(mind)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
