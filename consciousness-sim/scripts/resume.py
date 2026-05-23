"""Resume a previously persisted consciousness instance by name."""

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
import click
from dotenv import load_dotenv

from core.consciousness import Consciousness
from interfaces.cli import ConsciousnessCLI


@click.command()
@click.option("--name", required=True, type=str, help="Consciousness name")
def main(name: str) -> None:
    load_dotenv()
    config_path = Path(__file__).resolve().parents[1] / "config" / "default_consciousness.yaml"
    mind = Consciousness(name=name, config_path=str(config_path))
    cli = ConsciousnessCLI(mind)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
