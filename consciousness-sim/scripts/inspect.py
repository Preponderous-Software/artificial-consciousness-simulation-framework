"""Inspect a consciousness journal and memory state from local persistence."""

from __future__ import annotations

import os
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

from persistence.journal import Journal


@click.command()
@click.option("--name", required=True, type=str, help="Consciousness name")
@click.option("--limit", default=20, type=int, help="Number of recent events")
def main(name: str, limit: int) -> None:
    root = Path(os.path.expanduser(os.getenv("CONSCIOUSNESS_HOME", "~/.consciousness")))
    journal = Journal(root / name / "journal.jsonl")

    async def _run() -> None:
        events = await journal.recent(limit=limit)
        for e in events:
            print(f"{e['timestamp']} [{e['type']}] {e['content']}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
