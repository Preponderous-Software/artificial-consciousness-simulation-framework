"""Spawn a new named consciousness instance and start the live CLI."""

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
import logging.handlers
from pathlib import Path

import click
from dotenv import load_dotenv

from core.consciousness import Consciousness
from interfaces.cli import ConsciousnessCLI
from persistence.paths import consciousness_dir


def _configure_logging(name: str, level: str) -> Path:
    """Write logs to <CONSCIOUSNESS_HOME>/<name>/run.log; return the log path."""
    log_path = consciousness_dir(name) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.root.setLevel(getattr(logging, level.upper(), logging.WARNING))
    logging.root.addHandler(handler)
    return log_path


@click.command()
@click.option("--name", required=True, type=str, help="Consciousness name")
@click.option("--provider", default=None, type=str, help="LLM provider override")
@click.option("--model", default=None, type=str, help="Model override")
@click.option("--log-level", default="WARNING", show_default=True, help="Log level (DEBUG/INFO/WARNING/ERROR)")
def main(name: str, provider: str | None, model: str | None, log_level: str) -> None:
    load_dotenv()
    log_path = _configure_logging(name, log_level)
    logging.info("Spawn started — logs: %s", log_path)
    config_path = Path(__file__).resolve().parents[1] / "config" / "default_consciousness.yaml"

    if provider or model:
        import tempfile
        import yaml

        base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if provider:
            base["llm"]["provider"] = provider
        if model:
            base["llm"]["model"] = model
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".yaml",
            prefix="consciousness_override_",
            delete=False,
        ) as tmp:
            yaml.safe_dump(base, tmp)
            config_path = Path(tmp.name)

    mind = Consciousness(name=name, config_path=str(config_path))
    cli = ConsciousnessCLI(mind)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
