"""Spawn a new named consciousness instance.

No direct theory mapping — entry-point script.
Bootstraps config, wires logging to a per-instance rotating file, and
either hands off to ConsciousnessCLI (interactive) or runs headless
(no TUI). Use --bg to detach into the background immediately.
See core/consciousness.py for the theoretical grounding of the loop.

The web dashboard is a separate process — launch it with scripts/web.py
(issue #55). This script no longer hosts an in-process dashboard.
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
from persistence.paths import consciousness_dir
from scripts._logging import configure_logging


def _build_config_path(name: str, provider: str | None, model: str | None) -> Path:
    config_path = Path(__file__).resolve().parents[1] / "config" / "default_consciousness.yaml"
    if not (provider or model):
        return config_path

    import yaml

    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if provider:
        base["llm"]["provider"] = provider
    if model:
        base["llm"]["model"] = model
    # Persist the override into the consciousness dir rather than /tmp — the
    # previous tempfile-with-delete=False approach (#105) leaked a file on
    # every spawn. Writing into the instance dir is overwritten on each spawn
    # and makes the resolved config inspectable post-hoc.
    agent_dir = consciousness_dir(name)
    agent_dir.mkdir(parents=True, exist_ok=True)
    out_path = agent_dir / "_spawn_config.yaml"
    out_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return out_path


async def _run(mind: Consciousness, headless: bool) -> None:
    """Unified async entry point for all foreground modes."""
    if headless:
        from interfaces.event_relay import EventRelay
        sock_path = consciousness_dir(mind.name) / "events.sock"
        relay = EventRelay(mind, sock_path)
        relay_task = asyncio.create_task(relay.serve())
        try:
            await mind.run()
        except asyncio.CancelledError:
            pass  # signal-driven shutdown; mind.run() finally block saves state
        finally:
            relay_task.cancel()
            await asyncio.gather(relay_task, return_exceptions=True)
    else:
        cli = ConsciousnessCLI(mind)
        await cli.run()


@click.command()
@click.option("--name", required=True, type=str, help="Consciousness name")
@click.option("--provider", default=None, type=str, help="LLM provider override")
@click.option("--model", default=None, type=str, help="Model override")
@click.option("--log-level", default="WARNING", show_default=True, help="Log level (DEBUG/INFO/WARNING/ERROR)")
@click.option("--headless", is_flag=True, default=False, help="Skip TUI — run as a foreground log-only process")
@click.option("--bg", is_flag=True, default=False, help="Detach into background (implies --headless); logs to run.log")
def main(
    name: str,
    provider: str | None,
    model: str | None,
    log_level: str,
    headless: bool,
    bg: bool,
) -> None:
    load_dotenv()

    if bg:
        import os
        import subprocess

        agent_dir = consciousness_dir(name)
        agent_dir.mkdir(parents=True, exist_ok=True)
        log_path = agent_dir / "run.log"
        pid_path = agent_dir / "pid"

        # Re-invoke self as a headless foreground process in a new session.
        script = str(Path(__file__).resolve())
        args = [sys.executable, script] + [
            a for a in sys.argv[1:] if a not in ("--bg",)
        ] + ["--headless"]

        with open(log_path, "a") as log_fh:
            proc = subprocess.Popen(
                args,
                stdout=log_fh,
                stderr=log_fh,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ},
            )

        pid_path.write_text(str(proc.pid))
        click.echo(f"Started '{name}' in background  PID {proc.pid}")
        click.echo(f"Logs   : {log_path}")
        click.echo(f"Attach : python scripts/attach.py --name {name}")
        click.echo(f"Stop   : python scripts/stop.py --name {name}")
        return

    log_path = configure_logging(name, log_level)
    logging.info("Spawn started — logs: %s", log_path)
    config_path = _build_config_path(name, provider, model)
    mind = Consciousness(name=name, config_path=str(config_path))
    asyncio.run(_run(mind, headless))


if __name__ == "__main__":
    main()
