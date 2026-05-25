"""Launch the standalone web dashboard.

No direct theory mapping — entry-point script.
The dashboard owns no consciousness; it discovers and manages running
instances via CONSCIOUSNESS_HOME, tails their journals for live events,
and exposes a process-management HTTP surface (spawn / stop / archive).
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


@click.command()
@click.option("--port", default=8080, show_default=True, type=int, help="Port to bind")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Bind host. Use 0.0.0.0 to expose to your LAN (no auth — opt-in).",
)
@click.option(
    "--allow-remote-spawn",
    is_flag=True,
    default=False,
    help="Permit spawn/stop/archive from non-localhost callers. "
         "Dangerous — enables arbitrary process control over the network. "
         "Only use behind a trusted reverse proxy with its own authentication.",
)
@click.option("--log-level", default="INFO", show_default=True, help="Log level")
def main(port: int, host: str, allow_remote_spawn: bool, log_level: str) -> None:
    load_dotenv()
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if host != "127.0.0.1" and not allow_remote_spawn:
        click.echo(
            f"NOTE: bound to {host} but spawn/stop endpoints are still localhost-only. "
            "Pass --allow-remote-spawn to enable remote process control (no auth).",
            err=True,
        )
    if allow_remote_spawn:
        click.echo(
            "WARNING: --allow-remote-spawn enabled. Any host that can reach "
            f"http://{host}:{port}/ can spawn and kill consciousness processes.",
            err=True,
        )

    display_host = "localhost" if host == "127.0.0.1" else host
    click.echo(f"Dashboard: http://{display_host}:{port}")

    from interfaces.web.server import start
    asyncio.run(start(port=port, host=host, allow_remote_spawn=allow_remote_spawn))


if __name__ == "__main__":
    main()
