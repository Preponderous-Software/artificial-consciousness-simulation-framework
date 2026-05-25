"""Experiment harness CLI — `python scripts/experiment.py {run|list|replay-analysis}`.

See `experiments/__init__.py` for the architecture and issue #57 for the
motivating design. Phase 1 of #57: this CLI ships `run`, `list`,
`replay-analysis`. `compare` and the Claude skill are Phase 2 (deferred).

Examples:

    python scripts/experiment.py run experiments/manifests/sage-perception-baseline.yaml
    python scripts/experiment.py list
    python scripts/experiment.py replay-analysis experiments/sage-perception-baseline/2026-05-25T...Z/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import click
import yaml

from experiments.manifest import ExperimentManifest
from experiments.metrics import compute_all
from experiments.report import render_report
from experiments.runner import EXPERIMENTS_ROOT, run_experiment


@click.group()
def main() -> None:
    """Experiment harness for reproducible consciousness simulation runs."""


@main.command("run")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--max-wall-clock-minutes", default=30.0, show_default=True, type=float,
    help="Hard cap on wall-clock runtime; runs that exceed it are SIGTERM'd and reported as such.",
)
def cmd_run(manifest_path: Path, max_wall_clock_minutes: float) -> None:
    """Execute a manifest end-to-end and produce a versioned run record."""
    manifest = ExperimentManifest.from_yaml(manifest_path)
    click.echo(f"Running manifest: {manifest.name}")
    click.echo(f"  consciousness: {manifest.consciousness_name}")
    click.echo(f"  duration:      {manifest.duration}")
    click.echo(f"  tags:          {manifest.tags or '(none)'}")
    run_dir = run_experiment(manifest, max_wall_clock_minutes=max_wall_clock_minutes)
    click.echo(f"\nRun complete: {run_dir}")
    click.echo(f"  report:  {run_dir / 'report.md'}")
    click.echo(f"  metrics: {run_dir / 'metrics.json'}")


@main.command("list")
def cmd_list() -> None:
    """List every recorded run under `experiments/`, newest first."""
    if not EXPERIMENTS_ROOT.exists():
        click.echo("No experiments directory yet.")
        return
    rows: list[tuple[str, str, str, str]] = []
    for manifest_dir in sorted(EXPERIMENTS_ROOT.iterdir()):
        if not manifest_dir.is_dir() or manifest_dir.name in ("golden", "manifests"):
            continue
        for run_dir in sorted(manifest_dir.iterdir(), reverse=True):
            meta_path = run_dir / "meta.yaml"
            if not meta_path.exists():
                continue
            try:
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
            except (yaml.YAMLError, OSError):
                continue
            rows.append((
                manifest_dir.name,
                run_dir.name,
                meta.get("exit_reason", "?"),
                f"{meta.get('wall_clock_minutes', 0):.1f}m",
            ))
    if not rows:
        click.echo("No recorded runs found.")
        return
    width = max(len(r[0]) for r in rows)
    for name, ts, reason, wall in rows:
        click.echo(f"{name:<{width}}  {ts}  {wall:<8}  {reason}")


@main.command("replay-analysis")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def cmd_replay(run_dir: Path) -> None:
    """Re-compute metrics + re-render report.md from a stored run, using the
    current code. Useful for back-applying a fixed metric to old data without
    re-running the simulation."""
    journal = run_dir / "journal.jsonl"
    state = run_dir / "state.json"
    manifest_path = run_dir / "manifest.yaml"
    meta_path = run_dir / "meta.yaml"
    for p in (journal, state, manifest_path, meta_path):
        if not p.exists():
            raise click.ClickException(f"Missing artifact: {p}")

    manifest = ExperimentManifest.model_validate(yaml.safe_load(manifest_path.read_text()))
    meta = yaml.safe_load(meta_path.read_text())

    metrics = compute_all(journal, state)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(manifest, meta, metrics), encoding="utf-8")
    click.echo(f"Re-analyzed {run_dir}")
    click.echo(f"  report:  {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
