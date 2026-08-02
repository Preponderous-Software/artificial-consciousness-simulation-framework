"""Experiment harness CLI — `experiment.py {run|status|list|replay-analysis|compare|check-smoke|prune}`.

See `experiments/__init__.py` for the architecture and issue #57 for the
motivating design. Phase 1+2 of #57 ship `run` (with `--detach`), `status`,
`list`, `replay-analysis`, and `compare`. `check-smoke` (#87, Phase 3 of #57)
adds a CI regression gate over the mock-smoke-baseline manifest. The Claude
skills under `.claude/skills/` provide the narrative layer on top.

Examples:

    python scripts/experiment.py run experiments/manifests/sage-perception-baseline.yaml
    python scripts/experiment.py run <manifest> --detach   # fork; return immediately
    python scripts/experiment.py status experiments/<name>/<UTC-ts>/
    python scripts/experiment.py list
    python scripts/experiment.py replay-analysis experiments/<name>/<UTC-ts>/
    python scripts/experiment.py compare experiments/golden/Rafael/ experiments/golden/Echo/
    python scripts/experiment.py check-smoke experiments/mock-smoke-baseline/<UTC-ts>/
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

from experiments.compare import compare_runs
from experiments.manifest import ExperimentManifest
from experiments.metrics import compute_all
from experiments.prune import prune_runs
from experiments.regression import check_smoke_regression
from experiments.report import render_report
from experiments.runner import (
    EXPERIMENTS_ROOT,
    run_experiment,
    run_experiment_replicated,
    start_detached,
    status as runner_status,
)

GOLDEN_DIR = ROOT_DIR / "experiments" / "golden"


@click.group()
def main() -> None:
    """Experiment harness for reproducible consciousness simulation runs."""


@main.command("run")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--max-wall-clock-minutes", default=30.0, show_default=True, type=float,
    help="Hard cap on wall-clock runtime; runs that exceed it are SIGTERM'd and reported as such.",
)
@click.option(
    "--detach", is_flag=True, default=False,
    help="Fork the run as a background process and return immediately with the "
         "run dir. Use `experiment.py status <run-dir>` to monitor progress.",
)
def cmd_run(manifest_path: Path, max_wall_clock_minutes: float, detach: bool) -> None:
    """Execute a manifest end-to-end and produce a versioned run record."""
    manifest = ExperimentManifest.from_yaml(manifest_path)
    click.echo(f"Running manifest: {manifest.name}")
    click.echo(f"  consciousness: {manifest.consciousness_name}")
    click.echo(f"  duration:      {manifest.duration}")
    if manifest.resume_from:
        click.echo(f"  resume_from:   {manifest.resume_from}")
    if manifest.replicates and manifest.replicates > 1:
        click.echo(f"  replicates:    {manifest.replicates}")
    click.echo(f"  tags:          {manifest.tags or '(none)'}")

    if detach:
        run_dir = start_detached(
            manifest_path, max_wall_clock_minutes=max_wall_clock_minutes,
        )
        click.echo(f"\nDetached: {run_dir}")
        click.echo(f"  monitor: python scripts/experiment.py status {run_dir}")
        click.echo(f"  log:     {run_dir / '_detached.log'}")
        return

    run_dir = run_experiment_replicated(
        manifest, max_wall_clock_minutes=max_wall_clock_minutes,
    )
    click.echo(f"\nRun complete: {run_dir}")
    if manifest.replicates and manifest.replicates > 1:
        click.echo(f"  index:   {run_dir / 'replicates_index.md'}")
        click.echo(f"  ({manifest.replicates} replicates under {run_dir}/replicate-*/)")
    else:
        click.echo(f"  report:  {run_dir / 'report.md'}")
        click.echo(f"  metrics: {run_dir / 'metrics.json'}")


@main.command("status")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def cmd_status(run_dir: Path) -> None:
    """Report the state of a run directory (running / done / failed)."""
    info = runner_status(run_dir)
    state = info.get("state", "unknown")
    click.echo(f"{run_dir}: {state}")
    for k, v in info.items():
        if k == "state":
            continue
        click.echo(f"  {k}: {v}")


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


@main.command("compare")
@click.argument("run_a", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("run_b", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output", "output_path", type=click.Path(path_type=Path), default=None,
    help="Write the comparison markdown to this path instead of stdout.",
)
@click.option(
    "--samples", "k_samples", type=int, default=3, show_default=True,
    help="Number of thoughts to sample from each run (evenly spaced).",
)
def cmd_compare(run_a: Path, run_b: Path, output_path: Path | None, k_samples: int) -> None:
    """Render a side-by-side markdown comparison of two recorded runs.

    The pure-data layer. For interpretive narrative on top, use the
    `/compare-experiments` Claude skill, which calls this and adds prose.
    """
    md = compare_runs(run_a, run_b, k_samples=k_samples)
    if output_path is None:
        click.echo(md)
    else:
        output_path.write_text(md, encoding="utf-8")
        click.echo(f"Wrote comparison to {output_path}")


@main.command("check-smoke")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--expected", "expected_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Golden snapshot to compare against. Defaults to "
         "experiments/golden/_smoke_expected.json.",
)
def cmd_check_smoke(run_dir: Path, expected_path: Path | None) -> None:
    """CI regression gate (#87): compare a run's metrics.json against the
    pinned mock-smoke-baseline snapshot. Exits non-zero on drift."""
    expected_path = expected_path or (GOLDEN_DIR / "_smoke_expected.json")
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise click.ClickException(f"Missing artifact: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    failures = check_smoke_regression(metrics, expected)
    if failures:
        click.echo(f"Smoke regression check FAILED against {expected_path}:")
        for f in failures:
            click.echo(f"  - {f}")
        raise SystemExit(1)
    click.echo(f"Smoke regression check passed against {expected_path} ({len(expected)} field(s) pinned).")


@main.command("prune")
@click.option(
    "--older-than", "older_than_days", type=int, default=None,
    help="Delete runs whose mtime is more than this many days in the past.",
)
@click.option(
    "--keep-last", "keep_last", type=int, default=None,
    help="Keep only the N most-recent runs per manifest; prune the rest.",
)
@click.option(
    "--manifest", "manifest_filter", type=str, default=None,
    help="Scope pruning to one manifest's subdir (e.g. 'mock-smoke-baseline').",
)
@click.option(
    "--dry-run/--no-dry-run", default=True,
    help="Default ON: list what would be deleted without acting. --no-dry-run requires --yes.",
)
@click.option(
    "--yes", "-y", "confirm_delete", is_flag=True, default=False,
    help="Required to actually delete (overrides --dry-run). Reversibility safety.",
)
def cmd_prune(
    older_than_days: int | None,
    keep_last: int | None,
    manifest_filter: str | None,
    dry_run: bool,
    confirm_delete: bool,
) -> None:
    """Garbage-collect old run dirs under experiments/<manifest>/<UTC-ts>/.

    Default policy when no flags are given: `--keep-last 10` in dry-run mode.
    Pass `--yes` (or `--no-dry-run --yes`) to actually delete.

    Protected from deletion: `experiments/golden/`, `experiments/manifests/`,
    any run dir with a `.STARTED` marker, any run dir with a future mtime.
    """
    if older_than_days is None and keep_last is None:
        keep_last = 10
        click.echo("No retention flags passed — defaulting to --keep-last 10.")

    # Either explicit path drops the safety: --yes (short) or --no-dry-run (verbose).
    really_delete = confirm_delete or not dry_run

    targets = prune_runs(
        EXPERIMENTS_ROOT,
        older_than_days=older_than_days,
        keep_last=keep_last,
        manifest=manifest_filter,
        dry_run=not really_delete,
    )
    if not targets:
        click.echo("Nothing to prune.")
        return

    verb = "Deleted" if really_delete else "Would delete"
    click.echo(f"{verb} {len(targets)} run dir(s):")
    for t in targets:
        click.echo(f"  {t}")
    if not really_delete:
        click.echo("\nPass --yes to actually delete.")


if __name__ == "__main__":
    main()
