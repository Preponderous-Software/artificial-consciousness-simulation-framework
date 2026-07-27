"""Experiment runner — orchestrates spawn → wait → stop → snapshot → report.

Reads an `ExperimentManifest`, materialises a config (default + overrides) to a
tempfile, subprocess.Popen's a consciousness, polls `state.json` until the
duration is met, sends SIGTERM, copies the run artifacts into the experiment
dir, computes metrics, and renders a markdown report.

Features:
- Three duration modes: `minutes` (wall clock), `thoughts` (cumulative
  thought_count target), `add_thoughts` (produce this many more — pairs
  with resume_from).
- `resume_from` field: instead of wiping the consciousness dir, copy
  journal.jsonl + state.json from a source (a name in ~/.consciousness/
  or a path to a recorded run dir) into the new instance's dir before spawn.
- `replicates: N` field: run the manifest N times sequentially, each into
  its own `replicate-<i>/` subdir under the parent run dir.
- `--detach` mode (via `start_detached`): fork a child process that runs
  the experiment synchronously, parent returns immediately with the run dir.
  `status(run_dir)` reports running / done / failed.

Design notes:
- Mirrors `scripts/spawn.py` invocation rather than importing it; keeps the
  experiment harness decoupled from the spawn CLI's flag set.
- Uses `MockProvider` + `MockPerception` by default in the example manifest
  so CI runs and tests don't require Ollama or network.
- Captures git SHA at run time into `meta.yaml` for reproducibility.
- Wall-clock cap (`max_wall_clock_minutes`) hard-stops a hung run regardless
  of which duration mode is in use.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.manifest import ExperimentManifest
from experiments.metrics import compute_all
from experiments.report import render_report


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "consciousness-sim"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default_consciousness.yaml"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

# Status markers written into the run dir. Used by `status()` to disambiguate
# running / done / failed without parsing meta.yaml.
_STARTED_MARKER = ".STARTED"
_FAILED_MARKER = ".FAILED"


class RunnerError(RuntimeError):
    """Raised when a run cannot be completed cleanly."""


def _merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge overrides into base, returning a new dict. Lists replaced wholesale."""
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_config(out[k], v)
        else:
            out[k] = v
    return out


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:10]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _consciousness_dir(name: str) -> Path:
    """Resolve the on-disk directory for a consciousness, honoring the same
    name-sanitization rules the runtime uses (e.g. `-` → `_`).
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from persistence.paths import consciousness_dir as _real_dir
    return _real_dir(name)


def _read_thought_count(state_path: Path) -> int:
    try:
        return int(json.loads(state_path.read_text())["thought_count"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return 0


def _resolve_resume_source(resume_from: str) -> Path:
    """Resolve `resume_from` to a directory containing journal.jsonl + state.json.

    Accepts either:
    - A consciousness name (e.g. "Echo") → resolved via CONSCIOUSNESS_HOME
    - A path (absolute or relative) to a recorded run directory

    Raises RunnerError if the source is missing or incomplete.
    """
    candidate = Path(resume_from)
    if candidate.exists() and candidate.is_dir():
        source = candidate
    else:
        source = _consciousness_dir(resume_from)
    journal = source / "journal.jsonl"
    state = source / "state.json"
    if not (journal.exists() and state.exists()):
        raise RunnerError(
            f"resume_from={resume_from!r} resolved to {source} but it lacks "
            f"journal.jsonl ({journal.exists()}) or state.json ({state.exists()})"
        )
    return source


def _prepare_consciousness_dir(
    cons_dir: Path,
    resume_from: str | None,
) -> int:
    """Either wipe (fresh run) or seed from a resume source. Returns starting thought_count."""
    if resume_from is None:
        if cons_dir.exists():
            shutil.rmtree(cons_dir)
        return 0
    source = _resolve_resume_source(resume_from)
    if cons_dir.exists():
        shutil.rmtree(cons_dir)
    cons_dir.mkdir(parents=True)
    for fname in ("journal.jsonl", "state.json", "episodic.jsonl"):
        src = source / fname
        if src.exists():
            shutil.copy2(src, cons_dir / fname)
    return _read_thought_count(cons_dir / "state.json")


def run_experiment(
    manifest: ExperimentManifest,
    experiments_root: Path | None = None,
    max_wall_clock_minutes: float = 30.0,
    poll_interval_s: float = 1.0,
    run_dir: Path | None = None,
) -> Path:
    """Execute a single replicate of a manifest end-to-end.

    Returns the path to the run directory. If `run_dir` is provided (used by
    the replicates loop and detach mode), the runner writes into that dir;
    otherwise it generates a fresh timestamped dir under
    `experiments/<manifest.name>/`.

    Run directory layout:
        <run_dir>/
            manifest.yaml   # frozen copy of the spec (with schema_version)
            meta.yaml       # branch SHA, wall time, exit reason, starting count
            journal.jsonl   # copy from ~/.consciousness/<name>/
            state.json
            episodic.jsonl  # if present
            metrics.json    # compute_all() output + _schema_version
            report.md
    """
    experiments_root = experiments_root or EXPERIMENTS_ROOT
    if run_dir is None:
        # Microsecond-precision timestamp so back-to-back calls don't collide.
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
        run_dir = experiments_root / manifest.name / timestamp
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir()
    # Status marker — cleared on success
    (run_dir / _STARTED_MARKER).touch()

    try:
        return _run_one(manifest, run_dir, max_wall_clock_minutes, poll_interval_s)
    except Exception:
        (run_dir / _FAILED_MARKER).touch()
        raise
    finally:
        marker = run_dir / _STARTED_MARKER
        if marker.exists():
            marker.unlink()


def _run_one(
    manifest: ExperimentManifest,
    run_dir: Path,
    max_wall_clock_minutes: float,
    poll_interval_s: float,
) -> Path:
    # 1. Materialise merged config to a tempfile next to the run
    base_cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    merged_cfg = _merge_config(base_cfg, manifest.config_overrides)
    cfg_path = run_dir / "_spawn_config.yaml"
    cfg_path.write_text(yaml.safe_dump(merged_cfg, sort_keys=False), encoding="utf-8")

    # 2. Prepare consciousness dir — wipe (fresh) or copy-from-source (resume)
    cons_dir = _consciousness_dir(manifest.consciousness_name)
    starting_thought_count = _prepare_consciousness_dir(cons_dir, manifest.resume_from)

    # 3. Spawn the consciousness
    started_at = datetime.now(timezone.utc)
    launcher_code = (
        "import asyncio, sys, os\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from core.consciousness import Consciousness\n"
        f"mind = Consciousness(name={manifest.consciousness_name!r}, config_path={str(cfg_path)!r})\n"
        "asyncio.run(mind.run())\n"
    )
    log_path = run_dir / "spawn.log"
    log_fh = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-c", launcher_code],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env={**os.environ},
        start_new_session=True,
    )

    # 4. Compute targets from the manifest's duration mode
    target_thoughts = _compute_thought_target(manifest, starting_thought_count)
    target_minutes = manifest.duration.minutes
    state_path = cons_dir / "state.json"

    deadline = time.monotonic() + max_wall_clock_minutes * 60
    target_time = (
        time.monotonic() + target_minutes * 60
        if target_minutes is not None else None
    )

    exit_reason = "unknown"
    try:
        while True:
            if process.poll() is not None:
                exit_reason = f"process exited (code {process.returncode})"
                break
            if time.monotonic() > deadline:
                exit_reason = f"max_wall_clock_minutes={max_wall_clock_minutes} exceeded"
                break
            if target_thoughts is not None:
                if _read_thought_count(state_path) >= target_thoughts:
                    if manifest.duration.add_thoughts is not None:
                        exit_reason = (
                            f"reached target {manifest.duration.add_thoughts} added "
                            f"thoughts (start {starting_thought_count} → {target_thoughts})"
                        )
                    else:
                        exit_reason = f"reached target {target_thoughts} thoughts"
                    break
            if target_time is not None and time.monotonic() >= target_time:
                exit_reason = f"reached target {target_minutes} minutes"
                break
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        exit_reason = "KeyboardInterrupt"
        raise
    finally:
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_fh.close()

    ended_at = datetime.now(timezone.utc)

    # 5. Copy run artifacts
    for fname in ("journal.jsonl", "state.json", "episodic.jsonl"):
        src = cons_dir / fname
        if src.exists():
            shutil.copy2(src, run_dir / fname)

    journal_path = run_dir / "journal.jsonl"
    state_dest = run_dir / "state.json"
    if not (journal_path.exists() and state_dest.exists()):
        if exit_reason.startswith("max_wall_clock_minutes"):
            raise RunnerError(
                f"Run was capped by max_wall_clock_minutes={max_wall_clock_minutes} before "
                f"the consciousness could initialize (no journal.jsonl / state.json written "
                f"to {cons_dir}). The cap is likely too short for this host's subprocess "
                f"startup cost — try a larger --max-wall-clock-minutes. Spawn log: {log_path}"
            )
        raise RunnerError(
            f"Run completed but artifacts missing in {cons_dir}: "
            f"journal_exists={journal_path.exists()}, state_exists={state_dest.exists()}. "
            f"Spawn log: {log_path}"
        )
    metrics = compute_all(journal_path, state_dest)
    # _schema_version comes from compute_all itself — see metrics.METRICS_SCHEMA_VERSION

    # 6. Persist artifacts
    (run_dir / "manifest.yaml").write_text(manifest.to_yaml(), encoding="utf-8")
    meta = {
        "manifest_schema_version": manifest.schema_version,
        "branch_sha": manifest.branch_sha or _git_sha(),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "wall_clock_minutes": (ended_at - started_at).total_seconds() / 60,
        "exit_reason": exit_reason,
        "starting_thought_count": starting_thought_count,
        "spawn_log_path": str(log_path.relative_to(run_dir)),
    }
    if manifest.resume_from is not None:
        meta["resumed_from"] = manifest.resume_from
    (run_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(manifest, meta, metrics), encoding="utf-8")

    return run_dir


def _compute_thought_target(
    manifest: ExperimentManifest, starting_thought_count: int
) -> int | None:
    """Resolve `manifest.duration` to an absolute thought_count target, or None
    if the duration is wall-clock-based."""
    if manifest.duration.thoughts is not None:
        return manifest.duration.thoughts          # cumulative target
    if manifest.duration.add_thoughts is not None:
        return starting_thought_count + manifest.duration.add_thoughts
    return None                                      # minutes mode — target_time handles it


def run_experiment_replicated(
    manifest: ExperimentManifest,
    experiments_root: Path | None = None,
    max_wall_clock_minutes: float = 30.0,
    poll_interval_s: float = 1.0,
    parent_dir: Path | None = None,
) -> Path:
    """Run the manifest N times if `replicates` is set, otherwise once.

    Returns the path to the parent run dir (which contains per-replicate
    subdirs `replicate-0/`, `replicate-1/`, … and a `replicates_index.md`)
    when N>1, or the single run dir when N is None or 1.

    When `parent_dir` is provided (used by `start_detached`), populates that
    pre-created dir instead of generating a fresh timestamped one. This is
    what lets the detached child write into the same dir the parent process
    just returned to the caller.

    Aggregation across replicates (mean / stddev across metrics) is a Phase-2
    follow-up; for now `replicates_index.md` just lists the child reports
    and exit reasons.
    """
    experiments_root = experiments_root or EXPERIMENTS_ROOT
    n = manifest.replicates
    if n is None or n <= 1:
        return run_experiment(
            manifest, experiments_root=experiments_root,
            max_wall_clock_minutes=max_wall_clock_minutes,
            poll_interval_s=poll_interval_s,
            run_dir=parent_dir,
        )

    if parent_dir is None:
        parent_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
        parent_dir = experiments_root / manifest.name / parent_ts
        parent_dir.parent.mkdir(parents=True, exist_ok=True)
        parent_dir.mkdir()

    child_dirs: list[Path] = []
    for i in range(n):
        child_dir = parent_dir / f"replicate-{i}"
        child_dir.mkdir()
        run_experiment(
            manifest,
            experiments_root=experiments_root,
            max_wall_clock_minutes=max_wall_clock_minutes,
            poll_interval_s=poll_interval_s,
            run_dir=child_dir,
        )
        child_dirs.append(child_dir)

    # Write a simple index — aggregate metrics deferred per docstring
    lines = [
        f"# {manifest.name} — {n} replicates",
        "",
        f"Parent run: `{parent_dir.name}`",
        "",
        "| # | Replicate | Exit reason | Wall clock |",
        "|---|---|---|---|",
    ]
    for i, child in enumerate(child_dirs):
        meta = yaml.safe_load((child / "meta.yaml").read_text())
        lines.append(
            f"| {i} | `{child.name}` | {meta.get('exit_reason', '?')} "
            f"| {meta.get('wall_clock_minutes', 0):.1f}m |"
        )
    (parent_dir / "replicates_index.md").write_text("\n".join(lines), encoding="utf-8")
    return parent_dir


# ---------------------------------------------------------------------------
# Detach mode + status
# ---------------------------------------------------------------------------

def start_detached(
    manifest_path: Path,
    experiments_root: Path | None = None,
    max_wall_clock_minutes: float = 30.0,
) -> Path:
    """Fork a child process that runs the manifest; return immediately with
    the run dir.

    The child is detached via `start_new_session=True` so it survives the
    parent's exit. Status checks read marker files written by the child.

    Limitations: no daemonization (no double-fork), no log redirection to
    `~/.consciousness/`. The child writes its stdout/stderr to
    `<run_dir>/_detached.log`.
    """
    experiments_root = experiments_root or EXPERIMENTS_ROOT
    manifest = ExperimentManifest.from_yaml(Path(manifest_path))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    if manifest.replicates and manifest.replicates > 1:
        # For replicates, the timestamped dir is the parent; children land beneath
        run_dir = experiments_root / manifest.name / timestamp
    else:
        run_dir = experiments_root / manifest.name / timestamp
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    (run_dir / _STARTED_MARKER).touch()

    # Hand off to a worker script that re-imports runner and calls the
    # right entry point with the pre-created run_dir. We re-invoke the
    # module so the child runs in a fresh interpreter (no shared async
    # state, no inherited signal handlers).
    #
    # Critical: pass run_dir as parent_dir when replicates > 1, otherwise
    # run_experiment_replicated would create its own timestamped parent
    # and the pre-created dir we returned to the caller would be orphaned.
    worker_code = (
        "import sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from pathlib import Path\n"
        "from experiments.manifest import ExperimentManifest\n"
        "from experiments.runner import run_experiment, run_experiment_replicated\n"
        f"m = ExperimentManifest.from_yaml(Path({str(manifest_path)!r}))\n"
        f"run_dir = Path({str(run_dir)!r})\n"
        "if m.replicates and m.replicates > 1:\n"
        f"    run_experiment_replicated(m, experiments_root={str(experiments_root)!r},"
        f" max_wall_clock_minutes={max_wall_clock_minutes}, parent_dir=run_dir)\n"
        "else:\n"
        f"    run_experiment(m, max_wall_clock_minutes={max_wall_clock_minutes}, run_dir=run_dir)\n"
    )
    detached_log = run_dir / "_detached.log"
    log_fh = open(detached_log, "w", encoding="utf-8")
    try:
        subprocess.Popen(
            [sys.executable, "-c", worker_code],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env={**os.environ},
            start_new_session=True,
        )
    finally:
        # Close our parent-side fd immediately — the child holds its own
        # via the inherited fd. Keeps file-locking sane on Windows and
        # avoids ResourceWarning on long-lived parent processes.
        log_fh.close()
    # Parent does NOT wait — child runs until it writes report.md or fails.
    return run_dir


def status(run_dir: Path) -> dict[str, Any]:
    """Report the state of a run directory.

    Returns a dict with `state` ∈ {running, done, failed, unknown}
    plus optional `exit_reason`, `wall_clock_minutes`, `started_at`,
    `kind` (`single` or `replicated`).

    A run is considered "done" when either:
      - `report.md` exists (single-run case), or
      - `replicates_index.md` exists (parent dir of a replicated run)
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return {"state": "unknown", "reason": "run_dir does not exist"}
    if (run_dir / _FAILED_MARKER).exists():
        return {"state": "failed", "reason": "FAILED marker present; check _detached.log"}
    has_report = (run_dir / "report.md").exists()
    has_index = (run_dir / "replicates_index.md").exists()
    if has_report or has_index:
        info: dict[str, Any] = {
            "state": "done",
            "kind": "replicated" if has_index else "single",
        }
        meta_path = run_dir / "meta.yaml"
        if meta_path.exists():
            try:
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
                info.update({
                    "exit_reason": meta.get("exit_reason"),
                    "wall_clock_minutes": meta.get("wall_clock_minutes"),
                    "started_at": meta.get("started_at"),
                })
            except yaml.YAMLError:
                pass
        if has_index:
            n_replicates = len(list(run_dir.glob("replicate-*")))
            info["n_replicates"] = n_replicates
        return info
    if (run_dir / _STARTED_MARKER).exists():
        return {"state": "running", "reason": "STARTED marker present, no report.md / replicates_index.md yet"}
    return {"state": "unknown", "reason": "no markers and no report.md / replicates_index.md"}
