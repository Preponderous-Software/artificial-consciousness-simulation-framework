"""Experiment runner — orchestrates spawn → wait → stop → snapshot → report.

Reads an `ExperimentManifest`, materialises a config (default + overrides) to a
tempfile, subprocess.Popen's a fresh consciousness, polls `state.json` until
the duration is met, sends SIGTERM, copies the run artifacts into the
experiment dir, computes metrics, and renders a markdown report.

Design notes:
- Mirrors `scripts/spawn.py` invocation rather than importing it: keeps the
  experiment harness decoupled from the spawn CLI's flag set.
- Uses `MockProvider` + `MockPerception` by default in the example manifest so
  CI runs and tests don't require Ollama or network.
- Captures the current git SHA at run time into `meta.yaml` for reproducibility.
- Wall-clock cap (`max_wall_clock_minutes` arg) hard-stops a hung run even if
  the thought-count target isn't reached. Defaults to 30 minutes; raise for
  long real-LLM runs.
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
SPAWN_SCRIPT = PROJECT_ROOT / "scripts" / "spawn.py"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"


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
    # Ensure the project root is on sys.path before importing — runner.py is
    # importable from outside the consciousness-sim directory.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from persistence.paths import consciousness_dir as _real_dir
    return _real_dir(name)


def _read_thought_count(state_path: Path) -> int:
    try:
        return int(json.loads(state_path.read_text())["thought_count"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return 0


def run_experiment(
    manifest: ExperimentManifest,
    experiments_root: Path | None = None,
    max_wall_clock_minutes: float = 30.0,
    poll_interval_s: float = 1.0,
) -> Path:
    """Execute a manifest end-to-end, return the path to the run directory.

    Run directory layout:
        experiments/<manifest.name>/<UTC-timestamp>/
            manifest.yaml   # frozen copy of the spec
            meta.yaml       # branch SHA, wall time, exit reason
            journal.jsonl   # copy from ~/.consciousness/<name>/
            state.json
            episodic.jsonl  # if present
            metrics.json    # compute_all() output
            report.md
    """
    experiments_root = experiments_root or EXPERIMENTS_ROOT
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = experiments_root / manifest.name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Materialise merged config to a tempfile next to the run
    base_cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    merged_cfg = _merge_config(base_cfg, manifest.config_overrides)
    cfg_path = run_dir / "_spawn_config.yaml"
    cfg_path.write_text(yaml.safe_dump(merged_cfg, sort_keys=False), encoding="utf-8")

    # 2. Make sure we start from a clean ~/.consciousness/<name>/ — otherwise
    #    a previous run's state would resume and contaminate the experiment.
    cons_dir = _consciousness_dir(manifest.consciousness_name)
    if cons_dir.exists():
        shutil.rmtree(cons_dir)

    # 3. Spawn the consciousness — invoke spawn.py with overrides pointing at
    #    the merged config. spawn.py doesn't accept --config; instead we
    #    invoke the orchestrator directly via a tiny launcher so we can pass
    #    the merged config path.
    started_at = datetime.now(timezone.utc)
    launcher_code = (
        "import asyncio, sys, os\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from core.consciousness import Consciousness\n"
        f"mind = Consciousness(name={manifest.consciousness_name!r}, config_path={str(cfg_path)!r})\n"
        "asyncio.run(mind.run())\n"
    )
    log_path = run_dir / "spawn.log"
    process = subprocess.Popen(
        [sys.executable, "-c", launcher_code],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env={**os.environ},
        start_new_session=True,
    )

    # 4. Wait for the target — either thought count or wall-clock minutes
    target_thoughts = manifest.duration.thoughts
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
            # Process died unexpectedly?
            if process.poll() is not None:
                exit_reason = f"process exited (code {process.returncode})"
                break
            # Hit wall-clock cap?
            if time.monotonic() > deadline:
                exit_reason = f"max_wall_clock_minutes={max_wall_clock_minutes} exceeded"
                break
            # Hit thought target?
            if target_thoughts is not None:
                if _read_thought_count(state_path) >= target_thoughts:
                    exit_reason = f"reached target {target_thoughts} thoughts"
                    break
            # Hit time target?
            if target_time is not None and time.monotonic() >= target_time:
                exit_reason = f"reached target {target_minutes} minutes"
                break
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        exit_reason = "KeyboardInterrupt"
        raise
    finally:
        # 5. SIGTERM and wait for clean exit (5s grace per project convention)
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    ended_at = datetime.now(timezone.utc)

    # 6. Copy run artifacts into experiment dir (after process is stopped so
    #    state.json's _save_state() finally-block has run)
    for fname in ("journal.jsonl", "state.json", "episodic.jsonl"):
        src = cons_dir / fname
        if src.exists():
            shutil.copy2(src, run_dir / fname)

    # 7. Compute metrics
    journal_path = run_dir / "journal.jsonl"
    state_dest = run_dir / "state.json"
    if not (journal_path.exists() and state_dest.exists()):
        raise RunnerError(
            f"Run completed but artifacts missing in {cons_dir}: "
            f"journal_exists={journal_path.exists()}, state_exists={state_dest.exists()}. "
            f"Spawn log: {log_path}"
        )
    metrics = compute_all(journal_path, state_dest)

    # 8. Persist manifest, meta, metrics, report
    (run_dir / "manifest.yaml").write_text(manifest.to_yaml(), encoding="utf-8")
    meta = {
        "branch_sha": manifest.branch_sha or _git_sha(),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "wall_clock_minutes": (ended_at - started_at).total_seconds() / 60,
        "exit_reason": exit_reason,
        "spawn_log_path": str(log_path.relative_to(run_dir)),
    }
    (run_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(manifest, meta, metrics), encoding="utf-8")

    # Spawn config kept for forensics; remove if you want a smaller tree
    return run_dir
