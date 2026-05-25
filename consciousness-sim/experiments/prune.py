"""Garbage-collect old experiment run directories.

Per-run artifact dirs under `experiments/<manifest>/<UTC-ts>/` are ephemeral
(gitignored) and accumulate locally without bound. This module supplies the
pure retention logic; `scripts/experiment.py prune` is the CLI wrapper.

Safety rules (refusal cases — never deleted regardless of policy):
  - `experiments/golden/` and `experiments/manifests/` — committed to git
  - dirs with the `.STARTED` marker — in-progress runs
  - dirs whose mtime is in the future — clock-skew protection

Retention policies (intersected — a dir is prunable only if ALL active policies
say it is):
  - `older_than_days`: dir mtime is more than N days in the past
  - `keep_last`:       dir is not in the N most-recent under its manifest dir
  - `manifest`:        dir lives under the named manifest's subdir
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

_GIT_TRACKED_SUBDIRS: frozenset[str] = frozenset({"golden", "manifests"})
_STARTED_MARKER = ".STARTED"


def _is_protected(manifest_dir: Path) -> bool:
    return manifest_dir.name in _GIT_TRACKED_SUBDIRS


def _is_in_progress(run_dir: Path) -> bool:
    return (run_dir / _STARTED_MARKER).exists()


def _mtime(run_dir: Path) -> float:
    return run_dir.stat().st_mtime


def find_prunable_runs(
    experiments_root: Path,
    *,
    older_than_days: int | None = None,
    keep_last: int | None = None,
    manifest: str | None = None,
    now: datetime | None = None,
) -> list[Path]:
    """Return the run dirs that match the retention policy.

    All filters intersect — a dir is returned only if it satisfies every
    active rule. With no rules passed, returns an empty list (callers should
    supply at least one policy via the CLI defaults).
    """
    if not experiments_root.exists():
        return []
    now_ts = (now or datetime.now(timezone.utc)).timestamp()
    cutoff = now_ts - (older_than_days * 86400) if older_than_days is not None else None

    prunable: list[Path] = []
    for manifest_dir in sorted(experiments_root.iterdir()):
        if not manifest_dir.is_dir() or _is_protected(manifest_dir):
            continue
        if manifest is not None and manifest_dir.name != manifest:
            continue

        run_dirs = [d for d in manifest_dir.iterdir() if d.is_dir()]
        # Newest first by mtime so keep_last is unambiguous
        run_dirs.sort(key=_mtime, reverse=True)

        for idx, run_dir in enumerate(run_dirs):
            if _is_in_progress(run_dir):
                continue
            mt = _mtime(run_dir)
            if mt > now_ts:
                # Clock skew — refuse to act on future-dated dirs
                continue
            if cutoff is not None and mt >= cutoff:
                continue
            if keep_last is not None and idx < keep_last:
                continue
            if older_than_days is None and keep_last is None:
                # No active retention rule for this dir; nothing to prune
                continue
            prunable.append(run_dir)
    return prunable


def prune_runs(
    experiments_root: Path,
    *,
    older_than_days: int | None = None,
    keep_last: int | None = None,
    manifest: str | None = None,
    dry_run: bool = True,
    now: datetime | None = None,
) -> list[Path]:
    """Find prunable dirs and (optionally) delete them.

    Returns the list of dirs that were deleted (or would be deleted in
    dry-run mode), newest first.
    """
    targets = find_prunable_runs(
        experiments_root,
        older_than_days=older_than_days,
        keep_last=keep_last,
        manifest=manifest,
        now=now,
    )
    if dry_run:
        return targets
    for target in targets:
        shutil.rmtree(target)
    return targets
