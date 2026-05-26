"""Side-by-side comparison of two recorded experiment runs.

Reads each run's `metrics.json` + `state.json` + `journal.jsonl` and produces
a markdown report covering: vocabulary density delta, mood deltas across all
four dimensions, attractor-word rank changes, perception influence rate,
top-word table, sample-thought pair, and success-criteria pass/fail.

Pure-function layer — no LLM, no network. The Claude skill
`.claude/skills/compare-experiments.md` calls this for the data layer and
then writes the narrative interpretation on top.

Phase 2 of issue #57. Deferred Phase-2 work: aggregate-across-replicates
(mean / stddev), and the LLM-as-judge perception-influence metric noted
in #74.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.metrics import compute_all


@dataclass(frozen=True)
class RunRef:
    """Loaded artifacts for a single recorded run."""

    run_dir: Path
    manifest: dict[str, Any]
    meta: dict[str, Any]
    metrics: dict[str, Any]
    state: dict[str, Any]
    label: str

    @property
    def thoughts(self) -> int:
        return int(self.metrics.get("event_counts", {}).get("thought", 0))


def _read_yaml(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _read_json(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_run(run_dir: Path, label: str | None = None) -> RunRef:
    """Load the four artifacts a comparison reads from a recorded run."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    manifest = _read_yaml(run_dir / "manifest.yaml")
    meta = _read_yaml(run_dir / "meta.yaml")
    metrics = _read_json(run_dir / "metrics.json")
    state = _read_json(run_dir / "state.json")

    # Tolerate "journal-and-state-only" directories — including the golden
    # reference dirs, which intentionally don't ship metrics.json. If the
    # journal exists, compute metrics on the fly.
    if not metrics:
        journal_path = run_dir / "journal.jsonl"
        state_path = run_dir / "state.json"
        if journal_path.exists() and state_path.exists():
            metrics = compute_all(journal_path, state_path)
        else:
            raise ValueError(
                f"{run_dir} has no metrics.json and no journal.jsonl + state.json "
                "from which to compute them — is this actually a run directory?"
            )

    return RunRef(
        run_dir=run_dir,
        manifest=manifest,
        meta=meta,
        metrics=metrics,
        state=state,
        label=label or manifest.get("name") or run_dir.name,
    )


# ---------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------

def _safe_get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Walk a dotted path through nested dicts, returning default on miss."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def compute_diff(a: RunRef, b: RunRef) -> dict[str, Any]:
    """Compute the delta metrics that the markdown renderer formats.

    Always reports `b - a` (so "increase" means b had more than a).
    """
    diff: dict[str, Any] = {"a_label": a.label, "b_label": b.label}

    # Thoughts / events
    diff["n_thoughts"] = {"a": a.thoughts, "b": b.thoughts, "delta": b.thoughts - a.thoughts}

    # Top-word density
    da = _safe_get(a.metrics, "vocabulary.top_word_density_per_thought", 0.0) or 0.0
    db = _safe_get(b.metrics, "vocabulary.top_word_density_per_thought", 0.0) or 0.0
    diff["top_word_density"] = {"a": float(da), "b": float(db), "delta": float(db) - float(da)}

    # Mood (per dimension)
    mood_a = _safe_get(a.metrics, "mood.final", {}) or {}
    mood_b = _safe_get(b.metrics, "mood.final", {}) or {}
    dims = sorted(set(mood_a) | set(mood_b))
    diff["mood"] = {
        d: {
            "a": float(mood_a.get(d, 0.0)),
            "b": float(mood_b.get(d, 0.0)),
            "delta": float(mood_b.get(d, 0.0)) - float(mood_a.get(d, 0.0)),
        }
        for d in dims
    }
    diff["mood_non_degenerate"] = {
        "a": _safe_get(a.metrics, "mood.dimensions_non_degenerate", 0),
        "b": _safe_get(b.metrics, "mood.dimensions_non_degenerate", 0),
    }

    # Attractor ranks (top-10)
    ranks_a = _safe_get(a.metrics, "vocabulary.attractor_ranks_in_top_10", {}) or {}
    ranks_b = _safe_get(b.metrics, "vocabulary.attractor_ranks_in_top_10", {}) or {}
    diff["attractor_ranks"] = {
        word: {"a": ranks_a.get(word), "b": ranks_b.get(word)}
        for word in sorted(set(ranks_a) | set(ranks_b))
    }

    # Perception influence rate
    diff["perception_influence_rate"] = {
        "a": float(_safe_get(a.metrics, "perception.influence_rate", 0.0) or 0.0),
        "b": float(_safe_get(b.metrics, "perception.influence_rate", 0.0) or 0.0),
    }

    # Reflection rate
    diff["reflection_rate"] = {
        "a": float(_safe_get(a.metrics, "reflections.rate_per_thought", 0.0) or 0.0),
        "b": float(_safe_get(b.metrics, "reflections.rate_per_thought", 0.0) or 0.0),
    }

    return diff


# ---------------------------------------------------------------------------
# Sample thoughts — read from journals directly (metrics don't carry them)
# ---------------------------------------------------------------------------

def sample_thoughts(run: RunRef, k: int = 3) -> list[str]:
    """Pick k thoughts evenly across the journal (first, middle, last for k=3)."""
    journal = run.run_dir / "journal.jsonl"
    if not journal.exists():
        return []
    thoughts: list[str] = []
    with journal.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thought":
                thoughts.append(event.get("content", ""))
    n = len(thoughts)
    if n == 0:
        return []
    if n <= k:
        return thoughts
    # Evenly-spaced indices including first and last
    indices = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    return [thoughts[i] for i in indices]


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _fmt_signed(x: float, digits: int = 2) -> str:
    """Format a number with explicit sign for delta columns."""
    return f"{x:+.{digits}f}"


def _fmt_rank(r: int | None) -> str:
    return f"#{r}" if r is not None else "—"


def render_comparison(a: RunRef, b: RunRef, k_samples: int = 3) -> str:
    """Render a side-by-side markdown comparison of two runs.

    The output is intentionally dense and metric-first — the Claude skill
    is responsible for narrative interpretation. Sections:
      1. Summary table (thoughts, density, mood non-degenerate)
      2. Mood per dimension (before / after / delta)
      3. Vocabulary: top-word density delta + attractor-rank table
      4. Perception influence rate
      5. Sample thoughts (k per run, evenly spaced)
    """
    diff = compute_diff(a, b)
    lines: list[str] = [
        f"# Comparison — `{a.label}` ↔ `{b.label}`",
        "",
        f"- **A:** `{a.run_dir}` ({a.thoughts} thoughts; "
        f"branch `{a.meta.get('branch_sha', '?')}`; exit `{a.meta.get('exit_reason', '?')}`)",
        f"- **B:** `{b.run_dir}` ({b.thoughts} thoughts; "
        f"branch `{b.meta.get('branch_sha', '?')}`; exit `{b.meta.get('exit_reason', '?')}`)",
        "",
        "## Headline metrics",
        "",
        "| Metric | A | B | Δ (B − A) |",
        "|---|---|---|---|",
        f"| Thoughts | {a.thoughts} | {b.thoughts} | "
        f"{_fmt_signed(diff['n_thoughts']['delta'], 0)} |",
        f"| Top-word density (per thought) | "
        f"{diff['top_word_density']['a']:.2f} | "
        f"{diff['top_word_density']['b']:.2f} | "
        f"{_fmt_signed(diff['top_word_density']['delta'])} |",
        f"| Mood dimensions non-degenerate | "
        f"{diff['mood_non_degenerate']['a']} | "
        f"{diff['mood_non_degenerate']['b']} | "
        f"{_fmt_signed(diff['mood_non_degenerate']['b'] - diff['mood_non_degenerate']['a'], 0)} |",
        f"| Reflection rate (refl/thought) | "
        f"{diff['reflection_rate']['a']:.3f} | "
        f"{diff['reflection_rate']['b']:.3f} | "
        f"{_fmt_signed(diff['reflection_rate']['b'] - diff['reflection_rate']['a'], 3)} |",
        f"| Perception influence rate | "
        f"{diff['perception_influence_rate']['a']:.2f} | "
        f"{diff['perception_influence_rate']['b']:.2f} | "
        f"{_fmt_signed(diff['perception_influence_rate']['b'] - diff['perception_influence_rate']['a'])} |",
        "",
    ]

    # Mood per dimension
    if diff["mood"]:
        lines += [
            "## Mood per dimension",
            "",
            "| Dimension | A | B | Δ |",
            "|---|---|---|---|",
        ]
        for dim, vals in diff["mood"].items():
            lines.append(
                f"| {dim} | {vals['a']:.3f} | {vals['b']:.3f} | "
                f"{_fmt_signed(vals['delta'], 3)} |"
            )
        lines.append("")

    # Top-word table (side-by-side top 10 of each run)
    top_a = (_safe_get(a.metrics, "vocabulary.top_50", []) or [])[:10]
    top_b = (_safe_get(b.metrics, "vocabulary.top_50", []) or [])[:10]
    if top_a or top_b:
        lines += [
            "## Top 10 content words (side-by-side)",
            "",
            f"| # | A — `{a.label}` | count | B — `{b.label}` | count |",
            "|---|---|---|---|---|",
        ]
        rows = max(len(top_a), len(top_b))
        for i in range(rows):
            wa, ca = (top_a[i] if i < len(top_a) else ["", ""])
            wb, cb = (top_b[i] if i < len(top_b) else ["", ""])
            lines.append(f"| {i+1} | `{wa}` | {ca} | `{wb}` | {cb} |")
        lines.append("")

    # Success criteria — show each run's pass/fail against its own manifest's criteria
    crit_a = (a.manifest or {}).get("success_criteria") or []
    crit_b = (b.manifest or {}).get("success_criteria") or []
    if crit_a or crit_b:
        from experiments.manifest import SuccessCriterion, evaluate_success_criterion
        lines += ["## Success criteria status", ""]

        def _eval(crit_list, metrics):
            evals = []
            for raw in crit_list:
                try:
                    c = SuccessCriterion.model_validate(raw)
                except Exception:
                    continue
                passed, actual = evaluate_success_criterion(c, metrics)
                evals.append((c, passed, actual))
            return evals

        for label, evals in (
            (f"A — `{a.label}`", _eval(crit_a, a.metrics)),
            (f"B — `{b.label}`", _eval(crit_b, b.metrics)),
        ):
            lines.append(f"**{label}**")
            lines.append("")
            if not evals:
                lines.append("_(no criteria defined or all malformed)_")
                lines.append("")
                continue
            lines += ["| | Metric | Expected | Actual |", "|---|---|---|---|"]
            for c, passed, actual in evals:
                mark = "✅" if passed else "❌"
                actual_s = f"{actual:.3f}" if actual is not None else "missing"
                lines.append(f"| {mark} | `{c.kind}` | `{c.op} {c.value}` | {actual_s} |")
            lines.append("")

    # Attractor ranks
    if diff["attractor_ranks"]:
        lines += [
            "## Cosmic-attractor word ranks (top-10)",
            "",
            "_Lower rank = more attractor reassertion. `—` = out of top-10._",
            "",
            "| Word | A | B | Movement |",
            "|---|---|---|---|",
        ]
        for word, ranks in diff["attractor_ranks"].items():
            ra, rb = ranks["a"], ranks["b"]
            if ra is None and rb is None:
                movement = "both out"
            elif ra is None and rb is not None:
                movement = f"NEW entry at #{rb}"
            elif ra is not None and rb is None:
                movement = f"escaped from #{ra}"
            else:
                delta = rb - ra
                if delta == 0:
                    movement = "same"
                else:
                    direction = "↓ better (more escape)" if delta > 0 else "↑ worse (deeper attractor)"
                    movement = f"{delta:+d} {direction}"
            lines.append(f"| `{word}` | {_fmt_rank(ra)} | {_fmt_rank(rb)} | {movement} |")
        lines.append("")

    # Sample thoughts
    samples_a = sample_thoughts(a, k=k_samples)
    samples_b = sample_thoughts(b, k=k_samples)
    if samples_a or samples_b:
        lines += [f"## Sample thoughts (k={k_samples}, evenly spaced)", ""]
        for i in range(max(len(samples_a), len(samples_b))):
            lines.append(f"### Position {i+1}")
            lines.append("")
            if i < len(samples_a):
                lines.append(f"**A** — {a.label}:")
                lines.append("")
                lines.append("> " + textwrap.fill(samples_a[i], 95).replace("\n", "\n> "))
                lines.append("")
            if i < len(samples_b):
                lines.append(f"**B** — {b.label}:")
                lines.append("")
                lines.append("> " + textwrap.fill(samples_b[i], 95).replace("\n", "\n> "))
                lines.append("")

    return "\n".join(lines)


def compare_runs(run_a: Path, run_b: Path, k_samples: int = 3) -> str:
    """Top-level entry point: load two run dirs, return the markdown comparison."""
    a = load_run(run_a)
    b = load_run(run_b)
    return render_comparison(a, b, k_samples=k_samples)
