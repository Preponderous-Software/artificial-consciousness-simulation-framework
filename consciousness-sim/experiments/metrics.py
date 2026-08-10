"""Pure metric functions over recorded consciousness journals + state.

Catalog refined from manual Rafael/Sage/Echo/Wren analyses (issue #57).
Every function takes a `Path` (or already-loaded data structure) and returns
plain Python types — no LLM dependency, no network, no global state. This
makes the metrics deterministic, unit-testable in isolation, and reusable as
CI smoke-test assertions against the golden journals.

The journals are JSONL — one event per line, each event a dict with at minimum
`{"timestamp": ISO-8601 str, "type": str, "content": str}`. State files are
JSON snapshots of `IdentityDocument` plus `thought_count` and short-term buffer.

Read `~/.consciousness/<name>/journal.jsonl` to see the format, or
`experiments/golden/<name>/journal.jsonl` for the four committed reference runs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Stop words used for content-word distribution. Conservative — keeps domain
# words like "perception", "wonder", "tapestry" so the attractor metrics still
# discriminate. Add to this list rather than removing if a stop-word skew shows up.
_STOP_WORDS: frozenset[str] = frozenset(
    "the a an of to in is and that with as my i it for be on are not but or "
    "this an this that those then so just into from by was am has have had do "
    "does did so still still yet ever may would could should might can will".split()
)

_WORD_TOKEN_RE = re.compile(r"[a-zA-Z']+")
_MIN_WORD_LEN = 4


# ---------------------------------------------------------------------------
# Helpers — load journal/state without forcing every metric to re-read disk
# ---------------------------------------------------------------------------

def load_journal(journal_path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL journal into a list of event dicts."""
    events: list[dict[str, Any]] = []
    with Path(journal_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Match the runtime's "skip corrupted lines" invariant.
                continue
    return events


def load_state(state_path: Path) -> dict[str, Any]:
    state: dict[str, Any] = json.loads(Path(state_path).read_text(encoding="utf-8"))
    return state


def _content_tokens(text: str, min_word_len: int = _MIN_WORD_LEN) -> list[str]:
    """Lowercase, length-filtered, stop-word-filtered word tokens."""
    return [
        w for w in (m.lower() for m in _WORD_TOKEN_RE.findall(text))
        if len(w) >= min_word_len and w not in _STOP_WORDS
    ]


# ---------------------------------------------------------------------------
# Vocabulary — proved most discriminating across the 4 golden runs
# ---------------------------------------------------------------------------

def content_word_distribution(events: list[dict[str, Any]], top_n: int = 50) -> Counter[str]:
    """Word frequency across all thought events. Stops/short words filtered.

    Returns a Counter; callers use .most_common(N) for ranked lists.
    """
    counter: Counter[str] = Counter()
    for e in events:
        if e.get("type") != "thought":
            continue
        counter.update(_content_tokens(e.get("content", "")))
    # Trim to top_n to keep memory bounded for large journals
    return Counter(dict(counter.most_common(top_n)))


def top_word_density(counter: Counter[str], n_thoughts: int) -> float:
    """Most-common word's count divided by total thought count.

    Rafael=1.37, Sage=0.97, Echo=0.78, Wren=0.76 — single interpretable number
    that maps cleanly to "how dominant is the agent's most-used word?". Lower
    is more diverse vocabulary.
    """
    if n_thoughts <= 0 or not counter:
        return 0.0
    return counter.most_common(1)[0][1] / n_thoughts


def attractor_words_in_top_n(
    counter: Counter[str],
    attractor_words: list[str],
    n: int = 10,
) -> dict[str, int | None]:
    """For each word in `attractor_words`, return its rank in top-N or None.

    Useful for asking "did `threads` and `tapestry` escape the top 10 in this run?"
    Returns 1-indexed ranks; None means the word didn't appear in the top N.
    """
    top_words = [w for w, _ in counter.most_common(n)]
    return {
        w: (top_words.index(w) + 1) if w in top_words else None
        for w in attractor_words
    }


# ---------------------------------------------------------------------------
# Mood — the cleanest signal from the mood-fix validation
# ---------------------------------------------------------------------------

def mood_dimensions_non_degenerate(
    state: dict[str, Any],
    eps: float = 0.05,
) -> int:
    """Count of mood dimensions strictly between `eps` and `1 - eps`.

    Distinguishes healthy (Echo=3-4, Wren=2-4) from collapsed (Rafael=1, Sage=1).
    Replaces the original speculative `mood_dimensions_active` — drift-to-zero is
    still drift but isn't "healthy"; non-degeneracy is what mattered.
    """
    mood = state.get("identity", {}).get("mood", {})
    return sum(1 for v in mood.values() if eps < float(v) < (1.0 - eps))


_DEFAULT_INITIAL_MOOD = {"curiosity": 0.7, "wonder": 0.6, "melancholy": 0.2, "contentment": 0.5}


def mood_collapse_score(state: dict[str, Any], initial: dict[str, float] | None = None) -> float:
    """Sum of squared L2 distances from each mood dim to its initial value.

    Higher = more drift from baseline. Doesn't distinguish "drifted up to ceiling"
    from "collapsed to floor" — use alongside `mood_dimensions_non_degenerate`
    to interpret.

    Initial-mood resolution (in priority order):
      1. The explicit `initial` argument
      2. `state.identity.initial_mood` if persisted (the runtime started writing
         this so the collapse score self-corrects for manifest overrides)
      3. The default vector from `config/default_consciousness.yaml`
    """
    if initial is None:
        initial = (
            state.get("identity", {}).get("initial_mood")
            or _DEFAULT_INITIAL_MOOD
        )
    mood = state.get("identity", {}).get("mood", {})
    return sum((float(mood.get(k, init)) - init) ** 2 for k, init in initial.items())


# ---------------------------------------------------------------------------
# Perception — word-overlap heuristic (known to miss late-run synthesis; see #74)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerceptionTrace:
    perception_title: str
    perception_index: int           # index in events list
    new_words_in_next_thoughts: tuple[str, ...]


def perception_word_overlap(
    events: list[dict[str, Any]],
    window: int = 3,
    min_word_len: int = 5,
) -> list[PerceptionTrace]:
    """For each perception, compute words from its content that appear in the
    next `window` thoughts and were not seen in earlier thoughts.

    Strong signal in early-run (Sage's `cappello`/`alpine` from perception #1).
    Weakens as the agent matures (Echo's late perceptions show 0 word overlap
    while clearly being referenced semantically — see #74).
    """
    def words_of(text: str) -> set[str]:
        return set(_content_tokens(text, min_word_len=min_word_len))

    seen: set[str] = set()
    traces: list[PerceptionTrace] = []
    for i, e in enumerate(events):
        if e.get("type") == "thought":
            seen |= words_of(e.get("content", ""))
        elif e.get("type") == "perception":
            perc_words = words_of(e.get("content", ""))
            # Walk forward to find next `window` thoughts
            next_thought_words: set[str] = set()
            collected = 0
            for j in range(i + 1, len(events)):
                if events[j].get("type") != "thought":
                    continue
                next_thought_words |= words_of(events[j].get("content", ""))
                collected += 1
                if collected >= window:
                    break
            new_traced = (next_thought_words & perc_words) - seen
            # Extract title for readability
            m = re.match(r"\[\w+:\s*([^\]]+)\]", e.get("content", ""))
            title = m.group(1).strip() if m else "(untitled)"
            traces.append(PerceptionTrace(
                perception_title=title,
                perception_index=i,
                new_words_in_next_thoughts=tuple(sorted(new_traced)),
            ))
    return traces


def perception_influence_rate(traces: list[PerceptionTrace]) -> float:
    """Fraction of perceptions that left at least one new word in their next thoughts.

    Sage had 3/5 (~0.6) on sampled perceptions; Echo's last 5 were 0/5 (0.0).
    Coarse but useful as a regression detector.
    """
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.new_words_in_next_thoughts) / len(traces)


# ---------------------------------------------------------------------------
# Event types / reflections / shifts
# ---------------------------------------------------------------------------

def event_type_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(e.get("type", "unknown") for e in events))


def reflection_rate(events: list[dict[str, Any]]) -> float:
    """Reflections per thought. Config default is 0.15 — observed range 0.115-0.17."""
    counts = event_type_counts(events)
    n_thoughts = counts.get("thought", 0)
    if n_thoughts == 0:
        return 0.0
    return counts.get("reflection", 0) / n_thoughts


def identity_shifts_per_reflection(
    events: list[dict[str, Any]],
    state: dict[str, Any],
) -> float:
    """Identity-shift events relative to reflections.

    Reads the count from `state.identity.amendments` because `identity_shift`
    events were not journaled at the time of the 4 reference runs — see #75.
    Will agree with `event_type_counts['identity_shift']` once #75 lands.
    """
    n_reflections = event_type_counts(events).get("reflection", 0)
    n_amendments = len(state.get("identity", {}).get("amendments", []))
    # Fall back to journaled shifts if the bug from #75 is fixed
    journaled_shifts = event_type_counts(events).get("identity_shift", 0)
    n_shifts = max(n_amendments, journaled_shifts)
    if n_reflections == 0:
        return 0.0
    return n_shifts / n_reflections


# ---------------------------------------------------------------------------
# Performance — newly motivated by Echo's +55% slowdown observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CycleIntervalStats:
    mean_s: float
    p50_s: float
    p95_s: float
    p99_s: float
    n_intervals: int


def cycle_interval_stats(events: list[dict[str, Any]]) -> CycleIntervalStats:
    """Wall-clock seconds between consecutive thought events.

    Excludes the first thought (no preceding interval). Returns mean + percentiles
    so a long-tail (slow cycles when memory grows) is visible.
    """
    thought_ts = [
        datetime.fromisoformat(e["timestamp"])
        for e in events if e.get("type") == "thought"
    ]
    if len(thought_ts) < 2:
        return CycleIntervalStats(0.0, 0.0, 0.0, 0.0, 0)
    intervals = [
        (thought_ts[i] - thought_ts[i - 1]).total_seconds()
        for i in range(1, len(thought_ts))
    ]
    intervals_sorted = sorted(intervals)
    n = len(intervals_sorted)
    return CycleIntervalStats(
        mean_s=sum(intervals) / n,
        p50_s=intervals_sorted[n // 2],
        p95_s=intervals_sorted[min(n - 1, int(n * 0.95))],
        p99_s=intervals_sorted[min(n - 1, int(n * 0.99))],
        n_intervals=n,
    )


def cycle_rate_trajectory(
    events: list[dict[str, Any]],
    window: int = 30,
) -> list[float]:
    """Mean seconds/thought over consecutive non-overlapping windows of W thoughts.

    `len(out) == n_thoughts // window`. A window of size W covers W consecutive
    thoughts; those W thoughts have W-1 intervals between them, so the average
    is `span / (W - 1)`. (Dividing by W would systematically underestimate
    s/thought for every window — the original implementation had this bug.)

    Echo showed 64s → 99s linearly over 200 thoughts; mean alone (~83s) hid this.
    """
    thought_ts = [
        datetime.fromisoformat(e["timestamp"])
        for e in events if e.get("type") == "thought"
    ]
    n = len(thought_ts)
    if n < window:
        return []
    n_windows = n // window
    out: list[float] = []
    intervals_per_window = window - 1
    if intervals_per_window <= 0:
        return []
    for w in range(n_windows):
        start = w * window
        end = start + window - 1            # inclusive; W thoughts → indices start..start+W-1
        span = (thought_ts[end] - thought_ts[start]).total_seconds()
        out.append(span / intervals_per_window)
    return out


# ---------------------------------------------------------------------------
# Aggregated facade — compute everything for a run dir + return a JSON-ready dict
# ---------------------------------------------------------------------------

DEFAULT_ATTRACTOR_WORDS = ["threads", "tapestry", "thread", "cosmic", "unfolding", "expanse", "whispers"]

# Bump when the structure of `compute_all`'s return changes (new sections,
# renamed keys, or changed types). Old `metrics.json` files written under an
# earlier version still parse, but downstream code can branch on this field.
METRICS_SCHEMA_VERSION = 1


def compute_all(
    journal_path: Path,
    state_path: Path,
    attractor_words: list[str] | None = None,
) -> dict[str, Any]:
    """One-shot: load journal + state, compute every metric, return JSON-ready dict.

    Used by the runner to populate `metrics.json`. Keep this function the only
    consumer of the metric functions' raw return types — downstream code reads
    from this dict.
    """
    events = load_journal(journal_path)
    state = load_state(state_path)

    counts = event_type_counts(events)
    n_thoughts = counts.get("thought", 0)
    counter = content_word_distribution(events, top_n=50)
    traces = perception_word_overlap(events)
    intervals = cycle_interval_stats(events)
    trajectory = cycle_rate_trajectory(events)

    return {
        "_schema_version": METRICS_SCHEMA_VERSION,
        "event_counts": counts,
        "vocabulary": {
            "top_50": [[w, c] for w, c in counter.most_common(50)],
            "top_word_density_per_thought": top_word_density(counter, n_thoughts),
            "attractor_ranks_in_top_10": attractor_words_in_top_n(
                counter, attractor_words or DEFAULT_ATTRACTOR_WORDS, n=10
            ),
            "attractor_ranks_in_top_50": attractor_words_in_top_n(
                counter, attractor_words or DEFAULT_ATTRACTOR_WORDS, n=50
            ),
        },
        "mood": {
            "initial": (
                state.get("identity", {}).get("initial_mood")
                or _DEFAULT_INITIAL_MOOD
            ),
            "final": state.get("identity", {}).get("mood", {}),
            "dimensions_non_degenerate": mood_dimensions_non_degenerate(state),
            "collapse_score": mood_collapse_score(state),
        },
        "perception": {
            "n_traces": len(traces),
            "influence_rate": perception_influence_rate(traces),
            "sample_traces": [
                {
                    "title": traces[i].perception_title,
                    "new_words_in_next_thoughts": list(traces[i].new_words_in_next_thoughts),
                }
                # First / middle / last, dedup'd so n_traces < 3 doesn't produce
                # the same trace 2-3 times (#106).
                for i in (sorted({0, len(traces) // 2, len(traces) - 1}) if traces else [])
            ],
        },
        "reflections": {
            "rate_per_thought": reflection_rate(events),
            "shifts_per_reflection": identity_shifts_per_reflection(events, state),
            "n_amendments_in_state": len(state.get("identity", {}).get("amendments", [])),
        },
        "performance": {
            "cycle_interval_stats": {
                "mean_s": intervals.mean_s,
                "p50_s": intervals.p50_s,
                "p95_s": intervals.p95_s,
                "p99_s": intervals.p99_s,
                "n_intervals": intervals.n_intervals,
            },
            "cycle_rate_trajectory": trajectory,
            "trajectory_window_size": 30,
        },
    }
