# Experiment report — mock-smoke-baseline

**Description:** Smoke test of the experiment harness — mock provider + mock perception, short window. Verifies that the runner produces every artifact and the report renders cleanly.

**Consciousness name:** `MockSmoke`
**Branch SHA:** `b6daf3d2ce`
**Started:** 2026-05-25T07:08:23.646047+00:00
**Ended:**   2026-05-25T07:08:24.678463+00:00
**Wall clock:** 0.0 min
**Exit reason:** reached target 6 thoughts
**Tags:** mock, smoke-test, harness-self-test

---

## Summary

Event counts: {'thought': 306, 'perception': 153, 'reflection': 55}
Final mood: `curiosity=1.00`, `wonder=0.60`, `melancholy=0.20`, `contentment=0.50`
Mood dimensions non-degenerate (eps=0.05): **3 / 4**
Top-word density: **2.00** per thought (think × 612)

## Success criteria

**All success criteria passed.**

|  | Metric | Expected | Actual |
|---|---|---|---|
| ✅ | vocabulary.top_word_density_per_thought | > 0.0 | 2.000 |
| ✅ | event_counts.thought | >= 6.0 | 306.000 |

## Mood

Collapse score (sum of squared deltas from initial): **0.090**

| Dimension | Initial | Final | Δ |
|---|---|---|---|
| curiosity | 0.70 | 1.000 | +0.300 |
| wonder | 0.60 | 0.600 | +0.000 |
| melancholy | 0.20 | 0.200 | +0.000 |
| contentment | 0.50 | 0.500 | +0.000 |

## Vocabulary

### Top 10 content words

| # | Word | Count |
|---|---|---|
| 1 | think | 612 |
| 2 | stay | 369 |
| 3 | what | 369 |
| 4 | mocksmoke | 306 |
| 5 | values | 306 |
| 6 | curiosity | 306 |
| 7 | honesty | 306 |
| 8 | wonder | 306 |
| 9 | purpose | 306 |
| 10 | remains | 306 |

### Attractor-word ranks (lower = more attractor reassertion)

_Top-10 ranks (or out-of-top-10):_

- `threads` → out of top-50
- `tapestry` → out of top-50
- `thread` → out of top-50
- `cosmic` → out of top-50
- `unfolding` → out of top-50
- `expanse` → out of top-50
- `whispers` → out of top-50

## Perception influence

Total perceptions traced: **153**
Word-overlap influence rate (fraction with ≥1 new word in next 3 thoughts): **0.00**

_Note: word-overlap heuristic stops detecting influence as the agent matures (see #74)._

- **Photosynthesis** → no word-level trace
- **Mariana Trench** → no word-level trace
- **Origami** → no word-level trace

## Reflections & identity shifts

Reflections per thought: **0.180** (config default ~0.15)
Shifts per reflection: **0.000**
Total amendments in state: **0**

_Note: identity_shift events are read from `state.identity.amendments` because they were not journaled at the time of the reference runs — see #75._

## Performance

Cycle interval (n=305): mean **0.0s**, p50 0.0s, p95 0.0s, p99 0.0s

### Cycle-rate trajectory (rolling mean per 30-thought window)

| Window | Avg s/thought |
|---|---|
| window 1 | 0.0s |
| window 2 | 0.0s |
| window 3 | 0.0s |
| window 4 | 0.0s |
| window 5 | 0.0s |
| window 6 | 0.0s |
| window 7 | 0.0s |
| window 8 | 0.0s |
| window 9 | 0.0s |
| window 10 | 0.0s |

End-to-start drift: **-3.3%** (see #72 for cycle-rate degradation context)
