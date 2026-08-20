# Golden reference dataset

Four canonical consciousness runs frozen on disk. Used by:

1. **Regression tests** — `tests/test_experiment_metrics.py` asserts known values against these journals so future metric changes that silently alter behavior get caught.
2. **Bug fixture corpus** — issues #63, #70, #73, #75, #76 all needed real LLM output as fixtures; these journals supply them permanently.
3. **Empirical comparison** — new runs compare against these as the established baselines for the configurations they ran under.

## The four runs

| Instance | Config | N=thoughts | Final mood `(c, w, m, ct)` |
|---|---|---|---|
| `Rafael` | no perception, no mood fix | 83 | `(0.99, 0.10, 0.0, 0.0)` collapsed |
| `Sage` | perception #54, no mood fix | 76 | `(0.98, 0.0, 0.01, 0.0)` collapsed harder |
| `Echo` | perception + mood fix #66 + AST-1 #61 | 200 | `(0.99, 0.98, 0.34, 1.00)` healthy |
| `Wren` | identical config to Echo (N=2 replication) | 200 | `(1.00, 0.85, 0.58, 1.00)` healthy |

See `~/.consciousness/Echo/RUN_REPORT.md` and `~/.consciousness/Wren/RUN_REPORT.md`
for the original narrative analyses these journals support.

## Files

- `journal.jsonl` — append-only event log (thoughts, reflections, perceptions, memory_stored)
- `state.json` — final identity + mood + amendments snapshot
- `_smoke_expected.json` (this directory's top level, not per-instance) — pinned
  metrics snapshot for `experiments/manifests/mock-smoke-baseline.yaml`, used by
  `scripts/experiment.py check-smoke` (`experiments/regression.py`) as a CI
  regression gate (#87). Distinct from the four runs above: it isn't a full
  journal/state pair, just a handful of metric values that MockProvider's
  deterministic output makes predictable across runs. Each entry is either a
  **point pin** (a bare number, matched within `1e-6`) or a **range pin** (an
  object carrying `min` and/or `max`, plus an optional `note` since JSON has no
  comments). Range pins exist for metrics that only approach a stable value
  asymptotically — `mood.final.curiosity` climbs toward its equilibrium and
  never reaches it in finitely many cycles, so a point pin at that equilibrium
  passed on fast CI runners and failed on slower hosts (#173).

`episodic.jsonl` is intentionally excluded — the metrics functions don't read it
and including it ~doubles the size of this directory.
