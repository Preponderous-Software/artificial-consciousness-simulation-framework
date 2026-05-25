"""Experiment harness — reproducible consciousness simulation runs + analysis.

Implements issue #57: codify the manual Rafael/Sage/Echo/Wren analysis loop into
deterministic Python tools that ship as the `scripts/experiment.py` CLI.

Layer 1 — deterministic library (this package):
  - manifest.py: YAML-loaded experiment specs
  - metrics.py:  pure functions over journal + state (no LLM dependency)
  - runner.py:   spawn → wait → stop → copy → metrics → report
  - golden/:     reference journals from the four canonical runs

Layer 2 (Claude skill) and Layer 3 (CI integration) are deferred follow-ups
per the issue's phased plan.
"""
