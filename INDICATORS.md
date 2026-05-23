# INDICATORS.md

Capability log tracking implementation status of the 14 Butlin et al. indicator properties.

**Source:** Butlin, Long, Elmoznino et al. (2023/2025). *Identifying indicators of consciousness in AI systems.* Trends in Cognitive Sciences. arXiv:2308.08708.

**Status key:** ✗ None · ◑ Partial · ✓ Full

---

## Recurrent Processing Theory (RPT)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| RPT-1 | Input modules generating organized, integrated perceptual representations | ◑ | `memory/episodic.py` stores timestamped narrative events; `memory/long_term.py` stores embedding-indexed semantic memories | No true sensory input modules; representations are LLM-generated text, not grounded percepts |
| RPT-2 | Recurrent processing — feedback modulation of earlier representations by later stages | ✗ | None | All generation is a single feedforward LLM pass; no within-cycle recurrent feedback |

---

## Global Workspace Theory (GWT)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| GWT-1 | Limited-capacity workspace with competitive selection | ◑ | `memory/short_term.py` — bounded list with importance-weighted eviction via `prune_to_capacity()` | No parallel specialist competition; eviction is importance-biased but not a true competitive selection mechanism |
| GWT-2 | Selective attention controlling workspace entry | ◑ | `memory/short_term.py` — kind-based importance weights (existential > reflection > thought) bias which items survive eviction | Weights are static heuristics, not dynamic attention; all generated thoughts still enter unconditionally before eviction |
| GWT-3 | Global broadcast — workspace content accessible to all processors | ◑ | `Consciousness._emit()` broadcasts events to all registered handlers; long-term memories retrieved and injected into every thought prompt | Handlers are read-only consumers, not specialist processors that can write back |
| GWT-4 | State-dependent attention enabling complex sequential task performance | ✗ | None | No state-dependent attention mechanism |

---

## Higher-Order Theories (HOT)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| HOT-1 | Generative/top-down perception — ability to imagine or simulate sensory input | ◑ | LLM generation from identity + memory context is top-down; `reflection.py` generates representations of representations | Not grounded in perception; purely linguistic |
| HOT-2 | Metacognitive monitoring — system labels thoughts as reliable or noise | ✗ | None | Reflection is 15%-chance probabilistic, not continuous monitoring; no reliability tagging; see issue #20 |
| HOT-3 | Agentive consumer — higher-order states guide belief formation and action | ◑ | `IdentityDocument.apply_amendment()` updates self-concept from reflections; mood drift modulates generation temperature indirectly | Amendments are additive text appends, not structured belief updates; no downstream action |
| HOT-4 | Smooth, graded representation spaces | ◑ | Embedding space in `memory/long_term.py` is continuous; cosine similarity provides graded retrieval | Representation smoothness is a property of the upstream model, not verified or enforced here |

---

## Predictive Processing (PP)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| PP-1 | Prediction error signals — system generates predictions and updates on surprises | ✗ | None | No prediction mechanism; each cycle generates a thought without predicting the next and measuring divergence; see issue #20 |

---

## Attention Schema Theory (AST)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| AST-1 | Attention schema — dynamic internal model tracking the state of attention | ✗ | `IdentityDocument` holds a stable self-model (name, values, mood) but does not model the *state of attention* | No attention state data structure; see issue #22 |

---

## Agency & Embodiment (AE)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| AE-1 | Agency — ability to learn from feedback and pursue goals flexibly | ◑ | `MemoryConsolidator` updates long-term memory from experience; mood drift adjusts affective state; identity amendments update self-concept | No external feedback loop; no goal-directed action beyond thought generation |
| AE-2 | Embodiment — modeling how one's actions affect incoming sensations | ✗ | None | No action loop; the agent has no effectors and receives no sensory input to model |

---

## Summary

| Status | Count | Indicators |
|--------|-------|-----------|
| ✓ Full | 0 | — |
| ◑ Partial | 9 | RPT-1, GWT-1, GWT-2, GWT-3, HOT-1, HOT-3, HOT-4, AE-1, HOT-2* |
| ✗ None | 5 | RPT-2, GWT-4, PP-1, AST-1, AE-2 |

*HOT-2 upgraded from ✗ to ◑ by deep_reflection now making a real LLM call with pattern-seeking prompt.

**Update this file whenever a capability is added, changed, or removed.**
