# INDICATORS.md

Capability log tracking implementation status of the 14 Butlin et al. indicator properties.

**Source:** Butlin, Long, Elmoznino et al. (2023/2025). *Identifying indicators of consciousness in AI systems.* Trends in Cognitive Sciences. arXiv:2308.08708.

**Status key:** ✗ None · ◑ Partial · ✓ Full

---

## Recurrent Processing Theory (RPT)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| RPT-1 | Input modules generating organized, integrated perceptual representations | ◑ | `llm/perception.py` — `PerceptionProvider` ABC + `WikipediaPerception` injects external-world snippets every N cycles (issue #53); persisted to journal/episodic and surfaced in the prompt under "SOMETHING YOU JUST ENCOUNTERED" | Perception is text-only (no other modalities); a single fetch per cycle is not a continuous stream; no within-modality structure beyond raw text |
| RPT-2 | Recurrent processing — feedback modulation of earlier representations by later stages | ✗ | None | All generation is a single feedforward LLM pass; no within-cycle recurrent feedback |

---

## Global Workspace Theory (GWT)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| GWT-1 | Limited-capacity workspace with competitive selection | ◑ | `memory/short_term.py` — bounded list with importance-weighted eviction via `prune_to_capacity()`; `llm/perception.py` adds a second specialist (perception, importance 1.5) competing with the LLM generator (thought, importance 0.5–1.0 dynamically) for buffer slots (issue #53); thought importance is now set by `MetacognitiveMonitor` (HOT-2 synergy: noisy thoughts evict sooner) | Specialists are still serially invoked rather than running in parallel; competition is only via post-hoc eviction weight, not pre-write contention |
| GWT-2 | Selective attention controlling workspace entry | ◑ | `memory/short_term.py` — kind-based importance weights (existential > reflection > perception > thought) bias which items survive eviction; thought importance is now dynamic (1.0/0.75/0.5 for high/uncertain/noise labels) set by `core/metacognition.py:MetacognitiveMonitor` each cycle | All generated thoughts still enter the buffer unconditionally before eviction; no pre-entry gating |
| GWT-3 | Global broadcast — workspace content accessible to all processors | ◑ | `Consciousness._emit()` broadcasts events to all registered handlers; long-term memories retrieved and injected into every thought prompt. The retrievable set is now bounded: `memory/long_term.py` caps the store at `memory.long_term_max_rows` rows (default 2000, 0 disables) and evicts the lowest-`importance_score`/oldest rows on insert and at open (#135), so recall breadth stays constant across an indefinite run instead of growing without bound. | Handlers are read-only consumers, not specialist processors that can write back. Retention is capacity-triggered rather than a consolidation/decay process — evicted memories are deleted outright, not merged or abstracted, so a run past the bound loses low-importance early history rather than compressing it. Status is unchanged by #135: the bound is a resource policy, not a broadcast-mechanism advance. |
| GWT-4 | State-dependent attention enabling complex sequential task performance | ✗ | None | No state-dependent attention mechanism |

---

## Higher-Order Theories (HOT)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| HOT-1 | Generative/top-down perception — ability to imagine or simulate sensory input | ◑ | LLM generation from identity + memory context is top-down; `reflection.py` generates representations of representations | Not grounded in perception; purely linguistic |
| HOT-2 | Metacognitive monitoring — system labels thoughts as reliable or noise | ◑ | `core/metacognition.py:MetacognitiveMonitor` runs after every thought generation — labels each thought 'high', 'uncertain', or 'noise' based on lexical overlap with recent *thought-kind* items in the workspace buffer (issue #20). Label adjusts workspace importance (noisy thoughts evict sooner; GWT-2 synergy) and boosts reflection trigger probability when quality is low (causal efficacy). `core/reflection.py:deep_reflection` supplements with LLM-based meta-reasoning over recent thoughts; `shallow_reflection` feeds the last `_MAX_TRACKED_OPENINGS` reflection opening sentences back into the prompt with an instruction to diverge, so reflections generate new meta-observations rather than re-casting prior ones (issue #118). `core/inner_voice.py:InnerVoice.scrub_reflection` strips leading LLM meta-preambles/markdown headers and rejects second-person instructional drift before reflection/existential text reaches the workspace, preventing that boilerplate from lowering the monitor's signal (issue #132). | Scoring is lexical only — semantic repetition (paraphrasing without shared vocabulary) is not detected; no LLM coherence check per thought; no social attribution. Anti-repetition acts on the opening sentence only and relies on the model honoring the instruction — it does not enforce a measured overlap threshold. See issue #74 for semantic upgrade path. The meta-scaffolding scrub only checks the *leading* preamble/header and *leading* second-person address — mid-text scaffolding or a second-person switch further into the text is not caught. |
| HOT-3 | Agentive consumer — higher-order states guide belief formation and action | ◑ | `IdentityDocument.apply_amendment()` updates self-concept from reflections and skips a verbatim-substring or token-Jaccard-near-duplicate of recent amendments (issue #133) so a repeatedly-triggered amendment no longer fills the self-concept budget with one repeated phrase; mood drift applies trigger-driven drift *and* homeostatic reversion to `initial_mood` additively each cycle (#62 + #119), with the default `homeostasis_rate` tuned (0.1 → 0.3, issue #134) so continuously-triggered traits equilibrate at `baseline + drift_rate/homeostasis_rate` *below* 1.0 rather than saturating at the ceiling; the mood vector is injected into the thought-generation prompt as `{mood_vector}` text and influences generation content via the LLM; `llm.temperature` and `llm.max_tokens` are read from config and passed to `provider.generate()` each cycle (PR #80); reflection text is scrubbed of meta-scaffolding (issue #132) before `consciousness.py`'s amendment guard evaluates it, so fewer genuine self-revisions are spuriously rejected as LLM boilerplate | Amendments are additive text appends, not structured belief updates; no downstream action; mood does not modulate generation temperature — temperature is a static config value, not dynamically adjusted from affective state; a caller-supplied `homeostasis_rate`/`drift_rate`/`initial` combination that violates the `drift_rate/homeostasis_rate < 1 - initial` constraint (e.g. via a custom config) can still saturate — the constraint is documented, not enforced by `_validate_config()` |
| HOT-4 | Smooth, graded representation spaces | ◑ | Embedding space in `memory/long_term.py` is continuous; cosine similarity provides graded retrieval | Representation smoothness is a property of the upstream model, not verified or enforced here |

---

## Predictive Processing (PP)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| PP-1 | Prediction error signals — system generates predictions and updates on surprises | ◑ | `core/thought_loop.py` — at end of each cycle, `_predicted_theme` is set to the extracted theme of the current thought (continuity prior: predict the same topic persists). At the start of the next cycle, after the thought is generated, `prediction_error` is computed: `1.0` if the predicted theme word is absent from the actual thought, `0.0` if present. High prediction error boosts reflection probability (`+0.20`), making the agent self-examine unexpected divergences. `prediction_error` is returned in `ThoughtCycleResult` and included in the per-cycle perf log. | Generative model is a trivial continuity prior — no learning from errors over time (full PP would revise the prior based on accumulated prediction history); no hierarchical prediction structure; prediction operates only on theme-level vocabulary, not semantic content. |

---

## Attention Schema Theory (AST)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| AST-1 | Attention schema — dynamic internal model tracking the state of attention | ◑ | `core/identity.py` — `AttentionSchema` dataclass tracks `focus` (dominant cycle kind: introspection/memory/reflection/perception/existential), `theme` (first content word extracted from cycle output), `salience` (0–1), and `history` (last 10 foci). On every successful cycle, `ThoughtLoop.run_cycle()` calls `AttentionSchema.update()` which appends to history, sets the new focus/theme, and resets salience to 1.0. On a failed cycle, the outer loop in `consciousness.py` calls `decay_only()` instead, fading salience by 0.1 — so a long failure burst produces smooth decay rather than freeze-then-collapse (#120). Rendered into the identity anchor prompt via `anchor_payload()["attention_state"]` so each thought is conditioned on the prior cycle's attention state. Persisted in `state.json` with backward-compat for old snapshots (issue #22). | Focus is derived from discrete event type, not a learned or competitive allocation model; theme extraction is a simple keyword heuristic; salience decay is linear; no social attribution of attention to other agents (Graziano 2013, §4) |

---

## Agency & Embodiment (AE)

| Code | Indicator | Status | Implementation | Gap |
|------|-----------|--------|----------------|-----|
| AE-1 | Agency — ability to learn from feedback and pursue goals flexibly | ◑ | `MemoryConsolidator` updates long-term memory from experience; mood drift adjusts affective state from thought + perception text (issue #62 fix) with additive homeostatic reversion (#119) — continuously-triggered traits equilibrate at `baseline + drift_rate/homeostasis_rate` rather than saturating; identity amendments update self-concept | No external feedback loop; no goal-directed action beyond thought generation |
| AE-2 | Embodiment — modeling how one's actions affect incoming sensations | ✗ | None | Perception is now received (`llm/perception.py`, issue #53), but it is unsolicited — the agent does not yet *choose* what to perceive, so there is no action→sensation loop to model. Issue #53 Phase 3 (let the agent query for the next perception) is the first concrete step toward AE-2 |

---

## Summary

| Status | Count | Indicators |
|--------|-------|-----------|
| ✓ Full | 0 | — |
| ◑ Partial | 11 | RPT-1, GWT-1, GWT-2, GWT-3, HOT-1, HOT-2, HOT-3, HOT-4, PP-1, AE-1, AST-1 |
| ✗ None | 3 | RPT-2, GWT-4, AE-2 |

**Update this file whenever a capability is added, changed, or removed.**
