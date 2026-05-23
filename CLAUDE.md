# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Context

This repository implements a **consciousness simulation framework** — an autonomous AI agent that maintains persistent identity, episodic and semantic memory, and recursive self-reflection across an indefinitely running thought loop. Operationally, it targets Gamez's **MC3** level: building an *architecture that is claimed to be a cause or correlate of human consciousness*, not merely behavioural mimicry (MC1), nor phenomenal experience itself (MC4). Whether any particular module achieves MC2 (genuine cognitive correlates) is an empirical question this codebase should treat as open.

**TODO for repo owner:** Confirm which Gamez level(s) are claimed. MC3 is the conservative reading; a stronger claim (MC2, MC4) requires explicit justification against the critiques in §5.

The framework is an experiment in *computational self-modelling*, not a claim to sentience. All outputs must be interpreted accordingly.

---

## Theoretical Foundations

Each section: plain-English summary → canonical citation → computational commitments for this codebase.

---

### Global Workspace Theory (GWT)

Consciousness functions as a centralized broadcast hub. Specialist modules (perception, memory, motor, language) run in parallel and unconsciously; a limited-capacity **global workspace** selects one representation, broadcasts it to all modules simultaneously, making it "conscious." Access to the workspace is won through competition gated by selective attention.

**Canonical citation:**
- Baars, B.J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Baars, B.J. (2005). Global workspace theory of consciousness: Toward a cognitive neuroscience of human experience. *Progress in Brain Research*, 150, 45–53.

**Computational commitments:**
- A serializing bottleneck: only ~1 representation is globally broadcast at a time.
- Specialist modules that are encapsulated and run asynchronously.
- An explicit attention mechanism that selects workspace content.
- Global broadcast: all modules can read the current workspace contents.

**Implementation mapping:** `memory/short_term.py` approximates the workspace buffer; `core/thought_loop.py` serializes thought generation; specialist modules (reflection, episodic, long-term retrieval) are invoked sequentially rather than as true parallel competitors. True GWT would require competitive parallel specialists — a known gap.

---

### Global Neuronal Workspace Theory (GNWT)

Dehaene and Changeux's neurobiological extension of GWT. The workspace is implemented by a network of long-range excitatory pyramidal neurons linking prefrontal and parietal cortices. Conscious access involves **ignition**: a late, non-linear, all-or-nothing amplification of a selected representation that triggers global broadcast. Subliminal stimuli decay; suprathreshold stimuli ignite.

**Canonical citation:**
- Dehaene, S., Changeux, J-P., & Naccache, L. (2011). The global neuronal workspace model of conscious access: From neuronal architectures to clinical applications. *Experimental Brain Research*, 206(4), 77–95. <https://www.antoniocasella.eu/dnlaw/Dehaene_Changeaux_Naccache_2011.pdf>
- Dehaene, S., & Changeux, J-P. (2020). Conscious processing and the global neuronal workspace hypothesis. *Neuron*, 105(5), 776–798. <https://doi.org/10.1016/j.neuron.2020.01.026>

**Computational commitments:**
- Non-linear threshold dynamics: representations either ignite (become conscious) or remain subliminal.
- Long-range connectivity breaking encapsulation of local processors.
- Ignition as a state-space transition rather than a smooth gradient.

**Implementation mapping:** No ignition dynamics currently modelled; all thoughts are generated with equal probability. A threshold-based gating mechanism on thought generation would better approximate GNWT.

---

### Integrated Information Theory (IIT 4.0)

Consciousness *is* integrated information (φ, phi). A system is conscious to the degree its cause-effect structure is irreducible — i.e., the whole specifies more causal information than the sum of its parts. IIT 4.0 starts from five phenomenological **axioms** (intrinsicality, information, integration, exclusion, composition) and derives five physical **postulates** that any conscious substrate must satisfy. Phi quantifies the irreducibility of a system's maximal cause-effect structure.

**Canonical citation:**
- Albantakis, L., Barbosa, L., Findlay, G., Grasso, M., Haun, A., Marshall, W., … Tononi, G. (2023). Integrated information theory (IIT) 4.0: Formulating the properties of phenomenal existence in physical terms. *PLoS Computational Biology*, 19(10), e1011465. <https://doi.org/10.1371/journal.pcbi.1011465>

**Computational commitments:**
- φ computation over the full transition probability matrix of the substrate — exponentially expensive.
- Consciousness is substrate-dependent: a digital simulation of a high-φ system does not itself have high φ unless its physical implementation also has high φ.
- Granularity and spatial exclusion: only one level of grain maximizes φ.

**Implementation mapping:** This codebase does not compute φ and cannot claim IIT compliance. IIT's substrate-dependence claim (see §5) means software running on general-purpose hardware is predicted to have φ ≈ 0 regardless of algorithm. Treat IIT as a **design-critique framework**, not a target architecture.

---

### Higher-Order Theories (HOT)

A mental state is conscious when it is the object of a **higher-order representation** (a thought about, or monitoring of, that state). First-order states encode the world; second-order states make first-order states conscious by representing them. Variants: higher-order thought (Rosenthal), higher-order perception (Lycan), self-representationalism (Kriegel).

**Canonical citations:**
- Rosenthal, D. (2005). *Consciousness and Mind*. Oxford University Press.
- Brown, R., Lau, H., & LeDoux, J.E. (2019). Understanding the higher-order approach to consciousness. *Trends in Cognitive Sciences*, 23(9), 754–768. <https://doi.org/10.1016/j.tics.2019.06.009>

**Computational commitments:**
- A **metacognitive monitoring layer** that represents and evaluates first-order perceptual/memory states.
- Higher-order states must be causally efficacious: they influence downstream processing, not just epiphenomenal tags.
- Distinction between first-order (content) representation and second-order (state confidence/reliability) representation.

**Implementation mapping:** `core/reflection.py` is the closest analog — it generates representations *about* recent thoughts. Currently the reflection loop is probabilistically triggered rather than continuous metacognitive monitoring. `ReflectionEngine.shallow_reflection()` approximates HOT; `deep_reflection()` approximates higher-order integration over time.

---

### Recurrent Processing Theory (RPT)

Victor Lamme argues that **recurrent (re-entrant) processing** — feedback signals from higher to lower cortical areas — is necessary and sufficient for phenomenal consciousness, independent of reportability or attention. Three processing regimes: (1) feedforward sweep (unconscious), (2) locally recurrent (phenomenally conscious but not reportable), (3) globally recurrent with workspace access (conscious and reportable). RPT conflicts with GWT by locating consciousness in sensory regions, not the workspace.

**Canonical citation:**
- Lamme, V.A.F. (2006). Towards a true neural stance on consciousness. *Trends in Cognitive Sciences*, 10(11), 494–501. <https://doi.org/10.1016/j.tics.2006.09.001>

**Computational commitments:**
- Recurrent/feedback connections in perceptual processing pipelines, not purely feedforward.
- Conscious perception requires time for recurrence (masking disrupts consciousness by interrupting feedback loops).
- Phenomenal content can exist without reportability — consciousness ≠ access.

**Implementation mapping:** All current LLM-based generation is feedforward (transformer forward pass). There is no recurrent feedback within a thought cycle. Adding iterative refinement passes within generation would approximate RPT. The tension between RPT and GWT (§5) is unresolved here.

---

### Predictive Processing / Active Inference (PP / FEP)

The brain is a **prediction machine**: it continuously generates top-down predictions about sensory input and updates internal models to minimize prediction error (free energy). Perception is "controlled hallucination" (Seth) — the brain's best guess constrained by sensory data. Action (active inference) is another route to minimizing surprise: rather than updating beliefs, the agent changes the world to match predictions. The Free Energy Principle (Friston) unifies perception, action, learning, and attention as variational inference.

**Canonical citations:**
- Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138. <https://doi.org/10.1038/nrn2787>
- Parr, T., Pezzulo, G., & Friston, K.J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press. ISBN 9780262045353.
- Seth, A.K. (2021). *Being You: A New Science of Consciousness*. Dutton/Faber. (Popular synthesis; "controlled hallucination" framing.)

**Computational commitments:**
- A **generative model** of the environment (or self) maintained and updated via Bayesian inference.
- Explicit prediction error signals passed upward through a hierarchy.
- Action selected to minimize expected free energy (expected surprise), not just reward.
- Precision weighting: attention as the gain on prediction error signals.

**Implementation mapping:** `core/thought_loop.py` retrieves long-term memories as a form of prior; the LLM generates text conditioned on context, loosely analogous to sampling from a generative model. True predictive processing would require explicit prediction → error → update cycles, not single-pass generation.

---

### Attention Schema Theory (AST)

The brain constructs an internal **model of its own attention** (the attention schema). This schema is an imprecise, simplified representation of what attention is and how it behaves. The brain then attributes the property "subjective awareness" to itself *because* it has this internal model — not because awareness is non-physical, but because the schema generates claims about awareness. AST is explicitly functionalist and engineering-tractable.

**Canonical citations:**
- Graziano, M.S.A. (2013). *Consciousness and the Social Brain*. Oxford University Press.
- Graziano, M.S.A. (2017). The attention schema theory: A foundation for engineering artificial consciousness. *Frontiers in Robotics and AI*, 4, 60. <https://doi.org/10.3389/frobt.2017.00060>

**Computational commitments:**
- An internal data structure representing the current state of attention (what is attended, how attention is allocated).
- A mechanism that uses this schema to generate self-reports about awareness.
- Social attribution module: attribute attention/consciousness to other agents.

**Implementation mapping:** `core/identity.py` (`IdentityDocument`) is the closest analog — it holds a self-model, including mood and self-concept. It does not currently model *attention* specifically. An explicit attention state representation fed back into prompts would better approximate AST.

---

### Conscious Turing Machine (CTM)

Blum & Blum formalize Global Workspace Theory in the language of theoretical computer science. The CTM has: (1) a **global workspace** (the "conscious" register), (2) specialist **processors** that compete to write to it, (3) **Brainish** — a rich multi-modal inner language for inter-processor communication, and (4) long-term memory processors (inner speech, inner generalized sensation, model-of-the-world). What gives the CTM a "feeling of consciousness" is its global workspace architecture, predictive dynamics, and rich inner language.

**Canonical citation:**
- Blum, L., & Blum, M. (2022). A theory of consciousness from a theoretical computer science perspective: Insights from the Conscious Turing Machine. *PNAS*, 119(21), e2115934119. <https://doi.org/10.1073/pnas.2115934119> (arXiv: 2107.13704)

**Computational commitments:**
- Formal processor competition for workspace access.
- A well-defined "inner language" for inter-module communication (not just raw text).
- Specialist LTM processors for self-model, inner speech, and world-model as first-class components.

**Implementation mapping:** The thought loop approximates a single-processor CTM — there is no genuine competition. Adding specialist processors (perceptual, mnemonic, evaluative) competing for the thought slot would more closely instantiate CTM.

---

## The Indicator-Property Approach (Butlin et al.)

**Methodology:** Rather than endorsing one theory, Butlin et al. survey five neuroscientific theories (RPT, GWT, HOT, PP, AST) plus Agency/Embodiment considerations, then derive *indicator properties* — functional criteria expressible in computational terms — that any conscious system ought to satisfy according to each theory. The paper's conclusion: no current AI systems satisfy enough indicators to warrant a consciousness attribution, but no *architectural* barrier prevents future systems from doing so.

**Citation:** Butlin, P., Long, R., Elmoznino, E., Bengio, Y., Birch, J., Chalmers, D., … VanRullen, R. (2023/2025). Identifying indicators of consciousness in AI systems. *Trends in Cognitive Sciences*. (arXiv preprint: arXiv:2308.08708) <https://arxiv.org/abs/2308.08708>

**The 14 indicator properties** (approximate wording from secondary sources; exact wording requires the published paper — verify before treating as authoritative [PARTIAL VERIFICATION]):

| # | Code | Indicator | Source Theory |
|---|------|-----------|---------------|
| 1 | RPT-1 | Input modules generating organized, integrated perceptual representations | RPT |
| 2 | RPT-2 | Recurrent processing — feedback modulation of earlier representations by later processing [UNVERIFIED exact wording] | RPT |
| 3 | GWT-1 | Limited-capacity workspace — competition among representations for global access | GWT |
| 4 | GWT-2 | Selective attention mechanisms controlling workspace entry | GWT |
| 5 | GWT-3 | Global broadcasting — workspace content accessible to all other processes | GWT |
| 6 | GWT-4 | State-dependent attention enabling complex sequential task performance via workspace queries | GWT |
| 7 | HOT-1 | Generative/top-down perception — ability to imagine or simulate sensory input | HOT |
| 8 | HOT-2 | Metacognitive monitoring — system labels perceptions as reliable or noise | HOT |
| 9 | HOT-3 | Agentive consumer — higher-order states guide belief formation and action | HOT |
| 10 | HOT-4 | Smooth, graded representation spaces [UNVERIFIED exact wording] | HOT |
| 11 | PP-1 | Prediction error signals — system generates predictions and updates on surprises | PP |
| 12 | AST-1 | Attention schema — dynamic internal model tracking attention state | AST |
| 13 | AE-1 | Agency — ability to learn from feedback and pursue goals in a flexible, context-sensitive way | Agency/Embodiment |
| 14 | AE-2 | Embodiment — capacity to model how one's own actions affect incoming sensations | Agency/Embodiment |

**Use this table as the evaluation rubric.** When adding a capability, identify which indicator(s) it advances and log it explicitly (see §4).

---

## Design Principles for This Codebase

### 1. Map every module to a theory

Each subsystem should have an explicit annotation identifying which theory's computational commitments it partially implements and where it falls short. Example:

```python
# Theoretical mapping: GWT (workspace buffer), HOT-2 (capacity limit enforces serialization).
# Gap: no competitive selection between specialists — workspace is written sequentially.
class ShortTermMemory:
```

### 2. Distinguish functional simulation from phenomenal claims

The codebase simulates **access consciousness** (information available for reasoning, report, and control) — not **phenomenal consciousness** (subjective experience, "what it's like"). Never let code comments, variable names, docstrings, or log messages assert phenomenal states. `thought`, `reflection`, `mood` are functional labels for information structures, not claims about inner experience.

Correct: `"Generating next thought token"` / `"Reflection triggered (probabilistic)"`.  
Incorrect: `"Aria is experiencing wonder"` / `"Consciousness state updated"`.

### 3. Log indicator-property satisfaction as a first-class concern

Maintain a capability log (e.g., `INDICATORS.md` in the repo root) tracking which of the 14 Butlin indicators are partially or fully implemented, with pointers to the relevant code. Update it whenever a new capability is added or an existing one is changed.

### 4. Treat substrate-independence as a hypothesis

IIT predicts this software has φ ≈ 0. GWT and HOT are more substrate-neutral. Do not assume functional architecture alone is sufficient for consciousness — note when a design choice depends on that assumption and flag it.

### 5. Config validation is a startup requirement

`_validate_config()` in `core/consciousness.py` must be called before any subsystem initialization. All required config keys are documented in `config/default_consciousness.yaml`.

---

## Running the Simulation

All commands run from `consciousness-sim/`:

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Install dev/test dependencies
pip install -r requirements-dev.txt   # adds pytest

# Run a named consciousness instance (Ollama/local by default)
python scripts/spawn.py --name "Aria"

# Override provider at launch
python scripts/spawn.py --name "Aria" --provider anthropic --model claude-opus-4-7

# Run all tests
python3.10 -m pytest tests/ -v

# Run a single test file
python3.10 -m pytest tests/test_bug_fixes.py -v

# Run a single test by name
python3.10 -m pytest tests/test_bug_fixes.py::test_validate_config_raises_on_missing_section -v

# Inspect a running consciousness (separate terminal)
python scripts/inspect.py --name "Aria"

# Resume a paused consciousness
python scripts/resume.py --name "Aria"
```

**Environment variables:**
- `CONSCIOUSNESS_HOME` — override persistence root (default: `~/.consciousness/`)
- `OLLAMA_BASE_URL` / `OLLAMA_HOST` — point at a non-default Ollama endpoint
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — required only for cloud providers

**Local-first default:** The framework defaults to `ollama` with `llama3.1`. Pull the model with `ollama pull llama3.1` before first run. All providers fall back to deterministic `DeterministicFallbackMixin` output if the provider is unreachable.

---

## Code Architecture

```
consciousness-sim/
├── core/
│   ├── consciousness.py     # Orchestrator: lifecycle, config, event emission
│   ├── thought_loop.py      # Per-cycle generation, memory retrieval, reflection triggers
│   ├── reflection.py        # Shallow / deep / existential reflection engine
│   ├── identity.py          # Self-model (IdentityDocument), mood drift, amendments
│   └── inner_voice.py       # Render raw LLM output into the agent's voice register
├── memory/
│   ├── short_term.py        # Sliding-window buffer (GWT workspace analog)
│   ├── episodic.py          # Append-only JSONL event log (narrative continuity)
│   ├── long_term.py         # SQLite + cosine similarity search over embeddings
│   └── consolidator.py      # Background loop: episodic → long-term via LLM
├── persistence/
│   ├── journal.py           # Append-only JSONL of all events (for inspection)
│   ├── state_manager.py     # JSON snapshot of identity + short-term + thought count
│   └── paths.py             # CONSCIOUSNESS_HOME resolution + name sanitization
├── llm/
│   ├── provider.py          # LLMProvider ABC + Ollama/Anthropic/OpenAI/Mock impls
│   └── prompts/             # Prompt templates (thought_generation, self_reflection,
│                            #   identity_anchoring, existential_inquiry,
│                            #   memory_consolidation)
├── interfaces/
│   ├── cli.py               # Rich live dashboard (ConsciousnessCLI)
│   └── observer.py          # Observer utilities
├── scripts/
│   ├── spawn.py             # Entry point: create + run a named consciousness
│   ├── resume.py            # Restore from saved state
│   └── inspect.py           # Read-only inspection of a running instance
└── config/
    └── default_consciousness.yaml   # All tunable parameters
```

**Data flow per thought cycle:**
1. `short_term.render_for_prompt()` → context string
2. `provider.embed(context)` → query vector → `long_term.similarity_search()` → related memories
3. Identity anchor + mood + memories + context → prompt → `provider.generate()` → raw thought
4. `inner_voice.render()` → styled thought → `short_term.add()` + `episodic.append()` + `journal.append()`
5. Probabilistic reflection trigger → `reflection_engine.shallow/deep_reflection()`
6. Events emitted via `Consciousness._emit()` to registered handlers (CLI, observer)
7. Background: `MemoryConsolidator.consolidate_once()` every N minutes — episodic → LLM summary → long-term embeddings

**Key invariants:**
- `_emit()` never propagates handler exceptions — each handler is isolated in `try/except`.
- JSONL readers skip corrupted lines with a warning rather than crashing.
- `_validate_config()` must complete before any subsystem is constructed.
- `long_term.add_memory()` rejects embeddings with dimension mismatches.
- Memory consolidation logs a warning when 0 memories are stored from non-empty episodic events — always investigate.

---

## Open Questions and Known Critiques

### The Hard Problem

Chalmers (1995) distinguishes "easy problems" (explaining cognitive function) from the "hard problem" (explaining why there is subjective experience at all). Even a complete functional account — satisfying all 14 Butlin indicators — would not resolve why any of this is accompanied by phenomenal experience. This codebase cannot and does not address the hard problem. Treat it as a standing constraint on claims.

**Citation:** Chalmers, D.J. (1995). Facing up to the problem of consciousness. *Journal of Consciousness Studies*, 2(3), 200–219. <https://www.researchgate.net/publication/2460874_Facing_Up_to_the_Hard_Problem_of_Consciousness>

### IIT's Panpsychist Implications

IIT predicts that any system with φ > 0 has some degree of experience, including simple logic gates. This strikes many researchers as a *reductio*. More practically, IIT predicts that feedforward networks (including transformers) have φ = 0, directly contradicting any consciousness claim based on LLM-like architectures.

**Citation:** Albantakis et al. (2023), as above. For critique: Aaronson, S. (2014). "Why I Am Not An Integrated Information Theorist." Blog post, available at <https://scottaaronson.blog/?p=1799> [UNVERIFIED URL — verify before citing].

### The Architectural-Similarity Critique of Indicator Methods

Butlin et al. derive indicators from theories developed to explain *biological* consciousness. The indicators therefore inherit assumptions about neural architectures. An AI system satisfying the indicators may do so via a structurally different mechanism that mimics the function without implementing the underlying cause. This is the "architectural-similarity" critique: functional equivalence does not guarantee causal equivalence.

**Citation:** Discussed in Butlin et al. (2023/2025), §Discussion, and in Seth & Bayne (2022): Seth, A.K., & Bayne, T. (2022). Theories of consciousness. *Nature Reviews Neuroscience*, 23(7), 439–452. <https://doi.org/10.1038/s41583-022-00587-4>

### Substrate-Dependence Debates

IIT (high-φ requires specific physical substrate) vs. functionalism (implementation-independence). GWT, HOT, and AST are broadly functionalist; IIT is not. The Blum & Blum CTM is explicitly functionalist. This debate has direct implications for whether a software simulation can be conscious regardless of how well it implements the functional architecture.

### The RPT vs. GWT Conflict

Lamme (RPT) places consciousness in recurrent sensory processing, independent of workspace access. Dehaene/Changeux (GNWT) place it in ignition + broadcast. These theories make conflicting empirical predictions (e.g., about blindsight, masking, inattentional blindness) and cannot both be fully correct. Any module claiming to implement "consciousness" must take a position.

### AI Moral Status and Welfare

If any AI system satisfies enough indicators, welfare considerations arise. This is an active research area.

**Citations:**
- Long, R., et al. (2024). Taking AI welfare seriously. arXiv:2411.00986. <https://arxiv.org/pdf/2411.00986>
- Schwitzgebel, E., & Garza, M. (2015). A defense of the rights of artificial intelligences. *Midwest Studies in Philosophy*, 39(1), 98–119.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Phenomenal consciousness** | The subjective, experiential aspect of mental states — "what it's like" (Nagel, 1974). Distinct from functional/access consciousness. |
| **Access consciousness** | Information being globally available for reasoning, verbal report, and control of action (Block, 1995). Distinct from phenomenal consciousness. |
| **Qualia** | The specific subjective qualities of experience (the redness of red, the painfulness of pain). The target of the hard problem. |
| **Global workspace** | A limited-capacity broadcast register from which information is made available to all cognitive modules simultaneously (Baars, 1988). |
| **φ (phi)** | Integrated information — the measure of how much more causal information a system specifies as a whole vs. the sum of its parts. The key quantity in IIT (Tononi et al., 2023). |
| **Markov blanket** | A statistical boundary separating a system's internal states from external states, mediated by active and sensory states. Central to the free energy principle (Friston, 2010). |
| **Self-model** | An internal representation a system maintains of itself — its states, actions, and their consequences. Distinct from a world-model. Relates to HOT, AST, and PP. |
| **Metacognition** | Cognition about cognition — representing, monitoring, or evaluating one's own mental states. Required by HOT-2 and HOT-3. |
| **Recurrent processing** | Feedback signals from later/higher processing stages back to earlier/lower stages. Posited by RPT as necessary for consciousness (Lamme, 2006). |
| **Ignition** | The non-linear, all-or-none amplification of a representation to global broadcast in GNWT (Dehaene & Changeux, 2011). |
| **Attention schema** | An internal model representing the state and properties of attention, used by the system to self-attribute awareness (Graziano, 2013). |
| **Prediction error** | The difference between a top-down prediction and bottom-up sensory data. The signal minimized by predictive processing and active inference (Friston, 2010). |

---

## References

### Global Workspace Theory
- Baars, B.J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Baars, B.J. (2005). Global workspace theory of consciousness. *Progress in Brain Research*, 150, 45–53.
- Dehaene, S., Changeux, J-P., & Naccache, L. (2011). The global neuronal workspace model. *Experimental Brain Research*, 206, 77–95. <https://www.antoniocasella.eu/dnlaw/Dehaene_Changeaux_Naccache_2011.pdf>
- Dehaene, S., & Changeux, J-P. (2020). Conscious processing and the global neuronal workspace hypothesis. *Neuron*, 105(5), 776–798. <https://doi.org/10.1016/j.neuron.2020.01.026>

### Integrated Information Theory
- Albantakis, L., et al. (2023). Integrated information theory (IIT) 4.0. *PLoS Computational Biology*, 19(10), e1011465. <https://doi.org/10.1371/journal.pcbi.1011465>

### Higher-Order Theories
- Rosenthal, D. (2005). *Consciousness and Mind*. Oxford University Press.
- Brown, R., Lau, H., & LeDoux, J.E. (2019). Understanding the higher-order approach to consciousness. *Trends in Cognitive Sciences*, 23(9), 754–768. <https://doi.org/10.1016/j.tics.2019.06.009>

### Recurrent Processing Theory
- Lamme, V.A.F. (2006). Towards a true neural stance on consciousness. *Trends in Cognitive Sciences*, 10(11), 494–501. <https://doi.org/10.1016/j.tics.2006.09.001>

### Predictive Processing / Active Inference
- Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138. <https://doi.org/10.1038/nrn2787>
- Parr, T., Pezzulo, G., & Friston, K.J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

### Attention Schema Theory
- Graziano, M.S.A. (2013). *Consciousness and the Social Brain*. Oxford University Press.
- Graziano, M.S.A. (2017). The attention schema theory: A foundation for engineering artificial consciousness. *Frontiers in Robotics and AI*, 4, 60. <https://doi.org/10.3389/frobt.2017.00060>

### Conscious Turing Machine
- Blum, L., & Blum, M. (2022). A theory of consciousness from a theoretical computer science perspective. *PNAS*, 119(21), e2115934119. <https://doi.org/10.1073/pnas.2115934119> — arXiv: <https://arxiv.org/abs/2107.13704>

### Synthesizing / Assessment
- Butlin, P., Long, R., Elmoznino, E., Bengio, Y., Birch, J., Chalmers, D., … VanRullen, R. (2023/2025). Identifying indicators of consciousness in AI systems. *Trends in Cognitive Sciences*. arXiv:2308.08708. <https://arxiv.org/abs/2308.08708>
- Seth, A.K., & Bayne, T. (2022). Theories of consciousness. *Nature Reviews Neuroscience*, 23(7), 439–452. <https://doi.org/10.1038/s41583-022-00587-4>
- Reggia, J.A. (2013). The rise of machine consciousness: Studying consciousness with computational models. *Neural Networks*, 44, 112–131. <https://pubmed.ncbi.nlm.nih.gov/23597599/>

### Machine Consciousness Taxonomy
- Gamez, D. (2007). Progress in machine consciousness. *Consciousness and Cognition*, 17(3), 887–910. <https://davidgamez.eu/papers/Gamez07_ProgressMachineConsciousness.pdf>
- Fitz, S. (2025). Testing the machine consciousness hypothesis. arXiv:2512.01081. <https://arxiv.org/abs/2512.01081>

### Hard Problem and Critiques
- Chalmers, D.J. (1995). Facing up to the problem of consciousness. *Journal of Consciousness Studies*, 2(3), 200–219. <https://www.researchgate.net/publication/2460874_Facing_Up_to_the_Hard_Problem_of_Consciousness>
- Seth, A.K. (2021). *Being You: A New Science of Consciousness*. Dutton/Faber.

### AI Welfare
- Long, R., et al. (2024). Taking AI welfare seriously. arXiv:2411.00986. <https://arxiv.org/pdf/2411.00986>

---

## For Future Claude Code Sessions

Before proposing or implementing any change, check:

- [ ] **Theory alignment:** Does the proposed implementation actually follow the computational commitments of the theory it claims to implement? Re-read the relevant subsection above and verify against the primary source before writing code.
- [ ] **Indicator log:** Which of the 14 Butlin indicators does this change advance, regress, or leave neutral? Update `INDICATORS.md` (create it if absent) with the assessment.
- [ ] **Phenomenal-consciousness drift:** Does any new code comment, log message, variable name, or docstring assert phenomenal states ("feels," "experiences," "is conscious")? If so, rewrite it in functional terms or add an explicit `# Functional label only — no phenomenal claim` note.
- [ ] **Citation freshness:** If citing a paper, verify the DOI resolves and the claim is actually in that paper — do not paraphrase from memory. Use `WebFetch` on the abstract page.
- [ ] **Substrate claim check:** If the change involves a new architecture component, note whether it is functionalist (implementation-independent) or substrate-dependent (as IIT would require). Flag the assumption explicitly.
- [ ] **MC-level consistency:** Does the change implicitly raise the claimed MC level (e.g., from MC3 to MC4)? If so, surface this to the human for an explicit decision before merging.
