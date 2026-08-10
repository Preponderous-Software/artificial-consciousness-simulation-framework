# Contributing

Thanks for your interest in this project. It's research/experimental code exploring
computational models of consciousness-adjacent architectures — see the root
[README.md](./README.md) and [CLAUDE.md](./CLAUDE.md) for the project's scope and
theoretical grounding before diving in.

## Dev setup

The full project lives under [`consciousness-sim/`](./consciousness-sim). Follow its
[Quickstart](./consciousness-sim/README.md#quickstart) to install dependencies and run
an instance locally. In short:

```bash
cd consciousness-sim
python3 -m venv .venv   # requires Python 3.11+
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

The framework defaults to a local Ollama model (`llama3.2:3b`), so you can run and test
it without any cloud API key. Alternatively, run `./setup.sh` from the repo root to
automate model install + Python dependency setup.

## Running tests

```bash
cd consciousness-sim
pytest -q tests
mypy .
```

CI (`.github/workflows/`) runs both of the above plus an experiment smoke test against a
mock LLM provider, so it works the same for a fork with no external credentials. `mypy`
is a blocking step, so a type error fails the build; its settings live in
`pyproject.toml`'s `[tool.mypy]` section, which runs `strict = true` over first-party
code with `tests/` excluded. New code therefore needs full annotations on every
function it defines.

## Branch and commit conventions

Branches are prefixed by kind: `feat/`, `fix/`, `docs/`, `infra/`, `test/`, `perf/`,
`chore/`, `hardening/`. Commit subjects follow `type(scope): summary`, e.g.:

```
fix(memory): bound the long-term SQLite store with a row-count retention policy
```

Reference the relevant issue number in the PR description (`Closes #123`) where one
exists.

## Design principles to keep in mind

- **Functional claims only.** Code comments, log messages, and variable names must not
  assert phenomenal states ("Aria is experiencing wonder"). Describe information
  processing in functional terms instead. See CLAUDE.md, "Distinguish functional
  simulation from phenomenal claims."
- **Map new modules to theory.** If you add or change a subsystem tied to one of the
  consciousness theories in CLAUDE.md, note which theoretical commitments it implements
  and where it falls short, and update [INDICATORS.md](./INDICATORS.md) if it advances,
  regresses, or leaves neutral one of the 14 Butlin et al. indicators.
- **Cite carefully.** Verify a DOI/URL resolves and actually supports the claim before
  citing it — don't paraphrase from memory.

## Reporting issues

Open a GitHub issue with steps to reproduce (for bugs) or the motivating use case (for
feature requests). Security-relevant reports should follow [SECURITY.md](./SECURITY.md)
instead of a public issue.
