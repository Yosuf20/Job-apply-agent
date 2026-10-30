# AGENTS.md: Instructions for AI coding assistants

## Project
A multi-agent job application assistant in Python. It discovers jobs, scores
them against my profile, tailors a role-specific resume PDF, and (with my
approval) fills out applications. Built as a portfolio project that I also use
myself.

Read `docs/architecture.md` for design and `docs/roadmap.md` for the current step.

## Working rules
1. **One feature at a time.** Only implement what the current prompt asks for.
   Do not scaffold or stub future steps.
2. **Plan first.** Show a short plan before making changes. Wait for approval on
   anything touching more than a few files.
3. **Keep it simple.** Prefer plain Python and small functions over heavy
   frameworks. Ask before adding a new dependency.
4. **Explain non-obvious choices** in comments or `docs/decisions.md`. I must be
   able to explain every part of this code in an interview.
5. **Tests required.** Every feature ships with pytest tests. Run them and report results.

## Code standards
- Python 3.11+, full type hints, Pydantic v2 for all data passed between stages.
- Format and lint with `ruff`.
- Source in `src/jobagent/`, tests in `tests/`.
- Config comes from `config.py` (env vars). Never hardcode keys, paths, or model names.

## Architecture rules (do not violate)
- **All LLM calls go through `src/jobagent/llm/client.py`.** No direct SDK calls elsewhere.
- **Agents are small:** one job, one input model, one output model.
- **Plain code stays plain:** discovery, rendering, and tracking are NOT agents.
- **No invented facts.** Generated resume content may only use information from
  `data/profile.yaml`. Every generated bullet must trace to a source Bullet `id`.
- **Human approval is required** before any application is submitted.
- **No scraping or automating LinkedIn or Indeed.** Use public ATS endpoints
  (Greenhouse, Lever, Ashby) and official APIs only.

## Security and privacy
- `data/profile.yaml`, `.env`, and `data/output/` are gitignored. Never commit them.
- Never print or log API keys or personal data.
- Use fake data in tests and examples.

## Commands
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format .
```

## Git workflow
- Commit at the end of every completed step, once tests and `ruff` pass.
  One step = one commit (or a few small ones). Never commit failing code.
- Use Conventional Commit messages, e.g. `feat: add Pydantic models and example profile`,
  `test: cover profile validation`, `docs: update roadmap`, `fix: handle empty job description`.
- Before committing, run `git status` and confirm no secrets or personal data are
  staged (`.env`, `data/profile.yaml`, `data/output/`, `*.db`).
- Do NOT push to a remote, force-push, rewrite history, or create branches unless I ask.
- After committing, show me the commit message and `git log --oneline -3`.

## Definition of done for each step
- [ ] Code matches the prompt scope, nothing extra
- [ ] Tests pass
- [ ] `ruff` is clean
- [ ] `docs/roadmap.md` checkbox updated
- [ ] Any design choice recorded in `docs/decisions.md`
- [ ] Changes committed with a clear message (no secrets staged)
