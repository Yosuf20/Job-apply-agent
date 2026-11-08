# AGENTS.md: Instructions for AI coding assistants

## Project
A multi-agent job application assistant in Python. It discovers jobs, scores
them against my profile, tailors a role-specific resume PDF, and (with my
approval) fills out applications. Built as a portfolio project that I also use
myself.

Read `docs/architecture.md` for design and `docs/roadmap.md` for the current step.

## Tech stack
Only add a dependency in the step that needs it, and ask before adding anything not listed here.

| Area | Choice | Introduced in step |
|---|---|---|
| Language | Python 3.11+ | 1 |
| Package / env management | uv | 1 |
| Data models | Pydantic v2 | 1 |
| Config | python-dotenv, PyYAML | 1 |
| Testing / linting | pytest, ruff | 1 |
| LLM provider | Configured via LLM_PROVIDER (anthropic | openai | gemini); SDKs only, no LangChain wrappers | 2 |
| HTTP (job discovery) | httpx | 3 |
| Database | SQLite via SQLAlchemy | 6 |
| Resume templates | Jinja2 (HTML + CSS) | 9 |
| PDF rendering | Playwright (`page.pdf()`) | 9 |
| PDF text check | pypdf or pdfplumber | 9 |
| Agent orchestration | LangGraph (graph, retries, human-in-the-loop interrupts, checkpointing) | 10 |
| Tracing / observability | LangSmith (optional) plus own logging of cost and latency | 10 |
| CLI | Typer | 10 |
| UI | Streamlit or FastAPI (decide at step 11) | 11 |
| Browser automation | Playwright | 12 |

## LangGraph usage rules
- Each agent stays a plain, typed Python function or class (Pydantic in, Pydantic out)
  that works and is tested WITHOUT LangGraph. LangGraph only wires them together in Step 10.
- Define one graph state model (Pydantic) in `orchestrator.py`. Nodes read and update it.
- The Tailor <-> Validator loop is a conditional edge with a max retry count.
- Human approval gates use LangGraph interrupts with a checkpointer (SQLite).
- Do not pull in the full LangChain framework. Use `langgraph` plus the single LLM client in `llm/client.py`.
  Ask before adding any other `langchain-*` package.

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

## Commands (uv)
Package and environment management is done with **uv**. Do not use pip or edit
dependency lists by hand.
```bash
uv sync                        # create/refresh .venv from pyproject.toml + uv.lock
uv add <package>               # add a runtime dependency
uv add --dev <package>         # add a dev dependency
uv run pytest                  # run tests
uv run ruff check . && uv run ruff format .
```
Commit `uv.lock` along with `pyproject.toml` whenever dependencies change.

## Git workflow
- Commit each file separately, with its own Conventional Commit message
  that describes that file's change (e.g. `feat: add SHA-256 hashing helpers`,
  `test: cover tracker status transitions`, `docs: record tracker decisions`).
- Order commits so dependencies come first (a module before the code that
  imports it, source before its tests, docs last). Every commit should
  leave the project passing `uv run pytest` and `uv run ruff check .`.
- Exception: `pyproject.toml` and `uv.lock` change together, in one commit
  (e.g. `chore: add sqlalchemy dependency`).
- Stage by explicit path only (`git add path/to/file`). Never use
  `git add .` or `git add -A`.
- Before each commit, run `git status` and confirm no secrets or personal
  data are staged (`.env`, `data/profile.yaml`, `data/sources.yaml`,
  `data/output/`, `*.db`, `evals/matcher_labeled.json`).
- Do NOT push, force-push, rewrite history, or create branches unless I ask.
- Do not add trailers or co-author lines to commit messages.
- After the last commit of a step, show me `git log --oneline` for the step.

## Definition of done for each step
- [ ] Code matches the prompt scope, nothing extra
- [ ] Tests pass
- [ ] `ruff` is clean
- [ ] `docs/roadmap.md` checkbox updated
- [ ] Any design choice recorded in `docs/decisions.md`
- [ ] Changes committed with a clear message (no secrets staged)
- [ ] Each file committed separately with a clear message
