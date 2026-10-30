# Architecture

## Goal
Automate the repetitive parts of job hunting while keeping a human in control:
find jobs, judge fit, tailor a truthful resume, and prepare applications for review.

## Pipeline
```
Discovery -> Extractor -> Matcher -> (approve?) -> Tailor <-> Validator -> Renderer -> Review (human) -> Applier -> Tracker
```

## Components

| Stage | Type | Input -> Output |
|---|---|---|
| Discovery | Plain code | source config -> `list[Job]` |
| Extractor | LLM agent (small model) | `Job` -> `JobRequirements` |
| Matcher | LLM agent | `Profile` + `JobRequirements` -> `MatchResult` |
| Tailor | LLM agent (strong model) | `Profile` + `Job` + role preset -> `TailoredResume` |
| Validator | LLM agent (separate context) | `TailoredResume` + `Profile` -> `ValidationReport` |
| Renderer | Plain code | `TailoredResume` -> PDF |
| Applier | Agent with browser tools | `Job` + PDF + `Profile` -> filled form (not submitted) |
| Tracker | Plain code | all stage outputs -> SQLite |
| Orchestrator | LangGraph | runs the graph, retries, approval interrupts, checkpointed state |

## Key design ideas
- **Master profile, not master resume.** `profile.yaml` stores every job, project,
  and skill as tagged Bullets with stable ids. Tailoring selects and rephrases;
  it never writes from scratch.
- **Tailor <-> Validator loop.** The validator rejects any claim not supported by
  the profile and returns feedback. The tailor retries up to N times.
- **Typed messages.** Agents exchange Pydantic models, not free text.
- **Human gates.** Approval before tailoring (optional) and always before submitting.
- **Observability.** Every agent call logs input, output, latency, and cost.

## PDF rules
Single column, real text, standard fonts, one page. A text-extraction test
confirms the PDF parses cleanly for ATS systems.

## Evaluation
- Matcher: compare scores to my manual labels in `evals/labeled_jobs.json`.
- Validator: measure catch rate on deliberately planted fake claims.
- Applier: measure form-fill accuracy per ATS platform.

## Folder layout
```
src/jobagent/
  agents/        extractor, matcher, tailor, validator, applier
  tools/         profile lookup, browser actions, pdf helpers
  discovery/     greenhouse, lever, ashby
  rendering/     templates + render + checks
  tracking/      db + repository
  llm/client.py  single wrapper for all model calls
  orchestrator.py
  models.py
  config.py
```
