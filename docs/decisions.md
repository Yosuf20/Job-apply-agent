# Design decisions

Short log of why things are built the way they are. Add an entry whenever a
non-obvious choice is made.

## Template
**Date:** YYYY-MM-DD
**Decision:**
**Why:**
**Alternatives considered:**

---

## 001: Multiple small agents, not one big agent
**Decision:** Split into extractor, matcher, tailor, validator, applier.
**Why:** Each has one job, a typed input/output, and can be evaluated separately.
A separate validator catches hallucinations better with fresh context.
**Alternatives considered:** One agent with many tools (harder to debug and evaluate).

## 002: Plain code for deterministic stages
**Decision:** Discovery, rendering, and tracking are not agents.
**Why:** No judgment needed; plain code is cheaper, faster, and more reliable.

## 003: ATS public endpoints, not LinkedIn/Indeed
**Decision:** Use Greenhouse/Lever/Ashby APIs and career pages only.
**Why:** LinkedIn and Indeed prohibit automation and ban accounts; ATS endpoints
are public and more stable.

## 004: Bullet ids in the profile
**Decision:** Every profile bullet has a stable id.
**Why:** Lets the validator trace each generated claim back to its source.

## 005: LangGraph for orchestration
**Decision:** Build agents as plain typed functions first, then wire them with LangGraph in Step 10.
**Why:** The Tailor <-> Validator retry loop, human approval gates (interrupts), and
checkpointed state map directly onto LangGraph features, and it is a recognized tool.
Keeping agents independent of the framework keeps them testable and easy to explain.
**Alternatives considered:** Plain Python state machine (simpler, but I would rebuild
retries and persistence myself); CrewAI/AutoGen (hide too much of the control flow).

## 006: Bullet ID uniqueness across entire Profile
**Date:** 2026-10-30
**Decision:** Enforce global uniqueness of all Bullet IDs across both experience and project sections via a Pydantic model validator on `Profile`.
**Why:** The Validator agent needs unambiguous provenance for every claim in a generated resume. Unique IDs ensure bullet tracing is 1-to-1 without needing compound keys.
**Alternatives considered:** Scoping IDs locally per company/project, which would require compound keys throughout agents and prompts.

## 007: Provider-agnostic model configuration defaults
**Date:** 2026-10-30
**Decision:** Default `model_small` and `model_strong` to empty strings in `Settings` and support an optional `env_file` parameter in `get_settings()`.
**Why:** Defers LLM provider choice until Step 2 without locking into any vendor upfront, and keeps unit tests hermetic and independent of local developer environment variables.
**Alternatives considered:** Defaulting to OpenAI models (e.g., `gpt-4o-mini`), which could mislead users before provider selection.
