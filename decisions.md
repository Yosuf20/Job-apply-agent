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
