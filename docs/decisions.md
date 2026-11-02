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

## 008: Provider adapters with lazy SDK imports
**Date:** 2026-10-31
**Decision:** Use a thin `Provider` protocol and concrete adapters (`AnthropicProvider`, `OpenAIProvider`, `GeminiProvider`) importing SDKs lazily inside generation calls.
**Why:** Allows switching LLM providers via `.env` without code changes. Lazy imports ensure that unused SDKs or missing credentials for non-active providers never fail application startup or unit tests.
**Alternatives considered:** Direct SDK calls in agents (tight coupling); heavy multi-provider wrappers like LiteLLM or LangChain (adds unnecessary dependencies and hides control flow).

## 009: Centralized retry layer with disabled SDK retries
**Date:** 2026-10-31
**Decision:** Disable native SDK retries (`max_retries=0` for Anthropic and OpenAI, and `HttpRetryOptions(attempts=1)` for Google GenAI) and manage all retries centrally in `LLMClient`. Only transient errors (429, timeouts, network/connection errors, 5xx) trigger exponential backoff up to 3 attempts; non-transient errors (4xx auth, bad requests) fail immediately. Note that Google GenAI's `HttpOptions.timeout` is in milliseconds (converted from seconds), and `HttpRetryOptions(attempts=1)` sets tenacity's `stop_after_attempt(1)`, meaning 1 initial attempt and 0 retries.
**Why:** Multiple retry layers create nested retries, unpredictable latency spikes, and obscure failure tracking. Centralizing retries in `LLMClient` provides consistent logging, testability via injectable sleep functions, and single-point error monitoring.
**Alternatives considered:** Relying on default SDK retries (inconsistent across providers, difficult to observe or inject fake timers in unit tests).

## 010: Single-retry structured output repair loop
**Date:** 2026-10-31
**Decision:** In `complete_structured`, strip markdown code fences and attempt validation against a Pydantic model. On failure, retry exactly once with the error message appended back to the model, and raise `StructuredOutputError` if the second attempt fails.
**Why:** Models occasionally format JSON with minor syntax errors or missing required keys. Giving the model one targeted chance to self-correct dramatically improves reliability while capping cost and latency compared to infinite retry loops.
**Alternatives considered:** Relying purely on provider-specific JSON modes or function calling (incompatible across models and providers); unbounded validation retry loops (expensive and can loop indefinitely).

## 011: Public Greenhouse Job Board API only
**Date:** 2026-11-02
**Decision:** Discover Greenhouse jobs with `httpx` against
`GET https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true`.
No HTML scraping libraries, no LinkedIn/Indeed automation.
**Why:** The Job Board API is public, unauthenticated, and returns structured JSON.
Scraping career HTML is brittle and the project already forbids LinkedIn/Indeed.
**Alternatives considered:** BeautifulSoup/Playwright scraping of board HTML (fragile,
against the ATS-only rule).

## 012: Stdlib HTML-to-text for job descriptions
**Date:** 2026-11-02
**Decision:** Convert Greenhouse `content` with `html.unescape` plus `html.parser.HTMLParser`.
Skip `script`/`style`/`noscript`, turn paragraphs and `<br>` into line breaks, and prefix
list items with `- `.
**Why:** Descriptions are HTML-escaped HTML. The standard library is enough, keeps
dependencies unchanged, and is easy to explain in an interview.
**Alternatives considered:** BeautifulSoup or html2text (extra dependency for a small transform).

## 013: Per-board error isolation
**Date:** 2026-11-02
**Decision:** `discover()` catches `BoardNotFoundError` and `BoardFetchError` per board,
records them on `DiscoveryResult.errors`, and continues. That includes 404s, 4xx, exhausted
retries, non-JSON 200 bodies, and JSON missing the `jobs` key.
**Why:** One bad or unknown board token should not wipe out jobs from the rest of the list.
**Alternatives considered:** Fail the whole run on the first error (simpler, but unusable
with a mixed sources file).

## 014: Whole-word title and location matching
**Date:** 2026-11-02
**Decision:** `JobFilter` include/exclude terms use case-insensitive regex with `\\b` and
`re.escape`. `"intern"` drops "Software Engineering Intern" but keeps "Internal Tools Engineer".
`"india"` matches "Bengaluru, India" but not "Indiana, USA". Empty or whitespace-only
locations are treated as unknown and kept when `location_include` is set.
**Why:** Substring matching produces false positives on common stems (intern/internal,
india/indiana). Whole-word matching matches how people write filter keywords.
**Alternatives considered:** Substring `in` checks (too broad); LLM-based filtering (slow,
nondeterministic, and discovery is not an agent).
