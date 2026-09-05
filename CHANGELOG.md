# Changelog

All notable changes to the `agentcost` Python SDK.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.2.2] - 2026-09-02

### Changed
- `track_costs.tool()` now works outside a `workflow()`: calls made inside it
  carry `tool_name` (and nothing else) so guardrail compliance can see the tool
  boundary without a declared run. Inside a workflow, behaviour is unchanged.

## [0.2.1] — 2026-08-15

All changes are additive: the public API is unchanged and the new event fields are
optional, so a backend older than this release ignores them rather than storing them.

### Fixed

- **Cost was overstated on cached prompts.** `cached_tokens` was read off the provider
  response and put on the event, but the backend schema silently dropped it, so the full
  prompt was priced at the standard input rate. Client-side estimates now apply cache
  rates too, and the backend stores and reprices them. Where a provider publishes no
  cache rate, cached tokens fall back to the standard input rate — never a guessed
  discount.

- **Anthropic cache accounting was wrong in shape.** Anthropic reports
  `cache_read_input_tokens` *alongside* `input_tokens` (which holds only the uncached
  remainder), the opposite of OpenAI, where cached tokens are a subset of the prompt
  count. `_usage_from` now normalises both to one convention: `input_tokens` is the whole
  prompt, `cached_tokens` is the part read from cache. Cache *writes* stay separate,
  because they are billed at a premium over standard input rather than a discount —
  folding them in would have the wrong sign.

### Added

- **Capability fingerprinting.** Each call now records what it exercised —
  `{"vision": true, "tools": true, "tool_count": 3, "structured_output": true}` — under
  the reserved metadata key `_ac_caps`. This is what lets the optimizer tell whether a
  cheaper model would still serve the workload; without it, requirements resolved to
  unknown and unsafe downgrades could be suggested.

  **Booleans and counts only.** No prompt text, no tool definitions, no image data. The
  privacy posture is unchanged: nothing leaves the process that did not before. Omitted
  entirely when a call used none of these, so ordinary calls add no bytes. Written after
  caller metadata so a user's own keys can never be clobbered.

  Covers OpenAI, Anthropic, Gemini and their streaming paths. Gemini's nested `config`
  is flattened first, since tools and response schema live there rather than beside
  `contents`; Gemini image parts (`parts`, `inline_data`/`file_data`, PIL images passed
  directly) are detected alongside the OpenAI and Anthropic content shapes.

- **Trace inheritance from the environment, on every event.** A process that wraps an
  agent — a policy layer, an orchestrator, a CI job — exports `AGENTCOST_TRACE_ID`
  (and optionally `AGENTCOST_WORKFLOW`) and every event this SDK emits carries that run
  id, whether or not the agent uses `workflow()`. Inside `workflow()` the same
  environment id becomes the trace id unless an explicit `trace_id` argument is given;
  an active workflow always beats the environment. Ids are capped at 64 characters to
  match the backend column.

- `calculate_cost()` accepts `cached_tokens` and `cache_write_tokens`.

### Changed

- `anthropic_interceptor._usage_from()` returns a 4-tuple
  `(input, output, cache_read, cache_write)` instead of a 2-tuple. Internal, but noted
  because anything vendoring the module directly will need updating.

### Testing

- 26 new tests covering the fingerprint, environment inheritance, and — importantly —
  end-to-end interceptor enrichment: driving the real interceptors against fake provider
  clients and asserting on the emitted event. The unit tests alone could not prove the
  wiring, and a fingerprint computed but never attached would have passed them.

---

## [0.2.0] — 2026-08-11

- Workflow tracing with step-level cost attribution
- Pre-deployment cost analysis and loop detection via `agentcost analyze`
- Gemini (Google Gen AI) interceptor
