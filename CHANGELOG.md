# Changelog

All notable changes to the `agentcost` Python SDK.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.3.0] - 2026-10-12

### Added
- **Claude Code and Codex tracking.** `agentcost connect claude-code` (or
  `codex`) reads the session logs those agents already keep and sends each model
  call as an ordinary event: a session is the run, the tool a call invoked is its
  `tool_name`, and subagents are their own agents. A hook syncs after every turn.
  `agentcost sync --dry-run` shows what would be sent without an API key.
- **Guardrails that block.** `agentcost connect <agent> --enforce` installs a
  gate that runs before every tool call and prompt in Claude Code and Codex and
  refuses what the project's guardrails or hard budget cap do not allow:
  undeclared tools, writes by a read-only agent, undeclared models, per-run
  tool-call and cost limits, and a spent budget. It answers in the hooks'
  structured form, so the model is told why. It decides locally from a policy
  the sync keeps current, never waits on the network, and allows the call
  whenever it cannot decide. Needs a backend that serves
  `GET /v1/guardrails/policy`.
- Codex tools are reported under the names Codex's hooks use (`Bash`,
  `apply_patch`), read out of the script Codex wraps each call in.
- A request the API refused or dropped in Claude Code is recorded as a failed
  call against the session's model, so error rate is real.
- One-hour prompt-cache writes are priced at twice the input rate, as Anthropic
  bills them, in local estimates and (with a current backend) on the server.
- The sync hook starts the sync in its own process and returns, so a turn is
  sent even when the agent exits immediately after it.
- `agentcost` with no arguments lists every command and its main flags.
- **Antigravity tracking, as an estimate.** `agentcost connect antigravity`
  reads the per-conversation databases Antigravity keeps. The format is
  undocumented, so events are marked `metadata.estimated`, carry no tool
  attribution, and are never used for enforcement. Antigravity has no hooks; it
  is synced whenever another connected agent syncs, or on a schedule.
- A Claude Code call is identified by its API request id, the same id Claude
  Code's OpenTelemetry export carries, so a machine reporting both through
  `agentcost connect` and through the backend's OTLP receiver is counted once.
- The sync hook now syncs every connected agent, not only the one that fired it.
- `agentcost status` and `agentcost disconnect`.
- Events from coding agents carry `metadata.billing` (`subscription` or `api`),
  so usage covered by a plan can be told apart from metered spend.

Only usage numbers, model names, tool names and ids are read. Prompt text,
replies, tool arguments and tool output are never sent or kept. The working
folder's name and branch are sent only with `--include-repo`.

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
