# AgentCost SDK

**Zero-friction LLM cost tracking for OpenAI, Anthropic, Gemini, and LangChain applications.**

[AgentCost](https://agentcost.tech) is an open-source LLM cost observability platform. This is the
Python SDK — add two lines to your app and every OpenAI, Anthropic, Gemini, and LangChain call is
tracked with model, tokens, cost, and latency, attributed to the agent that made it.
Docs: [agentcost.tech/docs/sdk](https://agentcost.tech/docs/sdk)

## Installation

```bash
pip install agentcost
```

Or install from source:

```bash
cd agentcost-sdk
pip install -e .
```

## Quick Start

```python
from agentcost import track_costs

# 2 lines to add cost tracking!
track_costs.init(api_key="your_api_key", project_id="my-project")

# OpenAI — automatically tracked
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hello!"}])

# Anthropic — automatically tracked
from anthropic import Anthropic
client = Anthropic()
message = client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100, messages=[{"role": "user", "content": "Hello!"}])

# Gemini — automatically tracked (Google Gen AI SDK)
from google import genai
client = genai.Client()
response = client.models.generate_content(model="gemini-2.5-flash", contents="Hello!")

# LangChain — automatically tracked
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
response = llm.invoke("Hello!")
```

## Features

- **Zero Code Changes**: Monkey patches OpenAI, Anthropic, Gemini, and LangChain — your code works as-is
- **Automatic Tracking**: Captures `create()` (streaming and not), `parse()`, the OpenAI Responses API, Anthropic's `messages.stream()`, and LangChain `invoke()`/`ainvoke()`/`stream()`/`astream()` — sync and async
- **Accurate Tokens**: Uses provider-reported usage when available (including Gemini); estimates only when an SDK does not return usage
- **Real-Time Costs**: Calculates costs using up-to-date model pricing
- **Batched Sending**: Efficient network usage (size-based + time-based batching)
- **Rate Limiting**: Built-in rate limiter to protect your backend
- **Workflow Tracing**: Group a multi-step run and get cost per run, per step and per tool
- **Pre-deployment Analysis**: `agentcost analyze` estimates cost and finds loops before you ship — entirely offline
- **Local Mode**: Test without a backend

## Configuration

```python
track_costs.init(
    # Required for cloud mode
    api_key="sk_...",
    project_id="my-project",

    # Optional settings
    base_url="https://api.agentcost.tech",   # Your backend URL
    batch_size=10,                          # Events before auto-flush
    flush_interval=5.0,                     # Seconds between flushes
    debug=True,                             # Enable debug logging
    default_agent_name="my-agent",          # Default agent tag
    local_mode=False,                       # Store locally (no backend)
    enabled=True,                           # Enable/disable tracking

    # Custom pricing (overrides defaults)
    custom_pricing={
        "my-custom-model": {"input": 0.001, "output": 0.002}
    },

    # Global metadata (attached to all events)
    global_metadata={
        "environment": "production",
        "version": "1.0.0"
    }
)
```

## Agent Tagging

Tag LLM calls by agent for granular analytics:

```python
# Option 1: Set default agent
track_costs.set_agent_name("router-agent")

# Option 2: Context manager (recommended)
with track_costs.agent("technical-agent"):
    llm.invoke("How do I fix this?")  # Tagged as "technical-agent"

with track_costs.agent("billing-agent"):
    llm.invoke("What's my balance?")  # Tagged as "billing-agent"
```

## Workflows & Steps

Agent tagging answers *which agent* spent the money. Wrapping a multi-step run
answers *what one run costs*, which step dominates it, and whether it loops:

```python
with track_costs.workflow("support-triage"):

    with track_costs.step("classify"):
        llm.invoke("Which queue does this belong in?")

    with track_costs.tool("search_docs"):
        llm.invoke("Summarise these results")

    with track_costs.step("draft_reply"):
        llm.invoke("Write the response")
```

Every call inside shares one trace id and records its step, its parent and how
deeply it was nested. A nested `workflow()` joins the enclosing run rather than
starting a second one, and `step()` outside a workflow is a no-op — so
instrumenting a shared helper never depends on how it is called.

Entirely optional and entirely additive: without a `workflow()` your events are
exactly what they were before.

### Outcomes

Mark how a run ended and you get cost per completed outcome, which charges
failed runs to the successes they were paid for:

```python
with track_costs.workflow("support-triage"):
    ticket = handle(request)
    track_costs.outcome(ticket.resolved, label=ticket.status)
```

Sent once per run when the workflow closes, so a late failure overwrites an
earlier optimistic call.

## Pre-deployment Analysis

Estimate what an agent will cost, and find its loops, before it has spent
anything. The `agentcost` CLI ships with the package and runs entirely on your
machine — no network call, and no file content outlives the token count taken
from it.

```bash
# What do the prompt and skill files cost on every call?
agentcost analyze ./agent --model gpt-4o
```

For a cost-per-run figure, record one representative run in local mode and
project it to production volume:

```python
import json
from agentcost import track_costs

track_costs.init(local_mode=True)

with track_costs.workflow("support-triage"):
    run_agent(sample_request)

track_costs.flush()
json.dump(track_costs.get_local_events(), open("run.json", "w"))
```

```bash
agentcost analyze ./agent --events run.json --runs-per-day 2000
```

```
Prompt and skill files  (gpt-4o)
  3 file(s), 8,163 tokens, $0.020407 per call just to send them

Test run
  3 run(s), 4.0 calls per run, $0.044000 per run (worst $0.044000)
    $  0.022000  50.0%   2.0 calls  search_docs
    $  0.020000  45.5%   1.0 calls  draft_reply
    $  0.002000   4.5%   1.0 calls  classify

Projected at 2,000 runs/day: $2,640.00 per month

Findings (3)
  [  high] Step 'search_docs' ran 2.0 times per run; a loop or retry will multiply this in production
  [  high] 3 of 3 run(s) made the same call more than once (worst: 2x); the repeats are avoidable
  [medium] 2 files have identical content; sending both pays twice for the same context
```

| Flag | Purpose |
|---|---|
| `--model` | Model to price against (default `gpt-4o`) |
| `--events` | Events from a local-mode run (JSON array or JSONL) |
| `--runs-per-day` | Expected volume, to project a monthly cost |
| `--pattern` | Glob to include; repeatable (defaults cover prompt and doc files) |
| `--json` | Also write the full report as JSON |
| `--fail-on` | Exit non-zero on a finding at or above this severity |

Use `--fail-on high` in CI to block a deploy on a cost regression:

```bash
agentcost analyze ./agent --events run.json --fail-on high
```

## Metadata

Attach custom metadata for filtering/grouping:

```python
# Persistent metadata
track_costs.add_metadata("user_id", "user_123")
track_costs.add_metadata("tenant_id", "acme_corp")

# Temporary metadata (context manager)
with track_costs.metadata(conversation_id="conv_456", step="routing"):
    llm.invoke("Route this query")
```

## Local Testing

Test without running a backend:

```python
track_costs.init(local_mode=True, debug=True)

# Make LLM calls
llm.invoke("Hello!")
llm.invoke("World!")

# Retrieve captured events
events = track_costs.get_local_events()
for event in events:
    print(f"Model: {event['model']}")
    print(f"Tokens: {event['total_tokens']}")
    print(f"Cost: ${event['cost']:.6f}")
```

## Streaming Support

Streaming calls are automatically tracked:

```python
# Sync streaming
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="")
# Event recorded after stream completes

# Async streaming
async for chunk in llm.astream("Tell me a story"):
    print(chunk.content, end="")
# Event recorded after stream completes
```

## Event Structure

Each tracked event contains:

```python
{
    "agent_name": "my-agent",
    "model": "gpt-4",
    "input_tokens": 150,
    "output_tokens": 80,
    "total_tokens": 230,
    "cost": 0.0093,            # USD, real-time calculated
    "latency_ms": 1234,        # Measured latency
    "timestamp": "2026-01-23T10:30:45.123Z",
    "success": True,
    "error": None,
    "streaming": False,
    "metadata": {"conversation_id": "conv_456"}
}
```

## Dynamic Pricing (Real-Time Updates)

The SDK automatically fetches the latest pricing from the backend. This means:

- **No code changes** when model prices change
- Pricing is **cached for 24 hours** (efficient)
- Falls back to built-in defaults if backend is unavailable

### How It Works

```python
# SDK automatically fetches pricing from backend
track_costs.init(
    api_key="...",
    project_id="...",
    base_url="http://localhost:8000",  # If running locally
)

# Prices are fetched once and cached
# GET http://localhost:8000/v1/pricing → {"pricing": {"gpt-4": {"input": 0.03, ...}}}
```

### Manually Update Pricing

```python
from agentcost.cost_calculator import refresh_pricing, update_pricing

# Force refresh from backend
refresh_pricing()

# Or manually set pricing (doesn't require backend)
update_pricing({
    "my-custom-model": {"input": 0.001, "output": 0.002}
})
```

### Backend Pricing API

```bash
# Get all pricing
curl http://localhost:8000/v1/pricing

# Get specific model
curl http://localhost:8000/v1/pricing/gpt-4

# Update pricing (admin)
curl -X POST http://localhost:8000/v1/pricing \
  -H "Content-Type: application/json" \
  -d '{"gpt-4": {"input": 0.025, "output": 0.05}}'
```

## Pricing Coverage (3500+ models)

Costs are priced for 3500+ models across 45+ providers, synced from the backend and updated automatically when pricing changes.

Pricing coverage is not the same as automatic tracking. Calls are intercepted for the four integrations shown in the Quick Start — **OpenAI, Anthropic, Gemini and LangChain** (which also covers LangGraph and CrewAI when they run through LangChain). Calling another provider's SDK directly produces no events; route it through LangChain, or report the usage yourself. The table below is what the SDK can put a price on.

| Provider         | Models                                                              |
| ---------------- | ------------------------------------------------------------------- |
| OpenAI           | gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo, o1, o1-mini |
| Anthropic        | claude-3-opus/sonnet/haiku, claude-3.5-sonnet/haiku, claude-4-opus  |
| Google           | gemini-pro, gemini-1.5-pro/flash, gemini-2.0-flash/flash-lite, gemini-2.5-pro/flash/flash-lite, gemini-3-pro-preview |
| Groq             | llama-3.1-8b/70b, llama-3.2-3b, llama-3.3-70b, mixtral-8x7b         |
| DeepSeek         | deepseek-chat, deepseek-coder, deepseek-reasoner                    |
| Cohere           | command, command-light, command-r, command-r-plus                   |
| Mistral          | mistral-small/medium/large                                          |
| Together AI      | llama-3-70b/8b-chat, meta-llama models                              |
| Replicate        | Various open-source models                                          |
| OpenRouter       | Aggregated models from multiple providers                           |
| Perplexity       | pplx models                                                         |
| xAI              | Grok models                                                         |
| Amazon           | Amazon Nova, Titan models                                           |
| Azure            | Azure OpenAI models                                                 |
| AWS              | Bedrock models (Claude, Llama, Mistral)                             |
| Anyscale         | Anyscale endpoints                                                  |
| Cerebras         | Cerebras models                                                     |
| Cloudflare       | Workers AI models                                                   |
| Databricks       | DBRX, Meta Llama models                                             |
| DeepInfra        | Various hosted models                                               |
| Fireworks        | Fireworks AI models                                                 |
| Hyperbolic       | Hyperbolic models                                                   |
| Jina AI          | Embedding models                                                    |
| Lambda           | Lambda models                                                       |
| MiniMax          | MiniMax models                                                      |
| Moonshot         | Moonshot models                                                     |
| Sambanova        | Samba models                                                        |
| Voyage           | Embedding models                                                    |
| IBM              | watsonx models                                                      |
| AI21             | AI21 Labs models                                                    |
| Aleph Alpha      | Aleph Alpha models                                                  |
| Novita           | Novita hosted models                                                |
| Gradient AI      | Gradient endpoints                                                  |
| Dashscope        | Dashscope models (Alibaba)                                          |
| Friendliai       | Friendliai models                                                   |
| GMI              | GMI models                                                          |
| Llamagate        | Llamagate models                                                    |
| Morph            | Morph models                                                        |
| NLP Cloud        | NLP Cloud endpoints                                                 |
| Nscale           | Nscale models                                                       |
| Oracle           | OCI generative models                                               |
| OVHCloud         | OVHCloud models                                                     |
| Vercel           | Vercel AI Gateway, v0 models                                        |
| Weights & Biases | Wandb models                                                        |
| Zai              | Zai models                                                          |

**Note**: The full list of 3500+ models is dynamically loaded from the backend. Run `track_costs.init()` with a valid API key to access all supported models.

## Statistics

```python
stats = track_costs.get_stats()
print(f"Events sent: {stats['batcher']['events_sent']}")
print(f"Batches sent: {stats['batcher']['batches_sent']}")
```

## Graceful Shutdown

```python
track_costs.flush()     # Send pending events
track_costs.shutdown()  # Full shutdown
```

## License

MIT License
