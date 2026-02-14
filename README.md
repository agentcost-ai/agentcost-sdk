# AgentCost SDK

**Zero-friction LLM cost tracking for LangChain applications.**

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

# Your existing code works unchanged
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
response = llm.invoke("Hello!")  # Automatically tracked
```

## Features

- **Zero Code Changes**: Monkey patches LangChain - your code works as-is
- **Automatic Tracking**: Captures all `invoke()`, `ainvoke()`, `stream()`, `astream()` calls
- **Accurate Tokens**: Uses `tiktoken` for precise token counting
- **Real-Time Costs**: Calculates costs using up-to-date model pricing
- **Batched Sending**: Efficient network usage (size-based + time-based batching)
- **Rate Limiting**: Built-in rate limiter to protect your backend
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

## Supported Models (1900+)

The SDK supports 1900+ models across 45+ providers through dynamic pricing sync with the backend. Models are automatically updated when pricing changes.

| Provider      | Models                                                              |
| ------------- | ------------------------------------------------------------------- |
| OpenAI        | gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo, o1, o1-mini |
| Anthropic     | claude-3-opus/sonnet/haiku, claude-3.5-sonnet/haiku, claude-4-opus  |
| Google        | gemini-pro, gemini-1.5-pro/flash, gemini-2.0-flash                  |
| Groq          | llama-3.1-8b/70b, llama-3.2-3b, llama-3.3-70b, mixtral-8x7b         |
| DeepSeek      | deepseek-chat, deepseek-coder, deepseek-reasoner                    |
| Cohere        | command, command-light, command-r, command-r-plus                   |
| Mistral       | mistral-small/medium/large                                          |
| Together AI   | llama-3-70b/8b-chat, meta-llama models                              |
| Replicate     | Various open-source models                                          |
| OpenRouter    | Aggregated models from multiple providers                           |
| Perplexity    | pplx models                                                         |
| xAI           | Grok models                                                         |
| Amazon        | Amazon Nova, Titan models                                           |
| Azure         | Azure OpenAI models                                                 |
| AWS           | Bedrock models (Claude, Llama, Mistral)                             |
| Anyscale      | Anyscale endpoints                                                  |
| Cerebras      | Cerebras models                                                     |
| Cloudflare    | Workers AI models                                                   |
| Databricks    | DBRX, Meta Llama models                                             |
| DeepInfra     | Various hosted models                                               |
| Fireworks     | Fireworks AI models                                                 |
| Hyperbolic    | Hyperbolic models                                                   |
| Jina AI       | Embedding models                                                    |
| Lambda        | Lambda models                                                       |
| MiniMax       | MiniMax models                                                      |
| Moonshot      | Moonshot models                                                     |
| Sambanova     | Samba models                                                        |
| Voyage        | Embedding models                                                    |
| IBM           | watsonx models                                                      |
| AI21          | AI21 Labs models                                                    |
| Aleph Alpha   | Aleph Alpha models                                                  |
| Novita        | Novita hosted models                                                |
| Gradient AI   | Gradient endpoints                                                  |
| Dashscope     | Dashscope models (Alibaba)                                          |
| Friendliai    | Friendliai models                                                   |
| GMI           | GMI models                                                          |
| Llamagate     | Llamagate models                                                    |
| Morph         | Morph models                                                        |
| NLP Cloud     | NLP Cloud endpoints                                                 |
| Nscale        | Nscale models                                                       |
| Oracle        | OCI generative models                                               |
| OVHCloud      | OVHCloud models                                                     |
| Vercel        | Vercel AI Gateway, v0 models                                        |
| Weights & Biases | Wandb models                                                      |
| Zai           | Zai models                                                          |

**Note**: The full list of 1900+ models is dynamically loaded from the backend. Run `track_costs.init()` with a valid API key to access all supported models.

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
