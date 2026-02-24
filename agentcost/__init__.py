"""
AgentCost SDK

Track LLM costs across OpenAI, Anthropic, and LangChain with zero code changes.
Auto-detects installed SDKs and intercepts all LLM calls transparently.

Usage:
    from agentcost import track_costs
    
    ## Initialize tracking
    track_costs.init(
        api_key="your_api_key",
        project_id="your_project_id"
    )
    
    ## Works with any supported SDK — no code changes needed!
    
    # OpenAI
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(model="gpt-4o", messages=[...])
    
    # Anthropic
    from anthropic import Anthropic
    client = Anthropic()
    message = client.messages.create(model="claude-3-5-sonnet-20241022", ...)
    
    # LangChain / LangGraph
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke("Hello!")
    
    ## For local testing (no backend required)
    track_costs.init(local_mode=True)
    events = track_costs.get_local_events()

Installation:
    pip install agentcost              # Base (auto-detects installed SDKs)
    pip install agentcost[openai]      # With OpenAI SDK
    pip install agentcost[anthropic]   # With Anthropic SDK
    pip install agentcost[langchain]   # With LangChain
    pip install agentcost[all]         # All frameworks

For more information, visit: https://agentcost.tech
"""

from pathlib import Path


try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("agentcost")
except (ImportError, PackageNotFoundError):
    # Fallback for dev mode or old Python without importlib_metadata
    __version__ = "0.1.0"
__author__ = "AgentCost"

# Import main tracker module as track_costs
from . import tracker as track_costs

# Also expose key functions at package level for convenience
from .tracker import (
    init,
    shutdown,
    flush,
    get_stats,
    get_local_events,
    set_agent_name,
    add_metadata,
    session,
    agent,
    metadata,
    AgentCostTracker,
)

# Expose configuration
from .config import AgentCostConfig, DEFAULT_PRICING

# Expose components for advanced usage
from .token_counter import TokenCounter
from .cost_calculator import (
    CostCalculator,
    calculate_cost,
    get_pricing_manager,
    refresh_pricing,
    update_pricing,
)
from .batcher import HybridBatcher, LocalBatcher
from .http_client import AgentCostHTTPClient, MockHTTPClient

# Framework interceptors — imported safely since SDKs are optional deps
try:
    from .interceptor import LangChainInterceptor
except Exception:
    LangChainInterceptor = None  # type: ignore

try:
    from .openai_interceptor import OpenAIInterceptor
except Exception:
    OpenAIInterceptor = None  # type: ignore

try:
    from .anthropic_interceptor import AnthropicInterceptor
except Exception:
    AnthropicInterceptor = None  # type: ignore

__all__ = [
    # Version
    "__version__",
    
    # Main module
    "track_costs",
    
    # Tracker functions
    "init",
    "shutdown", 
    "flush",
    "get_stats",
    "get_local_events",
    "set_agent_name",
    "add_metadata",
    "session",
    "agent",
    "metadata",
    "AgentCostTracker",
    
    # Configuration
    "AgentCostConfig",
    "DEFAULT_PRICING",
    
    # Components (for advanced usage)
    "TokenCounter",
    "CostCalculator",
    "calculate_cost",
    "get_pricing_manager",
    "refresh_pricing",
    "update_pricing",
    "HybridBatcher",
    "LocalBatcher",
    "AgentCostHTTPClient",
    "MockHTTPClient",
    "LangChainInterceptor",
    "OpenAIInterceptor",
    "AnthropicInterceptor",
]
