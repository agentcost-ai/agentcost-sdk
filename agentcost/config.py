"""
AgentCost SDK Configuration

Contains default pricing data and SDK settings.
Prices are per 1,000 tokens in USD.

The SDK uses a tiered pricing lookup:
1. Custom pricing (user-provided)
2. Dynamic pricing from backend (/v1/pricing, 1900+ models)
3. DEFAULT_PRICING fallback (for offline/local mode)
"""

from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading


# Fallback pricing when backend is unreachable.
# Synced with backend's DEFAULT_PRICING for consistency.
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    'gpt-4': {'input': 0.03, 'output': 0.06},
    'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
    'gpt-4-turbo-preview': {'input': 0.01, 'output': 0.03},
    'gpt-4o': {'input': 0.0025, 'output': 0.01},
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
    'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004},
    'o1': {'input': 0.015, 'output': 0.06},
    'o1-preview': {'input': 0.015, 'output': 0.06},
    'o1-mini': {'input': 0.003, 'output': 0.012},
    
    # Anthropic
    'claude-3-opus': {'input': 0.015, 'output': 0.075},
    'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
    'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
    'claude-3-5-sonnet': {'input': 0.003, 'output': 0.015},
    'claude-3-5-haiku': {'input': 0.0008, 'output': 0.004},
    'claude-4-opus': {'input': 0.015, 'output': 0.075},
    
    # Groq
    'llama-3.1-8b-instant': {'input': 0.00005, 'output': 0.00008},
    'llama-3.1-70b-versatile': {'input': 0.00059, 'output': 0.00079},
    'llama-3.2-3b-preview': {'input': 0.00006, 'output': 0.00006},
    'llama-3.3-70b-versatile': {'input': 0.00059, 'output': 0.00079},
    'mixtral-8x7b-32768': {'input': 0.00024, 'output': 0.00024},
    
    # Google
    'gemini-pro': {'input': 0.00025, 'output': 0.0005},
    'gemini-1.5-pro': {'input': 0.00125, 'output': 0.005},
    'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
    'gemini-2.0-flash': {'input': 0.0001, 'output': 0.0004},
    
    # DeepSeek
    'deepseek-chat': {'input': 0.00014, 'output': 0.00028},
    'deepseek-coder': {'input': 0.00014, 'output': 0.00028},
    'deepseek-reasoner': {'input': 0.00055, 'output': 0.00219},
    
    # Mistral
    'mistral-small': {'input': 0.001, 'output': 0.003},
    'mistral-medium': {'input': 0.00275, 'output': 0.0081},
    'mistral-large': {'input': 0.004, 'output': 0.012},
    
    # Cohere
    'command': {'input': 0.001, 'output': 0.002},
    'command-light': {'input': 0.0003, 'output': 0.0006},
    'command-r': {'input': 0.0005, 'output': 0.0015},
    'command-r-plus': {'input': 0.003, 'output': 0.015},
    
    # Together AI
    'meta-llama/Llama-3-70b-chat-hf': {'input': 0.0009, 'output': 0.0009},
    'meta-llama/Llama-3-8b-chat-hf': {'input': 0.0002, 'output': 0.0002},
}


@dataclass
class AgentCostConfig:
    """Configuration for AgentCost SDK"""
    
    api_key: str
    project_id: str
    
    base_url: str = ""
    timeout: float = 10.0
    max_retries: int = 3
    
    batch_size: int = 10
    flush_interval: float = 5.0
    
    enabled: bool = True
    debug: bool = False
    
    default_agent_name: str = "default"
    
    def __post_init__(self):
        if not self.base_url:
            import os
            self.base_url = os.getenv("AGENTCOST_API_URL", "https://api.agentcost.tech")
    
    custom_pricing: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    global_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        """Mask sensitive fields in repr to avoid key leakage in logs/tracebacks."""
        masked_key = f"{self.api_key[:8]}..." if len(self.api_key) > 8 else "***"
        return (
            f"AgentCostConfig(api_key='{masked_key}', project_id='{self.project_id}', "
            f"base_url='{self.base_url}', enabled={self.enabled})"
        )
    
    def get_pricing(self, model: str) -> Dict[str, float]:
        """
        Get pricing for a model.
        
        Checks custom pricing first, then DEFAULT_PRICING (exact and fuzzy match).
        Returns zero pricing for unknown models.
        """
        if model in self.custom_pricing:
            return self.custom_pricing[model]
        
        if model in DEFAULT_PRICING:
            return DEFAULT_PRICING[model]
        
        # Fuzzy match for model variations
        model_lower = model.lower()
        for known_model, pricing in DEFAULT_PRICING.items():
            if known_model in model_lower or model_lower in known_model:
                return pricing
        
        if self.debug:
            print(f"[AgentCost] Unknown model '{model}'")
        return {'input': 0.0, 'output': 0.0}


# Global config instance (set by tracker.init())
_config: AgentCostConfig | None = None
_config_lock = threading.Lock()


def get_config() -> AgentCostConfig | None:
    """Get the current global configuration (thread-safe)"""
    with _config_lock:
        return _config


def set_config(config: AgentCostConfig) -> None:
    """Set the global configuration (thread-safe)"""
    global _config
    with _config_lock:
        _config = config
