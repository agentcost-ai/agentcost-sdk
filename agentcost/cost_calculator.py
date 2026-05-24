"""
AgentCost Cost Calculator

Calculates LLM API costs based on token usage and model pricing.
"""

import threading
import time
from typing import Dict, Optional
from .config import get_config, DEFAULT_PRICING


class DynamicPricingManager:
    """
    Manages dynamic pricing fetched from the backend.
    """
    
    def __init__(self):
        self._pricing_cache: Dict[str, Dict[str, float]] = {}
        self._last_fetch: Optional[float] = None
        self._fetch_interval = 86400  # 24 hours in seconds
        self._lock = threading.Lock()
        self._fetch_attempted = False
        self._fetch_in_progress = False
    
    @property
    def model_count(self) -> int:
        """Number of models in the cache."""
        return len(self._pricing_cache)
    
    @property
    def is_populated(self) -> bool:
        """Whether the cache has been populated from backend."""
        return bool(self._pricing_cache)
    
    def get_pricing(self, base_url: str = None) -> Dict[str, Dict[str, float]]:
        """
        Get current pricing (from cache or fetch from backend).
        
        Args:
            base_url: Backend URL to fetch from
            
        Returns:
            Pricing dictionary (2000+ models if synced from backend)
        """
        with self._lock:
            now = time.time()

            needs_fetch = (
                not self._pricing_cache or
                (self._last_fetch and now - self._last_fetch > self._fetch_interval)
            )

            if needs_fetch and base_url and not self._fetch_in_progress:
                self._fetch_in_progress = True
                threading.Thread(
                    target=self._fetch_pricing,
                    args=(base_url,),
                    daemon=True,
                ).start()

            return self._pricing_cache if self._pricing_cache else DEFAULT_PRICING
    
    def _fetch_pricing(self, base_url: str) -> None:
        """Fetch latest pricing from backend (non-blocking, with retry logic)."""
        with self._lock:
            if self._fetch_attempted and self._pricing_cache:
                # Already have cached data, don't retry aggressively
                self._fetch_in_progress = False
                return

        self._fetch_attempted = True
        
        try:
            import requests
            
            response = requests.get(
                f"{base_url.rstrip('/')}/v1/pricing",
                timeout=10,  # Increased timeout for large response
            )
            
            if response.status_code == 200:
                data = response.json()
                pricing_data = data.get('pricing', {})
                
                new_cache = {}
                for model, prices in pricing_data.items():
                    new_cache[model] = {
                        'input': prices.get('input', 0.0),
                        'output': prices.get('output', 0.0),
                    }
                
                self._pricing_cache = new_cache
                self._last_fetch = time.time()
                
                config = get_config()
                if config and config.debug:
                    source = data.get('source', 'unknown')
                    print(f"[AgentCost] Fetched pricing for {len(new_cache)} models (source: {source})")
            else:
                if get_config() and get_config().debug:
                    print(f"[AgentCost] Failed to fetch pricing: HTTP {response.status_code}")
                    
        except Exception as e:
            config = get_config()
            if config and config.debug:
                print(f"[AgentCost] Could not fetch pricing from backend: {e}")
        finally:
            self._fetch_in_progress = False
            # Reset fetch flag after 5 minutes to allow retry
            def reset_flag():
                time.sleep(300)
                self._fetch_attempted = False
            threading.Thread(target=reset_flag, daemon=True).start()
    
    def force_fetch(self, base_url: str) -> int:
        """
        Force an immediate fetch from the backend.
        
        Returns:
            Number of models fetched
        """
        self._fetch_attempted = False
        self._pricing_cache = {}
        self._last_fetch = None
        self.get_pricing(base_url)
        return len(self._pricing_cache)
    
    def update_pricing(self, pricing: Dict[str, Dict[str, float]]) -> None:
        """Manually update pricing cache."""
        with self._lock:
            self._pricing_cache.update(pricing)
            self._last_fetch = time.time()
    
    def clear_cache(self) -> None:
        """Clear pricing cache (forces re-fetch on next get_pricing call)."""
        with self._lock:
            self._pricing_cache = {}
            self._last_fetch = None
            self._fetch_attempted = False


# Global pricing manager
_pricing_manager = DynamicPricingManager()


def get_pricing_manager() -> DynamicPricingManager:
    """Get the global pricing manager instance."""
    return _pricing_manager


class CostCalculator:
    """Calculates LLM API costs based on token usage"""
    
    def __init__(self, custom_pricing: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Args:
            custom_pricing: Optional custom pricing dictionary to override defaults
        """
        self.custom_pricing = custom_pricing or {}
    
    def calculate_cost(
        self, 
        model: str, 
        input_tokens: int, 
        output_tokens: int
    ) -> float:
        """
        Calculate cost in USD
        
        Args:
            model: Model name (e.g., 'gpt-4')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Cost in USD (e.g., 0.00453)
        """
        pricing = self._get_model_pricing(model)
        
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        total_cost = input_cost + output_cost
        
        return round(total_cost, 8)  # Round to 8 decimal places for precision
    
    def _get_model_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for model, with fallback logic"""
        
        if model in self.custom_pricing:
            return self.custom_pricing[model]
        
        config = get_config()
        if config and model in config.custom_pricing:
            return config.custom_pricing[model]
        
        if config and config.base_url:
            dynamic_pricing = _pricing_manager.get_pricing(config.base_url)
            if model in dynamic_pricing:
                return dynamic_pricing[model]
            
            model_lower = model.lower()
            for known_model, pricing in dynamic_pricing.items():
                if known_model in model_lower:
                    return pricing
        
        if model in DEFAULT_PRICING:
            return DEFAULT_PRICING[model]
        
        model_lower = model.lower()
        for known_model, pricing in DEFAULT_PRICING.items():
            if known_model in model_lower:
                return pricing
        
        # Unknown model - log warning and return zero pricing
        config = get_config()
        if config and config.debug:
            print(f"[AgentCost] Warning: Unknown model '{model}' - cost will be $0.00. "
                  f"Add custom pricing via custom_pricing parameter or submit a request for the model to the team.")
        
        return {'input': 0.0, 'output': 0.0}
    
    def estimate_conversation_cost(
        self,
        model: str,
        avg_input_tokens: int,
        avg_output_tokens: int,
        num_turns: int
    ) -> float:
        """
        Estimate cost for a multi-turn conversation
        
        Args:
            model: Model name
            avg_input_tokens: Average tokens per input
            avg_output_tokens: Average tokens per output
            num_turns: Number of conversation turns
        
        Returns:
            Estimated total cost in USD
        """
        cost_per_turn = self.calculate_cost(model, avg_input_tokens, avg_output_tokens)
        return round(cost_per_turn * num_turns, 6)
    
    def get_cost_breakdown(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """
        Get detailed cost breakdown
        
        Returns:
            Dictionary with input_cost, output_cost, total_cost
        """
        pricing = self._get_model_pricing(model)
        
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        
        return {
            'input_cost': round(input_cost, 8),
            'output_cost': round(output_cost, 8),
            'total_cost': round(input_cost + output_cost, 8),
            'input_price_per_1k': pricing['input'],
            'output_price_per_1k': pricing['output']
        }


# Global calculator instance
_calculator: Optional[CostCalculator] = None


def get_calculator() -> CostCalculator:
    """Get or create the global calculator instance"""
    global _calculator
    if _calculator is None:
        _calculator = CostCalculator()
    return _calculator


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Convenience function to calculate cost"""
    return get_calculator().calculate_cost(model, input_tokens, output_tokens)


def refresh_pricing() -> None:
    """Force refresh pricing from backend"""
    _pricing_manager.clear_cache()


def update_pricing(pricing: Dict[str, Dict[str, float]]) -> None:
    """Manually update pricing without backend"""
    _pricing_manager.update_pricing(pricing)
