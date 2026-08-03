"""
AgentCost Cost Calculator

Calculates LLM API costs based on token usage and model pricing.
"""

import threading
import time
import warnings
from typing import Dict, Optional
from .config import get_config, DEFAULT_PRICING


class DynamicPricingManager:
    """
    Manages dynamic pricing fetched from the backend.
    """
    
    # Backoff after a failed fetch, so an unreachable backend cannot turn every
    # LLM call into another HTTP attempt.
    _RETRY_BACKOFF_START = 30.0
    _RETRY_BACKOFF_MAX = 900.0

    def __init__(self):
        self._pricing_cache: Dict[str, Dict[str, float]] = {}
        self._last_fetch: Optional[float] = None
        self._fetch_interval = 86400  # 24 hours in seconds
        self._lock = threading.Lock()
        self._fetch_in_progress = False
        self._next_attempt_at = 0.0
        self._retry_backoff = self._RETRY_BACKOFF_START
    
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
            Pricing dictionary (3500+ models if synced from backend)
        """
        with self._lock:
            now = time.time()

            needs_fetch = (
                not self._pricing_cache or
                (self._last_fetch and now - self._last_fetch > self._fetch_interval)
            )

            # The backoff gate matters most when the cache is EMPTY, or an
            # unreachable backend turns every calculate_cost() into another
            # thread and another 10s request.
            if (
                needs_fetch
                and base_url
                and not self._fetch_in_progress
                and now >= self._next_attempt_at
            ):
                self._fetch_in_progress = True
                threading.Thread(
                    target=self._fetch_pricing,
                    args=(base_url,),
                    daemon=True,
                ).start()

            return self._pricing_cache if self._pricing_cache else DEFAULT_PRICING
    
    def _fetch_pricing(self, base_url: str) -> None:
        """Fetch latest pricing from backend (non-blocking, with retry logic)."""
        succeeded = False

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
                
                with self._lock:
                    self._pricing_cache = new_cache
                    self._last_fetch = time.time()
                succeeded = True

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
            with self._lock:
                self._fetch_in_progress = False
                if succeeded:
                    self._retry_backoff = self._RETRY_BACKOFF_START
                    self._next_attempt_at = 0.0
                else:
                    # Grow the wait instead of hammering.
                    self._next_attempt_at = time.time() + self._retry_backoff
                    self._retry_backoff = min(
                        self._retry_backoff * 2, self._RETRY_BACKOFF_MAX
                    )

    def force_fetch(self, base_url: str) -> int:
        """
        Force an immediate, synchronous fetch from the backend.

        Returns:
            Number of models fetched
        """
        with self._lock:
            self._pricing_cache = {}
            self._last_fetch = None
            self._next_attempt_at = 0.0
            self._retry_backoff = self._RETRY_BACKOFF_START
            self._fetch_in_progress = True

        self._fetch_pricing(base_url)

        with self._lock:
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
            # Drop the backoff too, or a re-fetch stays gated for up to 15 min.
            self._next_attempt_at = 0.0
            self._retry_backoff = self._RETRY_BACKOFF_START


# Global pricing manager
_pricing_manager = DynamicPricingManager()


def get_pricing_manager() -> DynamicPricingManager:
    """Get the global pricing manager instance."""
    return _pricing_manager


# Model names already reported as unpriced, so the warning fires once each
# instead of on every call.
_warned_unknown_models = set()


def _best_substring_match(model: str, table: Dict[str, Dict[str, float]]):
    """
    Find the pricing entry whose key is the LONGEST substring of *model*,
    breaking ties towards the entry declared first.

    The first key that merely fits charges the wrong rate — 'gpt-4' precedes
    'gpt-4o-mini' in the table, so gpt-4o-mini would bill at 200x.
    """
    model_lower = model.lower()
    matches = [known for known in table if known in model_lower]
    if not matches:
        return None
    # max() returns the first of several equally-long keys, and dicts iterate in
    # insertion order, so this is the earliest-declared longest match.
    return table[max(matches, key=len)]


_warned_bad_pricing = set()


def _rate(pricing, key: str, model: str) -> float:
    """One side of a model's price, or 0.0 if the entry is unusable."""
    try:
        return float(pricing[key])
    except (KeyError, TypeError, ValueError):
        if model not in _warned_bad_pricing:
            _warned_bad_pricing.add(model)
            try:
                warnings.warn(
                    f"AgentCost: pricing for '{model}' has no usable '{key}' rate; "
                    f"that side is counted as $0.00. Expected "
                    f"{{'input': <per-1k USD>, 'output': <per-1k USD>}}.",
                    RuntimeWarning,
                    stacklevel=4,
                )
            except Exception:
                pass
        return 0.0


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

        # custom_pricing comes straight from the caller and is never shape-checked,
        # so a missing or non-numeric rate must degrade to 0.0 rather than raise:
        # this runs inside the interceptors' finally blocks, where an exception
        # would cost the caller their response and the event its record.
        input_cost = (input_tokens / 1000) * _rate(pricing, 'input', model)
        output_cost = (output_tokens / 1000) * _rate(pricing, 'output', model)

        return round(input_cost + output_cost, 8)
    
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
            
            match = _best_substring_match(model, dynamic_pricing)
            if match is not None:
                return match

        if model in DEFAULT_PRICING:
            return DEFAULT_PRICING[model]

        match = _best_substring_match(model, DEFAULT_PRICING)
        if match is not None:
            return match

        # Unknown model — warn once per model name, not only in debug mode.
        # These calls are recorded at $0.00, which looks identical to "nothing
        # was spent" on the dashboard; the user needs to know the figure is
        # missing rather than zero.
        if model not in _warned_unknown_models:
            _warned_unknown_models.add(model)
            try:
                warnings.warn(
                    f"AgentCost has no pricing for model '{model}' — it will be "
                    f"recorded at $0.00. Supply it via the custom_pricing argument "
                    f"to track_costs.init(), or sync pricing from the backend.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            except Exception:
                # warnings.warn raises under -W error / filterwarnings("error").
                # This runs from the interceptors' `finally:` blocks, wrapped
                # around the caller's own LLM call, so letting it escape would
                # break their request over a pricing-table gap AND skip the
                # exit_tracking() that follows it -- leaving the re-entrancy
                # depth stuck above zero so every later call is dropped as
                # nested. Telling the user about missing pricing is never worth
                # breaking their app.
                pass

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
