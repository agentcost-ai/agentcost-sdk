# AgentCost Tracker
# Main entry point for the AgentCost SDK.
# Coordinates all components: interceptor, batcher, HTTP client.

import atexit
import os
import contextvars
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from .config import AgentCostConfig, set_config, get_config
from .batcher import HybridBatcher, LocalBatcher
from .http_client import AgentCostHTTPClient, MockHTTPClient


# Default API URL
DEFAULT_API_URL = "https://api.agentcost.tech"


def _get_api_url(base_url: Optional[str] = None) -> str:
    """Get API URL from parameter, environment, or default."""
    if base_url:
        return base_url
    return os.environ.get("AGENTCOST_API_URL", DEFAULT_API_URL)


# Thread/async-safe context variable for agent name override
_agent_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    '_agent_name_var', default=None
)

# Thread/async-safe context variable for metadata override
_metadata_var: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    '_metadata_var', default=None
)


def get_effective_metadata() -> Dict[str, Any]:
    """
    Get the effective metadata dict, merging global config metadata
    with any ContextVar overrides from the metadata() context manager.
    Thread-safe: concurrent callers in different threads/async tasks
    see only their own overrides.
    """
    config = get_config()
    base = config.global_metadata.copy() if config and config.global_metadata else {}
    ctx_meta = _metadata_var.get(None)
    if ctx_meta:
        base.update(ctx_meta)
    return base


class AgentCostTracker:
    """
    Main tracker class that coordinates all AgentCost components.
    
    Usage:
        ## Simple initialization
        from agentcost import track_costs
        track_costs.init(api_key="sk_...", project_id="my-project")
        
        ## Your LangChain code runs normally
        llm = ChatOpenAI()
        response = llm.invoke("Hello!")  # Automatically tracked!
        
        ## Or use as context manager
        with track_costs.session(api_key="sk_...", project_id="my-project"):
            llm.invoke("Hello!")   # Tracked within this block
    """
    
    def __init__(self):
        self._config: Optional[AgentCostConfig] = None
        self._interceptors: List = []  # All active interceptors
        self._batcher: Optional[HybridBatcher] = None
        self._http_client: Optional[AgentCostHTTPClient] = None
        self._is_initialized = False
        self._local_mode = False
        self._atexit_registered = False
    
    def init(
        self,
        api_key: str = "",
        project_id: str = "",
        base_url: Optional[str] = None,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        enabled: bool = True,
        debug: bool = False,
        default_agent_name: str = "default",
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        global_metadata: Optional[Dict[str, Any]] = None,
        local_mode: bool = False,
        **kwargs
    ) -> "AgentCostTracker":
        """
        Initialize AgentCost tracking.
        
        Args:
            api_key: Your AgentCost API key (not needed in local_mode)
            project_id: Your project ID (not needed in local_mode)
            base_url: Backend API URL (default: AGENTCOST_API_URL env var or https://api.agentcost.tech)
            batch_size: Number of events before auto-flush
            flush_interval: Seconds between time-based flushes
            enabled: Enable/disable tracking
            debug: Enable debug logging
            default_agent_name: Default name for agents
            custom_pricing: Custom model pricing overrides
            global_metadata: Metadata to attach to all events
            local_mode: Store events locally instead of sending to backend
        
        Returns:
            Self for method chaining
        """
        if self._is_initialized:
            if debug:
                print("[AgentCost] Already initialized, reinitializing...")
            self.shutdown()
        
        self._local_mode = local_mode
        
        # Resolve API URL from parameter, env var, or default
        resolved_url = _get_api_url(base_url)
        
        self._config = AgentCostConfig(
            api_key=api_key,
            project_id=project_id,
            base_url=resolved_url,
            batch_size=batch_size,
            flush_interval=flush_interval,
            enabled=enabled,
            debug=debug,
            default_agent_name=default_agent_name,
            custom_pricing=custom_pricing or {},
            global_metadata=global_metadata or {},
        )
        
        set_config(self._config)
        
        if not enabled:
            if debug:
                print("[AgentCost] Tracking disabled")
            return self
        
        if local_mode:
            self._http_client = MockHTTPClient(debug=debug)
            if debug:
                print("[AgentCost] Running in local mode (events stored locally)")
        else:
            if not api_key or not project_id:
                print("[AgentCost] Warning: api_key and project_id required for cloud mode")
                print("[AgentCost] Use local_mode=True for local testing")
            
            self._http_client = AgentCostHTTPClient(
                api_key=api_key,
                base_url=resolved_url,
                debug=debug,
            )
        
        def send_callback(events: List[Dict]) -> bool:
            return self._http_client.send_events(project_id, events)
        
        if local_mode:
            self._batcher = LocalBatcher(
                batch_size=batch_size,
                flush_interval=flush_interval,
                debug=debug,
            )
        else:
            self._batcher = HybridBatcher(
                batch_size=batch_size,
                flush_interval=flush_interval,
                flush_callback=send_callback,
                debug=debug,
            )
        
        # Auto-detect and start all available framework interceptors.
        # Each interceptor patches the SDK it finds installed; if the SDK
        # is not present the interceptor returns False and is skipped.
        self._interceptors = []
        interceptor_classes = []
        
        # LangChain (covers LangGraph, CrewAI-via-LangChain)
        try:
            from .interceptor import LangChainInterceptor
            interceptor_classes.append(("LangChain", LangChainInterceptor))
        except Exception:
            pass
        
        # OpenAI SDK (direct openai.ChatCompletion calls)
        try:
            from .openai_interceptor import OpenAIInterceptor
            interceptor_classes.append(("OpenAI", OpenAIInterceptor))
        except Exception:
            pass
        
        # Anthropic SDK (direct anthropic.messages calls)
        try:
            from .anthropic_interceptor import AnthropicInterceptor
            interceptor_classes.append(("Anthropic", AnthropicInterceptor))
        except Exception:
            pass
        
        for name, cls in interceptor_classes:
            try:
                interceptor = cls(event_callback=self._batcher.add)
                if interceptor.start():
                    self._interceptors.append(interceptor)
                    if debug:
                        print(f"[AgentCost] {name} interceptor active")
            except Exception as e:
                if debug:
                    print(f"[AgentCost] {name} interceptor skipped: {e}")
        
        if self._interceptors:
            self._is_initialized = True
            if debug:
                names = [type(i).__name__.replace('Interceptor', '') for i in self._interceptors]
                print(f"[AgentCost] Tracking initialized ({', '.join(names)})")
        else:
            # Still mark as initialized -- config, batcher, and manual event
            # submission work fine.  The user just won't get automatic interception.
            self._is_initialized = True
            if debug:
                print("[AgentCost] No auto-interceptable SDK found. "
                      "Install openai, anthropic, or langchain-core for automatic tracking. "
                      "Manual event submission via the batcher still works.")
        
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True
        
        return self
    
    def shutdown(self) -> None:
        """Gracefully shutdown tracking"""
        if not self._is_initialized:
            return
        
        for interceptor in self._interceptors:
            interceptor.stop()
        self._interceptors = []
        
        if self._batcher:
            self._batcher.shutdown()
        
        if self._http_client:
            self._http_client.close()
        
        self._is_initialized = False
        
        if self._config and self._config.debug:
            print("[AgentCost] Shutdown complete")
    
    def flush(self) -> None:
        """Manually flush pending events"""
        if self._batcher:
            self._batcher.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        stats = {
            'initialized': self._is_initialized,
            'local_mode': self._local_mode,
        }
        
        if self._batcher:
            stats['batcher'] = self._batcher.get_stats()
        
        return stats
    
    def get_local_events(self) -> List[Dict]:
        """
        Get events stored locally (only in local_mode).
        
        Returns:
            List of event dictionaries
        """
        if self._local_mode and isinstance(self._batcher, LocalBatcher):
            return self._batcher.get_all_events()
        
        if self._local_mode and isinstance(self._http_client, MockHTTPClient):
            return self._http_client.get_all_events()
        
        return []
    
    def set_agent_name(self, name: str) -> None:
        """Set the default agent name for subsequent calls"""
        if self._config:
            self._config.default_agent_name = name
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to be attached to all subsequent events"""
        if self._config:
            self._config.global_metadata[key] = value
    
    def clear_metadata(self) -> None:
        """Clear all global metadata"""
        if self._config:
            self._config.global_metadata = {}
    
    @contextmanager
    def agent(self, name: str):
        """
        Context manager to set agent name for a block of code.
        
        Usage:
            with tracker.agent("router"):
                llm.invoke("Route this query")  # Tagged with agent="router"
        
        Thread-safe: uses contextvars so concurrent blocks don't interfere.
        """
        if not self._config:
            yield
            return
        
        token = _agent_name_var.set(name)
        try:
            yield
        finally:
            _agent_name_var.reset(token)
    
    @contextmanager
    def metadata(self, **kwargs):
        """
        Context manager to add temporary metadata.
        
        Usage:
            with tracker.metadata(conversation_id="abc123"):
                llm.invoke("Hello")  # Event includes conversation_id
        
        Thread-safe: uses contextvars so concurrent blocks don't interfere.
        """
        if not self._config:
            yield
            return
        
        # Merge with any existing context metadata (for nested calls)
        existing = _metadata_var.get(None) or {}
        merged = {**existing, **kwargs}
        token = _metadata_var.set(merged)
        try:
            yield
        finally:
            _metadata_var.reset(token)
    
    @property
    def is_active(self) -> bool:
        """Check if tracking is active"""
        return self._is_initialized and self._config and self._config.enabled
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


# Global tracker instance
_tracker = AgentCostTracker()


def init(**kwargs) -> AgentCostTracker:
    """Initialize global tracker"""
    return _tracker.init(**kwargs)


def shutdown() -> None:
    """Shutdown global tracker"""
    _tracker.shutdown()


def flush() -> None:
    """Flush pending events"""
    _tracker.flush()


def get_stats() -> Dict[str, Any]:
    """Get tracking statistics"""
    return _tracker.get_stats()


def get_local_events() -> List[Dict]:
    """Get locally stored events"""
    return _tracker.get_local_events()


def set_agent_name(name: str) -> None:
    """Set default agent name"""
    _tracker.set_agent_name(name)


def add_metadata(key: str, value: Any) -> None:
    """Add global metadata"""
    _tracker.add_metadata(key, value)


@contextmanager
def session(**kwargs):
    """
    Context manager for a tracking session.
    
    Usage:
        with track_costs.session(api_key="...", project_id="..."):
            llm.invoke("Hello")
    """
    tracker = AgentCostTracker()
    tracker.init(**kwargs)
    try:
        yield tracker
    finally:
        tracker.shutdown()


@contextmanager
def agent(name: str):
    """Context manager for agent name"""
    with _tracker.agent(name):
        yield


@contextmanager
def metadata(**kwargs):
    """Context manager for temporary metadata"""
    with _tracker.metadata(**kwargs):
        yield
