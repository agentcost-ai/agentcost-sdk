"""
AgentCost Interceptor

Monkey patches LangChain's BaseChatModel to intercept all LLM calls, zero code changes for users

Supports:
- Synchronous invoke()
- Async ainvoke()
- Streaming stream() and astream()
"""

import time
import hashlib
import threading
from functools import wraps
from typing import Any, Callable, Optional, Iterator, AsyncIterator
from datetime import datetime, timezone

from .token_counter import TokenCounter
from .cost_calculator import calculate_cost
from .config import get_config

# Shared cross-interceptor guard.  When a LangChain invoke() wraps an OpenAI
# or Anthropic SDK call, this depth counter prevents the lower-level
# interceptor from emitting a duplicate event.
try:
    from .openai_interceptor import _tracking_depth
except ImportError:
    _tracking_depth = threading.local()


def _get_effective_agent_name(config, explicit: Optional[str] = None) -> str:
    """Get the effective agent name, respecting context variable override."""
    if explicit:
        return explicit
    # Import here to avoid circular imports
    from .tracker import _agent_name_var
    ctx_name = _agent_name_var.get(None)
    if ctx_name:
        return ctx_name
    if config:
        return config.default_agent_name
    return "default"


def _hash_input(input_text: str) -> str:
    """
    Hash input text for caching pattern detection.
    Uses SHA-256 with normalized input.
    """
    normalized = input_text.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()

class LangChainInterceptor:
    """
    Intercepts LangChain LLM calls by monkey patching BaseChatModel.
    
    Usage:
        interceptor = LangChainInterceptor(event_callback=my_callback)
        interceptor.start()
        # ... user's LangChain code runs ...
        interceptor.stop()
    """
    
    def __init__(self, event_callback: Callable[[dict], None]):
        """
        Args:
            event_callback: Function to call with each captured event
        """
        self.event_callback = event_callback
        self.is_active = False
        
        # Store original methods
        self._original_invoke = None
        self._original_ainvoke = None
        self._original_stream = None
        self._original_astream = None
        
        # Reference to the class we're patching
        self._base_chat_model = None
    
    def start(self) -> bool:
        """
        Begin intercepting LLM calls.
        
        Returns:
            True if successfully started, False otherwise
        """
        if self.is_active:
            return True
        
        try:
            from langchain_core.language_models import BaseChatModel
            self._base_chat_model = BaseChatModel
            
            self._original_invoke = BaseChatModel.invoke
            self._original_ainvoke = getattr(BaseChatModel, 'ainvoke', None)
            self._original_stream = getattr(BaseChatModel, 'stream', None)
            self._original_astream = getattr(BaseChatModel, 'astream', None)
            
            wrapped_invoke = self._create_tracked_invoke()
            wrapped_ainvoke = self._create_tracked_ainvoke()
            wrapped_stream = self._create_tracked_stream()
            wrapped_astream = self._create_tracked_astream()
            
            BaseChatModel.invoke = wrapped_invoke
            if self._original_ainvoke:
                BaseChatModel.ainvoke = wrapped_ainvoke
            if self._original_stream:
                BaseChatModel.stream = wrapped_stream
            if self._original_astream:
                BaseChatModel.astream = wrapped_astream
            
            self.is_active = True
            
            config = get_config()
            if config and config.debug:
                print("[AgentCost] Interceptor started - tracking LLM calls")
            
            return True
            
        except ImportError as e:
            print(f"[AgentCost] Failed to import LangChain: {e}")
            return False
        except Exception as e:
            print(f"[AgentCost] Failed to start interceptor: {e}")
            return False
    
    def stop(self) -> None:
        """Stop intercepting, restore original methods"""
        if not self.is_active:
            return
        
        if self._base_chat_model and self._original_invoke:
            self._base_chat_model.invoke = self._original_invoke
            
            if self._original_ainvoke:
                self._base_chat_model.ainvoke = self._original_ainvoke
            if self._original_stream:
                self._base_chat_model.stream = self._original_stream
            if self._original_astream:
                self._base_chat_model.astream = self._original_astream
        
        self.is_active = False
        
        config = get_config()
        if config and config.debug:
            print("[AgentCost] Interceptor stopped")
    
    def _create_tracked_invoke(self) -> Callable:
        """Create the wrapped invoke method"""
        original_invoke = self._original_invoke
        event_callback = self.event_callback
        
        @wraps(original_invoke)
        def tracked_invoke(llm_self, input_data, *args, **kwargs):
            """Wrapped invoke that captures metrics"""
            
            config = get_config()
            
            if config and not config.enabled:
                return original_invoke(llm_self, input_data, *args, **kwargs)
            
            # Set guard so downstream OpenAI/Anthropic interceptors skip
            prev_depth = getattr(_tracking_depth, 'value', 0)
            _tracking_depth.value = prev_depth + 1
            
            model_name = _get_model_name(llm_self)
            
            explicit_agent = kwargs.pop('_agentcost_agent', None)
            agent_name = _get_effective_agent_name(config, explicit_agent)
            
            input_text = TokenCounter.extract_text_from_input(input_data)
            input_tokens = TokenCounter.count_tokens(input_text, model_name)
            
            error_message = None
            response = None
            start_time = time.time()
            
            try:
                response = original_invoke(llm_self, input_data, *args, **kwargs)
                return response
                
            except Exception as e:
                error_message = str(e)
                raise
                
            finally:
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                
                if response is not None:
                    output_text = TokenCounter.extract_text_from_output(response)
                    output_tokens = TokenCounter.count_tokens(output_text, model_name)
                else:
                    output_tokens = 0
                
                cost = calculate_cost(model_name, input_tokens, output_tokens)
                
                event = {
                    'agent_name': agent_name,
                    'model': model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'cost': cost,
                    'latency_ms': latency_ms,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'success': error_message is None,
                    'error': error_message,
                    'input_hash': _hash_input(input_text),
                }
                
                effective_meta = None
                try:
                    from .tracker import get_effective_metadata
                    effective_meta = get_effective_metadata()
                except ImportError:
                    if config and config.global_metadata:
                        effective_meta = config.global_metadata.copy()

                if effective_meta:
                    event['metadata'] = effective_meta
                
                try:
                    event_callback(event)
                except Exception as callback_error:
                    if config and config.debug:
                        print(f"[AgentCost] Event callback error: {callback_error}")
                
                _tracking_depth.value = prev_depth
        
        return tracked_invoke
    
    def _create_tracked_ainvoke(self) -> Callable:
        """Create the wrapped async ainvoke method"""
        original_ainvoke = self._original_ainvoke
        event_callback = self.event_callback
        
        if not original_ainvoke:
            return None
        
        @wraps(original_ainvoke)
        async def tracked_ainvoke(llm_self, input_data, *args, **kwargs):
            """Wrapped async invoke that captures metrics"""
            
            config = get_config()
            
            if config and not config.enabled:
                return await original_ainvoke(llm_self, input_data, *args, **kwargs)
            
            prev_depth = getattr(_tracking_depth, 'value', 0)
            _tracking_depth.value = prev_depth + 1
            
            model_name = _get_model_name(llm_self)
            
            explicit_agent = kwargs.pop('_agentcost_agent', None)
            agent_name = _get_effective_agent_name(config, explicit_agent)
            
            start_time = time.time()
            
            input_text = TokenCounter.extract_text_from_input(input_data)
            input_tokens = TokenCounter.count_tokens(input_text, model_name)
            
            error_message = None
            response = None
            
            try:
                # Call original async LLM method
                response = await original_ainvoke(llm_self, input_data, *args, **kwargs)
                return response
                
            except Exception as e:
                error_message = str(e)
                raise
                
            finally:
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                
                if response is not None:
                    output_text = TokenCounter.extract_text_from_output(response)
                    output_tokens = TokenCounter.count_tokens(output_text, model_name)
                else:
                    output_tokens = 0
                
                cost = calculate_cost(model_name, input_tokens, output_tokens)
                
                event = {
                    'agent_name': agent_name,
                    'model': model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'cost': cost,
                    'latency_ms': latency_ms,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'success': error_message is None,
                    'error': error_message,
                    'input_hash': _hash_input(input_text),
                }
                
                effective_meta = None
                try:
                    from .tracker import get_effective_metadata
                    effective_meta = get_effective_metadata()
                except ImportError:
                    if config and config.global_metadata:
                        effective_meta = config.global_metadata.copy()

                if effective_meta:
                    event['metadata'] = effective_meta
                
                try:
                    event_callback(event)
                except Exception as callback_error:
                    if config and config.debug:
                        print(f"[AgentCost] Event callback error: {callback_error}")
                
                _tracking_depth.value = prev_depth
        
        return tracked_ainvoke
    
    def _create_tracked_stream(self) -> Callable:
        """Create the wrapped stream method for streaming responses"""
        original_stream = self._original_stream
        event_callback = self.event_callback
        
        if not original_stream:
            return None
        
        @wraps(original_stream)
        def tracked_stream(llm_self, input_data, *args, **kwargs) -> Iterator:
            """Wrapped stream that captures metrics from streaming response"""
            
            config = get_config()
            
            if config and not config.enabled:
                yield from original_stream(llm_self, input_data, *args, **kwargs)
                return
            
            prev_depth = getattr(_tracking_depth, 'value', 0)
            _tracking_depth.value = prev_depth + 1
            
            model_name = _get_model_name(llm_self)
            explicit_agent = kwargs.pop('_agentcost_agent', None)
            agent_name = _get_effective_agent_name(config, explicit_agent)
            
            start_time = time.time()
            
            input_text = TokenCounter.extract_text_from_input(input_data)
            input_tokens = TokenCounter.count_tokens(input_text, model_name)
            
            accumulated_content = ""
            error_message = None
            
            try:
                for chunk in original_stream(llm_self, input_data, *args, **kwargs):
                    if hasattr(chunk, 'content'):
                        accumulated_content += str(chunk.content)
                    elif isinstance(chunk, str):
                        accumulated_content += chunk
                    yield chunk
                    
            except Exception as e:
                error_message = str(e)
                raise
                
            finally:
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                output_tokens = TokenCounter.count_tokens(accumulated_content, model_name)
                cost = calculate_cost(model_name, input_tokens, output_tokens)
                
                event = {
                    'agent_name': agent_name,
                    'model': model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'cost': cost,
                    'latency_ms': latency_ms,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'success': error_message is None,
                    'error': error_message,
                    'streaming': True,
                    'input_hash': _hash_input(input_text),
                }
                
                effective_meta = None
                try:
                    from .tracker import get_effective_metadata
                    effective_meta = get_effective_metadata()
                except ImportError:
                    if config and config.global_metadata:
                        effective_meta = config.global_metadata.copy()

                if effective_meta:
                    event['metadata'] = effective_meta
                
                try:
                    event_callback(event)
                except Exception as callback_error:
                    if config and config.debug:
                        print(f"[AgentCost] Event callback error: {callback_error}")
                
                _tracking_depth.value = prev_depth
        
        return tracked_stream
    
    def _create_tracked_astream(self) -> Callable:
        """Create the wrapped async stream method"""
        original_astream = self._original_astream
        event_callback = self.event_callback
        
        if not original_astream:
            return None
        
        @wraps(original_astream)
        async def tracked_astream(llm_self, input_data, *args, **kwargs) -> AsyncIterator:
            """Wrapped async stream that captures metrics"""
            
            config = get_config()
            
            # If tracking is disabled, just call original
            if config and not config.enabled:
                async for chunk in original_astream(llm_self, input_data, *args, **kwargs):
                    yield chunk
                return
            
            prev_depth = getattr(_tracking_depth, 'value', 0)
            _tracking_depth.value = prev_depth + 1
            
            # Extract model and agent info
            model_name = _get_model_name(llm_self)
            explicit_agent = kwargs.pop('_agentcost_agent', None)
            agent_name = _get_effective_agent_name(config, explicit_agent)
            
            # Start timing
            start_time = time.time()
            
            # Count input tokens
            input_text = TokenCounter.extract_text_from_input(input_data)
            input_tokens = TokenCounter.count_tokens(input_text, model_name)
            
            # Accumulate streamed content
            accumulated_content = ""
            error_message = None
            
            try:
                async for chunk in original_astream(llm_self, input_data, *args, **kwargs):
                    if hasattr(chunk, 'content'):
                        accumulated_content += str(chunk.content)
                    elif isinstance(chunk, str):
                        accumulated_content += chunk
                    yield chunk
                    
            except Exception as e:
                error_message = str(e)
                raise
                
            finally:
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                output_tokens = TokenCounter.count_tokens(accumulated_content, model_name)
                cost = calculate_cost(model_name, input_tokens, output_tokens)
                
                event = {
                    'agent_name': agent_name,
                    'model': model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'cost': cost,
                    'latency_ms': latency_ms,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'success': error_message is None,
                    'error': error_message,
                    'streaming': True,
                    'input_hash': _hash_input(input_text),
                }
                
                effective_meta = None
                try:
                    from .tracker import get_effective_metadata
                    effective_meta = get_effective_metadata()
                except ImportError:
                    if config and config.global_metadata:
                        effective_meta = config.global_metadata.copy()

                if effective_meta:
                    event['metadata'] = effective_meta
                
                try:
                    event_callback(event)
                except Exception as callback_error:
                    if config and config.debug:
                        print(f"[AgentCost] Event callback error: {callback_error}")
                
                _tracking_depth.value = prev_depth
        
        return tracked_astream


def _get_model_name(llm_instance: Any) -> str:
    """Extract model name from LLM instance"""
    for attr in ['model_name', 'model', '_model_name', 'model_id']:
        value = getattr(llm_instance, attr, None)
        if value:
            return str(value)
    
    return llm_instance.__class__.__name__
