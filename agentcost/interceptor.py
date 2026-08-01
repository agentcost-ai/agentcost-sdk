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
from functools import wraps
from typing import Any, Callable, Optional, Iterator, AsyncIterator
from datetime import datetime, timezone

from .token_counter import TokenCounter
from .cost_calculator import calculate_cost
from .config import get_config
# Shared cross-interceptor guard.  When a LangChain invoke() wraps an OpenAI
# or Anthropic SDK call, this depth counter prevents the lower-level
# interceptor from emitting a duplicate event.
from ._reentrancy import in_tracking, enter_tracking, exit_tracking


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


def _reported_usage(response: Any) -> Optional[tuple[int, int]]:
    """Provider-reported (input, output) token counts, or None if absent.

    ``ChatGoogleGenerativeAI`` exposes Gemini's authoritative usage as
    ``usage_metadata``. Other current LangChain integrations use the same
    shape or place the data under ``response_metadata.token_usage``.

    Returning None rather than zeros matters: only the caller knows whether a
    missing usage block should fall back to estimating from text or from the
    content accumulated across a stream.
    """
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        metadata = getattr(response, "response_metadata", None) or {}
        usage = metadata.get("token_usage") or metadata.get("usage_metadata")

    if not usage:
        return None

    def get(*names: str) -> int:
        for name in names:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if value is not None:
                return int(value or 0)
        return 0

    return (
        get("input_tokens", "prompt_tokens", "prompt_token_count"),
        get("output_tokens", "completion_tokens", "candidates_token_count"),
    )


def _extract_usage_tokens(response: Any, model: str) -> tuple[int, int]:
    """Prefer provider-reported usage exposed by LangChain over estimation."""
    reported = _reported_usage(response)
    if reported is not None:
        return reported

    output_text = TokenCounter.extract_text_from_output(response)
    return 0, TokenCounter.count_tokens(output_text, model)

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

    def _record(
        self,
        model_name: str,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        input_hash: str,
        error_message: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        """Build and emit one event, swallowing any bookkeeping failure.

        The wrappers call this from a ``finally`` block, so an exception
        escaping here would replace the caller's result and skip the
        exit_tracking() that follows it.
        """
        try:
            event = {
                'agent_name': agent_name,
                'model': model_name,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'cost': calculate_cost(model_name, input_tokens, output_tokens),
                'latency_ms': latency_ms,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'success': error_message is None,
                'error': error_message,
                'input_hash': input_hash,
            }
            if streaming:
                event['streaming'] = True

            from .tracker import get_effective_metadata
            metadata = get_effective_metadata()
            if metadata:
                event['metadata'] = metadata

            self.event_callback(event)
        except Exception as exc:
            config = get_config()
            if config and config.debug:
                print(f"[AgentCost] Tracking error: {exc}")

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
        interceptor = self
        
        @wraps(original_invoke)
        def tracked_invoke(llm_self, input_data, *args, **kwargs):
            """Wrapped invoke that captures metrics"""
            
            config = get_config()
            
            if config and not config.enabled:
                return original_invoke(llm_self, input_data, *args, **kwargs)

            # LangChain's own stream() falls back to calling invoke() for models
            # with no _stream implementation, so without this check one call
            # emitted two events and doubled the reported cost.
            if in_tracking():
                return original_invoke(llm_self, input_data, *args, **kwargs)

            # Set guard so downstream OpenAI/Anthropic interceptors skip
            depth_token = enter_tracking()
            
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
                    reported_input, output_tokens = _extract_usage_tokens(response, model_name)
                    if reported_input:
                        input_tokens = reported_input
                else:
                    output_tokens = 0

                interceptor._record(
                    model_name, agent_name, input_tokens, output_tokens,
                    latency_ms, _hash_input(input_text), error_message,
                )
                exit_tracking(depth_token)

        return tracked_invoke
    
    def _create_tracked_ainvoke(self) -> Callable:
        """Create the wrapped async ainvoke method"""
        original_ainvoke = self._original_ainvoke
        interceptor = self
        
        if not original_ainvoke:
            return None
        
        @wraps(original_ainvoke)
        async def tracked_ainvoke(llm_self, input_data, *args, **kwargs):
            """Wrapped async invoke that captures metrics"""
            
            config = get_config()
            
            if config and not config.enabled:
                return await original_ainvoke(llm_self, input_data, *args, **kwargs)

            # See tracked_invoke: astream() delegates to ainvoke() when the
            # model cannot stream, which double-counted the call.
            if in_tracking():
                return await original_ainvoke(llm_self, input_data, *args, **kwargs)

            depth_token = enter_tracking()
            
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
                    reported_input, output_tokens = _extract_usage_tokens(response, model_name)
                    if reported_input:
                        input_tokens = reported_input
                else:
                    output_tokens = 0
                
                interceptor._record(
                    model_name, agent_name, input_tokens, output_tokens,
                    latency_ms, _hash_input(input_text), error_message,
                )
                exit_tracking(depth_token)

        return tracked_ainvoke
    
    def _create_tracked_stream(self) -> Callable:
        """Create the wrapped stream method for streaming responses"""
        original_stream = self._original_stream
        interceptor = self
        
        if not original_stream:
            return None
        
        @wraps(original_stream)
        def tracked_stream(llm_self, input_data, *args, **kwargs) -> Iterator:
            """Wrapped stream that captures metrics from streaming response"""
            
            config = get_config()
            
            if config and not config.enabled:
                yield from original_stream(llm_self, input_data, *args, **kwargs)
                return
            
            model_name = _get_model_name(llm_self)
            explicit_agent = kwargs.pop('_agentcost_agent', None)
            agent_name = _get_effective_agent_name(config, explicit_agent)

            start_time = time.time()

            input_text = TokenCounter.extract_text_from_input(input_data)
            input_tokens = TokenCounter.count_tokens(input_text, model_name)

            accumulated_content = ""
            reported_input = 0
            reported_output = 0
            error_message = None

            try:
                chunks = iter(original_stream(llm_self, input_data, *args, **kwargs))
                while True:
                    # Guard next() only: that is where LangChain calls the
                    # provider SDK. Holding it across the caller's loop would
                    # suppress the caller's own direct SDK calls.
                    depth_token = enter_tracking()
                    try:
                        chunk = next(chunks)
                    except StopIteration:
                        break
                    finally:
                        exit_tracking(depth_token)

                    reported = _reported_usage(chunk)
                    if reported is not None:
                        # Providers split usage across chunks (Anthropic sends
                        # input tokens first and output tokens last), so keep
                        # the latest non-zero of each rather than whatever the
                        # final usage-bearing chunk happened to contain.
                        reported_input = reported[0] or reported_input
                        reported_output = reported[1] or reported_output
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
                if reported_input:
                    input_tokens = reported_input
                output_tokens = reported_output
                if not output_tokens:
                    # Nothing in the stream reported usage, so estimate from the
                    # accumulated text. ChatOpenAI does set stream_usage=True by
                    # default, but only on a default base_url/client; a custom
                    # endpoint, a proxy, or another chat model turns it off and
                    # then no chunk carries usage at all.
                    output_tokens = TokenCounter.count_tokens(accumulated_content, model_name)
                interceptor._record(
                    model_name, agent_name, input_tokens, output_tokens,
                    latency_ms, _hash_input(input_text), error_message,
                    streaming=True,
                )

        return tracked_stream
    
    def _create_tracked_astream(self) -> Callable:
        """Create the wrapped async stream method"""
        original_astream = self._original_astream
        interceptor = self
        
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
            reported_input = 0
            reported_output = 0
            error_message = None

            try:
                chunks = original_astream(llm_self, input_data, *args, **kwargs).__aiter__()
                while True:
                    # See tracked_stream: the guard covers only the await that
                    # actually reaches the provider, never the caller's loop
                    # body, so a concurrent direct SDK call is still recorded
                    # while a nested one is still suppressed.
                    depth_token = enter_tracking()
                    try:
                        chunk = await chunks.__anext__()
                    except StopAsyncIteration:
                        break
                    finally:
                        exit_tracking(depth_token)

                    reported = _reported_usage(chunk)
                    if reported is not None:
                        reported_input = reported[0] or reported_input
                        reported_output = reported[1] or reported_output
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
                if reported_input:
                    input_tokens = reported_input
                output_tokens = reported_output
                if not output_tokens:
                    # See tracked_stream: nothing reported usage.
                    output_tokens = TokenCounter.count_tokens(accumulated_content, model_name)
                interceptor._record(
                    model_name, agent_name, input_tokens, output_tokens,
                    latency_ms, _hash_input(input_text), error_message,
                    streaming=True,
                )

        return tracked_astream


def _get_model_name(llm_instance: Any) -> str:
    """Extract model name from LLM instance"""
    for attr in ['model_name', 'model', '_model_name', 'model_id']:
        value = getattr(llm_instance, attr, None)
        if value:
            return str(value)
    
    return llm_instance.__class__.__name__
