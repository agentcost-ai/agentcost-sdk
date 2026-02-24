"""
AgentCost OpenAI Interceptor

Monkey patches OpenAI SDK (v1.x+) to intercept all chat completion calls.
Uses exact token counts from API response — no estimation needed.

Supports:
- Synchronous create()
- Async create() via AsyncCompletions
- Streaming responses (sync and async)
"""

import time
import hashlib
import threading
from functools import wraps
from typing import Any, Callable, Optional
from datetime import datetime, timezone

from .cost_calculator import calculate_cost
from .config import get_config

# Thread-local guard to prevent double-counting when LangChain (or another
# higher-level interceptor) calls OpenAI under the hood.
_tracking_depth = threading.local()


def _hash_input(input_text: str) -> str:
    """Hash input text for caching pattern detection."""
    normalized = input_text.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _extract_messages_text(messages) -> str:
    """Extract text content from OpenAI messages format."""
    parts = []
    if not messages:
        return ""
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
        elif hasattr(msg, "content"):
            parts.append(str(msg.content or ""))
    return " ".join(parts)


def _get_effective_agent_name(config, explicit: Optional[str] = None) -> str:
    """Get the effective agent name, respecting context variable override."""
    if explicit:
        return explicit
    from .tracker import _agent_name_var
    ctx_name = _agent_name_var.get(None)
    if ctx_name:
        return ctx_name
    if config:
        return config.default_agent_name
    return "default"


class OpenAIInterceptor:
    """
    Intercepts OpenAI SDK (v1.x+) chat completion calls by monkey patching.

    Usage:
        interceptor = OpenAIInterceptor(event_callback=my_callback)
        interceptor.start()
        # ... user's OpenAI code runs normally ...
        interceptor.stop()
    """

    def __init__(self, event_callback: Callable[[dict], None]):
        self.event_callback = event_callback
        self.is_active = False
        self._original_create = None
        self._original_async_create = None
        self._completions_cls = None
        self._async_completions_cls = None

    def start(self) -> bool:
        """
        Begin intercepting OpenAI SDK calls.

        Returns:
            True if successfully started, False if OpenAI SDK is not installed.
        """
        if self.is_active:
            return True

        try:
            from openai.resources.chat.completions import Completions
            self._completions_cls = Completions
            self._original_create = Completions.create

            Completions.create = self._create_tracked_create()

            try:
                from openai.resources.chat.completions import AsyncCompletions
                self._async_completions_cls = AsyncCompletions
                self._original_async_create = AsyncCompletions.create
                AsyncCompletions.create = self._create_tracked_async_create()
            except ImportError:
                pass

            self.is_active = True

            config = get_config()
            if config and config.debug:
                print("[AgentCost] OpenAI interceptor started")

            return True

        except ImportError:
            return False
        except Exception as e:
            config = get_config()
            if config and config.debug:
                print(f"[AgentCost] Failed to start OpenAI interceptor: {e}")
            return False

    def stop(self) -> None:
        """Stop intercepting, restore original methods."""
        if not self.is_active:
            return

        if self._completions_cls and self._original_create:
            self._completions_cls.create = self._original_create
        if self._async_completions_cls and self._original_async_create:
            self._async_completions_cls.create = self._original_async_create

        self.is_active = False

        config = get_config()
        if config and config.debug:
            print("[AgentCost] OpenAI interceptor stopped")

    def _build_event(
        self,
        model: str,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        input_hash: str,
        error_message: Optional[str] = None,
        streaming: bool = False,
    ) -> dict:
        """Build a standardized event dict."""
        cost = calculate_cost(model, input_tokens, output_tokens)
        event = {
            "agent_name": agent_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": error_message is None,
            "error": error_message,
            "input_hash": input_hash,
        }
        if streaming:
            event["streaming"] = True

        try:
            from .tracker import get_effective_metadata
            meta = get_effective_metadata()
            if meta:
                event["metadata"] = meta
        except ImportError:
            pass

        return event

    def _emit(self, event: dict) -> None:
        """Emit event to callback, swallowing any callback errors."""
        try:
            self.event_callback(event)
        except Exception:
            config = get_config()
            if config and config.debug:
                import traceback
                traceback.print_exc()

    # ── Sync create ──────────────────────────────────────────────

    def _create_tracked_create(self) -> Callable:
        original = self._original_create
        interceptor = self

        @wraps(original)
        def tracked_create(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return original(client_self, *args, **kwargs)

            # Skip if we're already inside a higher-level interceptor
            depth = getattr(_tracking_depth, 'value', 0)
            if depth > 0:
                return original(client_self, *args, **kwargs)
            _tracking_depth.value = depth + 1

            model = kwargs.get("model", args[0] if args else "unknown")
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            agent_name = _get_effective_agent_name(config)
            input_text = _extract_messages_text(messages)
            input_hash = _hash_input(input_text)

            start_time = time.time()
            error_message = None
            response = None

            try:
                response = original(client_self, *args, **kwargs)

                if is_stream:
                    return _SyncStreamWrapper(
                        response, model, agent_name, input_hash, start_time, interceptor
                    )

                return response

            except Exception as e:
                error_message = str(e)
                raise

            finally:
                if not is_stream:
                    latency_ms = int((time.time() - start_time) * 1000)
                    input_tokens = 0
                    output_tokens = 0
                    if response and hasattr(response, "usage") and response.usage:
                        input_tokens = response.usage.prompt_tokens or 0
                        output_tokens = response.usage.completion_tokens or 0

                    event = interceptor._build_event(
                        model, agent_name, input_tokens, output_tokens,
                        latency_ms, input_hash, error_message,
                    )
                    interceptor._emit(event)
                _tracking_depth.value = getattr(_tracking_depth, 'value', 1) - 1

        return tracked_create

    # ── Async create ─────────────────────────────────────────────

    def _create_tracked_async_create(self) -> Callable:
        original = self._original_async_create
        interceptor = self

        @wraps(original)
        async def tracked_async_create(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return await original(client_self, *args, **kwargs)

            # Skip if we're already inside a higher-level interceptor
            depth = getattr(_tracking_depth, 'value', 0)
            if depth > 0:
                return await original(client_self, *args, **kwargs)
            _tracking_depth.value = depth + 1

            model = kwargs.get("model", args[0] if args else "unknown")
            messages = kwargs.get("messages", [])
            is_stream = kwargs.get("stream", False)
            agent_name = _get_effective_agent_name(config)
            input_text = _extract_messages_text(messages)
            input_hash = _hash_input(input_text)

            start_time = time.time()
            error_message = None
            response = None

            try:
                response = await original(client_self, *args, **kwargs)

                if is_stream:
                    return _AsyncStreamWrapper(
                        response, model, agent_name, input_hash, start_time, interceptor
                    )

                return response

            except Exception as e:
                error_message = str(e)
                raise

            finally:
                if not is_stream:
                    latency_ms = int((time.time() - start_time) * 1000)
                    input_tokens = 0
                    output_tokens = 0
                    if response and hasattr(response, "usage") and response.usage:
                        input_tokens = response.usage.prompt_tokens or 0
                        output_tokens = response.usage.completion_tokens or 0

                    event = interceptor._build_event(
                        model, agent_name, input_tokens, output_tokens,
                        latency_ms, input_hash, error_message,
                    )
                    interceptor._emit(event)
                _tracking_depth.value = getattr(_tracking_depth, 'value', 1) - 1

        return tracked_async_create


# ── Stream wrappers ──────────────────────────────────────────────


class _SyncStreamWrapper:
    """Wraps an OpenAI sync Stream to capture metrics after full consumption."""

    def __init__(self, stream, model, agent_name, input_hash, start_time, interceptor):
        self._stream = stream
        self._model = model
        self._agent_name = agent_name
        self._input_hash = input_hash
        self._start_time = start_time
        self._interceptor = interceptor
        self._output_tokens = 0
        self._input_tokens = 0
        self._accumulated_content = ""
        self._emitted = False

    def __iter__(self):
        try:
            for chunk in self._stream:
                self._process_chunk(chunk)
                yield chunk
        except Exception as e:
            self._emit_event(error_message=str(e))
            raise
        else:
            self._emit_event()

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)
        if not self._emitted:
            self._emit_event()

    def _process_chunk(self, chunk):
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                self._accumulated_content += delta.content
        if hasattr(chunk, "usage") and chunk.usage:
            self._input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
            self._output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

    def _emit_event(self, error_message=None):
        if self._emitted:
            return
        self._emitted = True

        latency_ms = int((time.time() - self._start_time) * 1000)

        # If no usage info from stream, estimate output tokens
        if self._output_tokens == 0 and self._accumulated_content:
            from .token_counter import TokenCounter
            self._output_tokens = TokenCounter.count_tokens(
                self._accumulated_content, self._model
            )

        event = self._interceptor._build_event(
            self._model, self._agent_name,
            self._input_tokens, self._output_tokens,
            latency_ms, self._input_hash, error_message,
            streaming=True,
        )
        self._interceptor._emit(event)

    # Delegate common Stream attributes
    @property
    def response(self):
        return getattr(self._stream, "response", None)


class _AsyncStreamWrapper:
    """Wraps an OpenAI async Stream to capture metrics after full consumption."""

    def __init__(self, stream, model, agent_name, input_hash, start_time, interceptor):
        self._stream = stream
        self._model = model
        self._agent_name = agent_name
        self._input_hash = input_hash
        self._start_time = start_time
        self._interceptor = interceptor
        self._output_tokens = 0
        self._input_tokens = 0
        self._accumulated_content = ""
        self._emitted = False

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                self._process_chunk(chunk)
                yield chunk
        except Exception as e:
            self._emit_event(error_message=str(e))
            raise
        else:
            self._emit_event()

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)
        if not self._emitted:
            self._emit_event()

    def _process_chunk(self, chunk):
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                self._accumulated_content += delta.content
        if hasattr(chunk, "usage") and chunk.usage:
            self._input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
            self._output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

    def _emit_event(self, error_message=None):
        if self._emitted:
            return
        self._emitted = True

        latency_ms = int((time.time() - self._start_time) * 1000)

        if self._output_tokens == 0 and self._accumulated_content:
            from .token_counter import TokenCounter
            self._output_tokens = TokenCounter.count_tokens(
                self._accumulated_content, self._model
            )

        event = self._interceptor._build_event(
            self._model, self._agent_name,
            self._input_tokens, self._output_tokens,
            latency_ms, self._input_hash, error_message,
            streaming=True,
        )
        self._interceptor._emit(event)

    @property
    def response(self):
        return getattr(self._stream, "response", None)
