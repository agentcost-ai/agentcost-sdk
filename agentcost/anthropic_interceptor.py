"""
AgentCost Anthropic Interceptor

Monkey patches Anthropic SDK (v0.18+) to intercept all message creation calls.
Uses exact token counts from API response — no estimation needed.

Supports:
- Synchronous create()
- Async create() via AsyncMessages
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

# Shared guard with openai_interceptor — prevents double-counting
try:
    from .openai_interceptor import _tracking_depth
except ImportError:
    _tracking_depth = threading.local()


def _hash_input(input_text: str) -> str:
    """Hash input text for caching pattern detection."""
    normalized = input_text.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _extract_messages_text(messages, system=None) -> str:
    """Extract text content from Anthropic messages format."""
    parts = []
    if system:
        if isinstance(system, str):
            parts.append(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    if not messages:
        return " ".join(parts)
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
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


class AnthropicInterceptor:
    """
    Intercepts Anthropic SDK message creation calls by monkey patching.

    Usage:
        interceptor = AnthropicInterceptor(event_callback=my_callback)
        interceptor.start()
        # ... user's Anthropic code runs normally ...
        interceptor.stop()
    """

    def __init__(self, event_callback: Callable[[dict], None]):
        self.event_callback = event_callback
        self.is_active = False
        self._original_create = None
        self._original_async_create = None
        self._original_stream = None
        self._original_async_stream = None
        self._messages_cls = None
        self._async_messages_cls = None

    def start(self) -> bool:
        """
        Begin intercepting Anthropic SDK calls.

        Returns:
            True if successfully started, False if Anthropic SDK is not installed.
        """
        if self.is_active:
            return True

        try:
            from anthropic.resources.messages import Messages
            self._messages_cls = Messages
            self._original_create = Messages.create

            Messages.create = self._create_tracked_create()

            # Patch stream method if it exists
            if hasattr(Messages, "stream"):
                self._original_stream = Messages.stream
                Messages.stream = self._create_tracked_stream()

            try:
                from anthropic.resources.messages import AsyncMessages
                self._async_messages_cls = AsyncMessages
                self._original_async_create = AsyncMessages.create
                AsyncMessages.create = self._create_tracked_async_create()

                if hasattr(AsyncMessages, "stream"):
                    self._original_async_stream = AsyncMessages.stream
                    AsyncMessages.stream = self._create_tracked_async_stream()
            except ImportError:
                pass

            self.is_active = True

            config = get_config()
            if config and config.debug:
                print("[AgentCost] Anthropic interceptor started")

            return True

        except ImportError:
            return False
        except Exception as e:
            config = get_config()
            if config and config.debug:
                print(f"[AgentCost] Failed to start Anthropic interceptor: {e}")
            return False

    def stop(self) -> None:
        """Stop intercepting, restore original methods."""
        if not self.is_active:
            return

        if self._messages_cls and self._original_create:
            self._messages_cls.create = self._original_create
        if self._messages_cls and self._original_stream:
            self._messages_cls.stream = self._original_stream
        if self._async_messages_cls and self._original_async_create:
            self._async_messages_cls.create = self._original_async_create
        if self._async_messages_cls and self._original_async_stream:
            self._async_messages_cls.stream = self._original_async_stream

        self.is_active = False

        config = get_config()
        if config and config.debug:
            print("[AgentCost] Anthropic interceptor stopped")

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

            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])
            system = kwargs.get("system", None)
            is_stream = kwargs.get("stream", False)
            agent_name = _get_effective_agent_name(config)
            input_text = _extract_messages_text(messages, system)
            input_hash = _hash_input(input_text)

            start_time = time.time()
            error_message = None
            response = None

            try:
                response = original(client_self, *args, **kwargs)
                return response

            except Exception as e:
                error_message = str(e)
                raise

            finally:
                latency_ms = int((time.time() - start_time) * 1000)
                input_tokens = 0
                output_tokens = 0

                if response and hasattr(response, "usage") and response.usage:
                    input_tokens = getattr(response.usage, "input_tokens", 0) or 0
                    output_tokens = getattr(response.usage, "output_tokens", 0) or 0

                event = interceptor._build_event(
                    model, agent_name, input_tokens, output_tokens,
                    latency_ms, input_hash, error_message,
                )
                interceptor._emit(event)
                _tracking_depth.value = getattr(_tracking_depth, 'value', 1) - 1

        return tracked_create

    # ── Sync stream ──────────────────────────────────────────────

    def _create_tracked_stream(self) -> Callable:
        original = self._original_stream
        interceptor = self

        @wraps(original)
        def tracked_stream(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return original(client_self, *args, **kwargs)

            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])
            system = kwargs.get("system", None)
            agent_name = _get_effective_agent_name(config)
            input_text = _extract_messages_text(messages, system)
            input_hash = _hash_input(input_text)

            start_time = time.time()
            stream_manager = original(client_self, *args, **kwargs)

            return _AnthropicStreamManagerWrapper(
                stream_manager, model, agent_name, input_hash,
                start_time, interceptor,
            )

        return tracked_stream

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

            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])
            system = kwargs.get("system", None)
            is_stream = kwargs.get("stream", False)
            agent_name = _get_effective_agent_name(config)
            input_text = _extract_messages_text(messages, system)
            input_hash = _hash_input(input_text)

            start_time = time.time()
            error_message = None
            response = None

            try:
                response = await original(client_self, *args, **kwargs)
                return response

            except Exception as e:
                error_message = str(e)
                raise

            finally:
                latency_ms = int((time.time() - start_time) * 1000)
                input_tokens = 0
                output_tokens = 0

                if response and hasattr(response, "usage") and response.usage:
                    input_tokens = getattr(response.usage, "input_tokens", 0) or 0
                    output_tokens = getattr(response.usage, "output_tokens", 0) or 0

                event = interceptor._build_event(
                    model, agent_name, input_tokens, output_tokens,
                    latency_ms, input_hash, error_message,
                )
                interceptor._emit(event)
                _tracking_depth.value = getattr(_tracking_depth, 'value', 1) - 1

        return tracked_async_create

    # ── Async stream ─────────────────────────────────────────────

    def _create_tracked_async_stream(self) -> Callable:
        original = self._original_async_stream
        interceptor = self

        @wraps(original)
        async def tracked_async_stream(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return await original(client_self, *args, **kwargs)

            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])
            system = kwargs.get("system", None)
            agent_name = _get_effective_agent_name(config)
            input_text = _extract_messages_text(messages, system)
            input_hash = _hash_input(input_text)

            start_time = time.time()
            stream_manager = await original(client_self, *args, **kwargs)

            return _AnthropicAsyncStreamManagerWrapper(
                stream_manager, model, agent_name, input_hash,
                start_time, interceptor,
            )

        return tracked_async_stream


# ── Stream wrappers ──────────────────────────────────────────────


class _AnthropicStreamManagerWrapper:
    """Wraps Anthropic's MessageStream context manager to capture metrics."""

    def __init__(self, stream_manager, model, agent_name, input_hash, start_time, interceptor):
        self._stream_manager = stream_manager
        self._model = model
        self._agent_name = agent_name
        self._input_hash = input_hash
        self._start_time = start_time
        self._interceptor = interceptor
        self._input_tokens = 0
        self._output_tokens = 0
        self._emitted = False

    def __enter__(self):
        result = self._stream_manager.__enter__()
        return _AnthropicEventStreamWrapper(result, self)

    def __exit__(self, *args):
        self._stream_manager.__exit__(*args)
        self._emit_event()

    def _emit_event(self, error_message=None):
        if self._emitted:
            return
        self._emitted = True
        latency_ms = int((time.time() - self._start_time) * 1000)
        event = self._interceptor._build_event(
            self._model, self._agent_name,
            self._input_tokens, self._output_tokens,
            latency_ms, self._input_hash, error_message,
            streaming=True,
        )
        self._interceptor._emit(event)

    # Delegate attribute access to the underlying stream manager
    def __getattr__(self, name):
        return getattr(self._stream_manager, name)


class _AnthropicAsyncStreamManagerWrapper:
    """Wraps Anthropic's async MessageStream context manager to capture metrics."""

    def __init__(self, stream_manager, model, agent_name, input_hash, start_time, interceptor):
        self._stream_manager = stream_manager
        self._model = model
        self._agent_name = agent_name
        self._input_hash = input_hash
        self._start_time = start_time
        self._interceptor = interceptor
        self._input_tokens = 0
        self._output_tokens = 0
        self._emitted = False

    async def __aenter__(self):
        result = await self._stream_manager.__aenter__()
        return _AnthropicAsyncEventStreamWrapper(result, self)

    async def __aexit__(self, *args):
        await self._stream_manager.__aexit__(*args)
        self._emit_event()

    def _emit_event(self, error_message=None):
        if self._emitted:
            return
        self._emitted = True
        latency_ms = int((time.time() - self._start_time) * 1000)
        event = self._interceptor._build_event(
            self._model, self._agent_name,
            self._input_tokens, self._output_tokens,
            latency_ms, self._input_hash, error_message,
            streaming=True,
        )
        self._interceptor._emit(event)

    def __getattr__(self, name):
        return getattr(self._stream_manager, name)


class _AnthropicEventStreamWrapper:
    """Wraps the iterable event stream to capture token usage from message_start/message_delta events."""

    def __init__(self, event_stream, parent):
        self._event_stream = event_stream
        self._parent = parent

    def __iter__(self):
        for event in self._event_stream:
            self._capture_usage(event)
            yield event

    def _capture_usage(self, event):
        # message_start contains input token count
        if hasattr(event, "type"):
            if event.type == "message_start" and hasattr(event, "message"):
                usage = getattr(event.message, "usage", None)
                if usage:
                    self._parent._input_tokens = getattr(usage, "input_tokens", 0) or 0
            elif event.type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    self._parent._output_tokens = getattr(usage, "output_tokens", 0) or 0

    def __getattr__(self, name):
        return getattr(self._event_stream, name)


class _AnthropicAsyncEventStreamWrapper:
    """Async version of the event stream wrapper."""

    def __init__(self, event_stream, parent):
        self._event_stream = event_stream
        self._parent = parent

    async def __aiter__(self):
        async for event in self._event_stream:
            self._capture_usage(event)
            yield event

    def _capture_usage(self, event):
        if hasattr(event, "type"):
            if event.type == "message_start" and hasattr(event, "message"):
                usage = getattr(event.message, "usage", None)
                if usage:
                    self._parent._input_tokens = getattr(usage, "input_tokens", 0) or 0
            elif event.type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    self._parent._output_tokens = getattr(usage, "output_tokens", 0) or 0

    def __getattr__(self, name):
        return getattr(self._event_stream, name)
