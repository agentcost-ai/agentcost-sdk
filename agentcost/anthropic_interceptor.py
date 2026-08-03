"""
AgentCost Anthropic Interceptor

Monkey patches Anthropic SDK (v0.18+) to intercept all message creation calls.
Uses exact token counts from API response — no estimation needed.

Supports:
- Synchronous create() and create(stream=True)
- Async create() and create(stream=True) via AsyncMessages
- The streaming helper client.messages.stream() (sync and async)

Streaming note: a streamed call has no usage at the moment create() returns —
the token counts arrive in the message_start / message_delta events, and for
the stream() helper they accumulate on the stream's current_message_snapshot.
So every streaming path here emits its event when the stream finishes, not
when the call returns.
"""

import time
import hashlib
from functools import wraps
from typing import Callable, Optional
from datetime import datetime, timezone

from .cost_calculator import calculate_cost
from .config import get_config
from ._reentrancy import in_tracking, enter_tracking, exit_tracking


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


def _request_text(kwargs) -> str:
    """Prompt text for hashing, materializing a one-shot ``messages`` in place.

    The SDK types the parameter as Iterable, so a generator is valid usage.
    Reading it here would consume it and Anthropic would receive an empty
    conversation, so replace it with a list before anything iterates it.
    """
    messages = kwargs.get("messages")
    if messages is not None and not isinstance(messages, (str, bytes, list, tuple, dict)):
        messages = kwargs["messages"] = list(messages)
    return _extract_messages_text(messages, kwargs.get("system"))


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


def _usage_from(obj) -> tuple:
    """Read (input_tokens, output_tokens) off anything carrying a usage block."""
    usage = getattr(obj, "usage", None)
    if not usage:
        return 0, 0
    return (
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )


def _exit_error(exc_info) -> Optional[str]:
    """Error message from __exit__ args, if the block left with an exception."""
    exc = exc_info[1] if len(exc_info) > 1 else None
    if not isinstance(exc, BaseException):
        return None
    return str(exc) or type(exc).__name__


def _is_stream_response(response) -> bool:
    """
    True for the Stream / AsyncStream returned by create(stream=True).

    Detected structurally rather than by importing Anthropic's private stream
    classes, which have moved between versions.
    """
    if response is None:
        return False
    if hasattr(response, "usage"):  # a fully-materialized Message
        return False
    return hasattr(response, "__iter__") or hasattr(response, "__aiter__")


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
            self._on_tracking_error()

    def _record_call(
        self,
        response,
        model: str,
        agent_name: str,
        input_hash: str,
        start_time: float,
        error_message: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        """Build and emit the event for a finished call.

        Every step runs under one guard because callers invoke this from a
        ``finally`` block: an exception here would replace the caller's
        response and skip the exit_tracking() that follows it.
        """
        try:
            input_tokens, output_tokens = _usage_from(response)
            self._emit(self._build_event(
                model, agent_name, input_tokens, output_tokens,
                int((time.time() - start_time) * 1000),
                input_hash, error_message, streaming=streaming,
            ))
        except Exception:
            self._on_tracking_error()

    @staticmethod
    def _on_tracking_error() -> None:
        """Report a tracking failure without disturbing the caller."""
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
            if in_tracking():
                return original(client_self, *args, **kwargs)
            token = enter_tracking()

            model = kwargs.get("model", "unknown")
            agent_name = _get_effective_agent_name(config)
            input_text = _request_text(kwargs)
            input_hash = _hash_input(input_text)

            start_time = time.time()
            error_message = None
            response = None
            deferred = False

            try:
                response = original(client_self, *args, **kwargs)

                # stream=True hands back an un-consumed event stream: usage is
                # not known yet, so hand ownership of the event to the wrapper.
                if _is_stream_response(response):
                    deferred = True
                    return _AnthropicRawStreamWrapper(
                        response, model, agent_name, input_hash,
                        start_time, interceptor,
                    )

                return response

            except Exception as e:
                error_message = str(e)
                raise

            finally:
                if not deferred:
                    interceptor._record_call(
                        response, model, agent_name, input_hash,
                        start_time, error_message,
                    )
                exit_tracking(token)

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
            # No enter_tracking(): the wrapper spans the caller's consumption,
            # where holding the guard would suppress their own direct calls.
            if in_tracking():
                return original(client_self, *args, **kwargs)

            model = kwargs.get("model", "unknown")
            agent_name = _get_effective_agent_name(config)
            input_text = _request_text(kwargs)
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
            if in_tracking():
                return await original(client_self, *args, **kwargs)
            token = enter_tracking()

            model = kwargs.get("model", "unknown")
            agent_name = _get_effective_agent_name(config)
            input_text = _request_text(kwargs)
            input_hash = _hash_input(input_text)

            start_time = time.time()
            error_message = None
            response = None
            deferred = False

            try:
                response = await original(client_self, *args, **kwargs)

                if _is_stream_response(response):
                    deferred = True
                    return _AnthropicAsyncRawStreamWrapper(
                        response, model, agent_name, input_hash,
                        start_time, interceptor,
                    )

                return response

            except Exception as e:
                error_message = str(e)
                raise

            finally:
                if not deferred:
                    interceptor._record_call(
                        response, model, agent_name, input_hash,
                        start_time, error_message,
                    )
                exit_tracking(token)

        return tracked_async_create

    # ── Async stream ─────────────────────────────────────────────

    def _create_tracked_async_stream(self) -> Callable:
        original = self._original_async_stream
        interceptor = self

        # NOT async: AsyncMessages.stream() is a plain method returning an
        # async context manager. Making this a coroutine (or awaiting the
        # result) breaks `async with client.messages.stream(...)` outright.
        @wraps(original)
        def tracked_async_stream(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return original(client_self, *args, **kwargs)
            # See tracked_stream: no enter_tracking() around a wrapper the
            # caller consumes on its own time.
            if in_tracking():
                return original(client_self, *args, **kwargs)

            model = kwargs.get("model", "unknown")
            agent_name = _get_effective_agent_name(config)
            input_text = _request_text(kwargs)
            input_hash = _hash_input(input_text)

            start_time = time.time()
            stream_manager = original(client_self, *args, **kwargs)

            return _AnthropicAsyncStreamManagerWrapper(
                stream_manager, model, agent_name, input_hash,
                start_time, interceptor,
            )

        return tracked_async_stream


# ── Stream wrappers ──────────────────────────────────────────────


class _StreamEventUsageMixin:
    """Accumulates usage from raw server-sent events."""

    def __init__(self, stream, model, agent_name, input_hash, start_time, interceptor):
        self._stream = stream
        self._model = model
        self._agent_name = agent_name
        self._input_hash = input_hash
        self._start_time = start_time
        self._interceptor = interceptor
        self._input_tokens = 0
        self._output_tokens = 0
        self._emitted = False

    def __getattr__(self, name):
        # object.__getattribute__ so a missing _stream raises instead of
        # recursing forever.
        return getattr(object.__getattribute__(self, "_stream"), name)

    def _emit_event(self, error_message=None):
        if self._emitted:
            return
        self._emitted = True
        # Guarded: this runs from the stream's finally/__exit__ paths.
        try:
            self._interceptor._emit(self._interceptor._build_event(
                self._model, self._agent_name,
                self._input_tokens, self._output_tokens,
                int((time.time() - self._start_time) * 1000),
                self._input_hash, error_message, streaming=True,
            ))
        except Exception:
            self._interceptor._on_tracking_error()

    def _capture_usage(self, event) -> None:
        etype = getattr(event, "type", None)
        if etype == "message_start":
            message = getattr(event, "message", None)
            if message is not None:
                self._input_tokens, out = _usage_from(message)
                # message_start carries a partial output count; keep it as a
                # floor in case no message_delta arrives.
                self._output_tokens = max(self._output_tokens, out)
                # The resolved model beats the alias the caller passed in.
                self._model = getattr(message, "model", None) or self._model
        elif etype == "message_delta":
            _, out = _usage_from(event)
            if out:
                self._output_tokens = out


class _AnthropicRawStreamWrapper(_StreamEventUsageMixin):
    """
    Wraps the Stream returned by messages.create(stream=True).

    The event is emitted when iteration ends: at create() time the usage is
    still unknown.
    """

    def __iter__(self):
        try:
            for event in self._stream:
                self._capture_usage(event)
                yield event
        except Exception as e:
            self._emit_event(error_message=str(e))
            raise
        finally:
            self._emit_event()

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        try:
            return self._stream.__exit__(*args)
        finally:
            self._emit_event()

    def close(self):
        try:
            return self._stream.close()
        finally:
            self._emit_event()


class _AnthropicAsyncRawStreamWrapper(_StreamEventUsageMixin):
    """Async counterpart of _AnthropicRawStreamWrapper."""

    async def __aiter__(self):
        try:
            async for event in self._stream:
                self._capture_usage(event)
                yield event
        except Exception as e:
            self._emit_event(error_message=str(e))
            raise
        finally:
            self._emit_event()

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        try:
            return await self._stream.__aexit__(*args)
        finally:
            self._emit_event()

    async def close(self):
        try:
            return await self._stream.close()
        finally:
            self._emit_event()


class _SnapshotUsageMixin:
    """
    Reads usage off the stream helper's accumulated message snapshot.

    Deliberately does NOT wrap the MessageStream itself: callers reach for
    .text_stream, .get_final_message(), .until_done() or plain iteration, and a
    proxy only captures the one path it wraps — every other path recorded zero
    tokens. The snapshot is populated no matter which one they use.
    """

    def _read_snapshot_usage(self) -> None:
        try:
            snapshot = getattr(self._entered_stream, "current_message_snapshot", None)
            if snapshot is None:
                return
            input_tokens, output_tokens = _usage_from(snapshot)
            if input_tokens or output_tokens:
                self._input_tokens = input_tokens
                self._output_tokens = output_tokens
            self._model = getattr(snapshot, "model", None) or self._model
        except Exception:
            # Never let bookkeeping break the caller's stream.
            pass

    def _emit_event(self, error_message=None):
        if self._emitted:
            return
        self._emitted = True
        self._read_snapshot_usage()
        # Guarded: this runs from the stream manager's __exit__ paths.
        try:
            self._interceptor._emit(self._interceptor._build_event(
                self._model, self._agent_name,
                self._input_tokens, self._output_tokens,
                int((time.time() - self._start_time) * 1000),
                self._input_hash, error_message, streaming=True,
            ))
        except Exception:
            self._interceptor._on_tracking_error()


class _AnthropicStreamManagerWrapper(_SnapshotUsageMixin):
    """Wraps Anthropic's MessageStreamManager to capture metrics."""

    def __init__(self, stream_manager, model, agent_name, input_hash, start_time, interceptor):
        self._stream_manager = stream_manager
        self._model = model
        self._agent_name = agent_name
        self._input_hash = input_hash
        self._start_time = start_time
        self._interceptor = interceptor
        self._input_tokens = 0
        self._output_tokens = 0
        self._entered_stream = None
        self._emitted = False

    def __enter__(self):
        # Hand back the real MessageStream so every consumption pattern keeps
        # working; usage comes from its snapshot at exit.
        self._entered_stream = self._stream_manager.__enter__()
        return self._entered_stream

    def __exit__(self, *args):
        try:
            return self._stream_manager.__exit__(*args)
        finally:
            self._emit_event(_exit_error(args))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_stream_manager"), name)


class _AnthropicAsyncStreamManagerWrapper(_SnapshotUsageMixin):
    """Wraps Anthropic's AsyncMessageStreamManager to capture metrics."""

    def __init__(self, stream_manager, model, agent_name, input_hash, start_time, interceptor):
        self._stream_manager = stream_manager
        self._model = model
        self._agent_name = agent_name
        self._input_hash = input_hash
        self._start_time = start_time
        self._interceptor = interceptor
        self._input_tokens = 0
        self._output_tokens = 0
        self._entered_stream = None
        self._emitted = False

    async def __aenter__(self):
        self._entered_stream = await self._stream_manager.__aenter__()
        return self._entered_stream

    async def __aexit__(self, *args):
        try:
            return await self._stream_manager.__aexit__(*args)
        finally:
            self._emit_event(_exit_error(args))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_stream_manager"), name)
