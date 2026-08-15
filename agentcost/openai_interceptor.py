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
from functools import wraps
from typing import Callable, Optional
from datetime import datetime, timezone

from .cost_calculator import calculate_cost
from .config import get_config
from .capabilities import CAPABILITY_KEY, fingerprint
# Guard preventing double-counting when LangChain (or another higher-level
# interceptor) calls OpenAI under the hood. Shared by every interceptor — see
# _reentrancy.py for why it is a ContextVar and not a thread-local.
from ._reentrancy import in_tracking, enter_tracking, exit_tracking


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


def _materialize_iterables(kwargs) -> None:
    """Replace one-shot iterables in the request with lists, in place.

    Both SDKs type these parameters as Iterable, so passing a generator is
    valid usage. Reading it to hash the prompt would consume it and the
    provider would then receive an empty conversation, so it must be
    materialized before either we or the SDK iterate it.
    """
    for key in ("messages", "input"):
        value = kwargs.get(key)
        if value is not None and not isinstance(value, (str, bytes, list, tuple, dict)):
            kwargs[key] = list(value)


def _extract_input_text(kwargs) -> str:
    """
    Extract prompt text from either API's request shape.

    chat.completions uses `messages`; the Responses API uses `input`, which may
    be a bare string or the same message-list structure. Materializes kwargs in
    place first — see _materialize_iterables.
    """
    _materialize_iterables(kwargs)
    if "messages" in kwargs:
        return _extract_messages_text(kwargs.get("messages"))
    value = kwargs.get("input")
    if isinstance(value, str):
        return value
    return _extract_messages_text(value)


def _usage_tokens(usage) -> tuple:
    """
    Read (input, output) tokens from either usage shape.

    chat.completions reports prompt_tokens/completion_tokens; the Responses API
    reports input_tokens/output_tokens.
    """
    if not usage:
        return 0, 0
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is not None or completion is not None:
        return prompt or 0, completion or 0
    return (
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )


def _cached_input_tokens(usage) -> int:
    """Cached prompt tokens; billed at a discount `cost` does not apply."""
    if not usage:
        return 0
    details = getattr(usage, "prompt_tokens_details", None) or getattr(
        usage, "input_tokens_details", None
    )
    if details is None:
        return 0
    return getattr(details, "cached_tokens", 0) or 0


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
        # (owner_class, attribute, original_callable) for every patched surface.
        self._patches = []

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

            # parse() (structured outputs) posts directly instead of going
            # through create(), and the Responses API is a separate resource.
            self._patch(Completions, "create", is_async=False)
            self._patch(Completions, "parse", is_async=False)

            try:
                from openai.resources.chat.completions import AsyncCompletions
                self._patch(AsyncCompletions, "create", is_async=True)
                self._patch(AsyncCompletions, "parse", is_async=True)
            except ImportError:
                pass

            try:
                from openai.resources.responses import Responses, AsyncResponses
                self._patch(Responses, "create", is_async=False)
                self._patch(AsyncResponses, "create", is_async=True)
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
            # Never leave the SDK half-patched: stop() bails on an interceptor
            # that never became active, so undo the surfaces ourselves.
            self._unpatch()
            config = get_config()
            if config and config.debug:
                print(f"[AgentCost] Failed to start OpenAI interceptor: {e}")
            return False

    def _unpatch(self) -> None:
        """Undo every surface patched by _patch."""
        for owner, attr, original in self._patches:
            setattr(owner, attr, original)
        self._patches = []

    def stop(self) -> None:
        """Stop intercepting, restore original methods."""
        if not self.is_active:
            return

        self._unpatch()

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
        cached_tokens: int = 0,
        capabilities: Optional[dict] = None,
    ) -> dict:
        """Build a standardized event dict."""
        cost = calculate_cost(
            model, input_tokens, output_tokens, cached_tokens=cached_tokens
        )
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
        if cached_tokens:
            event["cached_tokens"] = cached_tokens

        try:
            from .tracker import get_effective_metadata
            meta = get_effective_metadata()
            if meta:
                event["metadata"] = meta
        except ImportError:
            pass

        if capabilities:
            # Reserved namespace, merged after user metadata so a caller's own
            # keys can never collide with the capability fingerprint.
            event.setdefault("metadata", {})
            event["metadata"][CAPABILITY_KEY] = capabilities

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
        capabilities: Optional[dict] = None,
    ) -> None:
        """Build and emit the event for a finished non-streaming call.

        Every step runs under one guard because callers invoke this from a
        ``finally`` block: an exception here would replace the caller's
        response and skip the exit_tracking() that follows it.
        """
        try:
            usage = getattr(response, "usage", None)
            input_tokens, output_tokens = _usage_tokens(usage)
            self._emit(self._build_event(
                model, agent_name, input_tokens, output_tokens,
                int((time.time() - start_time) * 1000),
                input_hash, error_message,
                cached_tokens=_cached_input_tokens(usage),
                capabilities=capabilities,
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

    # ── Call surfaces ────────────────────────────────────────────

    def _patch(self, owner, attr: str, is_async: bool):
        """Patch one call surface, remembering how to restore it."""
        original = getattr(owner, attr, None)
        if original is None:
            return None
        wrapper = (
            self._make_async_tracked(original) if is_async
            else self._make_sync_tracked(original)
        )
        setattr(owner, attr, wrapper)
        self._patches.append((owner, attr, original))
        return original

    @staticmethod
    def _prepare_stream_usage(kwargs) -> bool:
        """
        Ask chat.completions for usage on streamed responses, which it omits
        otherwise. True when we injected it, so the extra chunk can be hidden.
        """
        if not kwargs.get("stream", False):
            return False
        # Test the value, not just the key. client.chat.completions.stream()
        # forwards stream_options unconditionally, defaulted to openai's `omit`
        # /NOT_GIVEN sentinel, so a membership check reads as "the caller set
        # this" on every .stream() call and we never ask for usage -- those
        # calls then report zero tokens. Anything that is not a real mapping
        # means the caller did not actually supply options.
        existing = kwargs.get("stream_options")
        if isinstance(existing, dict):
            return False
        # chat.completions only — identified by its `messages` argument. The
        # Responses API takes `input` and already reports usage on its terminal
        # event, so sending stream_options there is at best redundant.
        if "messages" not in kwargs:
            return False
        kwargs["stream_options"] = {"include_usage": True}
        return True

    # ── Sync create ──────────────────────────────────────────────

    def _make_sync_tracked(self, original: Callable) -> Callable:
        """Build the tracked wrapper shared by create(), parse() and responses."""
        interceptor = self

        @wraps(original)
        def tracked(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return original(client_self, *args, **kwargs)

            # Skip if we're already inside a higher-level interceptor
            if in_tracking():
                return original(client_self, *args, **kwargs)
            token = enter_tracking()

            model = kwargs.get("model", args[0] if args else "unknown")
            is_stream = kwargs.get("stream", False)
            agent_name = _get_effective_agent_name(config)
            input_hash = _hash_input(_extract_input_text(kwargs))
            caps = fingerprint(kwargs)
            injected_usage = interceptor._prepare_stream_usage(kwargs)

            start_time = time.time()
            error_message = None
            response = None

            try:
                try:
                    response = original(client_self, *args, **kwargs)
                except Exception as e:
                    # Some OpenAI-compatible gateways reject stream_options
                    # outright — retry as the caller originally wrote it rather
                    # than breaking their app.
                    if injected_usage and "stream_options" in str(e):
                        kwargs.pop("stream_options", None)
                        injected_usage = False
                        response = original(client_self, *args, **kwargs)
                    else:
                        raise

                if is_stream:
                    return _SyncStreamWrapper(
                        response, model, agent_name, input_hash, start_time,
                        interceptor, hide_usage_chunk=injected_usage,
                        capabilities=caps,
                    )

                return response

            except Exception as e:
                error_message = str(e)
                raise

            finally:
                # A streaming call that raised never produced a stream to wrap,
                # so its error is only recorded here.
                if not is_stream or error_message is not None:
                    interceptor._record_call(
                        response, model, agent_name, input_hash,
                        start_time, error_message, capabilities=caps,
                    )
                exit_tracking(token)

        return tracked

    # ── Async create ─────────────────────────────────────────────

    def _make_async_tracked(self, original: Callable) -> Callable:
        """Async twin of _make_sync_tracked."""
        interceptor = self

        @wraps(original)
        async def tracked(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return await original(client_self, *args, **kwargs)

            # Skip if we're already inside a higher-level interceptor
            if in_tracking():
                return await original(client_self, *args, **kwargs)
            token = enter_tracking()

            model = kwargs.get("model", args[0] if args else "unknown")
            is_stream = kwargs.get("stream", False)
            agent_name = _get_effective_agent_name(config)
            input_hash = _hash_input(_extract_input_text(kwargs))
            caps = fingerprint(kwargs)
            injected_usage = interceptor._prepare_stream_usage(kwargs)

            start_time = time.time()
            error_message = None
            response = None

            try:
                try:
                    response = await original(client_self, *args, **kwargs)
                except Exception as e:
                    # See the sync twin: retry without the injected option.
                    if injected_usage and "stream_options" in str(e):
                        kwargs.pop("stream_options", None)
                        injected_usage = False
                        response = await original(client_self, *args, **kwargs)
                    else:
                        raise

                if is_stream:
                    return _AsyncStreamWrapper(
                        response, model, agent_name, input_hash, start_time,
                        interceptor, hide_usage_chunk=injected_usage,
                        capabilities=caps,
                    )

                return response

            except Exception as e:
                error_message = str(e)
                raise

            finally:
                # See the sync wrapper: a failed stream is recorded here.
                if not is_stream or error_message is not None:
                    interceptor._record_call(
                        response, model, agent_name, input_hash,
                        start_time, error_message, capabilities=caps,
                    )
                exit_tracking(token)

        return tracked


# ── Stream wrappers ──────────────────────────────────────────────


class _StreamUsageMixin:
    """Bookkeeping shared by the sync and async Stream wrappers."""

    def __init__(self, stream, model, agent_name, input_hash, start_time,
                 interceptor, hide_usage_chunk=False, capabilities=None):
        self._stream = stream
        self._model = model
        self._agent_name = agent_name
        self._input_hash = input_hash
        self._start_time = start_time
        self._interceptor = interceptor
        self._capabilities = capabilities
        self._output_tokens = 0
        self._input_tokens = 0
        self._cached_tokens = 0
        self._accumulated_content = ""
        self._emitted = False
        self._hide_usage_chunk = hide_usage_chunk

    def __getattr__(self, name):
        # object.__getattribute__ so a missing _stream raises instead of
        # recursing forever.
        return getattr(object.__getattribute__(self, "_stream"), name)

    def _suppress(self, chunk) -> bool:
        """
        Hide the trailing usage-only chunk when WE asked for it.

        stream_options adds a final chunk with empty `choices`, which would
        crash caller code doing `chunk.choices[0]`. Callers who requested usage
        themselves still receive it.
        """
        if not self._hide_usage_chunk:
            return False
        return not getattr(chunk, "choices", None) and bool(getattr(chunk, "usage", None))

    def _process_chunk(self, chunk):
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                self._accumulated_content += delta.content
        # Responses API streams events rather than chat chunks; the terminal
        # one carries the completed response and its usage.
        if getattr(chunk, "type", None) in ("response.completed", "response.incomplete"):
            self._record_usage(getattr(getattr(chunk, "response", None), "usage", None))
        elif getattr(chunk, "usage", None):
            self._record_usage(chunk.usage)

    def _record_usage(self, usage):
        if not usage:
            return
        input_tokens, output_tokens = _usage_tokens(usage)
        if input_tokens:
            self._input_tokens = input_tokens
        if output_tokens:
            self._output_tokens = output_tokens
        self._cached_tokens = _cached_input_tokens(usage) or self._cached_tokens

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
            streaming=True, cached_tokens=self._cached_tokens,
            capabilities=self._capabilities,
        )
        self._interceptor._emit(event)


class _SyncStreamWrapper(_StreamUsageMixin):
    """Wraps an OpenAI sync Stream to capture metrics after full consumption."""

    def __iter__(self):
        # finally, not else: an early break (user hit stop, client
        # disconnected) throws GeneratorExit, a BaseException an `else` clause
        # never sees — and the call was billed regardless.
        try:
            for chunk in self._stream:
                self._process_chunk(chunk)
                if self._suppress(chunk):
                    continue
                yield chunk
        except Exception as e:
            self._emit_event(error_message=str(e))
            raise
        finally:
            self._emit_event()

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)
        self._emit_event()

    def close(self):
        try:
            if hasattr(self._stream, "close"):
                return self._stream.close()
        finally:
            self._emit_event()


class _AsyncStreamWrapper(_StreamUsageMixin):
    """Wraps an OpenAI async Stream to capture metrics after full consumption."""

    async def __aiter__(self):
        # See the sync wrapper: `else` misses GeneratorExit, so an abandoned
        # stream is never recorded.
        try:
            async for chunk in self._stream:
                self._process_chunk(chunk)
                if self._suppress(chunk):
                    continue
                yield chunk
        except Exception as e:
            self._emit_event(error_message=str(e))
            raise
        finally:
            self._emit_event()

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)
        self._emit_event()

    async def close(self):
        try:
            if hasattr(self._stream, "close"):
                return await self._stream.close()
        finally:
            self._emit_event()
