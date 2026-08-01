"""Instrumentation for the official Google Gen AI Python SDK.

The interceptor targets ``google-genai`` (imported as ``from google import
genai``), Google's current Gemini Developer API client.  It records usage
returned by Gemini rather than estimating tokens locally, including streamed
calls once the final usage-bearing chunk has been consumed.
"""

import hashlib
import inspect
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional

from .config import get_config
from .cost_calculator import calculate_cost
from ._reentrancy import in_tracking, enter_tracking, exit_tracking


def _hash_contents(contents: Any) -> str:
    """Create a stable hash without serialising potentially large binary media."""
    def extract(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            if isinstance(value.get("text"), str):
                return [value["text"]]
            return extract(value.get("parts", value.get("content", "")))
        if isinstance(value, (list, tuple)):
            result: list[str] = []
            for item in value:
                result.extend(extract(item))
            return result
        text = getattr(value, "text", None)
        if isinstance(text, str):
            return [text]
        parts = getattr(value, "parts", None)
        if parts is not None:
            return extract(parts)
        return []

    normalized = " ".join(extract(contents)).lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _effective_agent_name(config) -> str:
    from .tracker import _agent_name_var
    return _agent_name_var.get(None) or (config.default_agent_name if config else "default")


def _usage_counts(response: Any) -> tuple[int, int, int, int]:
    """Read the authoritative Gemini usage fields, tolerating SDK revisions.

    Returns ``(input, output, thinking, cached)``.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata") or response.get("usageMetadata")
    if usage is None:
        return 0, 0, 0, 0

    def value(*names: str) -> int:
        for name in names:
            candidate = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return candidate
        return 0

    # Gemini bills reasoning at the output rate but reports it *outside*
    # candidates_token_count, so reading only that field undercounts output
    # several-fold on gemini-2.5-*, where thinking is on by default.
    thinking = value("thoughts_token_count", "thoughtsTokenCount")
    return (
        value("prompt_token_count", "promptTokenCount"),
        value("candidates_token_count", "candidatesTokenCount") + thinking,
        thinking,
        # A subset of prompt_token_count, billed cheaper; reported so the
        # backend can correct a cost this module charges at the full rate.
        value("cached_content_token_count", "cachedContentTokenCount"),
    )


def _afc_turns(response: Any) -> int:
    """Count the billed round-trips ``generate_content`` made internally.

    Only the last turn's usage is recoverable, so the count is emitted to make
    the shortfall visible.
    """
    history = getattr(response, "automatic_function_calling_history", None)
    if not isinstance(history, (list, tuple)):
        return 1

    def is_tool_reply(part: Any) -> bool:
        if isinstance(part, dict):
            return part.get("function_response") is not None or part.get("functionResponse") is not None
        return getattr(part, "function_response", None) is not None

    # Each extra round-trip appends exactly one tool-reply Content to the
    # history; the turn that produced the final answer appends nothing.
    replies = 0
    for content in history:
        parts = content.get("parts") if isinstance(content, dict) else getattr(content, "parts", None)
        if isinstance(parts, (list, tuple)) and any(is_tool_reply(part) for part in parts):
            replies += 1
    return replies + 1


def _never_ran(stream: Any) -> bool:
    """True only when the wrapped stream provably never reached the network.

    ``generate_content_stream`` is a generator function (and its async twin
    returns an unstarted async generator), so no request is issued until the
    first ``next()``. Anything we cannot inspect is assumed to have run, because
    over-reporting a real call beats dropping one.
    """
    if inspect.isgenerator(stream):
        return inspect.getgeneratorstate(stream) == inspect.GEN_CREATED
    if inspect.isasyncgen(stream):
        if hasattr(inspect, "getasyncgenstate"):  # 3.12+
            return inspect.getasyncgenstate(stream) == inspect.AGEN_CREATED
        # Before 3.12 there is no accessor: an unstarted frame has not executed
        # an instruction yet, and an exhausted one has dropped its frame.
        frame = stream.ag_frame
        return frame is not None and frame.f_lasti < 0
    return False


class GeminiInterceptor:
    """Monkey-patch direct Gemini Developer API generation calls."""

    def __init__(self, event_callback: Callable[[dict], None]):
        self.event_callback = event_callback
        self.is_active = False
        self._models_cls = None
        self._async_models_cls = None
        self._original_generate_content = None
        self._original_generate_content_stream = None
        self._original_async_generate_content = None
        self._original_async_generate_content_stream = None

    def start(self) -> bool:
        if self.is_active:
            return True
        try:
            from google.genai.models import AsyncModels, Models

            self._models_cls = Models
            self._async_models_cls = AsyncModels
            self._original_generate_content = Models.generate_content
            self._original_generate_content_stream = Models.generate_content_stream
            self._original_async_generate_content = AsyncModels.generate_content
            self._original_async_generate_content_stream = AsyncModels.generate_content_stream

            Models.generate_content = self._tracked_generate_content()
            Models.generate_content_stream = self._tracked_generate_content_stream()
            AsyncModels.generate_content = self._tracked_async_generate_content()
            AsyncModels.generate_content_stream = self._tracked_async_generate_content_stream()
            self.is_active = True
            return True
        except ImportError:
            return False
        except Exception as error:
            config = get_config()
            if config and config.debug:
                print(f"[AgentCost] Failed to start Gemini interceptor: {error}")
            return False

    def stop(self) -> None:
        if not self.is_active:
            return
        self._models_cls.generate_content = self._original_generate_content
        self._models_cls.generate_content_stream = self._original_generate_content_stream
        self._async_models_cls.generate_content = self._original_async_generate_content
        self._async_models_cls.generate_content_stream = self._original_async_generate_content_stream
        self.is_active = False

    def _emit(self, model: str, agent_name: str, input_hash: str, start_time: float,
              response: Any = None, error_message: Optional[str] = None, streaming: bool = False) -> None:
        # One guard over the whole body: callers invoke this from a ``finally``
        # block, where an exception would replace the caller's response and
        # skip the exit_tracking() that follows it.
        try:
            input_tokens, output_tokens, thinking_tokens, cached_tokens = _usage_counts(response)
            event = {
                "agent_name": agent_name,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": calculate_cost(model, input_tokens, output_tokens),
                "latency_ms": int((time.time() - start_time) * 1000),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": error_message is None,
                "error": error_message,
                "input_hash": input_hash,
            }
            if streaming:
                event["streaming"] = True
            # Only attached when they apply, so the event shape is unchanged for
            # the non-thinking, non-cached, non-AFC calls that dominate traffic.
            if thinking_tokens:
                event["thinking_tokens"] = thinking_tokens
            if cached_tokens:
                event["cached_tokens"] = cached_tokens
            afc_turns = _afc_turns(response)
            if afc_turns > 1:
                event["afc_turns"] = afc_turns

            from .tracker import get_effective_metadata
            metadata = get_effective_metadata()
            if metadata:
                event["metadata"] = metadata
            self.event_callback(event)
        except Exception:
            config = get_config()
            if config and config.debug:
                import traceback
                traceback.print_exc()

    def _call_context(self, kwargs: dict) -> tuple[str, str, str, float]:
        config = get_config()
        return (
            kwargs.get("model", "unknown"),
            _effective_agent_name(config),
            _hash_contents(kwargs.get("contents")),
            time.time(),
        )

    def _tracked_generate_content(self) -> Callable:
        original, interceptor = self._original_generate_content, self
        @wraps(original)
        def wrapped(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return original(client_self, *args, **kwargs)
            if in_tracking():
                return original(client_self, *args, **kwargs)
            token = enter_tracking()
            model, agent, input_hash, started = interceptor._call_context(kwargs)
            response, error = None, None
            try:
                response = original(client_self, *args, **kwargs)
                return response
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                interceptor._emit(model, agent, input_hash, started, response, error)
                exit_tracking(token)
        return wrapped

    def _tracked_async_generate_content(self) -> Callable:
        original, interceptor = self._original_async_generate_content, self
        @wraps(original)
        async def wrapped(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return await original(client_self, *args, **kwargs)
            if in_tracking():
                return await original(client_self, *args, **kwargs)
            token = enter_tracking()
            model, agent, input_hash, started = interceptor._call_context(kwargs)
            response, error = None, None
            try:
                response = await original(client_self, *args, **kwargs)
                return response
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                interceptor._emit(model, agent, input_hash, started, response, error)
                exit_tracking(token)
        return wrapped

    def _tracked_generate_content_stream(self) -> Callable:
        original, interceptor = self._original_generate_content_stream, self
        @wraps(original)
        def wrapped(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return original(client_self, *args, **kwargs)
            if in_tracking():
                return original(client_self, *args, **kwargs)
            token = enter_tracking()
            model, agent, input_hash, started = interceptor._call_context(kwargs)
            try:
                stream = original(client_self, *args, **kwargs)
            except Exception as exc:
                interceptor._emit(model, agent, input_hash, started, error_message=str(exc), streaming=True)
                raise
            finally:
                # Release the recursion guard as soon as the underlying call
                # returns — never hold it across stream consumption. Holding it
                # meant a `break` (or an abandoned stream) leaked the depth and
                # silently killed tracking for the rest of the thread, and any
                # other LLM call made mid-iteration was skipped.
                exit_tracking(token)
            return _GeminiStream(stream, lambda response, error=None: interceptor._emit(
                model, agent, input_hash, started, response, error, streaming=True))
        return wrapped

    def _tracked_async_generate_content_stream(self) -> Callable:
        original, interceptor = self._original_async_generate_content_stream, self
        @wraps(original)
        async def wrapped(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return await original(client_self, *args, **kwargs)
            if in_tracking():
                return await original(client_self, *args, **kwargs)
            token = enter_tracking()
            model, agent, input_hash, started = interceptor._call_context(kwargs)
            try:
                stream = await original(client_self, *args, **kwargs)
            except Exception as exc:
                interceptor._emit(model, agent, input_hash, started, error_message=str(exc), streaming=True)
                raise
            finally:
                # See the sync variant: the guard must not span consumption.
                exit_tracking(token)
            return _GeminiAsyncStream(stream, lambda response, error=None: interceptor._emit(
                model, agent, input_hash, started, response, error, streaming=True))
        return wrapped


class _GeminiStream:
    def __init__(self, stream, emit):
        self._stream, self._emit = stream, emit
        self._last_response, self._done = None, False
        self._started = False

    def __iter__(self):
        # A generator body, so this runs on the first next() — exactly when the
        # underlying request is issued, not when iter() is merely called.
        self._started = True
        try:
            for response in self._stream:
                self._last_response = response
                yield response
        except Exception as exc:
            self._finish(str(exc))
            raise
        finally:
            # `finally` (not `else`) so that breaking out of the loop early —
            # which throws GeneratorExit, a BaseException — still records the
            # call. The request was billed by Gemini either way.
            self._finish()

    def _finish(self, error=None):
        if not self._done:
            self._done = True
            self._emit(self._last_response, error)

    def __del__(self):
        # A stream that was never iterated never hit the network, so there is
        # nothing to record. Only skip when that is provable (see _never_ran);
        # an unrecognisable stream is still recorded.
        try:
            if self._started or not _never_ran(self._stream):
                self._finish()
        except Exception:
            pass

    def __getattr__(self, name):
        # Never proxy private names — during __del__ or a half-built instance
        # this would recurse infinitely looking up self._stream.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._stream, name)


class _GeminiAsyncStream(_GeminiStream):
    async def __aiter__(self):
        # See the sync variant: first __anext__(), not the __aiter__() call.
        self._started = True
        try:
            async for response in self._stream:
                self._last_response = response
                yield response
        except Exception as exc:
            self._finish(str(exc))
            raise
        finally:
            self._finish()
