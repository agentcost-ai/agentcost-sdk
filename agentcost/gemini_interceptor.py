"""Instrumentation for the official Google Gen AI Python SDK.

The interceptor targets ``google-genai`` (imported as ``from google import
genai``), Google's current Gemini Developer API client.  It records usage
returned by Gemini rather than estimating tokens locally, including streamed
calls once the final usage-bearing chunk has been consumed.
"""

import hashlib
import threading
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional

from .config import get_config
from .cost_calculator import calculate_cost

try:
    from .openai_interceptor import _tracking_depth
except ImportError:
    _tracking_depth = threading.local()


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


def _usage_counts(response: Any) -> tuple[int, int]:
    """Read the authoritative Gemini usage fields, tolerating SDK revisions."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata") or response.get("usageMetadata")
    if usage is None:
        return 0, 0

    def value(*names: str) -> int:
        for name in names:
            candidate = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if candidate is not None:
                return int(candidate or 0)
        return 0

    return (
        value("prompt_token_count", "promptTokenCount"),
        value("candidates_token_count", "candidatesTokenCount"),
    )


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
        input_tokens, output_tokens = _usage_counts(response)
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
        try:
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
            if getattr(_tracking_depth, "value", 0):
                return original(client_self, *args, **kwargs)
            _tracking_depth.value = getattr(_tracking_depth, "value", 0) + 1
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
                _tracking_depth.value -= 1
        return wrapped

    def _tracked_async_generate_content(self) -> Callable:
        original, interceptor = self._original_async_generate_content, self
        @wraps(original)
        async def wrapped(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return await original(client_self, *args, **kwargs)
            if getattr(_tracking_depth, "value", 0):
                return await original(client_self, *args, **kwargs)
            _tracking_depth.value = getattr(_tracking_depth, "value", 0) + 1
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
                _tracking_depth.value -= 1
        return wrapped

    def _tracked_generate_content_stream(self) -> Callable:
        original, interceptor = self._original_generate_content_stream, self
        @wraps(original)
        def wrapped(client_self, *args, **kwargs):
            config = get_config()
            if config and not config.enabled:
                return original(client_self, *args, **kwargs)
            if getattr(_tracking_depth, "value", 0):
                return original(client_self, *args, **kwargs)
            _tracking_depth.value = getattr(_tracking_depth, "value", 0) + 1
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
                _tracking_depth.value = getattr(_tracking_depth, "value", 1) - 1
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
            if getattr(_tracking_depth, "value", 0):
                return await original(client_self, *args, **kwargs)
            _tracking_depth.value = getattr(_tracking_depth, "value", 0) + 1
            model, agent, input_hash, started = interceptor._call_context(kwargs)
            try:
                stream = await original(client_self, *args, **kwargs)
            except Exception as exc:
                interceptor._emit(model, agent, input_hash, started, error_message=str(exc), streaming=True)
                raise
            finally:
                # See the sync variant: the guard must not span consumption.
                _tracking_depth.value = getattr(_tracking_depth, "value", 1) - 1
            return _GeminiAsyncStream(stream, lambda response, error=None: interceptor._emit(
                model, agent, input_hash, started, response, error, streaming=True))
        return wrapped


class _GeminiStream:
    def __init__(self, stream, emit):
        self._stream, self._emit = stream, emit
        self._last_response, self._done = None, False

    def __iter__(self):
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
        # Stream created but never iterated at all: still record the call.
        try:
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
        try:
            async for response in self._stream:
                self._last_response = response
                yield response
        except Exception as exc:
            self._finish(str(exc))
            raise
        finally:
            self._finish()
