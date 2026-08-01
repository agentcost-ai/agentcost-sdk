"""
Tests for AgentCost SDK

Run with: pytest tests/ -v
"""

import asyncio
import json
import pytest
import time
from typing import Callable, Optional
from unittest.mock import Mock, patch

from agentcost import (
    track_costs,
    TokenCounter,
    CostCalculator,
    HybridBatcher,
    LocalBatcher,
    AgentCostConfig,
)

# langchain-core is optional, and an unguarded import of it takes the whole
# module down — including the OpenAI/Anthropic tests, which do not need it.
# pytest.importorskip cannot be used here: at module scope it skips the file.
try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk
    _HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover - only without langchain-core installed
    BaseChatModel, AIMessage, AIMessageChunk = object, None, None
    _HAS_LANGCHAIN = False


def _user_msgs():
    """The one-message prompt every interceptor test sends."""
    return [{"role": "user", "content": "hi"}]


def _assert_single_usage_event(events, input_tokens, output_tokens):
    assert len(events) == 1, f"expected exactly 1 event, got {len(events)}"
    assert events[0]["input_tokens"] == input_tokens
    assert events[0]["output_tokens"] == output_tokens
    assert events[0]["cost"] > 0


def _slow_async_transport(handler, delay=0.05):
    """A transport with real latency, so concurrent requests genuinely overlap."""
    import httpx

    class _SlowTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await request.aread()
            await asyncio.sleep(delay)
            return handler(request)

    return _SlowTransport()


def _gather_tracked(interceptor_cls, make_client, make_call, count=5):
    """Run `count` overlapping async calls under the interceptor, return events."""
    events = []
    interceptor = interceptor_cls(events.append)
    assert interceptor.start()

    async def run():
        client = make_client()
        await asyncio.gather(*[make_call(client) for _ in range(count)])

    try:
        asyncio.run(run())
    finally:
        interceptor.stop()
    return events


class TestTokenCounter:
    """Tests for token counting functionality"""
    
    def test_count_simple_text(self):
        """Test counting tokens in simple text"""
        text = "Hello, world!"
        count = TokenCounter.count_tokens(text, "gpt-4")
        assert count > 0
        assert count < 10  # "Hello, world!" is ~4 tokens
    
    def test_count_empty_text(self):
        """Test counting tokens in empty text"""
        count = TokenCounter.count_tokens("", "gpt-4")
        assert count == 0
    
    def test_count_long_text(self):
        """Test counting tokens in longer text"""
        text = "This is a longer piece of text that should have more tokens. " * 10
        count = TokenCounter.count_tokens(text, "gpt-4")
        assert count > 100
    
    def test_unknown_model_fallback(self):
        """Test that unknown models fall back to cl100k_base"""
        text = "Hello"
        count = TokenCounter.count_tokens(text, "unknown-model-xyz")
        assert count > 0
    
    def test_extract_text_from_string(self):
        """Test extracting text from string input"""
        text = TokenCounter.extract_text_from_input("Hello")
        assert text == "Hello"
    
    def test_extract_text_from_list(self):
        """Test extracting text from list of messages"""
        messages = [
            Mock(content="Hello"),
            Mock(content="World"),
        ]
        text = TokenCounter.extract_text_from_input(messages)
        assert "Hello" in text
        assert "World" in text


class TestCostCalculator:
    """Tests for cost calculation functionality"""
    
    def test_calculate_gpt4_cost(self):
        """Test GPT-4 cost calculation"""
        calc = CostCalculator()
        cost = calc.calculate_cost("gpt-4", 1000, 1000)
        # GPT-4: $0.03/1K input + $0.06/1K output = $0.09
        assert cost == pytest.approx(0.09, rel=0.01)
    
    def test_calculate_gpt35_cost(self):
        """Test GPT-3.5 cost calculation"""
        calc = CostCalculator()
        cost = calc.calculate_cost("gpt-3.5-turbo", 1000, 1000)
        # GPT-3.5: $0.0005/1K input + $0.0015/1K output = $0.002
        assert cost == pytest.approx(0.002, rel=0.01)
    
    def test_calculate_zero_tokens(self):
        """Test cost with zero tokens"""
        calc = CostCalculator()
        cost = calc.calculate_cost("gpt-4", 0, 0)
        assert cost == 0
    
    def test_custom_pricing(self):
        """Test custom pricing override"""
        custom = {"my-model": {"input": 0.1, "output": 0.2}}
        calc = CostCalculator(custom_pricing=custom)
        cost = calc.calculate_cost("my-model", 1000, 1000)
        assert cost == pytest.approx(0.3, rel=0.01)
    
    def test_cost_breakdown(self):
        """Test getting cost breakdown"""
        calc = CostCalculator()
        breakdown = calc.get_cost_breakdown("gpt-4", 1000, 500)
        
        assert "input_cost" in breakdown
        assert "output_cost" in breakdown
        assert "total_cost" in breakdown
        assert breakdown["total_cost"] == breakdown["input_cost"] + breakdown["output_cost"]


class TestLocalBatcher:
    """Tests for local batcher functionality"""
    
    def test_add_event(self):
        """Test adding events to batcher"""
        batcher = LocalBatcher(batch_size=10, flush_interval=60)
        
        batcher.add({"test": "event1"})
        batcher.add({"test": "event2"})
        
        batcher.flush()
        events = batcher.get_all_events()
        
        assert len(events) == 2
        batcher.shutdown()
    
    def test_auto_flush_on_size(self):
        """Test auto-flush when batch size is reached"""
        batcher = LocalBatcher(batch_size=3, flush_interval=60)
        
        for i in range(5):
            batcher.add({"event": i})
        
        batcher.flush()
        events = batcher.get_all_events()
        
        assert len(events) == 5
        batcher.shutdown()
    
    def test_stats(self):
        """Test getting batcher stats"""
        batcher = LocalBatcher(batch_size=10, flush_interval=60)
        
        batcher.add({"test": "event"})
        stats = batcher.get_stats()
        
        assert "events_added" in stats
        assert stats["events_added"] >= 1
        batcher.shutdown()


class TestAgentCostConfig:
    """Tests for configuration"""
    
    def test_default_config(self):
        """Test creating config with defaults"""
        config = AgentCostConfig(
            api_key="test_key",
            project_id="test_project"
        )
        
        assert config.api_key == "test_key"
        assert config.project_id == "test_project"
        assert config.batch_size == 10
        assert config.flush_interval == 5.0
        assert config.enabled == True
    
    def test_custom_config(self):
        """Test creating config with custom values"""
        config = AgentCostConfig(
            api_key="test_key",
            project_id="test_project",
            batch_size=20,
            flush_interval=10.0,
            debug=True,
        )
        
        assert config.batch_size == 20
        assert config.flush_interval == 10.0
        assert config.debug == True
    
    def test_get_pricing(self):
        """Test getting model pricing"""
        config = AgentCostConfig(
            api_key="test",
            project_id="test"
        )
        
        pricing = config.get_pricing("gpt-4")
        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] > 0


class TestTrackCostsLocalMode:
    """Tests for track_costs in local mode"""
    
    def test_init_local_mode(self):
        """Test initializing in local mode"""
        track_costs.init(local_mode=True, debug=False)
        
        assert track_costs._tracker.is_active
        
        track_costs.shutdown()
    
    def test_get_stats(self):
        """Test getting stats"""
        track_costs.init(local_mode=True, debug=False)
        
        stats = track_costs.get_stats()
        
        assert "initialized" in stats
        assert stats["initialized"] == True
        
        track_costs.shutdown()
    
    def test_agent_context_manager(self):
        """Test agent context manager"""
        track_costs.init(local_mode=True, debug=False)
        
        from agentcost.config import get_config
        
        with track_costs.agent("test-agent"):
            from agentcost.tracker import _agent_name_var
            assert _agent_name_var.get() == "test-agent"
        
        track_costs.shutdown()
    
    def test_metadata_context_manager(self):
        """Test metadata context manager"""
        track_costs.init(local_mode=True, debug=False)
        
        from agentcost.config import get_config
        
        with track_costs.metadata(conversation_id="conv-123", user_id="user-456"):
            from agentcost.tracker import get_effective_metadata
            metadata = get_effective_metadata()
            assert metadata.get("conversation_id") == "conv-123"
            assert metadata.get("user_id") == "user-456"
        
        # Metadata should be cleared after context exits
        from agentcost.tracker import get_effective_metadata
        assert "conversation_id" not in get_effective_metadata()
        
        track_costs.shutdown()


class TestInterceptor:
    """Tests for LangChain interceptor"""
    
    def test_interceptor_start_stop(self):
        """Test starting and stopping interceptor"""
        from agentcost.interceptor import LangChainInterceptor
        
        events = []
        interceptor = LangChainInterceptor(event_callback=lambda e: events.append(e))
        
        # Start
        success = interceptor.start()
        assert success == True
        assert interceptor.is_active == True
        
        # Stop
        interceptor.stop()
        assert interceptor.is_active == False
    
    def test_interceptor_idempotent_start(self):
        """Test that starting twice is safe"""
        from agentcost.interceptor import LangChainInterceptor
        
        events = []
        interceptor = LangChainInterceptor(event_callback=lambda e: events.append(e))
        
        # Start twice
        interceptor.start()
        interceptor.start()
        assert interceptor.is_active == True
        
        # Cleanup
        interceptor.stop()


class TestHTTPClient:
    """Tests for HTTP client"""
    
    def test_rate_limiter(self):
        """Test rate limiter allows requests within limit"""
        from agentcost.http_client import RateLimiter
        import time
        
        limiter = RateLimiter(max_requests=5, window_seconds=1.0)
        
        # First 5 requests should be instant
        for i in range(5):
            wait = limiter.acquire()
            assert wait == 0.0
        
        # 6th request should need to wait
        wait = limiter.acquire()
        assert wait > 0.0
    
    def test_mock_http_client(self):
        """Test mock HTTP client stores events"""
        from agentcost.http_client import MockHTTPClient
        
        client = MockHTTPClient(debug=False)
        
        # Send some events
        events = [{"test": 1}, {"test": 2}]
        success = client.send_events("proj-123", events)
        
        assert success == True
        assert len(client.get_all_events()) == 2
        assert client.send_count == 1
        
        # Clear
        client.clear()
        assert len(client.get_all_events()) == 0


class TestNewModelPricing:
    """Test that new model pricing is available"""
    
    def test_o1_pricing_exists(self):
        """Test OpenAI o1 model pricing"""
        from agentcost.config import DEFAULT_PRICING
        
        assert 'o1' in DEFAULT_PRICING
        assert 'o1-mini' in DEFAULT_PRICING
        assert DEFAULT_PRICING['o1']['input'] > 0
    
    def test_deepseek_pricing_exists(self):
        """Test DeepSeek model pricing"""
        from agentcost.config import DEFAULT_PRICING
        
        assert 'deepseek-chat' in DEFAULT_PRICING
        assert 'deepseek-reasoner' in DEFAULT_PRICING
    
    def test_gemini_flash_pricing_exists(self):
        """Test Gemini Flash pricing"""
        from agentcost.config import DEFAULT_PRICING
        
        assert 'gemini-1.5-flash' in DEFAULT_PRICING
        assert 'gemini-2.0-flash' in DEFAULT_PRICING
    
    def test_mistral_pricing_exists(self):
        """Test Mistral pricing"""
        from agentcost.config import DEFAULT_PRICING
        
        assert 'mistral-small' in DEFAULT_PRICING
        assert 'mistral-large' in DEFAULT_PRICING


class TestGeminiInterceptor:
    """Verify direct Google Gen AI SDK usage is captured without network calls."""

    @staticmethod
    def _response(prompt_tokens=12, output_tokens=8):
        return Mock(
            usage_metadata=Mock(
                prompt_token_count=prompt_tokens,
                candidates_token_count=output_tokens,
            )
        )

    def test_direct_generation_uses_gemini_usage_metadata(self):
        from agentcost.gemini_interceptor import GeminiInterceptor

        events = []
        interceptor = GeminiInterceptor(events.append)
        response = self._response()
        interceptor._original_generate_content = lambda _client, **_kwargs: response

        wrapped = interceptor._tracked_generate_content()
        assert wrapped(None, model="gemini-2.0-flash", contents="Hello") is response

        assert len(events) == 1
        assert events[0]["model"] == "gemini-2.0-flash"
        assert events[0]["input_tokens"] == 12
        assert events[0]["output_tokens"] == 8
        assert events[0]["total_tokens"] == 20
        assert events[0]["cost"] > 0

    def test_streaming_generation_emits_after_final_usage_chunk(self):
        from agentcost.gemini_interceptor import GeminiInterceptor

        events = []
        interceptor = GeminiInterceptor(events.append)
        chunks = [self._response(0, 0), self._response(15, 9)]
        interceptor._original_generate_content_stream = lambda _client, **_kwargs: iter(chunks)

        wrapped = interceptor._tracked_generate_content_stream()
        assert list(wrapped(None, model="gemini-2.0-flash", contents="Hello")) == chunks

        assert len(events) == 1
        assert events[0]["streaming"] is True
        assert events[0]["input_tokens"] == 15
        assert events[0]["output_tokens"] == 9

    def test_breaking_out_of_stream_still_records_and_frees_the_guard(self):
        """Regression: the recursion guard was held for the whole stream, so a
        `break` leaked it and permanently disabled tracking for that thread."""
        from agentcost.gemini_interceptor import GeminiInterceptor
        from agentcost._reentrancy import _tracking_depth

        _tracking_depth.set(0)
        events = []
        interceptor = GeminiInterceptor(events.append)
        interceptor._original_generate_content_stream = (
            lambda _client, **_kwargs: iter([self._response(0, 0), self._response(15, 9)])
        )
        interceptor._original_generate_content = lambda _client, **_kwargs: self._response()

        for _chunk in interceptor._tracked_generate_content_stream()(
            None, model="gemini-2.0-flash", contents="Hello"
        ):
            break  # abandon after the first chunk

        assert len(events) == 1, "an abandoned stream must still record the call"
        assert _tracking_depth.get() == 0, "recursion guard leaked"

        # The next ordinary call must still be tracked.
        interceptor._tracked_generate_content()(None, model="gemini-2.0-flash", contents="next")
        assert len(events) == 2, "tracking died after an abandoned stream"

    def test_stream_never_iterated_is_still_recorded(self):
        import gc

        from agentcost.gemini_interceptor import GeminiInterceptor
        from agentcost._reentrancy import _tracking_depth

        _tracking_depth.set(0)
        events = []
        interceptor = GeminiInterceptor(events.append)
        interceptor._original_generate_content_stream = (
            lambda _client, **_kwargs: iter([self._response(11, 4)])
        )

        stream = interceptor._tracked_generate_content_stream()(
            None, model="gemini-2.0-flash", contents="Hello"
        )
        del stream
        gc.collect()

        assert len(events) == 1
        assert _tracking_depth.get() == 0

    def test_call_made_during_stream_consumption_is_tracked(self):
        """The guard must not span consumption, or interleaved calls vanish."""
        from agentcost.gemini_interceptor import GeminiInterceptor
        from agentcost._reentrancy import _tracking_depth

        _tracking_depth.set(0)
        events = []
        interceptor = GeminiInterceptor(events.append)
        interceptor._original_generate_content_stream = (
            lambda _client, **_kwargs: iter([self._response(0, 0), self._response(15, 9)])
        )
        interceptor._original_generate_content = lambda _client, **_kwargs: self._response()

        for _chunk in interceptor._tracked_generate_content_stream()(
            None, model="gemini-2.0-flash", contents="outer"
        ):
            interceptor._tracked_generate_content()(
                None, model="gemini-2.0-flash", contents="inner"
            )
            break

        assert len([e for e in events if not e.get("streaming")]) == 1

    def test_stream_creation_failure_is_recorded(self):
        from agentcost.gemini_interceptor import GeminiInterceptor

        events = []
        interceptor = GeminiInterceptor(events.append)

        def fail(_client, **_kwargs):
            raise RuntimeError("Gemini unavailable")

        interceptor._original_generate_content_stream = fail
        with pytest.raises(RuntimeError, match="Gemini unavailable"):
            interceptor._tracked_generate_content_stream()(None, model="gemini-2.0-flash", contents="Hello")

        assert len(events) == 1
        assert events[0]["success"] is False
        assert events[0]["streaming"] is True

    @pytest.mark.asyncio
    async def test_async_generation_uses_gemini_usage_metadata(self):
        from agentcost.gemini_interceptor import GeminiInterceptor

        events = []
        interceptor = GeminiInterceptor(events.append)
        response = self._response(14, 6)

        async def generate(_client, **_kwargs):
            return response

        interceptor._original_async_generate_content = generate
        wrapped = interceptor._tracked_async_generate_content()
        assert await wrapped(None, model="gemini-2.0-flash", contents="Hello") is response

        assert len(events) == 1
        assert events[0]["input_tokens"] == 14
        assert events[0]["output_tokens"] == 6


class TestCostCalculatorEdgeCases:
    """Test edge cases in cost calculator"""
    
    def test_unknown_model_returns_zero(self):
        """Unknown model should return zero cost (not throw error)"""
        calc = CostCalculator()
        cost = calc.calculate_cost("totally-unknown-model-xyz", 1000, 1000)
        assert cost == 0.0
    
    def test_partial_model_match(self):
        """Test partial model name matching"""
        calc = CostCalculator()
        
        # 'gpt-4-0613' should match 'gpt-4'
        cost = calc.calculate_cost("gpt-4-0613", 1000, 500)
        assert cost > 0
    
    def test_claude_partial_match(self):
        """Test Claude partial matching"""
        calc = CostCalculator()
        
        # 'claude-3-5-sonnet-20241022' should match 'claude-3-5-sonnet'
        cost = calc.calculate_cost("claude-3-5-sonnet-20241022", 1000, 500)
        assert cost > 0


class TestAnthropicInterceptor:
    """Every way a real app calls Claude, against a mocked transport."""

    MODEL = "claude-3-5-sonnet-20241022"
    IN_TOK, OUT_TOK = 25, 15

    @classmethod
    def _message_json(cls):
        return {
            "id": "msg_1", "type": "message", "role": "assistant", "model": cls.MODEL,
            "content": [{"type": "text", "text": "Hello there"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": cls.IN_TOK, "output_tokens": cls.OUT_TOK},
        }

    @classmethod
    def _sse_bytes(cls):
        import json

        start = {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": cls.MODEL,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": cls.IN_TOK, "output_tokens": 1}}}
        frames = [
            ("message_start", start),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "Hello there"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": {"output_tokens": cls.OUT_TOK}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        return "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in frames).encode()

    @classmethod
    def _handler(cls, request):
        import json

        import httpx

        if json.loads(request.content.decode()).get("stream", False):
            return httpx.Response(200, content=cls._sse_bytes(),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=cls._message_json())

    @pytest.fixture
    def tracked(self):
        """(client, events) with the interceptor active against a mock transport."""
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        from anthropic import Anthropic

        from agentcost.anthropic_interceptor import AnthropicInterceptor

        events = []
        interceptor = AnthropicInterceptor(events.append)
        assert interceptor.start(), "Anthropic SDK not importable"
        client = Anthropic(
            api_key="test",
            http_client=httpx.Client(transport=httpx.MockTransport(self._handler)),
        )
        try:
            yield client, events
        finally:
            interceptor.stop()

    def _assert_usage(self, events):
        _assert_single_usage_event(events, self.IN_TOK, self.OUT_TOK)

    def test_plain_create_is_tracked(self, tracked):
        client, events = tracked
        client.messages.create(model=self.MODEL, max_tokens=100, messages=_user_msgs())
        self._assert_usage(events)

    def test_create_with_stream_true_records_usage(self, tracked):
        client, events = tracked
        for _ in client.messages.create(model=self.MODEL, max_tokens=100,
                                        messages=_user_msgs(), stream=True):
            pass
        self._assert_usage(events)
        assert events[0]["streaming"] is True

    def test_stream_helper_consumed_via_text_stream(self, tracked):
        client, events = tracked
        with client.messages.stream(model=self.MODEL, max_tokens=100,
                                    messages=_user_msgs()) as stream:
            for _ in stream.text_stream:
                pass
        self._assert_usage(events)

    def test_stream_helper_consumed_via_get_final_message(self, tracked):
        client, events = tracked
        with client.messages.stream(model=self.MODEL, max_tokens=100,
                                    messages=_user_msgs()) as stream:
            stream.get_final_message()
        self._assert_usage(events)

    def test_stream_that_dies_mid_iteration_records_the_error(self):
        """The request was billed before the connection dropped."""
        pytest.importorskip("anthropic")
        from agentcost.anthropic_interceptor import (
            AnthropicInterceptor, _AnthropicRawStreamWrapper)

        def exploding():
            yield Mock(spec=["type"], type="ping")
            raise RuntimeError("connection reset")

        events = []
        wrapper = _AnthropicRawStreamWrapper(
            exploding(), self.MODEL, "agent", "hash", time.time(),
            AnthropicInterceptor(events.append),
        )
        with pytest.raises(RuntimeError):
            list(wrapper)

        assert len(events) == 1
        assert events[0]["success"] is False
        assert "connection reset" in events[0]["error"]

    @pytest.mark.asyncio
    async def test_async_stream_that_dies_mid_iteration_records_the_error(self):
        pytest.importorskip("anthropic")
        from agentcost.anthropic_interceptor import (
            AnthropicInterceptor, _AnthropicAsyncRawStreamWrapper)

        async def exploding():
            yield Mock(spec=["type"], type="ping")
            raise RuntimeError("connection reset")

        events = []
        wrapper = _AnthropicAsyncRawStreamWrapper(
            exploding(), self.MODEL, "agent", "hash", time.time(),
            AnthropicInterceptor(events.append),
        )
        with pytest.raises(RuntimeError):
            async for _ in wrapper:
                pass

        assert len(events) == 1
        assert events[0]["success"] is False
        assert "connection reset" in events[0]["error"]

    def test_failure_inside_the_stream_block_is_recorded(self, tracked):
        client, events = tracked
        with pytest.raises(RuntimeError):
            with client.messages.stream(model=self.MODEL, max_tokens=100,
                                        messages=_user_msgs()) as stream:
                stream.get_final_message()
                raise RuntimeError("consumer blew up")

        assert len(events) == 1
        assert events[0]["success"] is False
        assert "consumer blew up" in events[0]["error"]

    def test_async_stream_is_a_context_manager_not_a_coroutine(self):
        """`async with client.messages.stream(...)` must not raise."""
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        from anthropic import AsyncAnthropic

        from agentcost.anthropic_interceptor import AnthropicInterceptor

        events = []
        interceptor = AnthropicInterceptor(events.append)
        assert interceptor.start()

        async def run():
            client = AsyncAnthropic(
                api_key="test",
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(self._handler)),
            )
            async with client.messages.stream(model=self.MODEL, max_tokens=100,
                                              messages=_user_msgs()) as stream:
                async for _ in stream.text_stream:
                    pass

        try:
            asyncio.run(run())
        finally:
            interceptor.stop()

        self._assert_usage(events)

    def test_concurrent_async_calls_are_all_tracked(self):
        """A thread-local guard let one in-flight call suppress its siblings."""
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        from anthropic import AsyncAnthropic

        from agentcost.anthropic_interceptor import AnthropicInterceptor

        events = _gather_tracked(
            AnthropicInterceptor,
            lambda: AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(
                transport=_slow_async_transport(self._handler))),
            lambda client: client.messages.create(
                model=self.MODEL, max_tokens=100, messages=_user_msgs()),
        )

        assert len(events) == 5, f"only {len(events)}/5 concurrent calls tracked"

    def test_concurrent_streaming_calls_are_all_tracked(self):
        """The reported production failure, end to end.

        A chatbot serves turns concurrently on one event loop and streams the
        reply. Streaming is the gap the non-streaming test above cannot cover:
        the event is emitted when the stream finishes, long after create()
        returned, so several turns are mid-stream at once. Under the old
        thread-local guard this reported 1/5.
        """
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        from anthropic import AsyncAnthropic

        from agentcost.anthropic_interceptor import AnthropicInterceptor

        async def one_turn(client):
            stream = await client.messages.create(
                model=self.MODEL, max_tokens=100, messages=_user_msgs(), stream=True
            )
            async for _ in stream:  # the chatbot relaying tokens to its user
                pass

        events = _gather_tracked(
            AnthropicInterceptor,
            lambda: AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(
                transport=_slow_async_transport(self._handler))),
            one_turn,
        )

        assert len(events) == 5, f"only {len(events)}/5 concurrent streamed turns tracked"
        assert all(e["output_tokens"] == self.OUT_TOK for e in events), \
            f"streamed usage lost: {[e['output_tokens'] for e in events]}"

    def test_concurrent_stream_helper_calls_are_all_tracked(self):
        """Same, via `async with client.messages.stream(...)`.

        This is the helper Anthropic's own docs lead with, so it is the most
        likely shape of a real integration.
        """
        pytest.importorskip("anthropic")
        httpx = pytest.importorskip("httpx")
        from anthropic import AsyncAnthropic

        from agentcost.anthropic_interceptor import AnthropicInterceptor

        async def one_turn(client):
            async with client.messages.stream(
                model=self.MODEL, max_tokens=100, messages=_user_msgs()
            ) as stream:
                async for _ in stream.text_stream:
                    pass

        events = _gather_tracked(
            AnthropicInterceptor,
            lambda: AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(
                transport=_slow_async_transport(self._handler))),
            one_turn,
        )

        assert len(events) == 5, f"only {len(events)}/5 concurrent stream() turns tracked"


class TestGeminiUsageAccuracy:
    """Regressions for Gemini calls that were billed for the wrong token counts."""

    MODEL = "gemini-2.0-flash"

    @staticmethod
    def _usage(**fields):
        """A usage_metadata double that only answers to the fields it was given.

        `spec` matters: a bare Mock invents a value for every attribute name, so
        it cannot show whether a field was really read off the response.
        """
        usage = Mock(spec=list(fields))
        for name, value in fields.items():
            setattr(usage, name, value)
        return usage

    @classmethod
    def _response(cls, prompt=10, output=5, afc_history=None, **extra_usage):
        usage = cls._usage(
            prompt_token_count=prompt, candidates_token_count=output, **extra_usage
        )
        spec = ["usage_metadata"] + (["automatic_function_calling_history"] if afc_history is not None else [])
        response = Mock(spec=spec)
        response.usage_metadata = usage
        if afc_history is not None:
            response.automatic_function_calling_history = afc_history
        return response

    @staticmethod
    def _afc_history(turns):
        """Mimic the history generate_content builds: one tool-reply per extra turn."""
        history = [Mock(spec=["parts"], parts=[Mock(spec=["text"], text="hi")])]
        for _ in range(turns - 1):
            history.append(Mock(spec=["parts"], parts=[Mock(spec=["function_call"], function_call=object())]))
            history.append(Mock(spec=["parts"], parts=[Mock(spec=["function_response"], function_response=object())]))
        return history

    def _emit_once(self, response):
        from agentcost.gemini_interceptor import GeminiInterceptor

        events = []
        interceptor = GeminiInterceptor(events.append)
        interceptor._original_generate_content = lambda _client, **_kwargs: response
        interceptor._tracked_generate_content()(None, model=self.MODEL, contents="Hello")
        assert len(events) == 1
        return events[0]

    # --- Bug 1: thinking tokens ------------------------------------------------

    def test_thinking_tokens_are_billed_as_output(self):
        """thoughts_token_count is charged at the output rate and is NOT part of
        candidates_token_count, so ignoring it undercounted output ~19x here."""
        from agentcost.cost_calculator import calculate_cost

        event = self._emit_once(self._response(1000, 50, thoughts_token_count=900))

        assert event["output_tokens"] == 950
        assert event["thinking_tokens"] == 900
        assert event["input_tokens"] == 1000
        assert event["total_tokens"] == 1950
        assert event["cost"] == calculate_cost(self.MODEL, 1000, 950)
        assert event["cost"] > calculate_cost(self.MODEL, 1000, 50)

    def test_response_without_thinking_keeps_the_old_event_shape(self):
        event = self._emit_once(self._response(12, 8))

        assert event["output_tokens"] == 8
        assert "thinking_tokens" not in event
        assert "cached_tokens" not in event
        assert "afc_turns" not in event

    def test_fields_the_response_does_not_carry_are_not_invented(self):
        """Absent usage fields must read as absent, not as a fabricated count."""
        event = self._emit_once(self._response(7, 3))

        assert event["input_tokens"] == 7
        assert event["output_tokens"] == 3
        assert "thinking_tokens" not in event
        assert "cached_tokens" not in event

    # --- Bug 2: cached input ---------------------------------------------------

    def test_cached_prompt_tokens_are_surfaced(self):
        """Cached input is far cheaper but is a subset of prompt_token_count; the
        count must reach the backend so the overcharge can be corrected there."""
        from agentcost.cost_calculator import calculate_cost

        event = self._emit_once(self._response(1000, 20, cached_content_token_count=800))

        assert event["cached_tokens"] == 800
        assert event["input_tokens"] == 1000, "cached tokens must not be double counted"
        # Documented limitation: still priced at the full input rate.
        assert event["cost"] == calculate_cost(self.MODEL, 1000, 20)

    # --- Bug 3: automatic function calling -------------------------------------

    def test_afc_round_trips_are_flagged(self):
        """generate_content loops internally and bills every turn, but reports
        only the last turn's usage. The count makes the shortfall visible."""
        event = self._emit_once(self._response(30, 10, afc_history=self._afc_history(3)))

        assert event["afc_turns"] == 3
        assert event["input_tokens"] == 30, "only the final turn's usage is recoverable"

    def test_single_turn_call_is_not_flagged_as_afc(self):
        assert "afc_turns" not in self._emit_once(self._response(30, 10, afc_history=[]))

    # --- Bug 4: phantom events for streams that never ran ----------------------

    @staticmethod
    def _generator_stream(chunks, calls):
        """Match the real SDK: generate_content_stream is a generator function,
        so no request is issued until the first next()."""
        def stream():
            calls.append("requested")
            for chunk in chunks:
                yield chunk
        return stream()

    def _stream_interceptor(self, chunks, calls):
        from agentcost.gemini_interceptor import GeminiInterceptor
        from agentcost._reentrancy import _tracking_depth

        _tracking_depth.set(0)
        events = []
        interceptor = GeminiInterceptor(events.append)
        interceptor._original_generate_content_stream = (
            lambda _client, **_kwargs: self._generator_stream(chunks, calls)
        )
        return interceptor, events

    def test_unconsumed_stream_emits_no_phantom_event(self):
        """Creating a stream and never iterating it made no API call, yet a fake
        0-token `success: True` event was emitted for it."""
        import gc

        calls = []
        interceptor, events = self._stream_interceptor([self._response(11, 4)], calls)

        stream = interceptor._tracked_generate_content_stream()(
            None, model=self.MODEL, contents="Hello"
        )
        del stream
        gc.collect()

        assert calls == [], "no request should have been made"
        assert events == [], f"phantom event for a call that never happened: {events}"

    def test_consumed_stream_is_still_recorded(self):
        calls = []
        chunks = [self._response(0, 0), self._response(15, 9)]
        interceptor, events = self._stream_interceptor(chunks, calls)

        assert list(interceptor._tracked_generate_content_stream()(
            None, model=self.MODEL, contents="Hello"
        )) == chunks

        assert calls == ["requested"]
        assert len(events) == 1
        assert events[0]["input_tokens"] == 15
        assert events[0]["output_tokens"] == 9

    def test_stream_abandoned_after_one_chunk_is_still_recorded(self):
        """The request was billed even though the caller walked away."""
        import gc

        calls = []
        interceptor, events = self._stream_interceptor(
            [self._response(0, 0), self._response(15, 9)], calls
        )

        for _chunk in interceptor._tracked_generate_content_stream()(
            None, model=self.MODEL, contents="Hello"
        ):
            break
        gc.collect()

        assert calls == ["requested"]
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_unconsumed_async_stream_emits_no_phantom_event(self):
        import gc

        from agentcost.gemini_interceptor import GeminiInterceptor
        from agentcost._reentrancy import _tracking_depth

        _tracking_depth.set(0)
        calls, events = [], []
        interceptor = GeminiInterceptor(events.append)

        async def create(_client, **_kwargs):
            async def stream():
                calls.append("requested")
                yield self._response(11, 4)
            return stream()

        interceptor._original_async_generate_content_stream = create
        stream = await interceptor._tracked_async_generate_content_stream()(
            None, model=self.MODEL, contents="Hello"
        )
        del stream
        gc.collect()

        assert calls == []
        assert events == [], f"phantom event for a call that never happened: {events}"


# ── LangChain streaming regression cover ─────────────────────────────
# BaseChatModel is `object` when langchain-core is absent; the test class below
# is skipped in that case, so these are never instantiated.


class _StreamingFakeChat(BaseChatModel):
    """A chat model shaped exactly like langchain-openai's default stream.

    Content chunks carry an empty ``response_metadata`` and the stream closes
    with a finish_reason-only chunk that has metadata but no token usage.
    """

    model_name: str = "gpt-4o-mini"
    pieces: list = []
    # chunk index -> usage_metadata payload, for providers that do report usage
    chunk_usage: dict = {}
    # Invoked from inside _stream(), i.e. from inside next() — the window in
    # which LangChain really talks to the provider SDK.
    during_provider_call: Optional[Callable] = None

    @property
    def _llm_type(self) -> str:
        return "streaming-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="".join(self.pieces)))]
        )

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGenerationChunk

        if self.during_provider_call:
            self.during_provider_call()
        for index, piece in enumerate(self.pieces):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=piece, usage_metadata=self.chunk_usage.get(index)
                )
            )
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=""),
            generation_info={"finish_reason": "stop", "model_name": self.model_name},
        )

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self._stream(messages, stop=stop, **kwargs):
            yield chunk


class _NonStreamingFakeChat(BaseChatModel):
    """No ``_stream``, so LangChain's stream() falls back to calling invoke()."""

    model_name: str = "gpt-4o-mini"

    @property
    def _llm_type(self) -> str:
        return "non-streaming-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="hello there"))])


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain-core is not installed")
class TestLangChainStreamingInterceptor:
    """Token counting and guard handling on the ``.stream()`` path."""

    PIECES = ["The quick brown fox ", "jumps over ", "the lazy dog. "] * 4

    @pytest.fixture
    def tracked(self):
        """(events, interceptor) with BaseChatModel patched and depth reset."""
        from agentcost._reentrancy import _tracking_depth
        from agentcost.interceptor import LangChainInterceptor

        _tracking_depth.set(0)
        events = []
        interceptor = LangChainInterceptor(events.append)
        assert interceptor.start()
        try:
            yield events
        finally:
            interceptor.stop()
            _tracking_depth.set(0)

    def _expected_output_tokens(self):
        return TokenCounter.count_tokens("".join(self.PIECES), "gpt-4o-mini")

    def test_stream_counts_output_when_the_last_chunk_reports_no_usage(self, tracked):
        """The default path: stream_usage is off, so nothing reports usage and
        the streamed text is all we have. Reporting 0 understated every bill."""
        events = tracked
        streamed = "".join(c.content for c in _StreamingFakeChat(pieces=self.PIECES).stream("hi"))

        assert streamed == "".join(self.PIECES)
        assert len(events) == 1
        assert events[0]["output_tokens"] == self._expected_output_tokens() > 0
        assert events[0]["cost"] > 0

    def test_stream_prefers_provider_reported_usage_over_the_text_estimate(self, tracked):
        """A real usage report must still win — estimation is the fallback."""
        events = tracked
        llm = _StreamingFakeChat(
            pieces=self.PIECES,
            chunk_usage={len(self.PIECES) - 1: {"input_tokens": 41, "output_tokens": 77,
                                                "total_tokens": 118}},
        )
        list(llm.stream("hi"))

        assert len(events) == 1
        assert events[0]["input_tokens"] == 41
        assert events[0]["output_tokens"] == 77

    def test_usage_split_across_chunks_survives_the_trailing_metadata_chunk(self, tracked):
        """Anthropic reports input tokens first and output tokens last, and a
        finish_reason chunk arrives after both — keeping only "the last chunk
        with metadata" threw the real numbers away."""
        events = tracked
        llm = _StreamingFakeChat(
            pieces=self.PIECES,
            chunk_usage={
                0: {"input_tokens": 33, "output_tokens": 0, "total_tokens": 33},
                len(self.PIECES) - 1: {"input_tokens": 0, "output_tokens": 64,
                                       "total_tokens": 64},
            },
        )
        list(llm.stream("hi"))

        assert len(events) == 1
        assert events[0]["input_tokens"] == 33
        assert events[0]["output_tokens"] == 64

    def test_guard_is_released_while_the_caller_consumes_chunks(self, tracked):
        """A sync generator runs in the caller's context, so holding the guard
        across the loop made every direct SDK call an app issued mid-stream
        look like a nested one and vanish from the ledger."""
        from agentcost._reentrancy import in_tracking

        observed = []
        for _chunk in _StreamingFakeChat(pieces=self.PIECES).stream("hi"):
            observed.append(in_tracking())

        assert observed, "stream yielded nothing"
        assert not any(observed), "guard still held while the caller consumes chunks"

    def test_guard_still_covers_the_provider_call_inside_next(self, tracked):
        """The other half of the trade-off: LangChain issues the provider
        request from inside next(), so the guard must be up there or the
        OpenAI/Anthropic interceptor emits a second event and doubles the cost.
        """
        from agentcost._reentrancy import in_tracking

        seen = []
        llm = _StreamingFakeChat(
            pieces=self.PIECES, during_provider_call=lambda: seen.append(in_tracking())
        )
        list(llm.stream("hi"))

        assert seen == [True], "provider call inside next() was left unguarded"

    def test_invoke_fallback_inside_stream_is_not_double_counted(self, tracked):
        """LangChain's stream() delegates to invoke() when a model cannot
        stream; that nested invoke must not add a second event."""
        events = tracked
        list(_NonStreamingFakeChat().stream("hi"))

        assert len(events) == 1, f"expected 1 event, got {len(events)}"

    def test_interleaved_streams_do_not_leak_guard_depth(self, tracked):
        """Two streams open at once used to reset their guard tokens out of
        order, leaving depth permanently above zero."""
        from agentcost._reentrancy import _tracking_depth

        events = tracked
        first = _StreamingFakeChat(pieces=self.PIECES).stream("a")
        second = _StreamingFakeChat(pieces=self.PIECES).stream("b")
        next(first)
        next(second)
        list(first)   # finish in the order opened, not LIFO
        list(second)

        assert len(events) == 2
        assert _tracking_depth.get() == 0, "recursion guard leaked"

    def test_abandoned_stream_records_and_leaves_no_residual_depth(self, tracked):
        """Breaking out of a stream must still record the call the provider
        already billed, and must not disable tracking afterwards."""
        from agentcost._reentrancy import _tracking_depth

        events = tracked
        for _chunk in _StreamingFakeChat(pieces=self.PIECES).stream("hi"):
            break

        assert len(events) == 1, "an abandoned stream must still record the call"
        assert _tracking_depth.get() == 0, "recursion guard leaked"

        _StreamingFakeChat(pieces=self.PIECES).invoke("next")
        assert len(events) == 2, "tracking died after an abandoned stream"

    @pytest.mark.asyncio
    async def test_astream_counts_output_and_releases_the_guard(self, tracked):
        """Same two failures on the async path."""
        from agentcost._reentrancy import _tracking_depth, in_tracking

        events = tracked
        observed = []
        async for _chunk in _StreamingFakeChat(pieces=self.PIECES).astream("hi"):
            observed.append(in_tracking())

        assert len(events) == 1
        assert events[0]["output_tokens"] == self._expected_output_tokens() > 0
        assert events[0]["cost"] > 0
        assert not any(observed), "guard still held while the caller consumes chunks"
        assert _tracking_depth.get() == 0


class TestOpenAIInterceptor:
    """Every OpenAI call surface a real app uses, against a mocked transport."""

    MODEL = "gpt-4o-mini"
    IN_TOK, OUT_TOK = 25, 15

    @classmethod
    def _completion(cls):
        return {
            "id": "1", "object": "chat.completion", "created": 1, "model": cls.MODEL,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": '{"text": "Hi"}'},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": cls.IN_TOK, "completion_tokens": cls.OUT_TOK,
                      "total_tokens": cls.IN_TOK + cls.OUT_TOK},
        }

    @classmethod
    def _responses_payload(cls):
        return {
            "id": "resp_1", "object": "response", "created_at": 1, "model": cls.MODEL,
            "status": "completed", "parallel_tool_calls": False, "tool_choice": "auto",
            "tools": [], "output": [{"id": "m1", "type": "message", "role": "assistant",
                                     "status": "completed",
                                     "content": [{"type": "output_text", "text": "Hi",
                                                  "annotations": []}]}],
            "usage": {"input_tokens": cls.IN_TOK, "output_tokens": cls.OUT_TOK,
                      "total_tokens": cls.IN_TOK + cls.OUT_TOK},
        }

    @classmethod
    def _sse(cls, include_usage):
        import json as _json

        frames = [
            {"id": "1", "object": "chat.completion.chunk", "created": 1, "model": cls.MODEL,
             "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]},
            {"id": "1", "object": "chat.completion.chunk", "created": 1, "model": cls.MODEL,
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        if include_usage:
            frames.append({"id": "1", "object": "chat.completion.chunk", "created": 1,
                           "model": cls.MODEL, "choices": [],
                           "usage": {"prompt_tokens": cls.IN_TOK,
                                     "completion_tokens": cls.OUT_TOK,
                                     "total_tokens": cls.IN_TOK + cls.OUT_TOK}})
        body = "".join(f"data: {_json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
        return body.encode()

    @classmethod
    def _handler(cls, request):
        import json as _json

        import httpx

        body = _json.loads(request.content.decode()) if request.content else {}
        if body.get("stream"):
            cls.last_stream_options = body.get("stream_options")
            return httpx.Response(200, content=cls._sse(bool(body.get("stream_options"))),
                                  headers={"content-type": "text/event-stream"})
        if "/responses" in request.url.path:
            return httpx.Response(200, json=cls._responses_payload())
        return httpx.Response(200, json=cls._completion())

    @pytest.fixture
    def tracked(self):
        pytest.importorskip("openai")
        httpx = pytest.importorskip("httpx")
        from openai import OpenAI

        from agentcost.openai_interceptor import OpenAIInterceptor

        type(self).last_stream_options = None
        events = []
        interceptor = OpenAIInterceptor(events.append)
        assert interceptor.start(), "OpenAI SDK not importable"
        client = OpenAI(
            api_key="test", base_url="https://api.openai.com/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self._handler)),
        )
        try:
            yield client, events
        finally:
            interceptor.stop()

    def _assert_usage(self, events):
        _assert_single_usage_event(events, self.IN_TOK, self.OUT_TOK)

    def test_plain_create_is_tracked(self, tracked):
        client, events = tracked
        client.chat.completions.create(model=self.MODEL, messages=_user_msgs())
        self._assert_usage(events)

    def test_streamed_call_records_input_tokens(self, tracked):
        client, events = tracked
        for _ in client.chat.completions.create(model=self.MODEL, messages=_user_msgs(),
                                                stream=True):
            pass
        self._assert_usage(events)
        assert self.last_stream_options == {"include_usage": True}

    def test_injected_usage_chunk_is_hidden_from_the_caller(self, tracked):
        """The extra chunk has empty choices and would crash chunk.choices[0]."""
        client, _ = tracked
        chunks = list(client.chat.completions.create(model=self.MODEL,
                                                     messages=_user_msgs(), stream=True))
        assert chunks and all(c.choices for c in chunks)

    def test_caller_supplied_stream_options_are_respected(self, tracked):
        client, _ = tracked
        chunks = list(client.chat.completions.create(
            model=self.MODEL, messages=_user_msgs(), stream=True,
            stream_options={"include_usage": True}))
        # Asked for it explicitly, so the usage chunk must still be delivered.
        assert any(not c.choices for c in chunks)

    def test_abandoned_stream_is_still_recorded(self, tracked):
        client, events = tracked
        for _ in client.chat.completions.create(model=self.MODEL, messages=_user_msgs(),
                                                stream=True):
            break
        assert len(events) == 1, "a billed call vanished when the caller stopped reading"

    def test_parse_is_tracked(self, tracked):
        from pydantic import BaseModel

        class Answer(BaseModel):
            text: str

        client, events = tracked
        client.chat.completions.parse(model=self.MODEL, messages=_user_msgs(),
                                      response_format=Answer)
        self._assert_usage(events)

    def test_responses_api_is_tracked(self, tracked):
        client, events = tracked
        client.responses.create(model=self.MODEL, input="hi")
        self._assert_usage(events)

    def test_stop_restores_every_patched_surface(self):
        pytest.importorskip("openai")
        from openai.resources.chat.completions import Completions
        from openai.resources.responses import Responses

        from agentcost.openai_interceptor import OpenAIInterceptor

        before = (Completions.create, Completions.parse, Responses.create)
        interceptor = OpenAIInterceptor(lambda e: None)
        assert interceptor.start()
        interceptor.stop()
        assert (Completions.create, Completions.parse, Responses.create) == before

    def test_concurrent_async_calls_are_all_tracked(self):
        pytest.importorskip("openai")
        httpx = pytest.importorskip("httpx")
        from openai import AsyncOpenAI

        from agentcost.openai_interceptor import OpenAIInterceptor

        events = _gather_tracked(
            OpenAIInterceptor,
            lambda: AsyncOpenAI(api_key="test", base_url="https://api.openai.com/v1",
                                http_client=httpx.AsyncClient(
                                    transport=_slow_async_transport(self._handler))),
            lambda client: client.chat.completions.create(
                model=self.MODEL, messages=_user_msgs()),
        )

        assert len(events) == 5, f"only {len(events)}/5 concurrent calls tracked"


class TestAuditRegressions:
    """Regressions for defects found in the 2026-07 audit.

    Each of these passed the whole suite before the fix, so they exist to keep
    the specific mechanism from coming back rather than to describe behaviour.
    """

    def test_unknown_model_warning_cannot_escape_under_warnings_as_errors(self):
        """A missing price must never break the caller's LLM call.

        calculate_cost runs from the interceptors' `finally:` blocks, so a
        warning that raises under -W error propagates into the user's request
        and skips the exit_tracking() after it, wedging the re-entrancy depth
        above zero and silently dropping every later call.
        """
        import warnings as _w

        from agentcost.cost_calculator import CostCalculator, _warned_unknown_models

        _warned_unknown_models.discard("audit-unpriced-model")
        with _w.catch_warnings():
            _w.simplefilter("error")
            cost = CostCalculator().calculate_cost("audit-unpriced-model", 10, 10)
        assert cost == 0.0

    def test_gpt4o_family_uses_o200k_encoding(self):
        """MODEL_ENCODINGS is consulted before the family rules, so a wrong
        entry there silently overrides them."""
        assert TokenCounter._get_encoding_name("gpt-4o") == "o200k_base"
        assert TokenCounter._get_encoding_name("gpt-4o-mini") == "o200k_base"
        # The cl100k families must not have been swept up in the change.
        assert TokenCounter._get_encoding_name("gpt-4") == "cl100k_base"
        assert TokenCounter._get_encoding_name("gpt-3.5-turbo") == "cl100k_base"

    def test_stream_options_sentinel_is_not_mistaken_for_caller_intent(self):
        """client.chat.completions.stream() always forwards stream_options,
        defaulted to a sentinel, so a key-membership test reads as "the caller
        set this" and usage is never requested."""
        from agentcost.openai_interceptor import OpenAIInterceptor

        sentinel_kwargs = {"stream": True, "messages": [], "stream_options": object()}
        assert OpenAIInterceptor._prepare_stream_usage(sentinel_kwargs) is True
        assert sentinel_kwargs["stream_options"] == {"include_usage": True}

        # A caller who really passes options still wins.
        caller = {"stream": True, "messages": [], "stream_options": {"include_usage": False}}
        assert OpenAIInterceptor._prepare_stream_usage(caller) is False
        assert caller["stream_options"] == {"include_usage": False}

    def test_shutdown_waits_for_in_flight_send(self):
        """Size-triggered flushes hand events to a daemon thread; without a
        handle, shutdown cannot wait and the interpreter kills them."""
        delivered = []

        def slow_ok(events):
            time.sleep(0.5)
            delivered.extend(events)
            return True

        batcher = HybridBatcher(batch_size=2, flush_interval=60, flush_callback=slow_ok)
        batcher.add({"e": 1})
        batcher.add({"e": 2})
        time.sleep(0.05)  # let the send thread start
        batcher.shutdown()
        assert len(delivered) == 2

    def test_shutdown_is_time_bounded_when_backend_is_unreachable(self):
        """shutdown() runs from atexit; an unbounded retry of a full queue is a
        multi-minute hang on process exit."""
        def slow_fail(events):
            time.sleep(0.3)
            return False

        batcher = HybridBatcher(batch_size=1000, flush_interval=60, flush_callback=slow_fail)
        batcher._failed_batches = [[{"x": i}] for i in range(100)]

        started = time.monotonic()
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            batcher.shutdown()
        elapsed = time.monotonic() - started

        # 100 batches x 0.3s = 30s unbounded; the budget is 5s plus one
        # in-flight attempt that cannot be interrupted mid-call.
        assert elapsed < 15, f"shutdown took {elapsed:.1f}s"

    def test_bad_custom_pricing_breaks_neither_the_call_nor_tracking(self):
        """Bookkeeping must never escape the wrapper's finally block.

        calculate_cost runs there, before exit_tracking(). A partial
        custom_pricing dict used to raise KeyError, which replaced the caller's
        response and wedged the re-entrancy depth so nothing was tracked again.
        """
        pytest.importorskip("openai")
        httpx = pytest.importorskip("httpx")
        from openai import OpenAI

        from agentcost._reentrancy import in_tracking
        from agentcost.config import AgentCostConfig, set_config
        from agentcost.openai_interceptor import OpenAIInterceptor

        completion = {
            "id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        from agentcost.config import get_config

        previous = get_config()
        events = []
        interceptor = OpenAIInterceptor(events.append)
        assert interceptor.start()
        try:
            set_config(AgentCostConfig(
                api_key="k", project_id="p", base_url="http://localhost:9",
                custom_pricing={"gpt-4o": {"input": 0.001}},  # 'output' missing
            ))
            client = OpenAI(api_key="t", http_client=httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, json=completion))))

            response = client.chat.completions.create(
                model="gpt-4o", messages=_user_msgs())

            assert response.choices[0].message.content == "hi"
            assert not in_tracking(), "re-entrancy depth leaked"
            # Degraded, not dropped: the priced side still counts.
            assert len(events) == 1
            assert events[0]["cost"] == pytest.approx(5 / 1000 * 0.001)
        finally:
            interceptor.stop()
            set_config(previous)

    @pytest.mark.parametrize("provider", ["openai", "anthropic"])
    def test_generator_messages_still_reach_the_provider(self, provider):
        """Hashing the prompt must not consume the caller's iterable.

        Both SDKs type `messages` as Iterable, so a generator is valid usage.
        Reading it to build the input hash drained it and the provider received
        an empty conversation.
        """
        httpx = pytest.importorskip("httpx")
        seen = {}

        def handler(payload):
            def _h(request):
                seen["messages"] = json.loads(request.content.decode())["messages"]
                return httpx.Response(200, json=payload)
            return _h

        if provider == "openai":
            pytest.importorskip("openai")
            from openai import OpenAI
            from agentcost.openai_interceptor import OpenAIInterceptor
            payload = {
                "id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
            interceptor = OpenAIInterceptor(lambda e: None)
            assert interceptor.start()
            client = OpenAI(api_key="t", http_client=httpx.Client(
                transport=httpx.MockTransport(handler(payload))))
            call = lambda msgs: client.chat.completions.create(model="gpt-4o", messages=msgs)
        else:
            pytest.importorskip("anthropic")
            from anthropic import Anthropic
            from agentcost.anthropic_interceptor import AnthropicInterceptor
            payload = {
                "id": "m1", "type": "message", "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 3},
            }
            interceptor = AnthropicInterceptor(lambda e: None)
            assert interceptor.start()
            client = Anthropic(api_key="t", http_client=httpx.Client(
                transport=httpx.MockTransport(handler(payload))))
            call = lambda msgs: client.messages.create(
                model="claude-3-5-sonnet-20241022", max_tokens=100, messages=msgs)

        try:
            call((m for m in _user_msgs()))
            assert seen["messages"] == _user_msgs(), "the generator was drained"
        finally:
            interceptor.stop()

    def test_failed_encoding_load_is_not_retried_every_call(self):
        """tiktoken fetches its BPE file over HTTP with no timeout, so a failed
        load must be negative-cached rather than retried per call."""
        import agentcost.token_counter as tc_module

        attempts = {"n": 0}
        real = tc_module.tiktoken.get_encoding

        def counting(name):
            attempts["n"] += 1
            if name == "o200k_base":
                raise RuntimeError("simulated CDN failure")
            return real(name)

        saved_cache = dict(TokenCounter._encoding_cache)
        saved_blocked = dict(TokenCounter._load_failed_until)
        tc_module.tiktoken.get_encoding = counting
        try:
            TokenCounter._encoding_cache.clear()
            TokenCounter._load_failed_until.clear()
            for _ in range(25):
                TokenCounter.count_tokens("hello world", "gpt-4o")
            # One failed o200k attempt plus one successful fallback load.
            assert attempts["n"] <= 2, f"{attempts['n']} load attempts for 25 calls"
        finally:
            tc_module.tiktoken.get_encoding = real
            TokenCounter._encoding_cache.clear()
            TokenCounter._encoding_cache.update(saved_cache)
            TokenCounter._load_failed_until.clear()
            TokenCounter._load_failed_until.update(saved_blocked)

    def test_config_limits_reach_the_batcher(self):
        """Clamping the config is worthless if the batcher is built from the
        raw arguments.

        The first attempt at this fix clamped AgentCostConfig and warned the
        user, while init() kept passing the caller's value straight to
        HybridBatcher — so batch_size=200 still produced 200-event batches that
        the server rejects with a permanent 422, recording nothing. Assert on
        the batcher, never on the config.
        """
        from agentcost.config import MAX_SERVER_BATCH_SIZE, MIN_FLUSH_INTERVAL
        from agentcost.tracker import AgentCostTracker

        import warnings as _w

        tracker = AgentCostTracker()
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                tracker.init(
                    api_key="k", project_id="p",
                    batch_size=200, flush_interval=0.0,
                    base_url="http://localhost:9",
                )
            assert tracker._batcher.batch_size == MAX_SERVER_BATCH_SIZE
            assert tracker._batcher.flush_interval == MIN_FLUSH_INTERVAL
        finally:
            tracker.shutdown()

    def test_shutdown_drop_warning_cannot_escape_atexit(self):
        """Losing events must not also corrupt the caller's shutdown."""
        import warnings as _w

        batcher = HybridBatcher(
            batch_size=1000, flush_interval=60, flush_callback=lambda e: False
        )
        batcher._failed_batches = [[{"x": 1}]]
        with _w.catch_warnings():
            _w.simplefilter("error")
            batcher.shutdown()  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
