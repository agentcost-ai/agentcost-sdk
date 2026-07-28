"""
Tests for AgentCost SDK

Run with: pytest tests/ -v
"""

import pytest
import time
from unittest.mock import Mock, patch

from agentcost import (
    track_costs,
    TokenCounter,
    CostCalculator,
    HybridBatcher,
    LocalBatcher,
    AgentCostConfig,
)


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
        from agentcost.gemini_interceptor import GeminiInterceptor, _tracking_depth

        _tracking_depth.value = 0
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
        assert getattr(_tracking_depth, "value", 0) == 0, "recursion guard leaked"

        # The next ordinary call must still be tracked.
        interceptor._tracked_generate_content()(None, model="gemini-2.0-flash", contents="next")
        assert len(events) == 2, "tracking died after an abandoned stream"

    def test_stream_never_iterated_is_still_recorded(self):
        import gc

        from agentcost.gemini_interceptor import GeminiInterceptor, _tracking_depth

        _tracking_depth.value = 0
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
        assert getattr(_tracking_depth, "value", 0) == 0

    def test_call_made_during_stream_consumption_is_tracked(self):
        """The guard must not span consumption, or interleaved calls vanish."""
        from agentcost.gemini_interceptor import GeminiInterceptor, _tracking_depth

        _tracking_depth.value = 0
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
