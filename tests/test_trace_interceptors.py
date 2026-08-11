"""
Trace attribution through the real interceptors.

tests/test_trace.py drives tracker._record_event directly, which proves the
stamping rule but not that any provider actually reaches it. These tests wire
each interceptor exactly as track_costs.init() does -- event_callback pointing
at the tracker -- and drive a genuine provider client against a mocked
transport. If an interceptor ever stops routing through that callback, the
trace fields vanish silently in production and only these tests notice.
"""

import json

import pytest

from agentcost import track_costs

MODEL_OPENAI = "gpt-4o-mini"
MODEL_ANTHROPIC = "claude-3-5-sonnet-20241022"
IN_TOK, OUT_TOK = 25, 15


@pytest.fixture
def tracker():
    track_costs.init(local_mode=True)
    yield track_costs._tracker
    track_costs.shutdown()


def _traced_events():
    track_costs.flush()
    return track_costs.get_local_events()


def _assert_fully_traced(event, *, step, tool=None):
    """Every trace field the backend indexes must be present and coherent."""
    assert event.get("trace_id"), "no trace_id — interceptor bypassed the tracker"
    assert event.get("span_id")
    assert event.get("parent_span_id")
    assert event["workflow"] == "e2e"
    assert event["step_name"] == step
    assert event["depth"] >= 1
    assert isinstance(event["step_index"], int)
    if tool:
        assert event["tool_name"] == tool
    else:
        assert "tool_name" not in event
    # The pre-existing fields must survive enrichment untouched.
    assert event["input_tokens"] == IN_TOK
    assert event["output_tokens"] == OUT_TOK
    assert event["model"]


# ── OpenAI ────────────────────────────────────────────────────────────────


def _openai_handler(request):
    import httpx

    body = json.loads(request.content.decode()) if request.content else {}
    payload = {
        "id": "1", "object": "chat.completion", "created": 1, "model": MODEL_OPENAI,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "Hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": IN_TOK, "completion_tokens": OUT_TOK,
                  "total_tokens": IN_TOK + OUT_TOK},
    }
    if body.get("stream"):
        frames = [
            {"id": "1", "object": "chat.completion.chunk", "created": 1,
             "model": MODEL_OPENAI,
             "choices": [{"index": 0, "delta": {"content": "Hi"},
                          "finish_reason": "stop"}]},
            {"id": "1", "object": "chat.completion.chunk", "created": 1,
             "model": MODEL_OPENAI, "choices": [],
             "usage": {"prompt_tokens": IN_TOK, "completion_tokens": OUT_TOK,
                       "total_tokens": IN_TOK + OUT_TOK}},
        ]
        content = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
        return httpx.Response(200, content=content.encode(),
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=payload)


@pytest.fixture
def openai_client(tracker):
    pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    from openai import OpenAI

    from agentcost.openai_interceptor import OpenAIInterceptor

    # The production wiring: interceptor -> tracker._record_event -> batcher.
    interceptor = OpenAIInterceptor(event_callback=tracker._record_event)
    assert interceptor.start(), "OpenAI SDK not importable"
    client = OpenAI(
        api_key="test", base_url="https://api.openai.com/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_openai_handler)),
    )
    try:
        yield client
    finally:
        interceptor.stop()


class TestOpenAITracing:
    def test_call_inside_a_step_is_traced(self, openai_client):
        with track_costs.workflow("e2e"):
            with track_costs.step("classify"):
                openai_client.chat.completions.create(
                    model=MODEL_OPENAI,
                    messages=[{"role": "user", "content": "hi"}],
                )

        events = _traced_events()
        assert len(events) == 1
        _assert_fully_traced(events[0], step="classify")

    def test_call_inside_a_tool_is_attributed_to_the_tool(self, openai_client):
        with track_costs.workflow("e2e"):
            with track_costs.tool("web_search"):
                openai_client.chat.completions.create(
                    model=MODEL_OPENAI,
                    messages=[{"role": "user", "content": "hi"}],
                )

        _assert_fully_traced(_traced_events()[0], step="web_search", tool="web_search")

    def test_streamed_call_consumed_in_context_is_traced(self, openai_client):
        """Streams emit at consumption; consumed inside the step, it must land."""
        with track_costs.workflow("e2e"):
            with track_costs.step("stream_step"):
                for _ in openai_client.chat.completions.create(
                    model=MODEL_OPENAI,
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                ):
                    pass

        _assert_fully_traced(_traced_events()[0], step="stream_step")

    def test_untraced_call_is_unchanged(self, openai_client):
        """The additive guarantee, proven through a real provider."""
        openai_client.chat.completions.create(
            model=MODEL_OPENAI, messages=[{"role": "user", "content": "hi"}]
        )

        event = _traced_events()[0]
        for field in (
            "trace_id", "span_id", "parent_span_id",
            "workflow", "step_name", "step_index", "depth", "tool_name",
        ):
            assert field not in event
        assert event["input_tokens"] == IN_TOK

    def test_multi_step_run_shares_one_trace(self, openai_client):
        with track_costs.workflow("e2e"):
            with track_costs.step("classify"):
                openai_client.chat.completions.create(
                    model=MODEL_OPENAI, messages=[{"role": "user", "content": "a"}])
            with track_costs.tool("search"):
                openai_client.chat.completions.create(
                    model=MODEL_OPENAI, messages=[{"role": "user", "content": "b"}])
            with track_costs.step("answer"):
                openai_client.chat.completions.create(
                    model=MODEL_OPENAI, messages=[{"role": "user", "content": "c"}])

        events = _traced_events()
        assert len(events) == 3
        assert len({e["trace_id"] for e in events}) == 1
        assert [e["step_name"] for e in events] == ["classify", "search", "answer"]
        # Ordinals must be distinct so the backend can order concurrent steps.
        assert len({e["step_index"] for e in events}) == 3


# ── Anthropic ─────────────────────────────────────────────────────────────


def _anthropic_handler(request):
    import httpx

    return httpx.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL_ANTHROPIC,
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": IN_TOK, "output_tokens": OUT_TOK},
    })


@pytest.fixture
def anthropic_client(tracker):
    pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")
    from anthropic import Anthropic

    from agentcost.anthropic_interceptor import AnthropicInterceptor

    interceptor = AnthropicInterceptor(event_callback=tracker._record_event)
    assert interceptor.start(), "Anthropic SDK not importable"
    client = Anthropic(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(_anthropic_handler)),
    )
    try:
        yield client
    finally:
        interceptor.stop()


class TestAnthropicTracing:
    def test_call_inside_a_step_is_traced(self, anthropic_client):
        with track_costs.workflow("e2e"):
            with track_costs.step("classify"):
                anthropic_client.messages.create(
                    model=MODEL_ANTHROPIC, max_tokens=64,
                    messages=[{"role": "user", "content": "hi"}],
                )

        events = _traced_events()
        assert len(events) == 1
        _assert_fully_traced(events[0], step="classify")

    def test_untraced_call_is_unchanged(self, anthropic_client):
        anthropic_client.messages.create(
            model=MODEL_ANTHROPIC, max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "trace_id" not in _traced_events()[0]


# ── LangChain ─────────────────────────────────────────────────────────────


class TestLangChainTracing:
    def test_call_inside_a_step_is_traced(self, tracker):
        pytest.importorskip("langchain_core")
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        from agentcost.interceptor import LangChainInterceptor

        class FakeChat(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                message = AIMessage(
                    content="Hi",
                    response_metadata={"model_name": "gpt-4o-mini"},
                    usage_metadata={
                        "input_tokens": IN_TOK,
                        "output_tokens": OUT_TOK,
                        "total_tokens": IN_TOK + OUT_TOK,
                    },
                )
                return ChatResult(generations=[ChatGeneration(message=message)])

        interceptor = LangChainInterceptor(event_callback=tracker._record_event)
        assert interceptor.start(), "LangChain not importable"
        try:
            with track_costs.workflow("e2e"):
                with track_costs.step("classify"):
                    FakeChat().invoke("hello")
        finally:
            interceptor.stop()

        events = _traced_events()
        assert len(events) == 1
        event = events[0]
        assert event["workflow"] == "e2e"
        assert event["step_name"] == "classify"
        assert event["trace_id"] and event["span_id"] and event["parent_span_id"]


# ── Gemini ────────────────────────────────────────────────────────────────


class TestGeminiTracing:
    """
    Gemini is patched at the method rather than the transport, so the mock
    sits one layer lower than the OpenAI/Anthropic fixtures. The wiring under
    test is the same: interceptor -> tracker._record_event.
    """

    @staticmethod
    def _response(prompt_tokens=IN_TOK, output_tokens=OUT_TOK):
        from unittest.mock import Mock

        return Mock(
            usage_metadata=Mock(
                prompt_token_count=prompt_tokens,
                candidates_token_count=output_tokens,
            )
        )

    def test_call_inside_a_step_is_traced(self, tracker):
        from agentcost.gemini_interceptor import GeminiInterceptor

        interceptor = GeminiInterceptor(event_callback=tracker._record_event)
        interceptor._original_generate_content = (
            lambda _client, **_kwargs: self._response()
        )
        wrapped = interceptor._tracked_generate_content()

        with track_costs.workflow("e2e"):
            with track_costs.step("classify"):
                wrapped(None, model="gemini-2.0-flash", contents="Hello")

        events = _traced_events()
        assert len(events) == 1
        _assert_fully_traced(events[0], step="classify")

    def test_streamed_call_is_traced(self, tracker):
        from agentcost.gemini_interceptor import GeminiInterceptor

        interceptor = GeminiInterceptor(event_callback=tracker._record_event)
        interceptor._original_generate_content_stream = lambda _client, **_kwargs: iter(
            [self._response(0, 0), self._response()]
        )
        wrapped = interceptor._tracked_generate_content_stream()

        with track_costs.workflow("e2e"):
            with track_costs.tool("lookup"):
                list(wrapped(None, model="gemini-2.0-flash", contents="Hello"))

        _assert_fully_traced(_traced_events()[0], step="lookup", tool="lookup")

    def test_untraced_call_is_unchanged(self, tracker):
        from agentcost.gemini_interceptor import GeminiInterceptor

        interceptor = GeminiInterceptor(event_callback=tracker._record_event)
        interceptor._original_generate_content = (
            lambda _client, **_kwargs: self._response()
        )
        interceptor._tracked_generate_content()(
            None, model="gemini-2.0-flash", contents="Hello"
        )

        assert "trace_id" not in _traced_events()[0]
