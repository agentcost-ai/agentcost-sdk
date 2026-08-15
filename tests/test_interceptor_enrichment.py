"""
End-to-end: does an intercepted call actually emit the new fields?

The unit tests cover ``fingerprint()`` in isolation and the backend covers
consuming ``_ac_caps``. Neither proves the wiring between them. A fingerprint
computed but never attached, or cache tokens read but never placed on the
event, would pass both and still ship nothing -- which is exactly the shape of
the bug this release fixes.

These drive the real interceptors against fake provider clients and assert on
the emitted event dict.
"""

import sys
import types

import pytest

from agentcost.capabilities import CAPABILITY_KEY


# ── Fake OpenAI ───────────────────────────────────────────────────────────

class _PromptTokensDetails:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class _OpenAIUsage:
    def __init__(self, prompt=1000, completion=100, cached=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.prompt_tokens_details = _PromptTokensDetails(cached)


class _OpenAIResponse:
    def __init__(self, **kwargs):
        self.usage = _OpenAIUsage(**kwargs)
        self.choices = []


# ── Fake Anthropic ────────────────────────────────────────────────────────

class _AnthropicUsage:
    def __init__(self, input_tokens=100, output_tokens=50, cache_read=0, cache_write=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_write


class _AnthropicMessage:
    def __init__(self, **kwargs):
        self.usage = _AnthropicUsage(**kwargs)
        self.model = "claude-sonnet-4"


@pytest.fixture
def captured():
    return []


@pytest.fixture
def openai_interceptor(captured, monkeypatch):
    """Patch a stand-in openai module, then start the real interceptor."""
    from agentcost.openai_interceptor import OpenAIInterceptor

    class Completions:
        def create(self, *args, **kwargs):
            return _OpenAIResponse(**getattr(self, "_next", {}))

    module = types.ModuleType("openai")
    resources = types.ModuleType("openai.resources")
    chat_mod = types.ModuleType("openai.resources.chat")
    completions_mod = types.ModuleType("openai.resources.chat.completions")
    completions_mod.Completions = Completions
    chat_mod.completions = completions_mod
    resources.chat = chat_mod
    module.resources = resources

    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setitem(sys.modules, "openai.resources", resources)
    monkeypatch.setitem(sys.modules, "openai.resources.chat", chat_mod)
    monkeypatch.setitem(sys.modules, "openai.resources.chat.completions", completions_mod)

    interceptor = OpenAIInterceptor(event_callback=captured.append)
    if not interceptor.start():
        pytest.skip("OpenAI interceptor could not attach to the stand-in module")
    yield Completions()
    interceptor.stop()


class TestOpenAIEnrichment:
    def test_cached_tokens_reach_the_event(self, openai_interceptor, captured):
        openai_interceptor._next = {"prompt": 1000, "completion": 100, "cached": 900}
        openai_interceptor.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        assert captured, "no event was emitted"
        event = captured[-1]
        assert event["input_tokens"] == 1000
        assert event["cached_tokens"] == 900

    def test_uncached_call_omits_the_field(self, openai_interceptor, captured):
        openai_interceptor._next = {"prompt": 1000, "completion": 100, "cached": 0}
        openai_interceptor.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        assert "cached_tokens" not in captured[-1]

    def test_tool_call_is_fingerprinted(self, openai_interceptor, captured):
        openai_interceptor._next = {}
        openai_interceptor.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        caps = captured[-1]["metadata"][CAPABILITY_KEY]
        assert caps["tools"] is True
        assert caps["tool_count"] == 1

    def test_vision_call_is_fingerprinted(self, openai_interceptor, captured):
        openai_interceptor._next = {}
        openai_interceptor.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        )

        assert captured[-1]["metadata"][CAPABILITY_KEY] == {"vision": True}

    def test_plain_call_carries_no_fingerprint(self, openai_interceptor, captured):
        openai_interceptor._next = {}
        openai_interceptor.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        metadata = captured[-1].get("metadata") or {}
        assert CAPABILITY_KEY not in metadata

    def test_no_prompt_text_is_emitted(self, openai_interceptor, captured):
        openai_interceptor._next = {}
        openai_interceptor.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "the secret is hunter2"}],
            tools=[{"type": "function", "function": {"name": "exfiltrate"}}],
        )

        serialized = repr(captured[-1])
        assert "hunter2" not in serialized
        assert "exfiltrate" not in serialized

    def test_user_metadata_is_not_clobbered_by_the_fingerprint(
        self, openai_interceptor, captured
    ):
        """The reserved key must coexist with whatever the caller tagged."""
        from agentcost.config import AgentCostConfig, get_config, set_config

        # Set config directly rather than calling track_costs.init(): init()
        # attaches its own interceptors, which would displace the one under
        # test and route events away from `captured`.
        previous = get_config()
        set_config(
            AgentCostConfig(
                api_key="sk_test",
                project_id="00000000-0000-4000-8000-000000000000",
                global_metadata={"user_id": "alice"},
            )
        )
        try:
            openai_interceptor._next = {}
            openai_interceptor.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "search"}}],
            )

            metadata = captured[-1]["metadata"]
            assert metadata["user_id"] == "alice"
            assert metadata[CAPABILITY_KEY]["tools"] is True
        finally:
            set_config(previous)


class TestAnthropicUsageNormalisation:
    """Anthropic reports cache reads *outside* input_tokens; OpenAI reports them
    inside. Both must reach the backend using the same convention."""

    def test_cache_read_is_folded_into_input_tokens(self):
        from agentcost.anthropic_interceptor import _usage_from

        message = _AnthropicMessage(input_tokens=100, output_tokens=50, cache_read=900)
        input_tokens, output_tokens, cached, written = _usage_from(message)

        # 100 uncached + 900 read from cache = 1000 prompt tokens total.
        assert input_tokens == 1000
        assert cached == 900
        assert output_tokens == 50
        assert written == 0

    def test_cache_writes_stay_separate(self):
        from agentcost.anthropic_interceptor import _usage_from

        message = _AnthropicMessage(input_tokens=100, output_tokens=50, cache_write=400)
        input_tokens, _out, cached, written = _usage_from(message)

        # Writes are billed at a premium, not a discount: they must not be
        # folded into either input_tokens or cached_tokens.
        assert input_tokens == 100
        assert cached == 0
        assert written == 400

    def test_absent_usage_is_all_zeroes(self):
        from agentcost.anthropic_interceptor import _usage_from

        assert _usage_from(object()) == (0, 0, 0, 0)
