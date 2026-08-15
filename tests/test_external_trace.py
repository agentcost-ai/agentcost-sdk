"""
Inheriting a run id from the environment.

This is the whole integration surface for a process that wraps an agent -- a
policy layer, an orchestrator, a CI job. It exports one variable and every
event this SDK emits carries the same run id, with no code change in the agent.
"""

import pytest

from agentcost import trace as trace_context
from agentcost.capabilities import fingerprint, merge_config


class TestTraceIdInheritance:
    def test_workflow_adopts_the_environment_run_id(self, monkeypatch):
        monkeypatch.setenv(trace_context.TRACE_ID_ENV, "run-from-control-plane")
        with trace_context.workflow("refactor") as trace_id:
            assert trace_id == "run-from-control-plane"

    def test_explicit_argument_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv(trace_context.TRACE_ID_ENV, "from-env")
        with trace_context.workflow("refactor", trace_id="explicit") as trace_id:
            assert trace_id == "explicit"

    def test_absent_variable_mints_a_local_id(self, monkeypatch):
        monkeypatch.delenv(trace_context.TRACE_ID_ENV, raising=False)
        with trace_context.workflow("refactor") as trace_id:
            assert trace_id
            assert trace_id != "from-env"

    def test_blank_variable_is_ignored(self, monkeypatch):
        """An unset variable exported as empty must not become a run id."""
        monkeypatch.setenv(trace_context.TRACE_ID_ENV, "   ")
        with trace_context.workflow("refactor") as trace_id:
            assert trace_id.strip()

    def test_uuid_shaped_ids_pass_through_intact(self, monkeypatch):
        run_id = "0532f9c4-a022-4e98-a543-d8e17c5b90a6"
        monkeypatch.setenv(trace_context.TRACE_ID_ENV, run_id)
        with trace_context.workflow("refactor") as trace_id:
            assert trace_id == run_id

    def test_overlong_id_is_truncated_to_the_column_width(self, monkeypatch):
        monkeypatch.setenv(trace_context.TRACE_ID_ENV, "x" * 200)
        with trace_context.workflow("refactor") as trace_id:
            assert len(trace_id) == trace_context.MAX_TRACE_ID

    def test_events_inside_carry_the_inherited_id(self, monkeypatch):
        monkeypatch.setenv(trace_context.TRACE_ID_ENV, "shared-run")
        with trace_context.workflow("refactor"):
            with trace_context.step("plan"):
                fields = trace_context.current_trace_fields()
        assert fields["trace_id"] == "shared-run"
        assert fields["step_name"] == "plan"

    def test_workflow_name_can_also_be_inherited(self, monkeypatch):
        monkeypatch.setenv(trace_context.WORKFLOW_ENV, "policy-wrapped-run")
        assert trace_context.inherited_workflow_name() == "policy-wrapped-run"


class TestTrackerLevelInheritance:
    """The zero-code-change case: the agent never calls workflow()."""

    @pytest.fixture
    def tracker(self):
        from agentcost import track_costs

        track_costs.init(local_mode=True)
        yield track_costs._tracker
        track_costs.shutdown()

    def _events(self):
        from agentcost import track_costs

        track_costs.flush()
        return track_costs.get_local_events()

    def test_events_outside_workflow_inherit_the_env_run_id(self, tracker, monkeypatch):
        monkeypatch.setenv(trace_context.TRACE_ID_ENV, "wrapped-run")
        monkeypatch.setenv(trace_context.WORKFLOW_ENV, "policy-wrapped")
        tracker._record_event({"model": "m"})
        event = self._events()[0]
        assert event["trace_id"] == "wrapped-run"
        assert event["workflow"] == "policy-wrapped"
        # Membership, not structure: an env-correlated run declares no spans.
        assert "span_id" not in event

    def test_workflow_name_alone_still_groups(self, tracker, monkeypatch):
        monkeypatch.delenv(trace_context.TRACE_ID_ENV, raising=False)
        monkeypatch.setenv(trace_context.WORKFLOW_ENV, "nightly-batch")
        tracker._record_event({"model": "m"})
        event = self._events()[0]
        assert event["workflow"] == "nightly-batch"
        assert "trace_id" not in event

    def test_active_workflow_beats_the_environment(self, tracker, monkeypatch):
        monkeypatch.setenv(trace_context.TRACE_ID_ENV, "from-env")
        monkeypatch.setenv(trace_context.WORKFLOW_ENV, "env-name")
        with trace_context.workflow("real-name", trace_id="explicit"):
            tracker._record_event({"model": "m"})
        event = self._events()[0]
        assert event["trace_id"] == "explicit"
        assert event["workflow"] == "real-name"

    def test_unset_environment_leaves_events_untouched(self, tracker, monkeypatch):
        monkeypatch.delenv(trace_context.TRACE_ID_ENV, raising=False)
        monkeypatch.delenv(trace_context.WORKFLOW_ENV, raising=False)
        tracker._record_event({"model": "m"})
        event = self._events()[0]
        assert "trace_id" not in event
        assert "workflow" not in event


class TestCapabilityFingerprint:
    """Only booleans and counts leave the process -- never payloads."""

    def test_plain_text_call_records_nothing(self):
        assert fingerprint({"messages": [{"role": "user", "content": "hi"}]}) is None

    def test_tools_are_counted_not_captured(self):
        caps = fingerprint({"messages": [], "tools": [{"name": "a"}, {"name": "b"}]})
        assert caps == {"tools": True, "tool_count": 2}

    def test_openai_style_image_part(self):
        caps = fingerprint(
            {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}
        )
        assert caps == {"vision": True}

    def test_anthropic_style_image_block(self):
        caps = fingerprint(
            {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}
        )
        assert caps == {"vision": True}

    def test_structured_output(self):
        caps = fingerprint({"messages": [], "response_format": {"type": "json_schema"}})
        assert caps == {"structured_output": True}

    def test_gemini_nested_config_is_flattened(self):
        """Tools live under `config`, so a naive read reports no capabilities."""
        caps = fingerprint(merge_config({"contents": "x", "config": {"tools": [{"n": 1}]}}))
        assert caps == {"tools": True, "tool_count": 1}

    def test_gemini_inline_data_part(self):
        caps = fingerprint(
            {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/png"}}]}]}
        )
        assert caps == {"vision": True}

    def test_gemini_file_data_part(self):
        caps = fingerprint(
            {"contents": [{"role": "user", "parts": [{"file_data": {"file_uri": "gs://x"}}]}]}
        )
        assert caps == {"vision": True}

    def test_gemini_part_object_with_inline_data(self):
        class Part:
            inline_data = object()
            file_data = None

        caps = fingerprint({"contents": [Part()]})
        assert caps == {"vision": True}

    def test_pil_image_passed_directly_in_contents(self):
        class FakeImage:
            pass

        FakeImage.__module__ = "PIL.Image"
        caps = fingerprint({"contents": ["describe this", FakeImage()]})
        assert caps == {"vision": True}

    def test_gemini_text_parts_record_nothing(self):
        caps = fingerprint({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
        assert caps is None

    def test_no_prompt_text_is_ever_included(self):
        caps = fingerprint(
            {
                "messages": [{"role": "user", "content": "my secret password is hunter2"}],
                "tools": [{"name": "delete_everything"}],
            }
        )
        assert "hunter2" not in str(caps)
        assert "delete_everything" not in str(caps)

    def test_malformed_input_never_raises(self):
        """Runs inside the interceptors' guarded paths; must degrade quietly."""
        assert fingerprint({"messages": object()}) is None
        assert fingerprint({}) is None
