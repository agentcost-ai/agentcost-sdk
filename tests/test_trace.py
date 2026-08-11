"""
Tests for AgentCost trace context.

Run with: pytest tests/test_trace.py -v
"""

import asyncio
import threading

import pytest

from agentcost import track_costs
from agentcost.trace import current_trace_fields, current_trace_id


@pytest.fixture
def tracker():
    """A local-mode tracker whose events stay in process."""
    track_costs.init(local_mode=True)
    t = track_costs._tracker
    yield t
    track_costs.shutdown()


def _events():
    track_costs.flush()
    return track_costs.get_local_events()


class TestTraceContext:
    def test_no_workflow_leaves_events_untouched(self, tracker):
        """The whole feature is additive: untraced code emits what it always did."""
        tracker._record_event({"model": "m", "cost": 1.0})
        event = _events()[0]
        assert event == {"model": "m", "cost": 1.0}

    def test_workflow_stamps_trace_id(self, tracker):
        with track_costs.workflow("triage"):
            tracker._record_event({"model": "m"})
        event = _events()[0]
        assert event["workflow"] == "triage"
        assert event["trace_id"]
        assert event["span_id"]

    def test_all_calls_in_one_run_share_a_trace_id(self, tracker):
        with track_costs.workflow("triage"):
            with track_costs.step("classify"):
                tracker._record_event({"model": "m"})
            with track_costs.step("answer"):
                tracker._record_event({"model": "m"})
        trace_ids = {e["trace_id"] for e in _events()}
        assert len(trace_ids) == 1

    def test_each_event_gets_its_own_span_id(self, tracker):
        with track_costs.workflow("w"):
            with track_costs.step("s"):
                tracker._record_event({"model": "m"})
                tracker._record_event({"model": "m"})
        span_ids = [e["span_id"] for e in _events()]
        assert len(set(span_ids)) == 2

    def test_step_records_name_and_parent(self, tracker):
        with track_costs.workflow("w"):
            with track_costs.step("classify"):
                tracker._record_event({"model": "m"})
        event = _events()[0]
        assert event["step_name"] == "classify"
        assert event["parent_span_id"]
        assert event["parent_span_id"] != event["span_id"]

    def test_nested_steps_increase_depth(self, tracker):
        with track_costs.workflow("w"):
            with track_costs.step("outer"):
                tracker._record_event({"model": "m", "tag": "outer"})
                with track_costs.step("inner"):
                    tracker._record_event({"model": "m", "tag": "inner"})
        by_tag = {e["tag"]: e for e in _events()}
        assert by_tag["inner"]["depth"] > by_tag["outer"]["depth"]

    def test_step_index_increments_across_the_run(self, tracker):
        with track_costs.workflow("w"):
            with track_costs.step("a"):
                tracker._record_event({"model": "m"})
            with track_costs.step("b"):
                tracker._record_event({"model": "m"})
            with track_costs.step("c"):
                tracker._record_event({"model": "m"})
        indexes = [e["step_index"] for e in _events()]
        assert indexes == sorted(indexes)
        assert len(set(indexes)) == 3

    def test_tool_marks_tool_name(self, tracker):
        with track_costs.workflow("w"):
            with track_costs.tool("web_search"):
                tracker._record_event({"model": "m"})
        event = _events()[0]
        assert event["tool_name"] == "web_search"
        assert event["step_name"] == "web_search"

    def test_llm_call_outside_a_tool_has_no_tool_name(self, tracker):
        with track_costs.workflow("w"):
            with track_costs.step("plain"):
                tracker._record_event({"model": "m"})
        assert "tool_name" not in _events()[0]

    def test_nested_workflow_joins_the_outer_trace(self, tracker):
        """A sub-agent must not fragment its caller's run into two traces."""
        with track_costs.workflow("outer") as outer_id:
            with track_costs.workflow("inner") as inner_id:
                tracker._record_event({"model": "m"})
        assert outer_id == inner_id
        assert _events()[0]["workflow"] == "outer"

    def test_explicit_trace_id_is_honoured(self, tracker):
        """Lets a trace span two processes."""
        with track_costs.workflow("w", trace_id="abc123"):
            tracker._record_event({"model": "m"})
        assert _events()[0]["trace_id"] == "abc123"

    def test_context_is_restored_after_the_block(self, tracker):
        with track_costs.workflow("w"):
            assert current_trace_id() is not None
        assert current_trace_id() is None
        assert current_trace_fields() == {}

    def test_step_outside_a_workflow_is_inert(self, tracker):
        """Instrumenting a helper must not depend on how it is called."""
        with track_costs.step("orphan"):
            tracker._record_event({"model": "m"})
        assert _events()[0] == {"model": "m"}

    def test_exception_still_restores_context(self, tracker):
        with pytest.raises(ValueError):
            with track_costs.workflow("w"):
                with track_costs.step("boom"):
                    raise ValueError("boom")
        assert current_trace_id() is None


class TestTraceIsolation:
    def test_threads_do_not_share_a_trace(self, tracker):
        """contextvars, not globals: two runs in flight must stay separate."""
        seen = {}

        def run(name):
            with track_costs.workflow(name):
                seen[name] = current_trace_id()

        threads = [threading.Thread(target=run, args=(n,)) for n in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert seen["a"] != seen["b"]

    def test_concurrent_async_tasks_do_not_share_a_trace(self, tracker):
        async def run(name):
            with track_costs.workflow(name):
                await asyncio.sleep(0)
                return current_trace_id()

        async def main():
            return await asyncio.gather(run("a"), run("b"))

        a, b = asyncio.run(main())
        assert a != b


class TestEnrichmentSafety:
    def test_event_survives_a_broken_trace_context(self, tracker, monkeypatch):
        """Trace enrichment must never cost the caller an event."""
        import agentcost.trace as trace_mod

        def explode():
            raise RuntimeError("context is broken")

        monkeypatch.setattr(trace_mod, "current_trace_fields", explode)
        tracker._record_event({"model": "m", "cost": 2.0})

        events = _events()
        assert len(events) == 1
        assert events[0]["cost"] == 2.0


class TestOutcomes:
    def test_outcome_is_emitted_when_the_workflow_closes(self, tracker):
        with track_costs.workflow("w"):
            tracker._record_event({"model": "m"})
            track_costs.outcome(True, label="resolved")

        records = _events()
        outcome = [r for r in records if r.get("record_type") == "outcome"]
        assert len(outcome) == 1
        assert outcome[0]["success"] is True
        assert outcome[0]["label"] == "resolved"
        assert outcome[0]["workflow"] == "w"

    def test_outcome_carries_the_trace_id_of_its_run(self, tracker):
        with track_costs.workflow("w") as trace_id:
            tracker._record_event({"model": "m"})
            track_costs.outcome(False)

        outcome = [r for r in _events() if r.get("record_type") == "outcome"][0]
        assert outcome["trace_id"] == trace_id

    def test_no_outcome_means_no_record(self, tracker):
        with track_costs.workflow("w"):
            tracker._record_event({"model": "m"})

        assert not [r for r in _events() if r.get("record_type") == "outcome"]

    def test_last_call_wins(self, tracker):
        """An optimistic success followed by a real failure."""
        with track_costs.workflow("w"):
            track_costs.outcome(True)
            track_costs.outcome(False, label="timeout")
            tracker._record_event({"model": "m"})

        outcome = [r for r in _events() if r.get("record_type") == "outcome"][0]
        assert outcome["success"] is False
        assert outcome["label"] == "timeout"

    def test_outcome_outside_a_workflow_is_a_noop(self, tracker):
        assert track_costs.outcome(True) is False
        assert not [r for r in _events() if r.get("record_type") == "outcome"]

    def test_nested_workflow_emits_one_outcome_at_the_outer_close(self, tracker):
        with track_costs.workflow("outer"):
            with track_costs.workflow("inner"):
                track_costs.outcome(True)
                tracker._record_event({"model": "m"})

        outcomes = [r for r in _events() if r.get("record_type") == "outcome"]
        assert len(outcomes) == 1
        assert outcomes[0]["workflow"] == "outer"

    def test_outcome_still_emitted_when_the_run_raises(self, tracker):
        with pytest.raises(ValueError):
            with track_costs.workflow("w"):
                track_costs.outcome(False, label="crashed")
                raise ValueError("boom")

        outcome = [r for r in _events() if r.get("record_type") == "outcome"][0]
        assert outcome["success"] is False


class TestRecordPartitioning:
    def test_outcomes_are_split_out_of_the_event_payload(self):
        from agentcost.http_client import partition_records

        events, outcomes = partition_records([
            {"model": "m", "cost": 1.0},
            {"record_type": "outcome", "trace_id": "t1", "success": True},
        ])

        assert len(events) == 1 and len(outcomes) == 1
        assert "record_type" not in outcomes[0]
        assert outcomes[0]["trace_id"] == "t1"

    def test_a_batch_without_outcomes_is_unchanged(self):
        from agentcost.http_client import partition_records

        events, outcomes = partition_records([{"model": "m"}])
        assert events == [{"model": "m"}]
        assert outcomes == []
