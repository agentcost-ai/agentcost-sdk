"""Tests for the pre-deployment analyzer."""

import json

import pytest

from agentcost.analyzer import (
    analyze,
    analyze_events,
    analyze_files,
    context_window,
    format_report,
    load_events,
)
from agentcost.cli import main

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@pytest.fixture
def agent_dir(tmp_path):
    (tmp_path / "system.md").write_text("You are a support agent. " * 200, encoding="utf-8")
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "triage.md").write_text("Triage rules. " * 50, encoding="utf-8")
    (tmp_path / "notes.py").write_text("# not a prompt file", encoding="utf-8")
    return tmp_path


def _event(**overrides):
    base = {"model": "gpt-4o", "cost": 0.01, "success": True}
    base.update(overrides)
    return base


class TestStaticAnalysis:
    def test_counts_and_prices_prompt_files(self, agent_dir):
        report = analyze_files(str(agent_dir))

        assert len(report.files) == 2
        assert report.total_tokens > 0
        assert report.cost_per_call > 0
        assert report.files[0].tokens >= report.files[1].tokens

    def test_ignores_files_outside_the_patterns(self, agent_dir):
        paths = [f.path for f in analyze_files(str(agent_dir)).files]
        assert not any(p.endswith(".py") for p in paths)

    def test_custom_patterns_are_honoured(self, agent_dir):
        report = analyze_files(str(agent_dir), patterns=("*.py",))
        assert [f.path for f in report.files] == ["notes.py"]

    def test_skips_vendored_directories(self, tmp_path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "big.md").write_text("x " * 100, encoding="utf-8")
        (tmp_path / "real.md").write_text("y " * 100, encoding="utf-8")

        paths = [f.path for f in analyze_files(str(tmp_path)).files]
        assert paths == ["real.md"]

    def test_flags_a_file_that_eats_the_context_window(self, tmp_path):
        (tmp_path / "huge.md").write_text("token " * 60_000, encoding="utf-8")

        codes = {f.code for f in analyze_files(str(tmp_path)).findings}
        assert "oversized_file" in codes

    def test_flags_duplicate_content(self, tmp_path):
        (tmp_path / "a.md").write_text("same content here", encoding="utf-8")
        (tmp_path / "b.md").write_text("same   content here", encoding="utf-8")

        codes = {f.code for f in analyze_files(str(tmp_path)).findings}
        assert "duplicate_content" in codes

    def test_clean_files_produce_no_findings(self, agent_dir):
        assert analyze_files(str(agent_dir)).findings == []

    def test_unknown_model_falls_back_to_a_default_window(self):
        assert context_window("some-unreleased-model") > 0

    def test_missing_path_yields_nothing_rather_than_raising(self, tmp_path):
        report = analyze_files(str(tmp_path / "does-not-exist"))
        assert report.files == []


class TestRunAnalysis:
    def test_projects_cost_per_run_from_traced_events(self):
        events = [
            _event(trace_id="r1", step_name="classify", cost=0.01),
            _event(trace_id="r1", step_name="answer", cost=0.03),
            _event(trace_id="r2", step_name="classify", cost=0.01),
            _event(trace_id="r2", step_name="answer", cost=0.03),
        ]
        report = analyze_events(events)

        assert report.runs_observed == 2
        assert report.calls_per_run == pytest.approx(2.0)
        assert report.cost_per_run == pytest.approx(0.04, rel=1e-6)

    def test_step_shares_sum_to_the_whole_run(self):
        events = [
            _event(trace_id="r1", step_name="a", cost=0.03),
            _event(trace_id="r1", step_name="b", cost=0.01),
        ]
        report = analyze_events(events)

        assert sum(s.share_of_run for s in report.steps) == pytest.approx(1.0)
        assert report.steps[0].step_name == "a"

    def test_flags_a_looping_step(self):
        events = [
            _event(trace_id="r1", step_name="search", cost=0.01),
            _event(trace_id="r1", step_name="search", cost=0.01),
            _event(trace_id="r1", step_name="answer", cost=0.01),
        ]
        codes = {f.code for f in analyze_events(events).findings}
        assert "step_loops" in codes

    def test_flags_identical_calls_inside_one_run(self):
        events = [
            _event(trace_id="r1", step_name="s", input_hash="same"),
            _event(trace_id="r1", step_name="s", input_hash="same"),
        ]
        codes = {f.code for f in analyze_events(events).findings}
        assert "repeated_call" in codes

    def test_identical_calls_across_runs_are_not_repeats(self):
        events = [
            _event(trace_id="r1", step_name="s", input_hash="same"),
            _event(trace_id="r2", step_name="s", input_hash="same"),
        ]
        codes = {f.code for f in analyze_events(events).findings}
        assert "repeated_call" not in codes

    def test_flags_failures_and_deep_nesting(self):
        events = [_event(trace_id="r1", step_name="s", success=False, depth=5)]

        codes = {f.code for f in analyze_events(events).findings}
        assert "failed_calls" in codes
        assert "deep_nesting" in codes

    def test_uninstrumented_run_still_reports_a_cost(self):
        report = analyze_events([_event(cost=0.02), _event(cost=0.03)])

        assert report.runs_observed == 1
        assert report.cost_per_run == pytest.approx(0.05, rel=1e-6)
        assert "not_instrumented" in {f.code for f in report.findings}

    def test_outcome_records_are_not_counted_as_calls(self):
        events = [
            _event(trace_id="r1", step_name="s", cost=0.01),
            {"record_type": "outcome", "trace_id": "r1", "success": True},
        ]
        assert analyze_events(events).calls_per_run == pytest.approx(1.0)

    def test_empty_run_is_handled(self):
        report = analyze_events([])
        assert report.runs_observed == 0
        assert report.cost_per_run == 0.0


class TestProjection:
    def test_monthly_projection_scales_by_volume(self):
        events = [_event(trace_id="r1", step_name="s", cost=0.01)]
        report = analyze(events=events, runs_per_day=1000)

        assert report.projected_runs_per_day == 1000
        assert report.projected_monthly_cost == pytest.approx(300.0, rel=1e-6)

    def test_no_volume_means_no_projection(self):
        report = analyze(events=[_event(trace_id="r1", cost=0.01)])
        assert report.projected_monthly_cost is None

    def test_findings_are_ordered_most_severe_first(self):
        events = [
            _event(trace_id="r1", step_name="s", success=False),
            _event(trace_id="r1", step_name="s"),
        ]
        severities = [f.severity for f in analyze(events=events).findings]
        assert severities == sorted(severities, key=lambda s: SEVERITY_ORDER[s])


class TestReportFormatting:
    def test_report_renders_without_error(self, agent_dir):
        events = [_event(trace_id="r1", step_name="s", cost=0.01)]
        text = format_report(analyze(str(agent_dir), events=events, runs_per_day=100))

        assert "pre-deployment analysis" in text
        assert "Nothing in this report was transmitted anywhere." in text

    def test_empty_report_renders(self):
        assert "No findings." in format_report(analyze(events=[]))


class TestEventLoading:
    def test_reads_a_json_array(self, tmp_path):
        path = tmp_path / "run.json"
        path.write_text(json.dumps([_event(trace_id="r1")]), encoding="utf-8")
        assert len(load_events(str(path))) == 1

    def test_reads_jsonl(self, tmp_path):
        path = tmp_path / "run.jsonl"
        lines = [json.dumps(_event(trace_id="r%d" % i)) for i in range(3)]
        path.write_text("\n".join(lines), encoding="utf-8")
        assert len(load_events(str(path))) == 3

    def test_empty_file_is_not_an_error(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("", encoding="utf-8")
        assert load_events(str(path)) == []


class TestCli:
    def test_analyze_a_directory(self, agent_dir, capsys):
        assert main(["analyze", str(agent_dir)]) == 0
        assert "pre-deployment analysis" in capsys.readouterr().out

    def test_writes_json_when_asked(self, agent_dir, tmp_path):
        out = tmp_path / "report.json"
        assert main(["analyze", str(agent_dir), "--json", str(out)]) == 0

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["static"]["total_tokens"] > 0
        assert "findings" in data

    def test_fail_on_returns_nonzero_for_a_matching_finding(self, tmp_path):
        events = tmp_path / "run.json"
        payload = [
            _event(trace_id="r1", step_name="s", input_hash="x"),
            _event(trace_id="r1", step_name="s", input_hash="x"),
        ]
        events.write_text(json.dumps(payload), encoding="utf-8")

        assert main(["analyze", "--events", str(events), "--fail-on", "high"]) == 1

    def test_fail_on_passes_a_clean_run(self, agent_dir):
        assert main(["analyze", str(agent_dir), "--fail-on", "high"]) == 0

    def test_requires_an_input(self):
        with pytest.raises(SystemExit):
            main(["analyze"])

    def test_unreadable_events_file_exits_cleanly(self, tmp_path, capsys):
        assert main(["analyze", "--events", str(tmp_path / "nope.json")]) == 2
        assert "could not read" in capsys.readouterr().err
