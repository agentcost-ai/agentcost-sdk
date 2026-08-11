"""
AgentCost Pre-deployment Analyzer

Estimates what an agent will cost, and where it will misbehave, before it has
spent anything. Two local inputs: the prompt and skill files on disk, and a
local-mode test run projected to production volume.

Nothing here opens a socket, and no file content outlives the token count
taken from it.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .cost_calculator import calculate_cost
from .token_counter import TokenCounter

DEFAULT_PATTERNS: Sequence[str] = (
    "*.md",
    "*.txt",
    "*.prompt",
    "*.tmpl",
    "*.j2",
    "*.jinja",
    "*.jinja2",
)

SKIP_DIRECTORIES = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".next", "dist", "build", ".tox",
})

# Beyond this share of a model's window, a single file leaves little room for
# conversation and retrieved context.
CONTEXT_WARN_RATIO = 0.25

DEFAULT_CONTEXT_WINDOW = 128_000

CONTEXT_WINDOWS: Dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4-turbo": 128_000,
    "claude-3-5-sonnet": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-1": 200_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
}


def context_window(model: str) -> int:
    """Best-known window for a model, falling back to a conservative default."""
    if model in CONTEXT_WINDOWS:
        return CONTEXT_WINDOWS[model]
    for known, window in CONTEXT_WINDOWS.items():
        if model.startswith(known):
            return window
    return DEFAULT_CONTEXT_WINDOW


@dataclass
class Finding:
    """One thing worth fixing before deploying."""

    severity: str  # "high" | "medium" | "low"
    code: str
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileReport:
    path: str
    tokens: int
    bytes: int
    cost_per_call: float
    share_of_context: float


@dataclass
class StaticReport:
    """What the prompt and skill files cost every time they are sent."""

    model: str
    files: List[FileReport] = field(default_factory=list)
    total_tokens: int = 0
    cost_per_call: float = 0.0
    findings: List[Finding] = field(default_factory=list)


@dataclass
class StepProjection:
    step_name: str
    calls_per_run: float
    cost_per_run: float
    share_of_run: float


@dataclass
class RunReport:
    """What a recorded test run implies about production."""

    runs_observed: int
    calls_per_run: float
    cost_per_run: float
    max_cost_per_run: float
    steps: List[StepProjection] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)


@dataclass
class AnalysisReport:
    static: Optional[StaticReport] = None
    run: Optional[RunReport] = None
    projected_monthly_cost: Optional[float] = None
    projected_runs_per_day: Optional[int] = None

    @property
    def findings(self) -> List[Finding]:
        found: List[Finding] = []
        if self.static:
            found.extend(self.static.findings)
        if self.run:
            found.extend(self.run.findings)
        order = {"high": 0, "medium": 1, "low": 2}
        return sorted(found, key=lambda f: order.get(f.severity, 3))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "static": asdict(self.static) if self.static else None,
            "run": asdict(self.run) if self.run else None,
            "projected_monthly_cost": self.projected_monthly_cost,
            "projected_runs_per_day": self.projected_runs_per_day,
            "findings": [asdict(f) for f in self.findings],
        }


def iter_files(
    root: str, patterns: Sequence[str] = DEFAULT_PATTERNS
) -> Iterable[str]:
    """Every file under root matching any pattern, skipping vendored trees."""
    if os.path.isfile(root):
        yield root
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRECTORIES]
        for filename in sorted(filenames):
            if any(fnmatch.fnmatch(filename, p) for p in patterns):
                yield os.path.join(dirpath, filename)


def analyze_files(
    root: str,
    model: str = "gpt-4o",
    patterns: Sequence[str] = DEFAULT_PATTERNS,
) -> StaticReport:
    """Token-count and price the prompt/skill files an agent ships with."""
    report = StaticReport(model=model)
    window = context_window(model)
    digests: Dict[str, List[str]] = {}

    for path in iter_files(root, patterns):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            continue

        tokens = TokenCounter.count_tokens(text, model)
        if tokens == 0:
            continue

        # Only the digest is kept; the text goes out of scope here.
        digest = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
        digests.setdefault(digest, []).append(path)

        report.files.append(
            FileReport(
                path=os.path.relpath(path, root) if os.path.isdir(root) else path,
                tokens=tokens,
                bytes=len(text.encode("utf-8")),
                cost_per_call=round(calculate_cost(model, tokens, 0), 6),
                share_of_context=round(tokens / window, 4),
            )
        )

    report.files.sort(key=lambda f: f.tokens, reverse=True)
    report.total_tokens = sum(f.tokens for f in report.files)
    report.cost_per_call = round(calculate_cost(model, report.total_tokens, 0), 6)
    report.findings = _static_findings(report, window, digests)
    return report


def _static_findings(
    report: StaticReport, window: int, digests: Dict[str, List[str]]
) -> List[Finding]:
    findings: List[Finding] = []

    for file_report in report.files:
        if file_report.share_of_context >= CONTEXT_WARN_RATIO:
            findings.append(
                Finding(
                    severity="high",
                    code="oversized_file",
                    message=(
                        f"{file_report.path} is {file_report.tokens:,} tokens, "
                        f"{file_report.share_of_context:.0%} of the model's context "
                        "window"
                    ),
                    detail={"path": file_report.path, "tokens": file_report.tokens},
                )
            )

    for paths in digests.values():
        if len(paths) > 1:
            findings.append(
                Finding(
                    severity="medium",
                    code="duplicate_content",
                    message=(
                        f"{len(paths)} files have identical content; sending both "
                        "pays twice for the same context"
                    ),
                    detail={"paths": sorted(paths)},
                )
            )

    if report.total_tokens >= window:
        findings.append(
            Finding(
                severity="high",
                code="context_overflow",
                message=(
                    f"All files together are {report.total_tokens:,} tokens, which "
                    f"exceeds the {window:,}-token context window"
                ),
                detail={"total_tokens": report.total_tokens, "window": window},
            )
        )

    return findings


def analyze_events(
    events: Sequence[Dict[str, Any]],
    runs_per_day: Optional[int] = None,
) -> RunReport:
    """
    Project a recorded run to production.

    Takes what ``track_costs.get_local_events()`` returns after a test run.
    Events without a trace are treated as one run, so an uninstrumented dry
    run still yields a cost per run rather than nothing.
    """
    llm_events = [e for e in events if e.get("record_type") != "outcome"]

    runs: Dict[str, List[Dict[str, Any]]] = {}
    for event in llm_events:
        runs.setdefault(event.get("trace_id") or "__untraced__", []).append(event)

    if not runs:
        return RunReport(
            runs_observed=0, calls_per_run=0.0, cost_per_run=0.0, max_cost_per_run=0.0
        )

    run_costs = [sum(float(e.get("cost") or 0.0) for e in calls) for calls in runs.values()]
    total_cost = sum(run_costs)
    run_count = len(runs)

    report = RunReport(
        runs_observed=run_count,
        calls_per_run=round(len(llm_events) / run_count, 2),
        cost_per_run=round(total_cost / run_count, 6),
        max_cost_per_run=round(max(run_costs), 6),
        steps=_step_projections(llm_events, run_count, total_cost),
    )
    report.findings = _run_findings(llm_events, runs, report)
    return report


def _step_projections(
    events: Sequence[Dict[str, Any]], run_count: int, total_cost: float
) -> List[StepProjection]:
    by_step: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        by_step.setdefault(event.get("step_name") or "(unnamed)", []).append(event)

    projections = []
    for name, calls in by_step.items():
        cost = sum(float(e.get("cost") or 0.0) for e in calls)
        projections.append(
            StepProjection(
                step_name=name,
                calls_per_run=round(len(calls) / run_count, 2),
                cost_per_run=round(cost / run_count, 6),
                share_of_run=round(cost / total_cost, 4) if total_cost else 0.0,
            )
        )
    projections.sort(key=lambda s: s.cost_per_run, reverse=True)
    return projections


def _run_findings(
    events: Sequence[Dict[str, Any]],
    runs: Dict[str, List[Dict[str, Any]]],
    report: RunReport,
) -> List[Finding]:
    findings: List[Finding] = []

    for step in report.steps:
        if step.calls_per_run >= 2:
            findings.append(
                Finding(
                    severity="high",
                    code="step_loops",
                    message=(
                        f"Step '{step.step_name}' ran {step.calls_per_run} times per "
                        "run; a loop or retry will multiply this in production"
                    ),
                    detail={"step": step.step_name, "calls_per_run": step.calls_per_run},
                )
            )

    # Identical inputs inside a single run: work already done, paid for twice.
    # Reported once across all runs -- one finding per affected run buries the
    # rest of the report on any realistic sample.
    affected_runs = 0
    worst_repeat = 0
    for calls in runs.values():
        seen: Dict[str, int] = {}
        for event in calls:
            digest = event.get("input_hash")
            if digest:
                seen[digest] = seen.get(digest, 0) + 1
        repeats = [n for n in seen.values() if n > 1]
        if repeats:
            affected_runs += 1
            worst_repeat = max(worst_repeat, max(repeats))

    if affected_runs:
        findings.append(
            Finding(
                severity="high",
                code="repeated_call",
                message=(
                    f"{affected_runs} of {len(runs)} run(s) made the same call more "
                    f"than once (worst: {worst_repeat}x); the repeats are avoidable"
                ),
                detail={"affected_runs": affected_runs, "worst_repeat": worst_repeat},
            )
        )

    failures = [e for e in events if e.get("success") is False]
    if failures:
        findings.append(
            Finding(
                severity="medium",
                code="failed_calls",
                message=f"{len(failures)} call(s) failed during the test run",
                detail={"count": len(failures)},
            )
        )

    depths = [int(e.get("depth") or 0) for e in events]
    if depths and max(depths) >= 4:
        findings.append(
            Finding(
                severity="medium",
                code="deep_nesting",
                message=(
                    f"Calls nested {max(depths)} levels deep; deep recursion is where "
                    "runaway cost usually starts"
                ),
                detail={"max_depth": max(depths)},
            )
        )

    if not any(e.get("trace_id") for e in events):
        findings.append(
            Finding(
                severity="low",
                code="not_instrumented",
                message=(
                    "No workflow() in the run, so every call was treated as one run. "
                    "Wrap the agent to get per-step figures"
                ),
            )
        )

    return findings


def analyze(
    path: Optional[str] = None,
    events: Optional[Sequence[Dict[str, Any]]] = None,
    model: str = "gpt-4o",
    runs_per_day: Optional[int] = None,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
) -> AnalysisReport:
    """Run whichever analyses the given inputs support."""
    report = AnalysisReport()

    if path:
        report.static = analyze_files(path, model=model, patterns=patterns)
    if events is not None:
        report.run = analyze_events(events, runs_per_day=runs_per_day)

    if runs_per_day and report.run and report.run.cost_per_run:
        report.projected_runs_per_day = runs_per_day
        report.projected_monthly_cost = round(
            report.run.cost_per_run * runs_per_day * 30, 2
        )

    return report


def format_report(report: AnalysisReport) -> str:
    """Human-readable report for a terminal."""
    lines: List[str] = ["", "AgentCost pre-deployment analysis", "=" * 34, ""]

    if report.static:
        static = report.static
        lines.append(f"Prompt and skill files  ({static.model})")
        lines.append(
            f"  {len(static.files)} file(s), {static.total_tokens:,} tokens, "
            f"${static.cost_per_call:.6f} per call just to send them"
        )
        for file_report in static.files[:10]:
            lines.append(
                f"    {file_report.tokens:>8,} tok  "
                f"{file_report.share_of_context:>5.1%} ctx  {file_report.path}"
            )
        if len(static.files) > 10:
            lines.append(f"    ... and {len(static.files) - 10} more")
        lines.append("")

    if report.run:
        run = report.run
        lines.append("Test run")
        lines.append(
            f"  {run.runs_observed} run(s), {run.calls_per_run} calls per run, "
            f"${run.cost_per_run:.6f} per run (worst ${run.max_cost_per_run:.6f})"
        )
        for step in run.steps[:10]:
            lines.append(
                f"    ${step.cost_per_run:>10.6f}  {step.share_of_run:>5.1%}  "
                f"{step.calls_per_run:>4} calls  {step.step_name}"
            )
        lines.append("")

    if report.projected_monthly_cost is not None:
        lines.append(
            f"Projected at {report.projected_runs_per_day:,} runs/day: "
            f"${report.projected_monthly_cost:,.2f} per month"
        )
        lines.append("")

    findings = report.findings
    if findings:
        lines.append(f"Findings ({len(findings)})")
        for finding in findings:
            lines.append(f"  [{finding.severity:>6}] {finding.message}")
    else:
        lines.append("No findings.")

    lines.append("")
    lines.append("Nothing in this report was transmitted anywhere.")
    lines.append("")
    return "\n".join(lines)


def load_events(path: str) -> List[Dict[str, Any]]:
    """Read events saved from a local-mode run (JSON array or JSONL)."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()

    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]
