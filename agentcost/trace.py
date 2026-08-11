"""
AgentCost Trace Context

Gives every tracked call a place in the shape of the run that produced it, so
two $0.02 calls are distinguishable as two pipeline steps or one retried step.

A trace is one run of a workflow; spans nest inside it. Each event carries its
own span id and its parent's, so the server rebuilds the tree without the SDK
ever sending one.

Contextvar-based, like ``agent()`` and ``metadata()``: concurrent tasks and
threads keep their own trace with nothing threaded through call signatures.
"""

import contextvars
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


def _new_id() -> str:
    """Short, collision-safe id. 16 hex chars is plenty inside one project."""
    return uuid.uuid4().hex[:16]


@dataclass
class _Trace:
    """One workflow run."""

    trace_id: str
    workflow: str
    # Lives on the trace, not the span: step_index must be comparable across
    # sibling branches.
    next_index: int = 0
    outcome: Optional["_Outcome"] = None

    def take_index(self) -> int:
        index = self.next_index
        self.next_index += 1
        return index


@dataclass
class _Outcome:
    """How one run ended, as declared by the caller."""

    success: bool
    label: Optional[str] = None


@dataclass
class _Span:
    """One step inside a trace."""

    span_id: str
    name: str
    parent_span_id: Optional[str] = None
    step_index: int = 0
    depth: int = 0
    # Set only by tool(); step() spans never carry a tool identity.
    tool_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


_trace_var: contextvars.ContextVar[Optional[_Trace]] = contextvars.ContextVar(
    "_agentcost_trace", default=None
)
_span_var: contextvars.ContextVar[Optional[_Span]] = contextvars.ContextVar(
    "_agentcost_span", default=None
)


def current_trace_fields() -> Dict[str, Any]:
    """Trace columns for an event created now; empty outside a workflow()."""
    trace = _trace_var.get(None)
    if trace is None:
        return {}

    fields: Dict[str, Any] = {
        "trace_id": trace.trace_id,
        "workflow": trace.workflow,
    }

    span = _span_var.get(None)
    if span is not None:
        fields["parent_span_id"] = span.span_id
        fields["step_name"] = span.name
        fields["step_index"] = span.step_index
        fields["depth"] = span.depth + 1
        if span.tool_name:
            fields["tool_name"] = span.tool_name
    else:
        # A call made directly under workflow() with no step around it.
        fields["depth"] = 0
        fields["step_index"] = trace.take_index()

    return fields


@contextmanager
def workflow(
    name: str,
    trace_id: Optional[str] = None,
    on_close: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """
    Open a trace for one run of a multi-step agent.

    A nested workflow() joins the enclosing trace rather than starting a new
    one: the cost question is always about the outermost run.

        with track_costs.workflow("support-triage"):
            with track_costs.step("classify"):
                llm.invoke(...)

    Pass ``trace_id`` to join a trace minted in another process. ``on_close``
    receives the outcome record, if one was declared, and fires only on the
    frame that owns the trace.
    """
    existing = _trace_var.get(None)
    if existing is not None:
        yield existing.trace_id
        return

    trace = _Trace(trace_id=trace_id or _new_id(), workflow=name)
    token = _trace_var.set(trace)
    try:
        yield trace.trace_id
    finally:
        _trace_var.reset(token)
        if on_close is not None:
            record = take_outcome_record(trace)
            if record is not None:
                on_close(record)


@contextmanager
def step(name: str, **extra: Any):
    """
    Open a span for one step of the enclosing workflow.

        with track_costs.step("retrieve"):
            llm.invoke(...)

    Steps nest, and ``depth`` records how deep. Outside a workflow() this
    yields None and records nothing, so instrumenting a helper never depends
    on how it is called.
    """
    trace = _trace_var.get(None)
    if trace is None:
        yield None
        return

    parent = _span_var.get(None)
    span = _Span(
        span_id=_new_id(),
        name=name,
        parent_span_id=parent.span_id if parent else None,
        step_index=trace.take_index(),
        depth=(parent.depth + 1) if parent else 0,
        extra=extra,
    )
    token = _span_var.set(span)
    try:
        yield span.span_id
    finally:
        _span_var.reset(token)


@contextmanager
def tool(name: str, **extra: Any):
    """
    Open a span for a tool call.

        with track_costs.tool("web_search"):
            results = search(query)

    Like step(), but marks the span with a tool name so calls made underneath
    are attributable to the tool.
    """
    trace = _trace_var.get(None)
    if trace is None:
        yield None
        return

    parent = _span_var.get(None)
    span = _Span(
        span_id=_new_id(),
        name=name,
        parent_span_id=parent.span_id if parent else None,
        step_index=trace.take_index(),
        depth=(parent.depth + 1) if parent else 0,
        tool_name=name,
        extra=extra,
    )
    token = _span_var.set(span)
    try:
        yield span.span_id
    finally:
        _span_var.reset(token)


def current_trace_id() -> Optional[str]:
    """The active trace id, or None outside a workflow()."""
    trace = _trace_var.get(None)
    return trace.trace_id if trace else None


def record_outcome(success: bool, label: Optional[str] = None) -> bool:
    """Mark how the enclosing run ended. Returns False outside a workflow()."""
    trace = _trace_var.get(None)
    if trace is None:
        return False
    trace.outcome = _Outcome(success=success, label=label)
    return True


def take_outcome_record(trace: "_Trace") -> Optional[Dict[str, Any]]:
    """The wire record for a finished trace, or None if none was declared."""
    if trace.outcome is None:
        return None
    return {
        "record_type": "outcome",
        "trace_id": trace.trace_id,
        "workflow": trace.workflow,
        "success": trace.outcome.success,
        "label": trace.outcome.label,
    }
