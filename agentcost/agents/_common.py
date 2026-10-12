"""Shared pieces of the session-log collectors."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# The backend's width for event_id and trace_id.
MAX_ID_LENGTH = 64


def read_rows(path: Path, offset: int) -> Iterator[Tuple[int, int, Dict[str, Any]]]:
    """Yield ``(start, end, row)`` for each complete JSON line after ``offset``.

    A line without its newline is still being written by the agent, so reading
    stops there and it is picked up whole on the next sync. Lines that are not
    JSON objects are skipped but still consumed.
    """
    with open(path, "rb") as handle:
        handle.seek(offset)
        position = offset
        for raw in handle:
            if not raw.endswith(b"\n"):
                return
            start, position = position, position + len(raw)
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield start, position, row


def bounded_id(prefix: str, value: str) -> str:
    """An id that fits the backend's column, hashed only when it would not."""
    candidate = f"{prefix}{value}"
    if len(candidate) <= MAX_ID_LENGTH:
        return candidate
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:40]


def repo_name(cwd: Optional[str]) -> Optional[str]:
    """The working directory's own name -- never the path that leads to it."""
    if not cwd:
        return None
    return os.path.basename(cwd.replace("\\", "/").rstrip("/")) or None


def build_event(
    *,
    source: str,
    event_id: str,
    session_id: str,
    agent_name: str,
    model: str,
    timestamp: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    tools: Optional[List[str]] = None,
    depth: int = 0,
    step_name: str = "main",
    billing: Optional[str] = None,
    cwd: Optional[str] = None,
    branch: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One model call as an AgentCost event.

    A session is the run: its id becomes the trace id, so the dashboard shows
    what each session cost and which calls made it expensive. ``tool_name`` is
    the tool the call invoked, which is what guardrails judge.
    """
    metadata: Dict[str, Any] = {
        "source": source,
        "session_id": session_id,
        # Filled in by the sync when the log does not say.
        "billing": billing,
        # Dropped by the sync unless the user opted in to sending them.
        "repo": repo_name(cwd),
        "branch": branch,
    }
    if tools:
        # Booleans only, matching the SDK's capability fingerprint: enough for
        # the optimizer to rule out models that cannot call tools.
        metadata["_ac_caps"] = {"tools": True}
        if len(tools) > 1:
            metadata["tools"] = tools
    if extra:
        metadata.update({key: value for key, value in extra.items() if value})

    event: Dict[str, Any] = {
        "event_id": event_id,
        "agent_name": agent_name,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "timestamp": timestamp,
        "success": True,
        "trace_id": session_id[:MAX_ID_LENGTH],
        "workflow": source,
        "step_name": step_name,
        "depth": depth,
        "metadata": metadata,
    }
    if cached_tokens:
        event["cached_tokens"] = cached_tokens
    if cache_write_tokens:
        event["cache_write_tokens"] = cache_write_tokens
    if tools:
        event["tool_name"] = tools[0]
    return event
