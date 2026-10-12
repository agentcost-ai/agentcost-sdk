"""
Codex collector.

Codex keeps one JSONL rollout per thread under
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``. Current versions write a
``token_usage_record`` per model response; older ones only the ``token_count``
event, whose ``last_token_usage`` is the same figure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._common import bounded_id, build_event, read_rows

NAME = "codex"
LABEL = "Codex"

HOOK_EVENTS = ("Stop", "SessionStart")
HOOK_NOTE = "Codex runs a new hook only after you trust it: open Codex and run /hooks once."


def home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def hook_file() -> Path:
    return home() / "hooks.json"


def session_files() -> List[Path]:
    return sorted((home() / "sessions").glob("**/rollout-*.jsonl"))


def detect_billing() -> Optional[str]:
    # Codex states the plan in the rollout itself; see parse().
    return None


def _tool_name(payload: Dict[str, Any]) -> Optional[str]:
    kind = payload.get("type") or ""
    if not kind.endswith("_call"):
        return None
    return payload.get("name") or kind[: -len("_call")]


def _usage_event(
    carry: Dict[str, Any], usage: Dict[str, Any], event_id: str, timestamp: str
) -> Optional[Dict[str, Any]]:
    # OpenAI convention already: cached and reasoning tokens are subsets of the
    # input and output counts, not additions to them.
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    tools, carry["tools"] = carry.get("tools") or [], []
    if not (input_tokens or output_tokens):
        return None

    thread = carry.get("thread") or "user"
    return build_event(
        source=NAME,
        event_id=event_id,
        session_id=carry["session"],
        agent_name=NAME if thread == "user" else f"{NAME}/{thread}",
        model=carry.get("model") or "unknown",
        timestamp=timestamp,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=int(usage.get("cached_input_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_write_input_tokens") or 0),
        tools=tools,
        depth=0 if thread == "user" else 1,
        step_name="main" if thread == "user" else thread,
        cwd=carry.get("cwd"),
        branch=carry.get("branch"),
        extra={
            "client_version": carry.get("version"),
            "surface": carry.get("surface"),
            "reasoning_tokens": usage.get("reasoning_output_tokens"),
        },
    )


def parse(
    path: Path, offset: int, carry: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
    """Events for the responses written after ``offset``, and where to resume.

    ``carry`` holds what earlier lines established -- session, model, plan --
    so a sync that starts mid-file still knows them.
    """
    carry = dict(carry)
    carry.setdefault("session", path.stem)
    events: List[Dict[str, Any]] = []
    end = offset

    for start, stop, row in read_rows(path, offset):
        end = stop
        kind, payload = row.get("type"), row.get("payload") or {}
        timestamp = row.get("timestamp") or ""
        event = None

        if kind == "session_meta":
            carry["session"] = payload.get("session_id") or payload.get("id") or carry["session"]
            carry["thread"] = payload.get("thread_source") or carry.get("thread")
            carry["cwd"] = payload.get("cwd") or carry.get("cwd")
            carry["version"] = payload.get("cli_version")
            carry["surface"] = payload.get("originator")
            carry["branch"] = (payload.get("git") or {}).get("branch")

        elif kind == "turn_context":
            carry["model"] = payload.get("model") or carry.get("model")
            carry["cwd"] = payload.get("cwd") or carry.get("cwd")

        elif kind == "response_item":
            name = _tool_name(payload)
            if name:
                carry.setdefault("tools", []).append(name)

        elif kind == "token_usage_record":
            carry["has_records"] = True
            response_id = payload.get("response_id") or f"{carry['session']}:{start}"
            event = _usage_event(
                carry, payload.get("usage") or {}, bounded_id("cx-", response_id), timestamp
            )

        elif kind == "event_msg" and payload.get("type") == "token_count":
            # A ChatGPT plan is named here; a session on an API key has none.
            carry["plan"] = (payload.get("rate_limits") or {}).get("plan_type") or ""
            info = payload.get("info") or {}
            total = (info.get("total_token_usage") or {}).get("total_tokens")
            # Codex repeats the event when nothing was spent; the running total
            # tells a repeat from a new response.
            if not carry.get("has_records") and info and total != carry.get("total"):
                carry["total"] = total
                event = _usage_event(
                    carry,
                    info.get("last_token_usage") or {},
                    bounded_id("cx-", f"{carry['session']}:{start}"),
                    timestamp,
                )

        if event:
            events.append(event)

    if "plan" in carry:
        for event in events:
            event["metadata"]["billing"] = "subscription" if carry["plan"] else "api"

    return events, end, carry
