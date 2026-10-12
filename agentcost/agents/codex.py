"""
Codex collector.

Codex keeps one JSONL rollout per thread under
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``. Current versions write a
``token_usage_record`` per model response; older ones only the ``token_count``
event, whose ``last_token_usage`` is the same figure.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._common import build_event, read_rows

NAME = "codex"
LABEL = "Codex"

HOOK_EVENTS = ("Stop", "SessionStart")
GATE_EVENTS = ("PreToolUse", "UserPromptSubmit")
# Tools are reported under the names Codex gives them in its hooks (see
# _tool_names), so a tool boundary declared from what the dashboard shows is
# enforceable as written.
GATES_TOOLS = True
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


# Codex's shell tool has had several names in its rollouts. Its hooks call it
# one thing, and that is the name a guardrail is written and enforced against.
_HOOK_NAMES = {"exec_command": "Bash", "shell": "Bash", "local_shell": "Bash"}

# In code mode Codex wraps every tool call in a script: `await tools.<name>(...)`.
_WRAPPED_TOOL = re.compile(r"\btools\.([A-Za-z_]\w*)\s*\(")


def _tool_names(payload: Dict[str, Any]) -> List[str]:
    """The tools one response item called, as Codex's hooks name them."""
    kind = payload.get("type") or ""
    if not kind.endswith("_call"):
        return []
    name = payload.get("name") or kind[: -len("_call")]
    names = [name]
    if name == "exec" and isinstance(payload.get("input"), str):
        # Only the tool names are taken from the script; the rest is not kept.
        names = _WRAPPED_TOOL.findall(payload["input"]) or names
    return [_HOOK_NAMES.get(found, found) for found in names]


def call_id(session: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> str:
    """An id for one response that every path to AgentCost can compute.

    Codex's OpenTelemetry export carries no response id, only the counts, so
    the rollout's response id cannot be the key if a machine reporting both
    ways is to be counted once. The prompt grows with every call in a session,
    which makes these four values distinct in practice.
    """
    identity = f"{session}|{input_tokens}|{cached_tokens}|{output_tokens}"
    return "cx-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:40]


def _usage_event(
    carry: Dict[str, Any], usage: Dict[str, Any], timestamp: str
) -> Optional[Dict[str, Any]]:
    # OpenAI convention already: cached and reasoning tokens are subsets of the
    # input and output counts, not additions to them.
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    tools, carry["tools"] = carry.get("tools") or [], []
    if not (input_tokens or output_tokens):
        return None

    cached_tokens = int(usage.get("cached_input_tokens") or 0)
    thread = carry.get("thread") or "user"
    return build_event(
        source=NAME,
        event_id=call_id(carry["session"], input_tokens, cached_tokens, output_tokens),
        session_id=carry["session"],
        agent_name=NAME if thread == "user" else f"{NAME}/{thread}",
        model=carry.get("model") or "unknown",
        timestamp=timestamp,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
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


def find_session(session_id: str) -> Optional[Path]:
    matches = list((home() / "sessions").glob(f"**/rollout-*{session_id}.jsonl"))
    return matches[0] if matches else None


def parse(
    path: Path, offset: int, carry: Dict[str, Any], include_open: bool = False
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
            carry.setdefault("tools", []).extend(_tool_names(payload))

        elif kind == "token_usage_record":
            carry["has_records"] = True
            event = _usage_event(carry, payload.get("usage") or {}, timestamp)

        elif kind == "event_msg" and payload.get("type") == "token_count":
            # A ChatGPT plan is named here; a session on an API key has none.
            carry["plan"] = (payload.get("rate_limits") or {}).get("plan_type") or ""
            info = payload.get("info") or {}
            total = (info.get("total_token_usage") or {}).get("total_tokens")
            # Codex repeats the event when nothing was spent; the running total
            # tells a repeat from a new response.
            if not carry.get("has_records") and info and total != carry.get("total"):
                carry["total"] = total
                event = _usage_event(carry, info.get("last_token_usage") or {}, timestamp)

        if event:
            events.append(event)

    if "plan" in carry:
        for event in events:
            event["metadata"]["billing"] = "subscription" if carry["plan"] else "api"

    return events, end, carry
