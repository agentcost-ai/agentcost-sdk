"""
Claude Code collector.

Claude Code keeps one JSONL transcript per session under
``~/.claude/projects/<project>/<session>.jsonl``, and one per subagent under
``<session>/subagents/``. Every assistant line carries the API message it came
from, including the provider's own usage block.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._common import bounded_id, build_event, read_rows

NAME = "claude-code"
LABEL = "Claude Code"

HOOK_EVENTS = ("Stop", "SessionStart")
GATE_EVENTS = ("PreToolUse", "UserPromptSubmit")
# The hook names a tool exactly as the transcript does, so a tool boundary
# declared from what the dashboard shows can be enforced as written.
GATES_TOOLS = True
HOOK_NOTE = "Claude Code picks the hook up in new sessions."

# A call that ended by asking for tools is only known to be fully written once
# the next call appears. If the session died first, take it after this long.
SETTLE_SECONDS = 600

# Written by Claude Code itself for local errors and interruptions; no API call.
_SYNTHETIC_MODEL = "<synthetic>"


def home() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def hook_file() -> Path:
    return home() / "settings.json"


def session_files() -> List[Path]:
    return sorted((home() / "projects").glob("**/*.jsonl"))


def detect_billing() -> Optional[str]:
    """Whether usage is paid per token or covered by a subscription.

    Read from the account summary Claude Code keeps beside its settings. The
    credentials file is never opened.
    """
    # Beside the config directory when one is set, otherwise in the home folder.
    folder = home() if os.environ.get("CLAUDE_CONFIG_DIR") else Path.home()
    try:
        account = json.loads((folder / ".claude.json").read_text(encoding="utf-8")).get("oauthAccount")
    except (OSError, ValueError, AttributeError):
        return None
    if not account:
        return "api"
    kind = f"{account.get('billingType', '')} {account.get('organizationType', '')}"
    return "subscription" if "subscription" in kind or "claude_" in kind else "api"


def _subagent_type(path: Path, row: Dict[str, Any]) -> str:
    if row.get("attributionAgent"):
        return str(row["attributionAgent"])
    try:
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        return str(meta.get("agentType") or "subagent")
    except (OSError, ValueError, AttributeError):
        return "subagent"


def _to_event(path: Path, call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    row, usage = call["row"], call["usage"]
    uncached = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    if not (uncached or cache_read or cache_write or output):
        return None

    session_id = row.get("sessionId") or path.stem
    is_subagent = path.parent.name == "subagents"
    subagent = _subagent_type(path, row) if is_subagent else None

    return build_event(
        source=NAME,
        # The API message id, not the transcript line: a resumed session copies
        # its history into a new file, and the call must not be counted twice.
        event_id=bounded_id("cc-", call["id"]),
        session_id=session_id,
        agent_name=f"{NAME}/{subagent}" if subagent else NAME,
        model=call["model"],
        timestamp=row.get("timestamp") or "",
        # Anthropic reports cache reads beside input_tokens; AgentCost counts
        # the whole prompt as input and the read part as a subset of it.
        input_tokens=uncached + cache_read,
        output_tokens=output,
        cached_tokens=cache_read,
        cache_write_tokens=cache_write,
        tools=call["tools"],
        depth=1 if is_subagent else 0,
        step_name=subagent or "main",
        cwd=row.get("cwd"),
        branch=row.get("gitBranch"),
        extra={
            "client_version": row.get("version"),
            "surface": row.get("entrypoint"),
            # Kept so 1-hour cache writes can be priced apart from 5-minute
            # ones; the event schema has a single cache-write count.
            "cache_write_1h_tokens": (usage.get("cache_creation") or {}).get(
                "ephemeral_1h_input_tokens"
            ),
        },
    )


def find_session(session_id: str) -> Optional[Path]:
    matches = list((home() / "projects").glob(f"*/{session_id}.jsonl"))
    return matches[0] if matches else None


def parse(
    path: Path, offset: int, carry: Dict[str, Any], include_open: bool = False
) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
    """Events for the calls written after ``offset``, and where to resume.

    ``include_open`` also returns a final call that may still be growing. The
    gate wants it, to count a run as it stands; a sync must not, because the
    server keeps the first version of an event it is sent.
    """
    calls: Dict[str, Dict[str, Any]] = {}
    last_id: Optional[str] = None
    end = offset

    for start, stop, row in read_rows(path, offset):
        end = stop
        if row.get("type") != "assistant":
            continue
        message = row.get("message") or {}
        message_id, usage, model = message.get("id"), message.get("usage"), message.get("model")
        if not message_id or not usage or not model or model == _SYNTHETIC_MODEL:
            continue

        # One API message is written as one line per content block, each
        # repeating the same usage, with tool results interleaved.
        call = calls.setdefault(
            message_id,
            {"id": message_id, "start": start, "row": row, "model": model, "tools": [], "stop": None},
        )
        call["usage"] = usage
        call["stop"] = message.get("stop_reason") or call["stop"]
        call["tools"] += [
            block["name"]
            for block in message.get("content") or []
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name")
        ]
        last_id = message_id

    # The final call may still be growing tool blocks. Leave it for the next
    # sync rather than send it short: event ids are write-once on the server.
    if last_id is not None and not include_open:
        last = calls[last_id]
        still_open = last["stop"] in (None, "tool_use")
        if still_open and time.time() - path.stat().st_mtime < SETTLE_SECONDS:
            del calls[last_id]
            end = last["start"]

    events = [event for event in (_to_event(path, call) for call in calls.values()) if event]
    return events, end, carry
