"""
Antigravity collector.

Antigravity publishes no usage export. What it does keep is one SQLite file per
conversation under ``~/.gemini/antigravity*/conversations/<id>.db``, whose
``gen_metadata`` table holds a protobuf record for every model call. Google
does not document that record; the fields read here were worked out from real
files and can change with any release. So these events are marked
``estimated`` and nothing here is relied on for enforcement.

Conversations from before June 2026 are ``.pb`` files, which are encrypted and
cannot be read at all.
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from ._common import bounded_id, build_event

NAME = "antigravity"
LABEL = "Antigravity"

# The IDE runs no hooks, so nothing can trigger a sync from inside it.
HOOK_EVENTS: Tuple[str, ...] = ()
GATE_EVENTS: Tuple[str, ...] = ()
GATES_TOOLS = False
HOOK_NOTE = (
    "Antigravity has no hook to trigger a sync. Its usage is sent whenever "
    "another connected agent syncs; otherwise run `agentcost sync` on a schedule."
)

# The newest call may still be streaming; take it once the file has gone quiet.
SETTLE_SECONDS = 120

# Field numbers in the undocumented record. One place, so a renumbering in a
# new Antigravity release is a one-line fix.
_CALL = 1  # top level -> the call
_USAGE = 4  # call -> usage
_MODEL = 19  # call -> model name
_TAGS = 20  # call -> repeated {1: key, 2: value}
_UNCACHED_INPUT, _OUTPUT, _CACHE_READ, _THINKING, _REQUEST_ID = 2, 3, 5, 9, 11
_STEP_STARTED = 1  # step metadata -> {1: unix seconds}


def home() -> Path:
    return Path(os.environ.get("AGENTCOST_ANTIGRAVITY_HOME") or Path.home() / ".gemini")


def hook_file() -> Optional[Path]:
    return None


def session_files() -> List[Path]:
    return sorted(home().glob("antigravity*/conversations/*.db"))


def find_session(session_id: str) -> Optional[Path]:
    matches = list(home().glob(f"antigravity*/conversations/{session_id}.db"))
    return matches[0] if matches else None


def detect_billing() -> Optional[str]:
    return "subscription"  # Antigravity is sold as a plan, never per token


def version(path: Path) -> int:
    """What changes when the conversation does.

    Not the file size: SQLite appends to the ``-wal`` file beside the database
    and the database itself may not grow for a long time.
    """
    stamps = [path.stat().st_mtime_ns]
    wal = path.with_name(path.name + "-wal")
    if wal.exists():
        stamps.append(wal.stat().st_mtime_ns)
    return max(stamps)


def _fields(data: bytes) -> List[Tuple[int, int, Any]]:
    """One level of a protobuf message as ``(field, wire type, value)``."""
    out: List[Tuple[int, int, Any]] = []
    position = 0

    def varint() -> int:
        nonlocal position
        value = shift = 0
        while True:
            byte = data[position]
            position += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return value

    while position < len(data):
        key = varint()
        field, wire = key >> 3, key & 7
        if wire == 0:
            value: Any = varint()
        elif wire == 2:
            length = varint()
            value = data[position : position + length]
            position += length
        elif wire in (1, 5):
            width = 8 if wire == 1 else 4
            value = data[position : position + width]
            position += width
        else:
            raise ValueError(f"unsupported wire type {wire}")
        out.append((field, wire, value))
    return out


def _first(fields: List[Tuple[int, int, Any]], number: int, wire: int) -> Any:
    return next((value for field, kind, value in fields if field == number and kind == wire), None)


def _decode(data: bytes) -> Optional[Dict[str, Any]]:
    """Usage, model and ids of one call, or None if it does not look like one."""
    try:
        call = _fields(_first(_fields(data), _CALL, 2) or b"")
        usage = _fields(_first(call, _USAGE, 2) or b"")
        model = _first(call, _MODEL, 2)
        if not usage or not model:
            return None
        counts = {field: value for field, kind, value in usage if kind == 0}
        tags = {}
        for field, kind, value in call:
            if field == _TAGS and kind == 2:
                pair = _fields(value)
                tags[(_first(pair, 1, 2) or b"").decode()] = (_first(pair, 2, 2) or b"").decode()
        request_id = _first(usage, _REQUEST_ID, 2)
        return {
            "model": model.decode(),
            "uncached": counts.get(_UNCACHED_INPUT, 0),
            "output": counts.get(_OUTPUT, 0),
            "cache_read": counts.get(_CACHE_READ, 0),
            "thinking": counts.get(_THINKING),
            "request_id": request_id.decode() if request_id else None,
            "step": int(tags["last_step_index"]) if tags.get("last_step_index", "").isdigit() else None,
        }
    except (IndexError, ValueError, UnicodeDecodeError, TypeError):
        return None


def _step_times(connection: sqlite3.Connection) -> Dict[int, str]:
    times: Dict[int, str] = {}
    for index, metadata in connection.execute("SELECT idx, metadata FROM steps"):
        try:
            seconds = _first(_fields(_first(_fields(metadata or b""), _STEP_STARTED, 2) or b""), 1, 0)
            if seconds:
                times[index] = datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
        except (IndexError, ValueError, OverflowError, OSError):
            continue
    return times


def parse(
    path: Path, offset: int, carry: Dict[str, Any], include_open: bool = False
) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
    """Events for the calls recorded after the last one seen."""
    carry = dict(carry)
    seen = version(path)
    last_sent = carry.get("idx", -1)
    try:
        # Read-only, so a conversation Antigravity has open is never disturbed.
        connection = sqlite3.connect(f"file:{quote(path.as_posix())}?mode=ro", uri=True, timeout=2)
        try:
            rows = connection.execute(
                "SELECT idx, data FROM gen_metadata WHERE idx > ? ORDER BY idx", (last_sent,)
            ).fetchall()
            times = _step_times(connection) if rows else {}
        finally:
            connection.close()
    except sqlite3.Error:
        return [], offset, carry  # locked, or not the layout expected: try again later

    if rows and not include_open and time.time() - seen / 1e9 < SETTLE_SECONDS:
        rows = rows[:-1]

    fallback = datetime.fromtimestamp(seen / 1e9, tz=timezone.utc).isoformat()
    surface = path.parent.parent.name
    events = []
    for index, data in rows:
        carry["idx"] = index
        call = _decode(data)
        if call is None or not (call["uncached"] or call["cache_read"] or call["output"]):
            continue
        events.append(
            build_event(
                source=NAME,
                event_id=bounded_id("ag-", call["request_id"] or f"{path.stem}:{index}"),
                session_id=path.stem,
                agent_name=NAME,
                model=call["model"],
                timestamp=times.get(call["step"], fallback),
                input_tokens=call["uncached"] + call["cache_read"],
                output_tokens=call["output"],
                cached_tokens=call["cache_read"],
                billing="subscription",
                extra={"estimated": True, "surface": surface, "reasoning_tokens": call["thinking"]},
            )
        )
    return events, seen, carry
