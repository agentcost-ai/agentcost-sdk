"""
Installs the hook that runs ``agentcost sync`` when a coding agent finishes a
turn. Claude Code's ``settings.json`` and Codex's ``hooks.json`` share one
shape, so one installer serves both.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

# How an AgentCost hook is recognised among the user's own.
MARKER = "agentcost.cli sync"


def sync_command(source: str) -> str:
    # Forward slashes, and quotes only when needed: the same string has to run
    # under bash, cmd and PowerShell, and PowerShell will not execute a quoted
    # path without a call operator.
    python = Path(sys.executable).as_posix()
    if " " in python:
        python = f'"{python}"'
    return f"{python} -m {MARKER} --quiet --source {source}"


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON ({exc}); left untouched") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object; left untouched")
    return data


def _save(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_name(path.name + ".agentcost-backup")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_name(path.name + ".agentcost-tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _is_ours(handler: Any) -> bool:
    return isinstance(handler, dict) and MARKER in str(handler.get("command", ""))


def _without_ours(data: Dict[str, Any]) -> bool:
    """Strip AgentCost handlers in place; report whether any were there."""
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return False
    found = False
    for event in list(hooks):
        groups = hooks[event] if isinstance(hooks[event], list) else []
        kept = []
        for group in groups:
            handlers = group.get("hooks") if isinstance(group, dict) else None
            if not isinstance(handlers, list) or not any(_is_ours(h) for h in handlers):
                kept.append(group)
                continue
            found = True
            rest = [handler for handler in handlers if not _is_ours(handler)]
            if rest:
                kept.append({**group, "hooks": rest})
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]
    if not hooks:
        del data["hooks"]
    return found


def is_installed(path: Path) -> bool:
    try:
        return _without_ours(_load(path))
    except ValueError:
        return False


def install(path: Path, events: Iterable[str], source: str) -> None:
    """Add the sync hook, replacing any earlier copy of it."""
    data = _load(path)
    _without_ours(data)
    hooks = data.setdefault("hooks", {})
    handler = {"type": "command", "command": sync_command(source), "async": True}
    for event in events:
        hooks.setdefault(event, []).append({"hooks": [dict(handler)]})
    _save(path, data)


def remove(path: Path) -> bool:
    """Take the sync hook out again. True if there was one."""
    data = _load(path)
    if not _without_ours(data):
        return False
    _save(path, data)
    return True
