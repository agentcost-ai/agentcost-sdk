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

# How AgentCost's hooks are recognised among the user's own.
MARKER = "agentcost.cli sync"
GATE_MARKER = "agentcost/agents/gate.py"


def _python() -> str:
    # Forward slashes, and quotes only when needed: the same string has to run
    # under bash, cmd and PowerShell, and PowerShell will not execute a quoted
    # path without a call operator.
    python = Path(sys.executable).as_posix()
    return f'"{python}"' if " " in python else python


def sync_command(source: str) -> str:
    return f"{_python()} -m {MARKER} --quiet --source {source}"


def gate_command(source: str) -> str:
    # By file, not `-m`: the gate runs before every tool call, and running it
    # as a module would import the whole SDK first. See gate.py.
    script = Path(__file__).with_name("gate.py").as_posix()
    if " " in script:
        script = f'"{script}"'
    return f"{_python()} {script} {source}"


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
    command = str(handler.get("command", "")) if isinstance(handler, dict) else ""
    return MARKER in command or GATE_MARKER in command


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


def is_installed(path: Path, marker: str = MARKER) -> bool:
    try:
        return marker in json.dumps(_load(path).get("hooks") or {})
    except ValueError:
        return False


def install(
    path: Path, events: Iterable[str], source: str, gate_events: Iterable[str] = ()
) -> None:
    """Add the sync hook, and the gate when asked, replacing earlier copies."""
    data = _load(path)
    _without_ours(data)
    hooks = data.setdefault("hooks", {})
    sync = {"type": "command", "command": sync_command(source), "async": True}
    for event in events:
        hooks.setdefault(event, []).append({"hooks": [dict(sync)]})
    # The gate answers before the tool runs, so it cannot be async. It reads
    # two small local files; the timeout only bounds a machine that is stuck.
    gate = {"type": "command", "command": gate_command(source), "timeout": 10}
    for event in gate_events:
        hooks.setdefault(event, []).append({"hooks": [dict(gate)]})
    _save(path, data)


def remove(path: Path) -> bool:
    """Take the sync hook out again. True if there was one."""
    data = _load(path)
    if not _without_ours(data):
        return False
    _save(path, data)
    return True
