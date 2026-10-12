"""
Collectors for coding agents.

Claude Code and Codex already write every model call to a session log on the
machine they run on. A collector reads those logs and turns each call into an
ordinary AgentCost event, so a coding agent needs no instrumentation at all.

Only usage numbers, model names, tool names and ids are read. Prompt text,
completions, tool arguments and tool output are never looked at.
"""

from . import claude_code, codex

SOURCES = {claude_code.NAME: claude_code, codex.NAME: codex}

__all__ = ["SOURCES"]
