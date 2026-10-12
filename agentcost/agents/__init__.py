"""
Collectors for coding agents.

Claude Code, Codex and Antigravity already write every model call to a session
log on the machine they run on. A collector reads those logs and turns each call into an
ordinary AgentCost event, so a coding agent needs no instrumentation at all.

Only usage numbers, model names, tool names and ids are read. Prompt text,
completions, tool arguments and tool output are never sent or kept. The one
place an argument is read at all is Codex, which wraps each tool call in a
script: the script is scanned for the tool name and nothing else is taken.
"""

from . import antigravity, claude_code, codex

SOURCES = {claude_code.NAME: claude_code, codex.NAME: codex, antigravity.NAME: antigravity}

__all__ = ["SOURCES"]
