"""
The gate: refuses a coding agent's tool call or prompt when the project's
guardrails or budget say no.

Installed by ``agentcost connect <agent> --enforce`` as a hook that runs before
every tool call and every prompt. It decides from two local files -- the policy
the last sync fetched and the session's own log -- and never waits on the
network. Anything it cannot work out, it allows: a broken gate must not stop
anyone working.

Run by file path rather than ``python -m``. As a module it would import the
whole SDK before answering, tripling the delay added to each tool call.
"""

from __future__ import annotations

import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PREFIX = "AgentCost guardrail: "


def _in_period(budget: Dict[str, Any]) -> bool:
    """A cap from last month must not block the first prompt of this one."""
    try:
        ends = datetime.fromisoformat(str(budget["period_ends_at"]).replace("Z", "+00:00"))
        return datetime.now(timezone.utc) < ends
    except (KeyError, ValueError, TypeError):
        return False


def _run_totals(source: Any, sync: Any, hook: Dict[str, Any], priced: bool) -> Dict[str, Any]:
    """Tool calls and cost of this session's main thread so far.

    Read from the session log, picking up where the last call to the gate
    stopped, so the totals are current to the call being judged rather than to
    the last sync.
    """
    session = str(hook.get("session_id") or "")
    name = "".join(ch for ch in session if ch.isalnum() or ch == "-") or "unknown"
    state_path = sync.home() / "runs" / f"{name}.json"
    state = sync.read_json(state_path)
    totals = {
        "tool_calls": state.get("tool_calls", 0),
        "cost": state.get("cost", 0.0),
        "billing": state.get("billing"),
        "model": state.get("model"),
    }

    path = hook.get("transcript_path") or source.find_session(session)
    if not path or not os.path.exists(path):
        return totals
    if priced:
        sync.prepare_pricing(sync.load_config(), fetch=False)

    def add(events: Any) -> None:
        for event in events:
            totals["tool_calls"] += 1 if event.get("tool_name") else 0
            totals["cost"] += sync.price(event) if priced else 0.0
            totals["billing"] = event["metadata"].get("billing") or totals["billing"]
            totals["model"] = event["model"]

    settled, offset, carry = source.parse(
        Path(path), state.get("offset", 0), state.get("carry") or {}
    )
    add(settled)
    sync.write_json(state_path, {**totals, "offset": offset, "carry": carry})

    # The call asking for this very tool is in the log but not yet settled.
    # Counted for the decision, not remembered, or it would be counted twice.
    still_open, _, _ = source.parse(Path(path), offset, carry, include_open=True)
    add(still_open)
    return totals


def decide(source_name: str, hook: Dict[str, Any]) -> Optional[str]:
    """Why this tool call or prompt is refused, or None to let it through."""
    from agentcost.agents import SOURCES, sync

    policy = sync.read_json(sync.home() / "policy.json")
    source = SOURCES.get(source_name)
    if not policy or source is None:
        return None

    event, tool = hook.get("hook_event_name"), hook.get("tool_name")
    rails = policy.get("guardrails") or {}
    subagent = hook.get("agent_type")
    agent = f"{source_name}/{subagent}" if subagent else source_name
    # A subagent without a boundary of its own answers to the agent's.
    rail = rails.get(agent) or rails.get(source_name) or {}

    if event == "PreToolUse" and tool and source.GATES_TOOLS:
        allowed = rail.get("allowed_tools")
        if allowed is not None and tool not in allowed:
            return f"{tool} is not among the tools this project allows {agent} to use."
        if rail.get("read_only") and (policy.get("tool_tags") or {}).get(tool) == "write":
            return f"{agent} is read-only in this project, and {tool} writes."

    max_calls = rail.get("max_tool_calls_per_run")
    max_cost = rail.get("max_cost_per_run_usd")
    models = rail.get("allowed_models")
    budget = policy.get("budget") or {}
    capped = (
        event == "UserPromptSubmit"
        and budget.get("exhausted")
        and budget.get("mode") == "hard_cap"
        and _in_period(budget)
    )
    if not (max_calls or max_cost or capped or models is not None):
        return None

    run = _run_totals(source, sync, hook, priced=bool(max_cost))

    # Codex names the model in the hook; Claude Code's is the one its latest
    # call used. A session's very first prompt has neither and is let through.
    model = hook.get("model") or run["model"]
    if models is not None and model and model not in models:
        return (
            f"{model} is not among the models this project allows {source_name} to use "
            f"({', '.join(models) or 'none'}). Switch model to continue."
        )

    # The call being asked about is already in the log and so already counted:
    # the limit is passed only when the count goes over it.
    if max_calls and event == "PreToolUse" and run["tool_calls"] > max_calls:
        return (
            f"this session has reached the limit of {max_calls} tool "
            f"call{'' if max_calls == 1 else 's'} this project allows {source_name} "
            f"in one run. Start a new session."
        )
    if max_cost and run["cost"] > max_cost:
        return (
            f"this session has used ${run['cost']:.2f}, over the ${max_cost:.2f} this "
            f"project allows {source_name} per run. Start a new session."
        )
    if capped:
        settings = (sync.load_config()["sources"].get(source_name)) or {}
        billing = settings.get("billing") or run["billing"] or source.detect_billing()
        # Only metered usage spends the budget. Unknown is not refused.
        if billing == "api":
            return (
                f"the project's monthly budget of {budget.get('budget')} "
                f"{budget.get('currency') or 'USD'} is spent."
            )
    return None


def refusal(hook: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """The hook's answer, in the form Claude Code and Codex both take.

    Not an exit code: both stop the call on exit 2, but Codex then tells the
    model only that the call was rejected. This form carries the reason to the
    model, which is what lets it explain the refusal instead of retrying.
    """
    message = PREFIX + reason
    if hook.get("hook_event_name") == "PreToolUse":
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": message,
            }
        }
    return {"decision": "block", "reason": message}


def main(argv: Any) -> int:
    try:
        hook = json.loads(sys.stdin.buffer.read() or b"{}")
        reason = decide(argv[1], hook)
        if reason:
            sys.stdout.write(json.dumps(refusal(hook, reason)))
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    # Stand in for the package so importing the collectors does not run the
    # SDK's __init__, which pulls in the provider interceptors and an HTTP
    # stack this script never uses.
    if "agentcost" not in sys.modules:
        package = types.ModuleType("agentcost")
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        sys.modules["agentcost"] = package
    raise SystemExit(main(sys.argv))
