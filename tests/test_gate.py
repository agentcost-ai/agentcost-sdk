"""Tests for the gate: the hook that refuses a tool call or a prompt."""
# ruff: noqa: F811  (the imported `homes` fixture is requested by name below)

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

from agentcost.agents import claude_code, gate, hooks
from agentcost.agents import sync as agent_sync
from agentcost.cli import main

from .test_agents import (  # noqa: F401  (homes is an autouse fixture)
    CODEX_HEAD,
    SESSION,
    TEXT,
    TOOL_RESULT,
    USAGE,
    Recorder,
    _append,
    _assistant,
    _claude_session,
    _codex_row,
    _codex_session,
    _config,
    _tool,
    homes,
)


def _policy(homes, rails=(), tags=None, budget=None):
    path = homes / "agentcost" / "policy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    policy = {
        "at": time.time(),
        "guardrails": {rail["agent_name"]: rail for rail in rails},
        "tool_tags": tags or {},
        "budget": budget or {},
    }
    path.write_text(json.dumps(policy), encoding="utf-8")


def _pre_tool(tool, path=None, **extra):
    hook = {"hook_event_name": "PreToolUse", "session_id": SESSION, "tool_name": tool, **extra}
    if path:
        hook["transcript_path"] = str(path)
    return hook


def _prompt(path=None):
    hook = {"hook_event_name": "UserPromptSubmit", "session_id": SESSION}
    if path:
        hook["transcript_path"] = str(path)
    return hook


# -- Tool boundaries -------------------------------------------------------


def test_allows_everything_without_a_policy(homes):
    assert gate.decide("claude-code", _pre_tool("Bash")) is None


def test_refuses_a_tool_outside_the_boundary(homes):
    _policy(homes, [{"agent_name": "claude-code", "allowed_tools": ["Read", "Grep"]}])

    assert gate.decide("claude-code", _pre_tool("Read")) is None
    reason = gate.decide("claude-code", _pre_tool("Bash"))
    assert "Bash" in reason and "claude-code" in reason
    # A subagent with no boundary of its own answers to the main one.
    assert gate.decide("claude-code", _pre_tool("Bash", agent_type="Plan"))


def test_subagent_boundary_wins_over_the_main_one(homes):
    _policy(
        homes,
        [
            {"agent_name": "claude-code", "allowed_tools": ["Read"]},
            {"agent_name": "claude-code/Plan", "allowed_tools": ["Read", "Bash"]},
        ],
    )

    assert gate.decide("claude-code", _pre_tool("Bash", agent_type="Plan")) is None
    assert gate.decide("claude-code", _pre_tool("Bash"))


def test_read_only_refuses_only_tools_known_to_write(homes):
    _policy(
        homes,
        [{"agent_name": "claude-code", "read_only": True}],
        tags={"Edit": "write", "Read": "read"},
    )

    assert "read-only" in gate.decide("claude-code", _pre_tool("Edit"))
    assert gate.decide("claude-code", _pre_tool("Read")) is None
    assert gate.decide("claude-code", _pre_tool("Untagged")) is None


def test_codex_tool_names_are_not_judged(homes):
    _policy(homes, [{"agent_name": "codex", "allowed_tools": []}])

    assert gate.decide("codex", _pre_tool("Bash")) is None


# -- Per-run limits --------------------------------------------------------


def test_counts_tool_calls_up_to_the_one_being_asked(homes):
    path = _claude_session(
        homes,
        [_assistant("msg_1", [_tool("Read")]), TOOL_RESULT, _assistant("msg_2", [_tool("Bash")])],
    )
    _policy(homes, [{"agent_name": "claude-code", "max_tool_calls_per_run": 3}])
    assert gate.decide("claude-code", _pre_tool("Bash", path)) is None
    # Asked again, the open call is still one call, not two.
    assert gate.decide("claude-code", _pre_tool("Bash", path)) is None

    _append(path, [TOOL_RESULT, _assistant("msg_3", [_tool("Bash")])])
    reason = gate.decide("claude-code", _pre_tool("Bash", path))
    assert "3 tool calls" in reason and "new session" in reason
    # A limit on tool calls does not stop the conversation itself.
    assert gate.decide("claude-code", _prompt(path)) is None


def test_stops_a_run_over_its_cost_limit(homes):
    # Each call in the fixture is worth $0.00224 at the fixture rates.
    path = _claude_session(homes, [_assistant("msg_1", [_tool("Read")])])
    _policy(homes, [{"agent_name": "claude-code", "max_cost_per_run_usd": 0.003}])
    assert gate.decide("claude-code", _pre_tool("Read", path)) is None

    _append(path, [TOOL_RESULT, _assistant("msg_2", [_tool("Read")])])
    assert "per run" in gate.decide("claude-code", _pre_tool("Read", path))
    assert gate.decide("claude-code", _prompt(path))


def test_finds_a_codex_session_by_id(homes):
    _codex_session(
        homes,
        CODEX_HEAD
        + [
            _codex_row("response_item", {"type": "custom_tool_call", "name": "exec"}),
            _codex_row(
                "token_usage_record",
                {"session_id": SESSION, "response_id": "resp_1", "usage": USAGE},
            ),
        ],
    )
    _policy(homes, [{"agent_name": "codex", "max_tool_calls_per_run": 1}])

    assert "1 tool calls" in gate.decide("codex", _pre_tool("Bash"))


# -- Budget ----------------------------------------------------------------


def _budget(**overrides):
    budget = {
        "exhausted": True,
        "mode": "hard_cap",
        "budget": 100.0,
        "currency": "USD",
        "period_ends_at": (datetime.now(timezone.utc) + timedelta(days=3)).isoformat(),
    }
    budget.update(overrides)
    return budget


def _connected(**settings):
    agent_sync.save_config(_config(sources={"claude-code": settings}))


def test_stops_prompts_when_a_metered_budget_is_spent(homes):
    path = _claude_session(homes, [_assistant("msg_1", [TEXT], stop="end_turn")])
    _policy(homes, budget=_budget())
    _connected(billing="api")

    assert "budget" in gate.decide("claude-code", _prompt(path))
    # Tool calls mid-turn are not cut off; the next prompt is.
    assert gate.decide("claude-code", _pre_tool("Read", path)) is None


def test_budget_never_stops_a_subscription(homes):
    path = _claude_session(homes, [_assistant("msg_1", [TEXT], stop="end_turn")])
    _policy(homes, budget=_budget())

    _connected(billing="subscription")
    assert gate.decide("claude-code", _prompt(path)) is None
    _connected()  # nothing known about how it is paid for
    assert gate.decide("claude-code", _prompt(path)) is None


def test_ignores_a_cap_that_is_not_in_force(homes):
    path = _claude_session(homes, [_assistant("msg_1", [TEXT], stop="end_turn")])
    _connected(billing="api")

    _policy(homes, budget=_budget(period_ends_at="2026-01-01T00:00:00+00:00"))
    assert gate.decide("claude-code", _prompt(path)) is None
    _policy(homes, budget=_budget(mode="warn"))
    assert gate.decide("claude-code", _prompt(path)) is None


# -- As a hook -------------------------------------------------------------


def _run_gate(homes, source, stdin):
    env = {
        **os.environ,
        "AGENTCOST_HOME": str(homes / "agentcost"),
        "CLAUDE_CONFIG_DIR": str(homes / "claude"),
        "CODEX_HOME": str(homes / "codex"),
    }
    # Exactly as the hook runs it: by file path, with the event on stdin.
    return subprocess.run(
        [sys.executable, gate.__file__, source],
        input=stdin,
        capture_output=True,
        env=env,
        timeout=60,
    )


def test_blocks_with_exit_2_and_a_reason(homes):
    _policy(homes, [{"agent_name": "claude-code", "allowed_tools": ["Read"]}])

    blocked = _run_gate(homes, "claude-code", json.dumps(_pre_tool("Bash")).encode())
    assert blocked.returncode == 2
    assert b"AgentCost guardrail: Bash" in blocked.stderr

    allowed = _run_gate(homes, "claude-code", json.dumps(_pre_tool("Read")).encode())
    assert allowed.returncode == 0 and allowed.stderr == b""


def test_run_limits_work_without_the_sdk_loaded(homes):
    """The script stands in for the package; the cost path must still import."""
    path = _claude_session(homes, [_assistant("msg_1", [_tool("Read")], stop="end_turn")])
    _policy(homes, [{"agent_name": "claude-code", "max_cost_per_run_usd": 0.000001}])
    pricing = {"claude-opus-5-5": {"input": 0.004, "output": 0.02}}
    cache = {"base_url": agent_sync.DEFAULT_BASE_URL, "at": time.time(), "pricing": pricing}
    (homes / "agentcost" / "pricing.json").write_text(json.dumps(cache), encoding="utf-8")

    blocked = _run_gate(homes, "claude-code", json.dumps(_pre_tool("Read", path)).encode())

    assert blocked.returncode == 2, blocked.stderr
    assert b"per run" in blocked.stderr


def test_lets_work_continue_when_it_cannot_decide(homes):
    _policy(homes, [{"agent_name": "claude-code", "allowed_tools": ["Read"]}])
    event = json.dumps(_pre_tool("Bash")).encode()

    assert _run_gate(homes, "claude-code", b"not json").returncode == 0
    assert _run_gate(homes, "no-such-agent", event).returncode == 0
    (homes / "agentcost" / "policy.json").write_text("{ broken", encoding="utf-8")
    assert _run_gate(homes, "claude-code", event).returncode == 0


# -- Installing and refreshing --------------------------------------------


def test_install_with_gate_and_clean_removal(homes):
    path = claude_code.hook_file()

    hooks.install(path, claude_code.HOOK_EVENTS, "claude-code", claude_code.GATE_EVENTS)

    data = json.loads(path.read_text(encoding="utf-8"))["hooks"]
    assert set(data) == {"Stop", "SessionStart", "PreToolUse", "UserPromptSubmit"}
    gate_hook = data["PreToolUse"][0]["hooks"][0]
    assert gate_hook["command"].endswith("gate.py claude-code") and "async" not in gate_hook
    assert hooks.is_installed(path, hooks.GATE_MARKER)

    # Connecting again without enforcement takes the gate out and keeps the sync.
    hooks.install(path, claude_code.HOOK_EVENTS, "claude-code")
    assert not hooks.is_installed(path, hooks.GATE_MARKER) and hooks.is_installed(path)

    hooks.install(path, claude_code.HOOK_EVENTS, "claude-code", claude_code.GATE_EVENTS)
    assert hooks.remove(path)
    assert "hooks" not in json.loads(path.read_text(encoding="utf-8"))


class _Served:
    def __init__(self, body, ok=True):
        self.ok, self._body = ok, body

    def json(self):
        return self._body


SERVED = {
    "guardrails": [
        {"agent_name": "claude-code", "allowed_tools": ["Read"], "max_cost_per_run_usd": 5}
    ],
    "tool_tags": {"Edit": "write"},
    "budget": {"mode": "hard_cap"},
}


def test_refresh_policy_keeps_the_last_good_copy(homes, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Served(SERVED))
    policy = agent_sync.refresh_policy(_config(), force=True)
    assert policy["guardrails"]["claude-code"]["allowed_tools"] == ["Read"]
    assert gate.decide("claude-code", _pre_tool("Bash"))

    # An older backend, or one that is down, changes nothing.
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Served(None, ok=False))
    assert agent_sync.refresh_policy(_config(), force=True) == policy
    assert gate.decide("claude-code", _pre_tool("Bash"))


def test_sync_refreshes_policy_only_when_enforcing(homes, monkeypatch):
    import requests

    calls = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append(a) or _Served(SERVED))

    agent_sync.run_sync(_config(), sender=Recorder(), fetch_pricing=False)
    assert calls == []

    enforcing = _config(sources={"claude-code": {"enforce": True}})
    agent_sync.run_sync(enforcing, sender=Recorder(), fetch_pricing=False)
    assert len(calls) == 1
    agent_sync.run_sync(enforcing, sender=Recorder(), fetch_pricing=False)
    assert len(calls) == 1  # still fresh


def test_cli_connect_enforce(homes, capsys, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Served(SERVED))
    monkeypatch.setattr(agent_sync, "_http_sender", lambda config: Recorder())
    monkeypatch.setattr(agent_sync, "load_pricing", lambda base_url, fetch: None)

    code = main(
        ["connect", "claude-code", "--api-key", "sk_test", "--project-id", "proj", "--enforce"]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "only these tools: Read" in out
    assert "$5.00 per session" in out and "monthly budget" in out
    assert hooks.is_installed(claude_code.hook_file(), hooks.GATE_MARKER)
    assert main(["status"]) == 0
    assert "enforcing guardrails" in capsys.readouterr().out
