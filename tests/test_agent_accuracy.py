"""Failed calls, one-hour cache pricing and model boundaries for coding agents."""
# ruff: noqa: F811  (the imported `homes` fixture is requested by name below)

import json
import time

from agentcost.agents import claude_code, gate
from agentcost.agents import sync as agent_sync

from .test_agents import (  # noqa: F401
    SESSION,
    TEXT,
    Recorder,
    _assistant,
    _claude_session,
    _config,
    _tool,
    homes,
)


def _api_error(error="rate_limit", status=429, uuid="e1"):
    return {
        "type": "assistant",
        "isApiErrorMessage": True,
        "error": error,
        "apiErrorStatus": status,
        "uuid": uuid,
        "sessionId": SESSION,
        "timestamp": "2026-10-01T10:05:00.000Z",
        "message": {"id": "synthetic-1", "model": "<synthetic>", "content": [TEXT], "usage": {}},
    }


def test_a_failed_request_is_an_event_against_the_sessions_model(homes):
    path = _claude_session(
        homes,
        [
            _assistant("msg_1", [TEXT], stop="end_turn"),
            _api_error(),
            _api_error(error="server_error", status=None, uuid="e2"),
        ],
    )

    events, _, _ = claude_code.parse(path, 0, {})

    ok, limited, dropped = events
    assert ok["success"] is True and "error" not in ok
    assert limited["success"] is False and limited["error"] == "rate_limit (429)"
    assert limited["model"] == "claude-opus-5-5"
    assert limited["input_tokens"] == 0 and limited["trace_id"] == SESSION
    assert limited["event_id"] == "cc-err-e1"
    assert dropped["error"] == "server_error"
    assert "never read" not in json.dumps(events)


def test_a_failure_before_any_call_has_no_model_to_blame(homes):
    path = _claude_session(homes, [_api_error()])

    assert claude_code.parse(path, 0, {})[0] == []


def test_the_model_is_remembered_across_syncs(homes):
    path = _claude_session(homes, [_assistant("msg_1", [TEXT], stop="end_turn")])
    _, offset, carry = claude_code.parse(path, 0, {})

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_api_error()) + "\n")
    events, _, _ = claude_code.parse(path, offset, json.loads(json.dumps(carry)))

    assert [event["error"] for event in events] == ["rate_limit (429)"]


def test_one_hour_cache_writes_cost_twice_the_input_rate(homes):
    _claude_session(homes, [_assistant("msg_1", [TEXT], stop="end_turn")])
    sender = Recorder()

    agent_sync.run_sync(_config(), sources=["claude-code"], sender=sender, fetch_pricing=False)

    # 10 uncached + 1000 read + 50 out, as before, but the 200 written tokens
    # are one-hour writes: 200 x 2 x $0.004/1k instead of 200 x $0.005/1k.
    event = sender.batches[0][0]
    assert event["metadata"]["cache_write_1h_tokens"] == 200
    assert round(event["cost"], 6) == round(0.00004 + 0.0002 + 0.001 + 0.0016, 6)


def test_codex_tools_are_named_as_its_hooks_name_them():
    """Shapes taken from a real rollout: code mode wraps each call in a script."""
    from agentcost.agents import codex

    wrapped_shell = {"type": "custom_tool_call", "name": "exec",
                     "input": 'const r = await tools.exec_command({cmd:"echo never-read"});'}
    wrapped_patch = {"type": "custom_tool_call", "name": "exec",
                     "input": 'const r = await tools.apply_patch("*** Begin Patch");'}
    both = {"type": "custom_tool_call", "name": "exec",
            "input": "await tools.apply_patch(p); await tools.exec_command({cmd})"}

    assert codex._tool_names(wrapped_shell) == ["Bash"]
    assert codex._tool_names(wrapped_patch) == ["apply_patch"]
    assert codex._tool_names(both) == ["apply_patch", "Bash"]
    # Older rollouts call the tool directly.
    assert codex._tool_names({"type": "function_call", "name": "shell"}) == ["Bash"]
    assert codex._tool_names({"type": "local_shell_call"}) == ["Bash"]
    assert codex._tool_names({"type": "function_call", "name": "wait"}) == ["wait"]
    assert codex._tool_names({"type": "custom_tool_call", "name": "exec", "input": "1 + 1"}) == ["exec"]
    assert codex._tool_names({"type": "custom_tool_call_output"}) == []


def test_codex_call_id_matches_what_the_otlp_receiver_derives():
    """Both paths hash session and counts, so one call sent both ways is one event."""
    import hashlib

    from agentcost.agents import codex

    session = "01a12f9e-5f24-7f40-884e-be95cfff30ae"
    expected = "cx-" + hashlib.sha256(f"{session}|13677|1792|34".encode()).hexdigest()[:40]

    assert codex.call_id(session, 13677, 1792, 34) == expected


def _policy(homes, rail):
    path = homes / "agentcost" / "policy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    policy = {"at": time.time(), "guardrails": {rail["agent_name"]: rail}, "tool_tags": {}, "budget": {}}
    path.write_text(json.dumps(policy), encoding="utf-8")


def _hook(event, path=None, **extra):
    hook = {"hook_event_name": event, "session_id": SESSION, **extra}
    if path:
        hook["transcript_path"] = str(path)
    return hook


def test_gate_refuses_a_model_outside_the_boundary(homes):
    path = _claude_session(homes, [_assistant("msg_1", [_tool("Read")])])
    _policy(homes, {"agent_name": "claude-code", "allowed_models": ["claude-sonnet-5"]})

    reason = gate.decide("claude-code", _hook("PreToolUse", path, tool_name="Read"))
    assert "claude-opus-5-5" in reason and "claude-sonnet-5" in reason
    assert gate.decide("claude-code", _hook("UserPromptSubmit", path))

    _policy(homes, {"agent_name": "claude-code", "allowed_models": ["claude-opus-5-5"]})
    assert gate.decide("claude-code", _hook("UserPromptSubmit", path)) is None


def test_gate_lets_the_first_prompt_through_when_no_model_is_known_yet(homes):
    path = _claude_session(homes, [])
    _policy(homes, {"agent_name": "claude-code", "allowed_models": ["claude-sonnet-5"]})

    assert gate.decide("claude-code", _hook("UserPromptSubmit", path)) is None


def test_gate_uses_the_model_codex_names_in_the_hook(homes):
    _policy(homes, {"agent_name": "codex", "allowed_models": ["gpt-6-luna"]})

    assert gate.decide("codex", _hook("UserPromptSubmit", model="gpt-6-luna")) is None
    assert "gpt-6-sol" in gate.decide("codex", _hook("UserPromptSubmit", model="gpt-6-sol"))
