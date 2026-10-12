"""Tests for the Claude Code and Codex collectors."""

import json
import os
import time

import pytest

from agentcost.agents import claude_code, codex, hooks
from agentcost.agents import sync as agent_sync
from agentcost.cli import main
from agentcost.cost_calculator import get_pricing_manager

SESSION = "0532f9c4-a022-4e98-a543-d8e17c5b90a6"


@pytest.fixture(autouse=True)
def homes(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("AGENTCOST_HOME", str(tmp_path / "agentcost"))
    monkeypatch.delenv("AGENTCOST_API_KEY", raising=False)
    monkeypatch.delenv("AGENTCOST_API_URL", raising=False)
    # Priced from memory, so no test reaches for the network.
    manager = get_pricing_manager()
    manager.update_pricing(
        {
            "claude-opus-5-5": {"input": 0.004, "output": 0.02, "cached_input": 0.0002, "cache_write": 0.005},
            "gpt-6-luna": {"input": 0.0001, "output": 0.0005, "cached_input": 0.00001},
        }
    )
    yield tmp_path
    manager.clear_cache()


def _write_lines(path, rows, complete=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row) for row in rows)
    path.write_text(text + ("\n" if complete else ""), encoding="utf-8")
    return path


def _append(path, rows):
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _assistant(message_id, blocks, stop="tool_use", model="claude-opus-5-5", **row):
    usage = {
        "input_tokens": 10,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 200,
        "output_tokens": 50,
        "cache_creation": {"ephemeral_1h_input_tokens": 200, "ephemeral_5m_input_tokens": 0},
    }
    return {
        "type": "assistant",
        "sessionId": SESSION,
        "timestamp": "2026-10-01T10:00:00.000Z",
        "cwd": "C:\\work\\billing-svc",
        "gitBranch": "main",
        "version": "2.1.300",
        "entrypoint": "cli",
        "message": {
            "id": message_id,
            "model": model,
            "stop_reason": stop,
            "usage": usage,
            "content": blocks,
        },
        **row,
    }


def _tool(name):
    return {"type": "tool_use", "name": name, "id": "toolu_1", "input": {"secret": "never read"}}


TEXT = {"type": "text", "text": "never read"}
TOOL_RESULT = {"type": "user", "message": {"content": "never read"}}


def _claude_session(tmp_path, rows, name=SESSION):
    return _write_lines(tmp_path / "claude" / "projects" / "C--work" / f"{name}.jsonl", rows)


# -- Claude Code -----------------------------------------------------------


def test_claude_one_event_per_api_message(homes):
    path = _claude_session(
        homes,
        [
            _assistant("msg_1", [TEXT]),
            _assistant("msg_1", [_tool("Read")]),
            TOOL_RESULT,
            _assistant("msg_1", [_tool("Grep")]),
            _assistant("msg_2", [TEXT], stop="end_turn"),
        ],
    )

    events, offset, _ = claude_code.parse(path, 0, {})

    assert [event["event_id"] for event in events] == ["cc-msg_1", "cc-msg_2"]
    assert offset == path.stat().st_size
    first = events[0]
    # The whole prompt is input; cache reads are a subset of it.
    assert first["input_tokens"] == 1010
    assert first["cached_tokens"] == 1000
    assert first["cache_write_tokens"] == 200
    assert first["output_tokens"] == 50
    assert first["trace_id"] == SESSION
    assert first["workflow"] == "claude-code"
    assert first["agent_name"] == "claude-code"
    assert first["tool_name"] == "Read"
    assert first["metadata"]["tools"] == ["Read", "Grep"]
    assert first["metadata"]["cache_write_1h_tokens"] == 200
    assert first["metadata"]["_ac_caps"] == {"tools": True}
    assert "tool_name" not in events[1]


def test_claude_never_carries_content(homes):
    path = _claude_session(homes, [_assistant("msg_1", [TEXT, _tool("Bash")], stop="end_turn")])

    events, _, _ = claude_code.parse(path, 0, {})

    assert "never read" not in json.dumps(events)


def test_claude_holds_a_call_that_may_still_be_growing(homes):
    path = _claude_session(homes, [_assistant("msg_1", [_tool("Read")])])

    events, offset, _ = claude_code.parse(path, 0, {})
    assert events == [] and offset == 0

    _append(path, [_assistant("msg_1", [_tool("Grep")]), _assistant("msg_2", [TEXT], stop="end_turn")])
    events, offset, _ = claude_code.parse(path, offset, {})

    assert [event["event_id"] for event in events] == ["cc-msg_1", "cc-msg_2"]
    assert events[0]["metadata"]["tools"] == ["Read", "Grep"]
    assert offset == path.stat().st_size


def test_claude_takes_an_abandoned_call_once_the_file_is_idle(homes):
    path = _claude_session(homes, [_assistant("msg_1", [_tool("Read")])])
    old = time.time() - claude_code.SETTLE_SECONDS - 5
    os.utime(path, (old, old))

    events, offset, _ = claude_code.parse(path, 0, {})

    assert [event["event_id"] for event in events] == ["cc-msg_1"]
    assert offset == path.stat().st_size


def test_claude_skips_synthetic_and_half_written_lines(homes):
    path = _claude_session(homes, [_assistant("msg_x", [TEXT], stop="end_turn", model="<synthetic>")])
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_assistant("msg_2", [TEXT], stop="end_turn"))[:-20])
    complete = path.stat().st_size - len(json.dumps(_assistant("msg_2", [TEXT], stop="end_turn"))[:-20])

    events, offset, _ = claude_code.parse(path, 0, {})

    assert events == []
    assert offset == complete


def test_claude_subagent_is_its_own_agent(homes):
    path = _write_lines(
        homes / "claude" / "projects" / "C--work" / SESSION / "subagents" / "agent-a1.jsonl",
        [_assistant("msg_s", [TEXT], stop="end_turn", attributionAgent="Plan", isSidechain=True)],
    )

    events, _, _ = claude_code.parse(path, 0, {})

    assert events[0]["agent_name"] == "claude-code/Plan"
    assert events[0]["step_name"] == "Plan"
    assert events[0]["depth"] == 1
    assert events[0]["trace_id"] == SESSION
    assert path in claude_code.session_files()


def test_claude_billing_from_account_summary(homes):
    assert claude_code.detect_billing() is None
    (homes / "claude").mkdir(exist_ok=True)
    account = {"oauthAccount": {"billingType": "stripe_subscription", "organizationType": "claude_pro"}}
    (homes / "claude" / ".claude.json").write_text(json.dumps(account), encoding="utf-8")
    assert claude_code.detect_billing() == "subscription"
    (homes / "claude" / ".claude.json").write_text("{}", encoding="utf-8")
    assert claude_code.detect_billing() == "api"


# -- Codex -----------------------------------------------------------------

USAGE = {
    "input_tokens": 14565,
    "cached_input_tokens": 11008,
    "cache_write_input_tokens": 0,
    "output_tokens": 103,
    "reasoning_output_tokens": 40,
    "total_tokens": 14668,
}


def _codex_row(kind, payload, timestamp="2026-10-01T10:00:00.000Z"):
    return {"timestamp": timestamp, "type": kind, "payload": payload}


def _token_count(total, plan="go", last=USAGE):
    return _codex_row(
        "event_msg",
        {
            "type": "token_count",
            "info": {"total_token_usage": {"total_tokens": total}, "last_token_usage": last},
            "rate_limits": {"plan_type": plan},
        },
    )


CODEX_HEAD = [
    _codex_row(
        "session_meta",
        {
            "session_id": SESSION,
            "cwd": "c:\\work\\PricingEngine",
            "cli_version": "0.155.0",
            "originator": "codex_vscode",
            "thread_source": "user",
            "git": {"branch": "master", "repository_url": "https://example.com/private/repo"},
        },
    ),
    _codex_row("turn_context", {"model": "gpt-6-luna", "cwd": "c:\\work\\PricingEngine"}),
]


def _codex_session(tmp_path, rows):
    return _write_lines(
        tmp_path / "codex" / "sessions" / "2026" / "10" / "01" / f"rollout-2026-10-01T10-00-00-{SESSION}.jsonl",
        rows,
    )


def test_codex_one_event_per_response(homes):
    path = _codex_session(
        homes,
        CODEX_HEAD
        + [
            _codex_row("response_item", {"type": "custom_tool_call", "name": "exec", "input": "never read"}),
            _codex_row("token_usage_record", {"session_id": SESSION, "response_id": "resp_1", "usage": USAGE}),
            _codex_row("response_item", {"type": "custom_tool_call_output", "output": "never read"}),
            _token_count(14668),
            _codex_row("token_usage_record", {"session_id": SESSION, "response_id": "resp_2", "usage": USAGE}),
            _token_count(29336),
        ],
    )

    events, offset, carry = codex.parse(path, 0, {})

    # The token_count beside each record is the same response, not another one.
    assert [event["event_id"] for event in events] == ["cx-resp_1", "cx-resp_2"]
    first = events[0]
    assert first["model"] == "gpt-6-luna"
    assert first["input_tokens"] == 14565
    assert first["cached_tokens"] == 11008
    assert first["output_tokens"] == 103
    assert first["tool_name"] == "exec"
    assert "tool_name" not in events[1]
    assert first["trace_id"] == SESSION
    assert first["agent_name"] == "codex"
    assert first["metadata"]["billing"] == "subscription"
    assert first["metadata"]["reasoning_tokens"] == 40
    assert "never read" not in json.dumps(events)
    assert offset == path.stat().st_size
    assert carry["model"] == "gpt-6-luna"


def test_codex_older_rollouts_use_token_count(homes):
    path = _codex_session(
        homes,
        CODEX_HEAD + [_token_count(14668, plan=None), _token_count(14668, plan=None), _token_count(29336, plan=None)],
    )

    events, _, _ = codex.parse(path, 0, {})

    assert len(events) == 2  # the repeated total is a repeat, not a response
    assert len({event["event_id"] for event in events}) == 2
    assert events[0]["metadata"]["billing"] == "api"


def test_codex_resumes_mid_file_with_what_it_learned(homes):
    path = _codex_session(homes, CODEX_HEAD)
    _, offset, carry = codex.parse(path, 0, {})
    carry = json.loads(json.dumps(carry))  # as it comes back from state.json

    _append(path, [_codex_row("token_usage_record", {"session_id": SESSION, "response_id": "resp_9", "usage": USAGE})])
    events, _, _ = codex.parse(path, offset, carry)

    assert events[0]["model"] == "gpt-6-luna"
    assert events[0]["trace_id"] == SESSION


# -- Sync ------------------------------------------------------------------


def _config(**overrides):
    config = {
        "api_key": "sk_test",
        "project_id": "proj",
        "base_url": "http://127.0.0.1:9",
        "sources": {"claude-code": {}, "codex": {}},
    }
    config.update(overrides)
    return config


class Recorder:
    def __init__(self, accept=True):
        self.accept = accept
        self.batches = []

    def __call__(self, batch):
        self.batches.append(batch)
        return self.accept

    @property
    def ids(self):
        return [event["event_id"] for batch in self.batches for event in batch]


def _both_sessions(homes):
    claude = _claude_session(homes, [_assistant("msg_1", [TEXT], stop="end_turn")])
    _codex_session(
        homes,
        CODEX_HEAD
        + [_codex_row("token_usage_record", {"session_id": SESSION, "response_id": "resp_1", "usage": USAGE})],
    )
    return claude


def test_sync_sends_once(homes):
    claude = _both_sessions(homes)
    sender = Recorder()

    result = agent_sync.run_sync(_config(), sender=sender, fetch_pricing=False)

    assert result["ok"] and result["sent"] == 2
    assert sorted(sender.ids) == ["cc-msg_1", "cx-resp_1"]
    assert result["sources"]["claude-code"]["calls"] == 1
    assert result["sources"]["claude-code"]["cost"] > 0

    again = Recorder()
    agent_sync.run_sync(_config(), sender=again, fetch_pricing=False)
    assert again.batches == []

    _append(claude, [_assistant("msg_2", [TEXT], stop="end_turn")])
    agent_sync.run_sync(_config(), sender=again, fetch_pricing=False)
    assert again.ids == ["cc-msg_2"]


def test_sync_keeps_what_was_refused(homes):
    _both_sessions(homes)

    refused = Recorder(accept=False)
    result = agent_sync.run_sync(_config(), sender=refused, fetch_pricing=False)
    assert not result["ok"] and result["sent"] == 0

    retry = Recorder()
    agent_sync.run_sync(_config(), sender=retry, fetch_pricing=False)
    assert sorted(retry.ids) == ["cc-msg_1", "cx-resp_1"]


def test_sync_dry_run_sends_and_remembers_nothing(homes):
    _both_sessions(homes)
    sender = Recorder()

    result = agent_sync.run_sync(_config(), dry_run=True, sender=sender, fetch_pricing=False)

    assert sender.batches == []
    assert result["sources"]["codex"]["calls"] == 1
    assert agent_sync.load_state()["files"] == {}


def test_sync_repo_is_opt_in_and_never_a_path(homes):
    _both_sessions(homes)

    private = Recorder()
    agent_sync.run_sync(_config(), dry_run=False, sender=private, fetch_pricing=False)
    for batch in private.batches:
        for event in batch:
            assert "repo" not in event["metadata"] and "branch" not in event["metadata"]
    assert "example.com" not in json.dumps(private.batches)


def test_sync_sends_repo_name_when_asked(homes):
    _both_sessions(homes)
    sender = Recorder()

    agent_sync.run_sync(_config(include_repo=True, user="alice"), sender=sender, fetch_pricing=False)

    by_id = {event["event_id"]: event["metadata"] for batch in sender.batches for event in batch}
    assert by_id["cc-msg_1"]["repo"] == "billing-svc"
    assert by_id["cc-msg_1"]["branch"] == "main"
    assert by_id["cx-resp_1"]["repo"] == "PricingEngine"
    assert by_id["cx-resp_1"]["user_id"] == "alice"
    assert "C:\\\\work" not in json.dumps(sender.batches)


def test_sync_ignores_history_before_the_backfill_window(homes):
    _both_sessions(homes)
    sender = Recorder()
    config = _config(sources={"claude-code": {"since": "2026-10-02T00:00:00Z"}, "codex": {}})

    agent_sync.run_sync(config, sender=sender, fetch_pricing=False)

    assert sender.ids == ["cx-resp_1"]


def test_sync_steps_aside_for_a_running_sync(homes):
    _both_sessions(homes)
    lock = homes / "agentcost" / "sync.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    sender = Recorder()

    result = agent_sync.run_sync(_config(), sender=sender, fetch_pricing=False)
    assert result["busy"] and sender.batches == []

    stale = time.time() - agent_sync.LOCK_STALE_SECONDS - 5
    os.utime(lock, (stale, stale))
    result = agent_sync.run_sync(_config(), sender=sender, fetch_pricing=False)
    assert not result["busy"] and len(sender.ids) == 2
    assert not lock.exists()


# -- Hooks -----------------------------------------------------------------


def test_hook_install_keeps_everything_else(homes):
    path = homes / "claude" / "settings.json"
    path.parent.mkdir(parents=True)
    theirs = {"hooks": [{"type": "command", "command": "echo mine"}]}
    original = {"theme": "dark", "hooks": {"Stop": [theirs]}}
    path.write_text(json.dumps(original), encoding="utf-8")

    hooks.install(path, claude_code.HOOK_EVENTS, "claude-code")
    hooks.install(path, claude_code.HOOK_EVENTS, "claude-code")  # twice is once

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["theme"] == "dark"
    assert data["hooks"]["Stop"][0] == theirs
    assert len(data["hooks"]["Stop"]) == 2 and len(data["hooks"]["SessionStart"]) == 1
    ours = data["hooks"]["Stop"][1]["hooks"][0]
    assert ours["async"] is True and ours["command"].endswith("--quiet --source claude-code")
    assert hooks.is_installed(path)
    assert json.loads((homes / "claude" / "settings.json.agentcost-backup").read_text()) == original

    assert hooks.remove(path)
    assert json.loads(path.read_text(encoding="utf-8")) == original
    assert not hooks.is_installed(path)
    assert not hooks.remove(path)


def test_hook_install_refuses_a_file_it_cannot_parse(homes):
    path = homes / "codex" / "hooks.json"
    path.parent.mkdir(parents=True)
    path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ValueError):
        hooks.install(path, codex.HOOK_EVENTS, "codex")
    assert path.read_text(encoding="utf-8") == "{ not json"


def test_hook_creates_the_file_when_there_is_none(homes):
    path = codex.hook_file()

    hooks.install(path, codex.HOOK_EVENTS, "codex")

    assert set(json.loads(path.read_text(encoding="utf-8"))["hooks"]) == {"Stop", "SessionStart"}


# -- CLI -------------------------------------------------------------------


def test_cli_quiet_sync_is_silent_when_nothing_is_connected(homes, capsys):
    assert main(["sync", "--quiet"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_cli_sync_says_when_nothing_is_connected(homes, capsys):
    assert main(["sync"]) == 2
    assert "agentcost connect" in capsys.readouterr().err


def test_cli_connect_needs_a_key(homes, capsys):
    assert main(["connect", "codex"]) == 2
    assert not codex.hook_file().exists()


def test_cli_connect_then_disconnect(homes, capsys, monkeypatch):
    _both_sessions(homes)
    sent = Recorder()
    monkeypatch.setattr(agent_sync, "_http_sender", lambda config: sent)
    monkeypatch.setattr(agent_sync, "load_pricing", lambda base_url, fetch: None)

    code = main(["connect", "codex", "--api-key", "sk_test", "--project-id", "proj", "--backfill-days", "36500"])

    assert code == 0
    assert sent.ids == ["cx-resp_1"]  # only the agent that was connected
    assert hooks.is_installed(codex.hook_file())
    assert "/hooks" in capsys.readouterr().out
    assert agent_sync.load_config()["sources"].keys() == {"codex"}

    assert main(["status"]) == 0
    assert "hook installed, 0 call(s) waiting" in capsys.readouterr().out

    assert main(["disconnect", "codex"]) == 0
    assert not hooks.is_installed(codex.hook_file())
    assert agent_sync.load_config()["sources"] == {}
