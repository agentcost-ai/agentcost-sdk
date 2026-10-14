"""Session names: sent only when asked for, once per title, for past chats too."""
# ruff: noqa: F811  (the imported `homes` fixture is requested by name below)

import json
import os

from agentcost.agents import claude_code, codex
from agentcost.agents import sync as agent_sync
from agentcost.cli import main

from .test_agents import (  # noqa: F401
    CODEX_HEAD,
    SESSION,
    TEXT,
    USAGE,
    Recorder,
    _append,
    _assistant,
    _claude_session,
    _codex_row,
    _codex_session,
    _config,
    homes,
)


def _title(text):
    return {"type": "ai-title", "aiTitle": text, "sessionId": SESSION}


def _names(sender):
    return [
        (record["session_id"], record["name"])
        for batch in sender.batches
        for record in batch
        if record.get("record_type") == "session"
    ]


def _titled_session(homes):
    return _claude_session(
        homes,
        [_title("Fix the billing export"), _assistant("msg_1", [TEXT], stop="end_turn")],
    )


def test_claude_reads_the_latest_title(homes):
    path = _titled_session(homes)
    _append(path, [_title("Fix the billing export and its tests")])

    assert claude_code.session_name(path) == (SESSION, "Fix the billing export and its tests")


def test_claude_untitled_chats_and_subagents_have_no_name(homes):
    untitled = _claude_session(homes, [_assistant("msg_1", [TEXT], stop="end_turn")], name="plain")
    subagent = homes / "claude" / "projects" / "C--work" / SESSION / "subagents" / "agent-a1.jsonl"
    subagent.parent.mkdir(parents=True)
    subagent.write_text(json.dumps(_title("Not a chat")) + "\n", encoding="utf-8")

    assert claude_code.session_name(untitled) is None
    assert claude_code.session_name(subagent) is None


def test_codex_reads_the_thread_name_from_its_index(homes):
    path = _codex_session(homes, CODEX_HEAD)
    index = homes / "codex" / "session_index.jsonl"
    index.write_text(
        json.dumps({"id": SESSION, "thread_name": "Raise the limit"})
        + "\nnot json\n"
        + json.dumps({"id": SESSION, "thread_name": "Raise the generation limit"})
        + "\n"
        + json.dumps({"id": "someone-else", "thread_name": "Other"})
        + "\n",
        encoding="utf-8",
    )

    assert codex.session_name(path) == (SESSION, "Raise the generation limit")
    index.unlink()
    assert codex.session_name(path) is None


def test_names_are_not_sent_unless_asked_for(homes):
    _titled_session(homes)
    sender = Recorder()

    agent_sync.run_sync(_config(), sources=["claude-code"], sender=sender, fetch_pricing=False)

    assert _names(sender) == []
    assert "Fix the billing export" not in json.dumps(sender.batches)


def test_a_name_is_sent_once_and_again_only_when_it_changes(homes):
    path = _titled_session(homes)
    config = _config(session_names=True)
    sender = Recorder()

    result = agent_sync.run_sync(config, sources=["claude-code"], sender=sender, fetch_pricing=False)
    assert result["named"] == 1
    assert _names(sender) == [(SESSION, "Fix the billing export")]

    agent_sync.run_sync(config, sources=["claude-code"], sender=sender, fetch_pricing=False)
    assert len(_names(sender)) == 1  # untouched file: not even reopened

    _append(path, [_assistant("msg_2", [TEXT], stop="end_turn")])
    agent_sync.run_sync(config, sources=["claude-code"], sender=sender, fetch_pricing=False)
    assert len(_names(sender)) == 1  # the chat moved on, its title did not

    _append(path, [_title("Fix the billing export and its tests")])
    agent_sync.run_sync(config, sources=["claude-code"], sender=sender, fetch_pricing=False)
    assert _names(sender)[-1] == (SESSION, "Fix the billing export and its tests")
    assert len(_names(sender)) == 2


def test_chats_already_synced_get_their_names_when_the_option_is_turned_on(homes):
    _titled_session(homes)
    agent_sync.run_sync(_config(), sources=["claude-code"], sender=Recorder(), fetch_pricing=False)

    sender = Recorder()
    agent_sync.run_sync(
        _config(session_names=True), sources=["claude-code"], sender=sender, fetch_pricing=False
    )

    # No new calls to send, only the name of a chat that was sent before.
    assert _names(sender) == [(SESSION, "Fix the billing export")]
    assert all(record.get("record_type") == "session" for batch in sender.batches for record in batch)


def test_a_refused_name_is_offered_again(homes):
    _titled_session(homes)
    config = _config(session_names=True)

    class NamesRefused(Recorder):
        def __call__(self, batch):
            super().__call__(batch)
            return not any(record.get("record_type") == "session" for record in batch)

    agent_sync.run_sync(config, sources=["claude-code"], sender=NamesRefused(), fetch_pricing=False)

    sender = Recorder()
    agent_sync.run_sync(config, sources=["claude-code"], sender=sender, fetch_pricing=False)
    assert _names(sender) == [(SESSION, "Fix the billing export")]


def test_names_go_to_the_backend_as_their_own_list(homes, monkeypatch):
    from agentcost.http_client import AgentCostHTTPClient

    posted = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"status": "ok"}

    client = AgentCostHTTPClient(api_key="sk_test", base_url="http://127.0.0.1:9")
    monkeypatch.setattr(
        client.session, "post", lambda url, json, headers, timeout: posted.update(json) or Response()
    )

    assert client.send_events(
        "proj",
        [
            {"model": "m", "input_tokens": 1, "output_tokens": 1},
            {"record_type": "session", "session_id": SESSION, "name": "Fix the billing export"},
        ],
    )

    assert posted["sessions"] == [{"session_id": SESSION, "name": "Fix the billing export"}]
    assert posted["events"] == [{"model": "m", "input_tokens": 1, "output_tokens": 1}]
    assert "outcomes" not in posted


def test_connect_remembers_the_choice(homes, capsys, monkeypatch):
    _titled_session(homes)
    sent = Recorder()
    monkeypatch.setattr(agent_sync, "_http_sender", lambda config: sent)
    monkeypatch.setattr(agent_sync, "load_pricing", lambda base_url, fetch: None)

    code = main(
        ["connect", "claude-code", "--api-key", "sk_test", "--project-id", "proj",
         "--backfill-days", "36500", "--session-names", "--no-hook"]
    )

    assert code == 0
    assert "Named 1 session(s)." in capsys.readouterr().out
    assert agent_sync.load_config()["session_names"] is True
    assert os.path.exists(homes / "agentcost" / "state.json")
