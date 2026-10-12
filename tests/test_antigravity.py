"""Tests for the Antigravity collector, against databases built to its layout."""
# ruff: noqa: F811  (the imported `homes` fixture is requested by name below)

import os
import sqlite3
import time

import pytest

from agentcost.agents import antigravity
from agentcost.agents import sync as agent_sync
from agentcost.cli import main

from .test_agents import Recorder, _config, homes  # noqa: F401

CONVERSATION = "9f4752e8-f7ba-46ee-8009-a61e75031214"
STARTED = 1783578283  # 2026-07-09T06:24:43Z


def _varint(value):
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _int(field, value):
    return _varint(field << 3) + _varint(value)


def _bytes(field, value):
    value = value.encode() if isinstance(value, str) else value
    return _varint(field << 3 | 2) + _varint(len(value)) + value


def _call(model="claude-opus-4-6-thinking", uncached=3961, output=394, cache_read=66495,
          request_id="req_vrtx_011CcrCdYkdwWep3MopeazHD", step=1, thinking=None):
    usage = _int(1, 1026) + _int(2, uncached) + _int(3, output)
    if cache_read:
        usage += _int(5, cache_read)
    if thinking:
        usage += _int(9, thinking)
    if request_id:
        usage += _bytes(11, request_id)
    call = _int(3, 1026) + _bytes(4, usage) + _bytes(19, model)
    call += _bytes(20, _bytes(1, "last_step_index") + _bytes(2, str(step)))
    # Fields the collector does not know sit around the ones it does.
    return _bytes(2, b"\x10") + _bytes(4, "76669465-4c60") + _bytes(1, call) + _int(10, 1)


@pytest.fixture
def conversation(homes, monkeypatch):
    monkeypatch.setenv("AGENTCOST_ANTIGRAVITY_HOME", str(homes / "gemini"))
    path = homes / "gemini" / "antigravity-ide" / "conversations" / f"{CONVERSATION}.db"
    path.parent.mkdir(parents=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE gen_metadata (idx integer PRIMARY KEY, data blob, size integer)")
    db.execute("CREATE TABLE steps (idx integer, metadata blob)")
    for index in range(3):
        stamp = _bytes(1, _int(1, STARTED + index * 60) + _int(2, 725183600))
        db.execute("INSERT INTO steps VALUES (?, ?)", (index, stamp))
    db.commit()
    db.close()
    return path


def _add(path, *calls, idle=True):
    db = sqlite3.connect(path)
    start = db.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM gen_metadata").fetchone()[0]
    for offset, data in enumerate(calls):
        db.execute("INSERT INTO gen_metadata VALUES (?, ?, ?)", (start + offset, data, len(data)))
    db.commit()
    db.close()
    if idle:
        old = time.time() - antigravity.SETTLE_SECONDS - 5
        os.utime(path, (old, old))


def test_reads_usage_out_of_the_undocumented_record(conversation):
    _add(conversation, _call(), _call(model="gemini-pro-default", cache_read=0, thinking=610,
                                      request_id=None, step=2))

    events, _, carry = antigravity.parse(conversation, 0, {})

    first, second = events
    assert first["model"] == "claude-opus-4-6-thinking"
    # Cache reads are reported beside the input, and counted inside it here.
    assert first["input_tokens"] == 3961 + 66495
    assert first["cached_tokens"] == 66495
    assert first["output_tokens"] == 394
    assert first["event_id"] == "ag-req_vrtx_011CcrCdYkdwWep3MopeazHD"
    assert first["trace_id"] == CONVERSATION and first["workflow"] == "antigravity"
    assert first["timestamp"].startswith("2026-07-09T06:25:43")
    assert first["metadata"]["estimated"] is True
    assert first["metadata"]["billing"] == "subscription"
    assert first["metadata"]["surface"] == "antigravity-ide"
    assert second["event_id"] == f"ag-{CONVERSATION}:1"
    assert second["metadata"]["reasoning_tokens"] == 610
    assert "cached_tokens" not in second
    assert carry["idx"] == 1


def test_only_new_calls_are_read_and_the_newest_waits(conversation):
    _add(conversation, _call(request_id="req_a"), _call(request_id="req_b"), idle=False)

    events, version, carry = antigravity.parse(conversation, 0, {})
    assert [event["event_id"] for event in events] == ["ag-req_a"]

    _add(conversation, _call(request_id="req_c"))
    events, _, carry = antigravity.parse(conversation, version, carry)
    assert [event["event_id"] for event in events] == ["ag-req_b", "ag-req_c"]

    assert antigravity.parse(conversation, version, carry)[0] == []


def test_a_record_it_cannot_read_is_skipped_not_fatal(conversation):
    _add(conversation, b"\xff\xff\xff", _bytes(1, _bytes(19, "no-usage")), _call(request_id="req_ok"))

    events, _, carry = antigravity.parse(conversation, 0, {})

    assert [event["event_id"] for event in events] == ["ag-req_ok"]
    assert carry["idx"] == 2


def test_a_file_that_is_not_the_expected_database_yields_nothing(conversation):
    other = conversation.with_name("not-a-conversation.db")
    other.write_bytes(b"this is not sqlite")

    assert antigravity.parse(other, 0, {}) == ([], 0, {})


def test_sync_notices_a_change_the_file_size_does_not_show(conversation):
    _add(conversation, _call(request_id="req_a"))
    config = _config(sources={"antigravity": {}})
    sender = Recorder()
    agent_sync.run_sync(config, sender=sender, fetch_pricing=False)
    assert sender.ids == ["ag-req_a"]

    size = conversation.stat().st_size
    _add(conversation, _call(request_id="req_b"))
    assert conversation.stat().st_size == size  # the new row fitted in a free page

    agent_sync.run_sync(config, sender=sender, fetch_pricing=False)
    assert sender.ids == ["ag-req_a", "ag-req_b"]


def test_connect_installs_no_hook_and_says_how_it_syncs(conversation, capsys, monkeypatch):
    _add(conversation, _call())
    sent = Recorder()
    monkeypatch.setattr(agent_sync, "_http_sender", lambda config: sent)
    monkeypatch.setattr(agent_sync, "load_pricing", lambda base_url, fetch: None)

    code = main(
        ["connect", "antigravity", "--api-key", "sk_test", "--project-id", "proj",
         "--backfill-days", "36500", "--enforce"]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert len(sent.ids) == 1
    assert "no hook to trigger a sync" in captured.out
    assert "tracking only" in captured.err
    assert agent_sync.load_config()["sources"]["antigravity"]["enforce"] is False
    assert main(["status"]) == 0 and "Antigravity: no hook" in capsys.readouterr().out
    assert main(["disconnect", "antigravity"]) == 0
