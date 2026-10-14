"""
Reads what the connected coding agents wrote since the last sync and sends it
to AgentCost.

Each session log has a byte cursor in ``~/.agentcost/state.json``. A cursor
only moves once the events before it were accepted, and every event carries an
id the server stores once, so a sync that is interrupted, repeated or run from
two sessions at the same moment neither loses a call nor counts one twice.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import MAX_SERVER_BATCH_SIZE, AgentCostConfig, set_config
from ..cost_calculator import calculate_cost, get_pricing_manager
from . import SOURCES

DEFAULT_BASE_URL = "https://api.agentcost.tech"

# A sync still holding the lock after this long has died.
LOCK_STALE_SECONDS = 300

# The server allows 100 requests a minute per key; a long backfill stays under.
PACED_ABOVE_BATCHES = 60
PACE_SECONDS = 0.7

# How long the gate's copy of the policy is trusted before a sync refetches it.
POLICY_MAX_AGE_SECONDS = 60

# How long a finished session's running totals are kept for the gate.
RUN_STATE_DAYS = 7

Sender = Callable[[List[Dict[str, Any]]], bool]


def home() -> Path:
    return Path(os.environ.get("AGENTCOST_HOME") or Path.home() / ".agentcost")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)  # holds the API key; a no-op on Windows
    except OSError:
        pass
    os.replace(temporary, path)


def load_config() -> Dict[str, Any]:
    config = read_json(home() / "config.json")
    config.setdefault("sources", {})
    config["api_key"] = os.environ.get("AGENTCOST_API_KEY") or config.get("api_key")
    config["base_url"] = (
        os.environ.get("AGENTCOST_API_URL") or config.get("base_url") or DEFAULT_BASE_URL
    )
    return config


def save_config(config: Dict[str, Any]) -> None:
    write_json(home() / "config.json", config)


def load_state() -> Dict[str, Any]:
    state = read_json(home() / "state.json")
    state.setdefault("files", {})
    return state


def since_days(days: int) -> str:
    moment = datetime.now(timezone.utc) - timedelta(days=days)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_pricing(base_url: str, fetch: bool) -> None:
    """Price estimates from the backend's catalogue, cached on disk for a day.

    The SDK's own fetch runs in the background with a short timeout, which
    suits a long-lived app but not a command that exits in a second -- and the
    hosted API can take longer than that to wake. A hook never waits for it:
    without a cache it sends estimates of zero, and the server reprices anyway.
    """
    path = home() / "pricing.json"
    cached = read_json(path)
    fresh = cached.get("base_url") == base_url and time.time() - cached.get("at", 0) < 86400
    if not fresh and fetch:
        import requests

        try:
            response = requests.get(f"{base_url.rstrip('/')}/v1/pricing", timeout=90)
            pricing = response.json().get("pricing") if response.ok else None
        except (requests.RequestException, ValueError):
            pricing = None
        if pricing:
            cached = {"base_url": base_url, "at": time.time(), "pricing": pricing}
            write_json(path, cached)
    if cached.get("pricing"):
        get_pricing_manager().update_pricing(cached["pricing"])


class _Lock:
    """One sync at a time: every open session fires the hook on its own."""

    def __init__(self) -> None:
        self.path = home() / "sync.lock"
        self.held = False

    def __enter__(self) -> "_Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                self.held = True
                break
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime < LOCK_STALE_SECONDS:
                        break
                    self.path.unlink()
                except OSError:
                    break
        return self

    def __exit__(self, *_: Any) -> None:
        if self.held:
            try:
                self.path.unlink()
            except OSError:
                pass


def _finalize(event: Dict[str, Any], config: Dict[str, Any], billing: Optional[str]) -> None:
    metadata = event["metadata"]
    metadata["billing"] = metadata.get("billing") or billing or "unknown"
    if not config.get("include_repo"):
        metadata.pop("repo", None)
        metadata.pop("branch", None)
    if config.get("user"):
        metadata["user_id"] = config["user"]
    event["metadata"] = {key: value for key, value in metadata.items() if value is not None}
    # An estimate only: the server reprices every event from its catalogue.
    event["cost"] = price(event)


def price(event: Dict[str, Any]) -> float:
    written = event.get("cache_write_tokens", 0)
    # One-hour cache writes cost twice the input rate, not the catalogue's
    # five-minute write rate. The server prices them the same way.
    one_hour = min(event["metadata"].get("cache_write_1h_tokens") or 0, written)
    return calculate_cost(
        event["model"],
        event["input_tokens"],
        event["output_tokens"],
        cached_tokens=event.get("cached_tokens", 0),
        cache_write_tokens=written - one_hour,
    ) + calculate_cost(event["model"], 2 * one_hour, 0)


def prepare_pricing(config: Dict[str, Any], fetch: bool) -> None:
    # The calculator only consults the backend's catalogue once it has a config.
    set_config(
        AgentCostConfig(
            api_key=config.get("api_key") or "",
            project_id=config.get("project_id") or "",
            base_url=config["base_url"],
        )
    )
    load_pricing(config["base_url"], fetch=fetch)


def refresh_policy(config: Dict[str, Any], force: bool = False) -> Optional[Dict[str, Any]]:
    """Fetch what the gate enforces and keep it where the gate can read it.

    The gate runs before every tool call and never touches the network; it
    decides from this file. The sync after each turn keeps it current, so a
    guardrail changed in the dashboard applies from the next turn. If the
    backend cannot be reached the previous policy stays in force.
    """
    path = home() / "policy.json"
    held = read_json(path)
    if held and not force and time.time() - held.get("at", 0) < POLICY_MAX_AGE_SECONDS:
        return held

    import requests

    try:
        response = requests.get(
            f"{config['base_url'].rstrip('/')}/v1/guardrails/policy",
            headers={"Authorization": f"Bearer {config['api_key']}"},
            timeout=90 if force else 10,
        )
        served = response.json() if response.ok else None
    except (requests.RequestException, ValueError):
        served = None
    if not isinstance(served, dict):
        return held or None

    policy = {
        "at": time.time(),
        "guardrails": {rail["agent_name"]: rail for rail in served.get("guardrails") or []},
        "tool_tags": served.get("tool_tags") or {},
        "budget": served.get("budget") or {},
    }
    write_json(path, policy)
    return policy


def _prune_runs() -> None:
    """Drop the gate's per-session totals for sessions long finished."""
    cutoff = time.time() - RUN_STATE_DAYS * 86400
    for path in (home() / "runs").glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def _new_session_names(
    names: Sequence[str], state: Dict[str, Any]
) -> "tuple[List[Dict[str, Any]], Dict[str, Any]]":
    """Names not yet sent, and what to remember once they have been.

    Kept apart from the event cursors: a chat is usually titled after its first
    calls were sent, and a chat that finished before names were switched on has
    nothing new to send but its name.
    """
    seen = state.get("names") or {}
    records: List[Dict[str, Any]] = []
    remember: Dict[str, Any] = {}
    for name in names:
        source = SOURCES[name]
        if not hasattr(source, "session_name"):
            continue
        for path in source.session_files():
            key = str(path)
            known = seen.get(key) or {}
            try:
                modified = path.stat().st_mtime_ns
                if known.get("mtime") == modified:
                    continue
                found = source.session_name(path)
            except OSError:
                continue
            title = found[1] if found else known.get("name")
            remember[key] = {"mtime": modified, "name": title}
            if found and found[1] != known.get("name"):
                records.append(
                    {"record_type": "session", "session_id": found[0], "name": found[1]}
                )
    return records, remember


def _summarize(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    models: Dict[str, Dict[str, Any]] = {}
    for event in events:
        row = models.setdefault(event["model"], {"calls": 0, "tokens": 0, "cost": 0.0})
        row["calls"] += 1
        row["tokens"] += event["total_tokens"]
        row["cost"] += event["cost"]
    return {
        "calls": len(events),
        "sessions": len({event["trace_id"] for event in events}),
        "tokens": sum(event["total_tokens"] for event in events),
        "cost": sum(event["cost"] for event in events),
        "billing": sorted({event["metadata"]["billing"] for event in events}),
        "models": models,
    }


def run_sync(
    config: Dict[str, Any],
    sources: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    sender: Optional[Sender] = None,
    fetch_pricing: bool = True,
) -> Dict[str, Any]:
    """Sync the given sources (default: the connected ones).

    Returns ``{"ok", "busy", "sources": {name: summary}}``. A dry run reads and
    prices everything but sends nothing and moves no cursor.
    """
    names = list(sources) if sources else list(config["sources"])
    result: Dict[str, Any] = {"ok": True, "busy": False, "sources": {}}

    prepare_pricing(config, fetch=fetch_pricing)

    with _Lock() as lock:
        if not lock.held and not dry_run:
            result["busy"] = True
            return result

        state = load_state()
        events: List[Dict[str, Any]] = []
        # (events collected so far, state key, the cursor to store once sent)
        marks: List[Any] = []
        by_source: Dict[str, List[Dict[str, Any]]] = {name: [] for name in names}

        for name in names:
            source = SOURCES[name]
            settings = config["sources"].get(name) or {}
            since = settings.get("since") or ""
            billing = settings.get("billing") or source.detect_billing()

            for path in source.session_files():
                key = str(path)
                entry = state["files"].get(key) or {}
                try:
                    # Bytes for a log that only grows; a source whose file does
                    # not (SQLite) says for itself what marks a change.
                    size = source.version(path) if hasattr(source, "version") else path.stat().st_size
                    offset = entry.get("offset", 0)
                    carry = entry.get("carry") or {}
                    if offset > size:  # replaced by a shorter file: start over
                        offset, carry = 0, {}
                    if offset == size:
                        continue
                    found, offset, carry = source.parse(path, offset, carry)
                except OSError:
                    continue  # deleted or locked mid-read; the next sync retries

                # Both sides are UTC ISO 8601, which sorts as text.
                found = [e for e in found if e["timestamp"] and e["timestamp"] >= since]
                for event in found:
                    _finalize(event, config, billing)
                events += found
                by_source[name] += found
                marks.append((len(events), key, {"offset": offset, "carry": carry}))

        sent = 0
        if not dry_run:
            send = sender or _http_sender(config)
            batches = [
                events[index : index + MAX_SERVER_BATCH_SIZE]
                for index in range(0, len(events), MAX_SERVER_BATCH_SIZE)
            ]
            for batch in batches:
                if not send(batch):
                    result["ok"] = False
                    break
                sent += len(batch)
                if len(batches) > PACED_ABOVE_BATCHES:
                    time.sleep(PACE_SECONDS)

            for collected, key, entry in marks:
                if collected <= sent:
                    state["files"][key] = entry
            if config.get("session_names") and result["ok"]:
                records, remember = _new_session_names(names, state)
                batches = [
                    records[index : index + MAX_SERVER_BATCH_SIZE]
                    for index in range(0, len(records), MAX_SERVER_BATCH_SIZE)
                ]
                if all(send(batch) for batch in batches):
                    state.setdefault("names", {}).update(remember)
                    result["named"] = len(records)

            state["last_sync"] = datetime.now(timezone.utc).isoformat()
            write_json(home() / "state.json", state)

            if any((config["sources"].get(name) or {}).get("enforce") for name in names):
                refresh_policy(config)
                _prune_runs()

        for name in names:
            result["sources"][name] = _summarize(by_source[name])
        result["sent"] = sent
        return result


def _http_sender(config: Dict[str, Any]) -> Sender:
    from ..http_client import AgentCostHTTPClient

    client = AgentCostHTTPClient(api_key=config["api_key"], base_url=config["base_url"])
    return lambda batch: client.send_events(config["project_id"], batch)
