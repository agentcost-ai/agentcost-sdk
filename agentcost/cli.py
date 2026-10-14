"""
AgentCost command line interface.

    agentcost analyze ./agent --model gpt-4o --runs-per-day 2000
    agentcost analyze --events run.json --json report.json

    agentcost connect claude-code --api-key sk_...
    agentcost sync --dry-run
    agentcost status

``analyze`` runs entirely locally. ``connect`` and ``sync`` send usage from
Claude Code and Codex sessions on this machine to AgentCost.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from . import __version__
from .agents import SOURCES
from .agents import hooks as agent_hooks
from .agents import sync as agent_sync
from .analyzer import DEFAULT_PATTERNS, analyze, format_report, load_events


# What `agentcost` prints on its own: every command, grouped by what it is for,
# with the flags people reach for first. `--help` on a command has the rest.
OVERVIEW = """\
AgentCost {version}

TRACK A CODING AGENT ON THIS MACHINE          <agent>: {agents}

  agentcost sync --dry-run          See what would be sent. Needs no key; sends nothing.
  agentcost connect <agent>         Start tracking: imports 30 days, then syncs after every turn.
      --api-key sk_...                Project API key (first time only, or set AGENTCOST_API_KEY)
      --enforce                       Also block calls that break the project's guardrails or budget
      --user NAME                     Attribute this machine's usage to a developer
      --include-repo                  Also send the working folder's name and git branch
      --session-names                 Also send each chat's title, to list sessions by name
      --backfill-days N               How much history to import (default 30)
      --base-url URL                  A self-hosted AgentCost
  agentcost status                  What is connected, whether hooks are in place, what is waiting
  agentcost sync                    Send what is new right now (the hook does this for you)
  agentcost disconnect <agent>      Stop tracking. What was already sent stays.

  Only token counts, model, tool names and ids are read. Never prompts or replies.

ESTIMATE AN AGENT BEFORE YOU DEPLOY IT        runs offline, sends nothing

  agentcost analyze ./agent --model gpt-4o        What the prompt and skill files cost per call
      --events run.json --runs-per-day 2000         Cost per run and per month, from a recorded run
      --fail-on high                                Exit non-zero on a finding; for CI
      --json report.json                            Also write the full report

TRACK YOUR OWN PYTHON APP

  from agentcost import track_costs
  track_costs.init(api_key="sk_...", project_id="...")

  agentcost <command> --help        Every flag for one command
  Docs                              https://agentcost.tech/docs
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentcost",
        description="Estimate what an agent will cost, and track what coding agents spend.",
        epilog="Run `agentcost` with no arguments for an overview of every command.",
    )
    parser.add_argument("--version", action="version", version=f"agentcost {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyse prompt/skill files and a recorded test run.",
    )
    analyze_parser.add_argument(
        "path",
        nargs="?",
        help="Directory or file of prompts and skill files.",
    )
    analyze_parser.add_argument(
        "--events",
        help="Events from a local-mode run (JSON array or JSONL).",
    )
    analyze_parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to price against (default: gpt-4o).",
    )
    analyze_parser.add_argument(
        "--runs-per-day",
        type=int,
        help="Expected production volume, to project a monthly cost.",
    )
    analyze_parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help="Glob to include; repeatable. Defaults cover prompt and doc files.",
    )
    analyze_parser.add_argument(
        "--json",
        dest="json_out",
        help="Also write the full report as JSON to this path.",
    )
    analyze_parser.add_argument(
        "--fail-on",
        choices=["high", "medium", "low"],
        help="Exit non-zero if a finding at or above this severity is present.",
    )

    sources = sorted(SOURCES)

    connect_parser = subparsers.add_parser(
        "connect",
        help="Track a coding agent's sessions on this machine.",
    )
    connect_parser.add_argument("source", choices=sources)
    connect_parser.add_argument("--api-key", help="Project API key (or AGENTCOST_API_KEY).")
    connect_parser.add_argument("--project-id", help="Looked up from the key when omitted.")
    connect_parser.add_argument("--base-url", help="Backend URL, for a self-hosted AgentCost.")
    connect_parser.add_argument(
        "--backfill-days",
        type=int,
        default=30,
        help="How much existing history to import (default: 30).",
    )
    connect_parser.add_argument(
        "--include-repo",
        action="store_true",
        help="Also send the working folder's name and git branch.",
    )
    connect_parser.add_argument(
        "--session-names",
        action="store_true",
        help="Also send each chat's title, so sessions are listed by name. "
        "A title summarises what was asked.",
    )
    connect_parser.add_argument("--user", help="Attribute this machine's usage to a user id.")
    connect_parser.add_argument(
        "--billing",
        choices=["api", "subscription"],
        help="Override how this agent is paid for (detected when omitted).",
    )
    connect_parser.add_argument(
        "--enforce",
        action="store_true",
        help="Also refuse tool calls and prompts that break the project's "
        "guardrails or budget, before they run.",
    )
    connect_parser.add_argument(
        "--no-hook",
        action="store_true",
        help="Do not install the hook; run `agentcost sync` yourself.",
    )

    sync_parser = subparsers.add_parser("sync", help="Send new usage to AgentCost.")
    sync_parser.add_argument("--source", choices=sources, help="Default: every connected agent.")
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent. Sends nothing; needs no API key.",
    )
    sync_parser.add_argument("--quiet", action="store_true", help="No output; for hooks.")

    subparsers.add_parser("status", help="Show connected agents and what is waiting.")

    disconnect_parser = subparsers.add_parser("disconnect", help="Stop tracking a coding agent.")
    disconnect_parser.add_argument("source", choices=sources)

    return parser


def _print_sync(result: dict, dry_run: bool) -> None:
    verb = "would send" if dry_run else "sent"
    for name, summary in result["sources"].items():
        label = SOURCES[name].LABEL
        if not summary["calls"]:
            print(f"{label}: nothing new")
            continue
        print(
            f"{label}: {verb} {summary['calls']:,} calls from {summary['sessions']:,} "
            f"session(s), {summary['tokens']:,} tokens, ${summary['cost']:,.2f} at API rates"
        )
        ranked = sorted(summary["models"].items(), key=lambda item: -item[1]["cost"])
        for model, row in ranked:
            print(f"    ${row['cost']:>10,.2f}  {row['calls']:>6,} calls  {model}")
        if "subscription" in summary["billing"]:
            print("    Covered by a subscription: this is what the usage is worth, not a bill.")
    if result.get("named"):
        print(f"Named {result['named']:,} session(s).")
    if not dry_run and not result["ok"]:
        print(
            "agentcost: the backend did not accept everything; the rest is kept "
            "and goes with the next sync.",
            file=sys.stderr,
        )


def _resolve_project(config: dict) -> Optional[str]:
    """The project a key belongs to. Doubles as the check that the key works."""
    import requests

    try:
        response = requests.get(
            f"{config['base_url'].rstrip('/')}/v1/projects/me",
            headers={"Authorization": f"Bearer {config['api_key']}"},
            timeout=90,  # the hosted API can take a minute to wake from idle
        )
        return response.json().get("id") if response.ok else None
    except (requests.RequestException, ValueError):
        return None


def _print_enforcement(name: str, policy: Optional[dict]) -> None:
    source = SOURCES[name]
    if policy is None:
        print(
            "agentcost: this backend did not return a policy, so nothing is "
            "enforced yet. It is fetched again on every sync.",
            file=sys.stderr,
        )
        return
    rail = policy["guardrails"].get(name)
    limits = []
    if rail:
        if rail.get("allowed_tools") is not None and source.GATES_TOOLS:
            limits.append(f"only these tools: {', '.join(rail['allowed_tools']) or 'none'}")
        if rail.get("read_only") and source.GATES_TOOLS:
            limits.append("no tool tagged as writing")
        if rail.get("max_tool_calls_per_run"):
            limits.append(f"at most {rail['max_tool_calls_per_run']} tool calls per session")
        if rail.get("max_cost_per_run_usd"):
            limits.append(f"at most ${rail['max_cost_per_run_usd']:.2f} per session")
    if policy["budget"].get("mode") == "hard_cap":
        limits.append("no new prompts once the monthly budget is spent (metered usage only)")
    if limits:
        print(f"Enforcing for {source.LABEL}: " + "; ".join(limits) + ".")
    else:
        print(
            f"Enforcement is on, but the project declares no guardrail for the agent "
            f"'{name}' and no hard budget cap. Declare one in the dashboard and it "
            "applies from the next turn."
        )


def _connect(args) -> int:
    source = SOURCES[args.source]
    config = agent_sync.load_config()
    config["api_key"] = args.api_key or config.get("api_key")
    config["base_url"] = args.base_url or config["base_url"]
    if not config["api_key"]:
        print("agentcost: give --api-key, or set AGENTCOST_API_KEY", file=sys.stderr)
        return 2

    project_id = args.project_id or _resolve_project(config)
    if not project_id:
        print(
            f"agentcost: {config['base_url']} did not accept that API key. "
            "Nothing was changed.",
            file=sys.stderr,
        )
        return 2
    config["project_id"] = project_id

    config["include_repo"] = args.include_repo or config.get("include_repo", False)
    config["session_names"] = args.session_names or config.get("session_names", False)
    config["user"] = args.user or config.get("user")
    settings = config["sources"].get(args.source) or {}
    settings.setdefault("since", agent_sync.since_days(args.backfill_days))
    if args.billing:
        settings["billing"] = args.billing
    settings["enforce"] = (args.enforce or settings.get("enforce", False)) and not args.no_hook
    if settings["enforce"] and not source.GATE_EVENTS:
        print(f"agentcost: {source.LABEL} has no hook to enforce with; tracking only.", file=sys.stderr)
        settings["enforce"] = False
    config["sources"][args.source] = settings
    agent_sync.save_config(config)
    print(f"Connected {source.LABEL} to project {project_id}.")

    if not source.HOOK_EVENTS:
        print(source.HOOK_NOTE)
    elif not args.no_hook:
        gate_events = source.GATE_EVENTS if settings["enforce"] else ()
        try:
            agent_hooks.install(source.hook_file(), source.HOOK_EVENTS, args.source, gate_events)
            print(f"Hook installed in {source.hook_file()}. {source.HOOK_NOTE}")
        except (OSError, ValueError) as exc:
            print(f"agentcost: could not install the hook: {exc}", file=sys.stderr)
            print("Run `agentcost sync` yourself, or on a schedule, instead.")
    if settings["enforce"]:
        _print_enforcement(args.source, agent_sync.refresh_policy(config, force=True))

    result = agent_sync.run_sync(config, sources=[args.source])
    _print_sync(result, dry_run=False)
    return 0 if result["ok"] else 1


def _sync(args) -> int:
    config = agent_sync.load_config()
    sources = [args.source] if args.source else None
    if args.dry_run:
        # Useful before connecting at all: look at every agent with history.
        sources = sources or list(config["sources"]) or sorted(SOURCES)
    elif not config.get("api_key") or not config.get("project_id") or not config["sources"]:
        if not args.quiet:
            print("agentcost: nothing is connected. Run `agentcost connect <agent>`.", file=sys.stderr)
        return 0 if args.quiet else 2
    elif sources and args.source not in config["sources"]:
        if not args.quiet:
            print(f"agentcost: {args.source} is not connected.", file=sys.stderr)
        return 0 if args.quiet else 2

    if not args.quiet:
        result = agent_sync.run_sync(config, sources=sources, dry_run=args.dry_run)
        if result["busy"]:
            print("agentcost: another sync is running.")
            return 0
        _print_sync(result, args.dry_run)
        return 0 if result["ok"] else 1

    # From a hook: whatever goes wrong here must never reach the agent's session.
    import logging
    import warnings

    logging.disable(logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            agent_sync.run_sync(config, sources=sources, fetch_pricing=False)
        except Exception:  # noqa: BLE001
            pass
    return 0


def _status() -> int:
    config = agent_sync.load_config()
    if not config["sources"]:
        print("No coding agent is connected. Run `agentcost connect <agent>`.")
        return 0
    print(f"Project {config.get('project_id')} at {config['base_url']}")
    print(f"Last sync: {agent_sync.load_state().get('last_sync') or 'never'}")
    pending = agent_sync.run_sync(config, dry_run=True)["sources"]
    for name in config["sources"]:
        source = SOURCES[name]
        target = source.hook_file()
        hook = "hook installed" if target and agent_hooks.is_installed(target) else "no hook"
        if target and agent_hooks.is_installed(target, agent_hooks.GATE_MARKER):
            hook += ", enforcing guardrails"
        print(f"  {source.LABEL}: {hook}, {pending[name]['calls']:,} call(s) waiting to sync")
    return 0


def _disconnect(args) -> int:
    source = SOURCES[args.source]
    config = agent_sync.load_config()
    try:
        if source.hook_file():
            agent_hooks.remove(source.hook_file())
    except (OSError, ValueError) as exc:
        print(f"agentcost: could not remove the hook: {exc}", file=sys.stderr)
    if config["sources"].pop(args.source, None) is not None:
        agent_sync.save_config(config)
    print(f"Disconnected {source.LABEL}. Usage already sent stays in AgentCost.")
    return 0


def _should_fail(findings, threshold: str) -> bool:
    rank = {"high": 0, "medium": 1, "low": 2}
    limit = rank[threshold]
    return any(rank.get(f.severity, 3) <= limit for f in findings)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "connect":
        return _connect(args)
    if args.command == "sync":
        return _sync(args)
    if args.command == "status":
        return _status()
    if args.command == "disconnect":
        return _disconnect(args)
    if args.command != "analyze":
        print(OVERVIEW.format(version=__version__, agents=", ".join(sorted(SOURCES))))
        return 0

    if not args.path and not args.events:
        parser.error("give a path to analyse, --events, or both")

    events: Optional[List[dict]] = None
    if args.events:
        try:
            events = load_events(args.events)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"agentcost: could not read {args.events}: {exc}", file=sys.stderr)
            return 2

    report = analyze(
        path=args.path,
        events=events,
        model=args.model,
        runs_per_day=args.runs_per_day,
        patterns=tuple(args.patterns) if args.patterns else DEFAULT_PATTERNS,
    )

    print(format_report(report))

    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(report.to_dict(), handle, indent=2)
        except OSError as exc:
            print(f"agentcost: could not write {args.json_out}: {exc}", file=sys.stderr)
            return 2

    if args.fail_on and _should_fail(report.findings, args.fail_on):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
