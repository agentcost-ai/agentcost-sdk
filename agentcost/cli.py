"""
AgentCost command line interface.

    agentcost analyze ./agent --model gpt-4o --runs-per-day 2000
    agentcost analyze --events run.json --json report.json

Runs entirely locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from . import __version__
from .analyzer import DEFAULT_PATTERNS, analyze, format_report, load_events


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentcost",
        description="Estimate what an agent will cost before deploying it.",
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
    return parser


def _should_fail(findings, threshold: str) -> bool:
    rank = {"high": 0, "medium": 1, "low": 2}
    limit = rank[threshold]
    return any(rank.get(f.severity, 3) <= limit for f in findings)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "analyze":
        parser.print_help()
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
