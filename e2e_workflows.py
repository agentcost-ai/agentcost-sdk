"""
End-to-end proof: real SDK -> real HTTP -> real backend -> real analytics.

Everything else in the suites stops at a boundary. The SDK tests stop at the
batcher, the backend tests inject JSON directly into the ASGI app. This script
is the only thing that exercises the whole chain the way production does:
a genuine provider client, monkey-patched by a genuine interceptor, batched,
POSTed over HTTP with an API key, persisted through the real ingestion path,
and read back through the real analytics endpoints.

Run against a backend already listening on BASE_URL.
"""

import json
import os
import sys
import uuid

import httpx

BASE_URL = os.environ["AGENTCOST_E2E_URL"].rstrip("/")
IN_TOK, OUT_TOK = 25, 15
MODEL = "gpt-4o-mini"

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def openai_handler(request):
    """Stands in for OpenAI. The AgentCost path around it is entirely real."""
    body = json.loads(request.content.decode()) if request.content else {}
    if body.get("stream"):
        frames = [
            {"id": "1", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
             "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}]},
            {"id": "1", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
             "choices": [], "usage": {"prompt_tokens": IN_TOK,
                                      "completion_tokens": OUT_TOK,
                                      "total_tokens": IN_TOK + OUT_TOK}},
        ]
        content = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
        return httpx.Response(200, content=content.encode(),
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={
        "id": "1", "object": "chat.completion", "created": 1, "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": IN_TOK, "completion_tokens": OUT_TOK,
                  "total_tokens": IN_TOK + OUT_TOK},
    })


def main():
    print("\n=== 1. Backend health ===")
    health = httpx.get(f"{BASE_URL}/v1/health", timeout=30)
    check("backend is up", health.status_code == 200, f"HTTP {health.status_code}")

    print("\n=== 2. Register a real user and create a project over HTTP ===")
    signup = httpx.post(
        f"{BASE_URL}/v1/auth/register",
        json={
            "email": f"e2e-{uuid.uuid4().hex[:10]}@example.com",
            "password": "E2e-Password-123",
            "name": "E2E Runner",
            "accept_terms": True,
            "accept_privacy": True,
            "terms_version": "1.0",
            "privacy_version": "1.0",
        },
        timeout=60,
    )
    check("user registered", signup.status_code in (200, 201),
          f"HTTP {signup.status_code} {signup.text[:200]}")
    if signup.status_code not in (200, 201):
        return
    jwt = {"Authorization": f"Bearer {signup.json()['access_token']}"}

    created = httpx.post(
        f"{BASE_URL}/v1/projects",
        json={"name": f"e2e-{uuid.uuid4().hex[:8]}"},
        headers=jwt,
        timeout=30,
    )
    check("project created", created.status_code in (200, 201),
          f"HTTP {created.status_code} {created.text[:160]}")
    if created.status_code not in (200, 201):
        return
    project = created.json()
    project_id, api_key = project["id"], project["api_key"]
    print(f"  project_id={project_id}")

    print("\n=== 3. Run a real agent through the real SDK ===")
    from agentcost import track_costs
    from openai import OpenAI

    track_costs.init(api_key=api_key, project_id=project_id, base_url=BASE_URL,
                     batch_size=100, flush_interval=60)

    client = OpenAI(api_key="test", base_url="https://api.openai.com/v1",
                    http_client=httpx.Client(transport=httpx.MockTransport(openai_handler)))

    def ask(text):
        return client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": text}])

    # Two runs of the same workflow. The second loops: it asks the identical
    # question three times inside one run, which is what repeated-work exists
    # to surface.
    with track_costs.workflow("support-triage"):
        with track_costs.step("classify"):
            ask("which queue?")
        with track_costs.tool("search_docs"):
            ask("find the docs")
        with track_costs.step("draft_reply"):
            ask("write it")
        track_costs.outcome(True, label="resolved")

    with track_costs.workflow("support-triage"):
        with track_costs.step("classify"):
            ask("which queue?")
        with track_costs.tool("search_docs"):
            ask("looping query")
            ask("looping query")
            ask("looping query")
        track_costs.outcome(False, label="escalated")

    # An untracked call: must ingest fine and must NOT appear in workflows.
    ask("no workflow around me")

    track_costs.flush()
    track_costs.shutdown()
    print("  8 calls made (7 inside workflows, 1 outside) and flushed over HTTP")

    auth = {"Authorization": f"Bearer {api_key}"}

    def get(path):
        r = httpx.get(f"{BASE_URL}{path}", headers=auth, timeout=60)
        r.raise_for_status()
        return r.json()

    print("\n=== 4. Ingestion landed ===")
    overview = get("/v1/analytics/overview?range=24h")
    check("all 8 calls persisted", overview["total_calls"] == 8,
          f"got {overview['total_calls']}")
    check("tokens aggregated", overview["total_tokens"] == 8 * (IN_TOK + OUT_TOK),
          f"got {overview['total_tokens']}")

    print("\n=== 5. Workflow analytics ===")
    workflows = get("/v1/analytics/workflows?range=24h")
    check("one workflow found", len(workflows) == 1, f"got {len(workflows)}")
    if workflows:
        w = workflows[0]
        check("workflow named correctly", w["workflow"] == "support-triage", w["workflow"])
        check("two runs counted", w["runs"] == 2, f"got {w['runs']}")
        check("untraced call excluded from workflow", w["total_calls"] == 7,
              f"got {w['total_calls']} (8 calls made, 1 outside the workflow)")
        check("per-run average divides by runs", w["avg_calls_per_run"] == 3.5,
              f"got {w['avg_calls_per_run']} (7 traced calls / 2 runs)")
        check("nesting depth recorded", w["max_depth"] >= 1, f"got {w['max_depth']}")

    print("\n=== 6. Step analytics ===")
    steps = {s["step_name"]: s for s in get("/v1/analytics/workflows/steps?range=24h")}
    check("all three steps present",
          {"classify", "search_docs", "draft_reply"} <= set(steps),
          f"got {sorted(steps)}")
    if "search_docs" in steps:
        check("looping step shows calls_per_run > 1",
              steps["search_docs"]["calls_per_run"] == 2.0,
              f"got {steps['search_docs']['calls_per_run']} (4 calls / 2 runs)")
    if "classify" in steps:
        check("non-looping step stays at 1.0",
              steps["classify"]["calls_per_run"] == 1.0,
              f"got {steps['classify']['calls_per_run']}")

    print("\n=== 7. Tool analytics ===")
    tools = get("/v1/analytics/workflows/tools?range=24h")
    check("tool attributed", len(tools) == 1 and tools[0]["tool_name"] == "search_docs",
          str([t["tool_name"] for t in tools]))
    if tools:
        check("tool captured all its nested calls", tools[0]["calls"] == 4,
              f"got {tools[0]['calls']}")

    print("\n=== 8. Loop detection ===")
    repeats = get("/v1/analytics/workflows/repeated-work?range=24h")
    check("the loop was caught", len(repeats) >= 1, f"got {len(repeats)} findings")
    if repeats:
        r = repeats[0]
        check("repeat count correct", r["occurrences"] == 3, f"got {r['occurrences']}")
        # Costs are rounded to 6dp throughout the codebase, so compare at
        # that granularity rather than to floating-point exactness.
        check("charges only the redundant repeats",
              abs(r["wasted_cost"] - round(r["spend"] * 2 / 3, 6)) <= 1e-6,
              f"wasted={r['wasted_cost']} of spend={r['spend']}")

    print("\n=== 9. Trace detail ===")
    traces = get("/v1/analytics/traces?range=24h")
    check("both runs listed", len(traces) == 2, f"got {len(traces)}")
    if traces:
        detail = get(f"/v1/analytics/traces/{traces[0]['trace_id']}")
        check("span tree returned", len(detail["spans"]) == traces[0]["calls"],
              f"{len(detail['spans'])} spans")
        parented = [s for s in detail["spans"] if s["parent_span_id"]]
        check("spans carry parents", len(parented) == len(detail["spans"]),
              f"{len(parented)}/{len(detail['spans'])}")
        ids = {s["span_id"] for s in detail["spans"]}
        check("span ids are unique", len(ids) == len(detail["spans"]))

    print("\n=== 10. Cross-project isolation ===")
    other = httpx.post(f"{BASE_URL}/v1/projects",
                       json={"name": f"e2e-other-{uuid.uuid4().hex[:6]}"},
                       headers=jwt, timeout=30)
    if other.status_code in (200, 201) and traces:
        other_auth = {"Authorization": f"Bearer {other.json()['api_key']}"}
        leaked = httpx.get(f"{BASE_URL}/v1/analytics/traces/{traces[0]['trace_id']}",
                           headers=other_auth, timeout=30)
        check("another project cannot read this trace", leaked.status_code == 404,
              f"HTTP {leaked.status_code}")

    print("\n=== 11. Cost per completed outcome ===")
    outcomes = get("/v1/analytics/workflows/outcomes?range=24h")
    check("outcomes recorded", len(outcomes) == 1, f"got {len(outcomes)}")
    if outcomes:
        o = outcomes[0]
        check("one success and one failure",
              o["succeeded"] == 1 and o["failed"] == 1,
              f"succeeded={o['succeeded']} failed={o['failed']}")
        check("no runs left unreported", o["unknown"] == 0, f"got {o['unknown']}")
        check("failure spend is charged to the success",
              o["cost_per_success"] is not None
              and o["cost_per_success"] > o["cost_on_success"],
              f"cost_per_success={o['cost_per_success']} "
              f"cost_on_success={o['cost_on_success']}")

    print("\n=== 12. Run cost distribution ===")
    dist = get("/v1/analytics/workflows/distribution?range=24h")
    check("distribution computed", dist is not None and dist["runs"] == 2,
          f"runs={dist['runs'] if dist else None}")
    if dist:
        check("histogram accounts for every run",
              sum(b["count"] for b in dist["histogram"]) == dist["runs"],
              f"buckets sum to {sum(b['count'] for b in dist['histogram'])}")

    print("\n=== 13. Optimizations endpoint still healthy ===")
    suggestions = get("/v1/optimizations")
    check("optimizations endpoint responds", isinstance(suggestions, list),
          f"{len(suggestions) if isinstance(suggestions, list) else '?'} suggestions")

    print("\n" + "=" * 62)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL END-TO-END CHECKS PASSED")


if __name__ == "__main__":
    main()
