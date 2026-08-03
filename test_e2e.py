"""
End-to-End Test: SDK → Backend

This script tests the full flow:
1. Create a project in the backend
2. Configure SDK with project API key
3. Make LLM calls that get tracked
4. Query analytics from backend
"""

import sys
sys.path.insert(0, 'agentcost-sdk')

import httpx
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

from agentcost import track_costs

BACKEND_URL = "http://127.0.0.1:8000"


def main():
    print("=" * 60)
    print("AgentCost End-to-End Test")
    print("=" * 60)
    
    # Step 1: Create a project
    print("\nStep 1: Creating project...")
    with httpx.Client() as client:
        response = client.post(
            f"{BACKEND_URL}/v1/projects",
            json={"name": "test-project", "description": "E2E test project"}
        )
        project = response.json()
    
    project_id = project["id"]
    api_key = project["api_key"]
    print(f"   Project ID: {project_id}")
    print(f"   API Key: {api_key}")
    
    # Step 2: Initialize SDK with real backend
    print("\nStep 2: Initializing SDK...")
    track_costs.init(
        api_key=api_key,
        project_id=project_id,
        base_url=BACKEND_URL,
        debug=True,
        batch_size=3,  # Small batch for testing
        flush_interval=2.0,
    )
    
    # Step 3: Make LLM calls
    print("\nStep 3: Making LLM calls...")
    llm = ChatGroq(model="llama-3.1-8b-instant")
    
    with track_costs.agent("router-agent"):
        response1 = llm.invoke("Route this: billing question")
        print(f"   Response 1: {response1.content[:50]}...")
    
    with track_costs.agent("billing-agent"):
        response2 = llm.invoke("What are the payment options?")
        print(f"   Response 2: {response2.content[:50]}...")
    
    with track_costs.agent("technical-agent"):
        response3 = llm.invoke("How do I reset my password?")
        print(f"   Response 3: {response3.content[:50]}...")
    
    # Step 4: Flush and wait
    print("\nStep 4: Flushing events...")
    track_costs.flush()
    
    import time
    print("   Waiting for events to be sent...")
    time.sleep(5)  # Wait longer for background thread to send
    
    # Print batcher stats
    stats = track_costs.get_stats()
    print(f"   Batcher stats: {stats.get('batcher', {})}")
    
    # Step 5: Query analytics
    print("\nStep 5: Querying analytics...")
    with httpx.Client() as client:
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # Get overview
        response = client.get(
            f"{BACKEND_URL}/v1/analytics/overview?range=24h",
            headers=headers
        )
        overview = response.json()
        print(f"\n   Overview:")
        print(f"   - Total Calls: {overview['total_calls']}")
        print(f"   - Total Tokens: {overview['total_tokens']}")
        print(f"   - Total Cost: ${overview['total_cost']:.6f}")
        print(f"   - Avg Latency: {overview['avg_latency_ms']:.0f}ms")
        
        # Get agent stats
        response = client.get(
            f"{BACKEND_URL}/v1/analytics/agents?range=24h",
            headers=headers
        )
        agents = response.json()
        print(f"\n   Per-Agent Stats:")
        for agent in agents:
            print(f"   - {agent['agent_name']}: {agent['total_calls']} calls, ${agent['total_cost']:.6f}")
        
        # Get events list
        response = client.get(
            f"{BACKEND_URL}/v1/events",
            headers=headers
        )
        events = response.json()
        print(f"\n   Events stored: {len(events)}")
    
    # Cleanup
    track_costs.shutdown()
    
    print("\n" + "=" * 60)
    print("End-to-End Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
