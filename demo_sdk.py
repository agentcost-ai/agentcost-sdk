"""
AgentCost SDK Demo

This demo shows how to use the AgentCost SDK with LangChain.
Run this after installing the SDK with: pip install -e agentcost-sdk/
"""

import sys
sys.path.insert(0, 'agentcost-sdk')

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

# Import AgentCost
from agentcost import track_costs


def main():
    print("=" * 60)
    print("AgentCost SDK Demo")
    print("=" * 60)
    
    # Initialize tracking in local mode (no backend needed)
    track_costs.init(
        local_mode=True,
        debug=True,
        default_agent_name="demo-agent"
    )
    
    print("\nTracking initialized!\n")
    
    # Create LangChain LLM
    llm = ChatGroq(model="llama-3.1-8b-instant")
    
    # Make some LLM calls - they're automatically tracked!
    print("Making LLM calls...\n")
    
    # Call 1: Simple question
    with track_costs.agent("question-agent"):
        response1 = llm.invoke("What is Python?")
        print(f"Response 1: {response1.content[:100]}...\n")
    
    # Call 2: With metadata
    with track_costs.metadata(conversation_id="demo-123"):
        response2 = llm.invoke("Explain decorators in one sentence.")
        print(f"Response 2: {response2.content[:100]}...\n")
    
    # Call 3: Another agent
    with track_costs.agent("technical-agent"):
        response3 = llm.invoke("What is monkey patching?")
        print(f"Response 3: {response3.content[:100]}...\n")
    
    # Flush any pending events
    track_costs.flush()
    
    # Get all captured events
    events = track_costs.get_local_events()
    
    print("=" * 60)
    print(f"Captured {len(events)} LLM calls:")
    print("=" * 60)
    
    total_cost = 0
    total_tokens = 0
    
    for i, event in enumerate(events, 1):
        print(f"\nCall {i}:")
        print(f"   Agent: {event['agent_name']}")
        print(f"   Model: {event['model']}")
        print(f"   Input tokens: {event['input_tokens']}")
        print(f"   Output tokens: {event['output_tokens']}")
        print(f"   Total tokens: {event['total_tokens']}")
        print(f"   Cost: ${event['cost']:.6f}")
        print(f"   Latency: {event['latency_ms']}ms")
        print(f"   Success: {event['success']}")
        
        if 'metadata' in event:
            print(f"   Metadata: {event['metadata']}")
        
        total_cost += event['cost']
        total_tokens += event['total_tokens']
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"   Total calls: {len(events)}")
    print(f"   Total tokens: {total_tokens}")
    print(f"   Total cost: ${total_cost:.6f}")
    print("=" * 60)
    
    # Show stats
    stats = track_costs.get_stats()
    print(f"\nBatcher stats: {stats.get('batcher', {})}")
    
    # Shutdown
    track_costs.shutdown()
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
