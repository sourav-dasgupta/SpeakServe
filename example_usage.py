#!/usr/bin/env python3
"""
Example usage of AgentSession class for appointment scheduling.
This demonstrates how the agent handles conversation flow using OpenAI function-calling.
"""

import asyncio
import os
import sys
from agent_session import AgentSession

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 Please set these variables in your .env file:")
        print("   Copy .env.example to .env and fill in your values")
        print("   Example: cp .env.example .env")
        return False
    
    return True

async def main():
    """Example conversation with the AgentSession"""
    
    # Check environment variables
    if not check_environment():
        return
    
    # Initialize the agent
    openai_api_key = os.getenv("OPENAI_API_KEY")
    agent = AgentSession(openai_api_key=openai_api_key, client_id="example_user_123")
    
    # Example conversation flow
    conversation_messages = [
        "Hi, I need to schedule an appointment",
        "I want it for tomorrow at 2 PM",
        "It's for a dental checkup",
        "My name is John Smith",
        "Yes, that works for me",
        "Thank you, that's all I need"
    ]
    
    print("🤖 AI Agent: Hello! I'm your appointment scheduling assistant.")
    print("=" * 60)
    
    for i, message in enumerate(conversation_messages, 1):
        print(f"\n👤 User {i}: {message}")
        
        # Process the message
        result = await agent.process_message(message)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        print(f"🤖 Agent: {result.get('response_message', 'Processing...')}")
        
        if result.get('next_action') == 'to_speech':
            print("🔊 Converting response to speech...")
            speech_result = await agent.to_speech(result['response_message'])
            if speech_result.get('audio_file'):
                print(f"🎵 Audio saved to: {speech_result['audio_file']}")
        
        # Show current state
        summary = agent.get_conversation_summary()
        print(f"📊 State: Intent={summary['current_intent']}, "
              f"Filled slots={len(summary['filled_slots'])}, "
              f"Missing slots={len(summary['missing_slots'])}")
        
        # Small delay to simulate real conversation
        await asyncio.sleep(1)
    
    print("\n" + "=" * 60)
    print("📋 Final Conversation Summary:")
    final_summary = agent.get_conversation_summary()
    print(f"Client ID: {final_summary['client_id']}")
    print(f"Final Intent: {final_summary['current_intent']}")
    print(f"Filled Slots: {final_summary['filled_slots']}")
    print(f"Conversation Length: {final_summary['conversation_length']} messages")

async def interactive_mode():
    """Interactive mode for testing the agent"""
    
    # Check environment variables
    if not check_environment():
        return
    
    # Initialize the agent
    openai_api_key = os.getenv("OPENAI_API_KEY")
    agent = AgentSession(openai_api_key=openai_api_key, client_id="interactive_user")
    
    print("🤖 AI Agent: Hello! I'm your appointment scheduling assistant.")
    print("Type 'quit' to exit, 'reset' to start over, 'status' to see current state")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("🤖 Agent: Goodbye!")
                break
            elif user_input.lower() == 'reset':
                agent.reset_session()
                print("🤖 Agent: Session reset. How can I help you?")
                continue
            elif user_input.lower() == 'status':
                summary = agent.get_conversation_summary()
                print(f"📊 Current State:")
                print(f"  Intent: {summary['current_intent']}")
                print(f"  Filled Slots: {summary['filled_slots']}")
                print(f"  Missing Slots: {summary['missing_slots']}")
                continue
            elif not user_input:
                continue
            
            # Process the message
            result = await agent.process_message(user_input)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            print(f"🤖 Agent: {result.get('response_message', 'Processing...')}")
            
            # Handle next action
            next_action = result.get('next_action')
            if next_action == 'to_speech':
                print("🔊 Converting to speech...")
                speech_result = await agent.to_speech(result['response_message'])
                if speech_result.get('audio_file'):
                    print(f"🎵 Audio saved to: {speech_result['audio_file']}")
            
        except KeyboardInterrupt:
            print("\n🤖 Agent: Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(main()) 