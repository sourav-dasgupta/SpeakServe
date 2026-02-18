# AgentSession Class Documentation

## Overview

The `AgentSession` class is a sophisticated AI agent that uses OpenAI function-calling to handle conversation flow, intent recognition, slot filling, calendar operations, and text-to-speech conversion. It's designed for appointment scheduling and calendar management applications.

## Features

### 🎯 **Intent Recognition**
- Detects user intents using OpenAI function-calling
- Supports multiple intent types: greeting, appointment requests, cancellations, etc.
- Provides confidence scores for intent detection

### 📝 **Slot Filling**
- Manages conversation slots (date, time, name, purpose, etc.)
- Validates slot values using regex patterns
- Handles required vs optional slots
- Provides contextual prompts for missing information

### 📅 **Calendar API Integration**
- Check availability
- Book appointments
- Cancel appointments
- Reschedule appointments
- List existing appointments

### 🔊 **Text-to-Speech**
- Converts responses to speech using OpenAI TTS
- Multiple voice options (alloy, echo, fable, onyx, nova, shimmer)
- Adjustable speech speed
- Audio file generation

### 🔄 **Conversation Management**
- Maintains conversation history
- Context-aware responses
- Session state management
- Graceful error handling

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Basic Usage

### Simple Example

```python
import asyncio
from agent_session import AgentSession

async def main():
    # Initialize the agent
    agent = AgentSession(
        openai_api_key="your-api-key",
        client_id="user_123"
    )
    
    # Process a user message
    result = await agent.process_message("I need to schedule an appointment for tomorrow")
    
    print(f"Response: {result['response_message']}")
    print(f"Next Action: {result['next_action']}")

asyncio.run(main())
```

### Interactive Mode

```bash
python example_usage.py interactive
```

## Class Structure

### Core Methods

#### `process_message(user_message: str) -> Dict[str, Any]`
Main method to process user input and determine next actions using OpenAI function-calling.

#### `handle_intent(intent_type, confidence, extracted_slots, next_action, response_message) -> Dict[str, Any]`
Handles detected user intent and extracts slots from the message.

#### `fill_slots(missing_slots, prompt_message, next_action) -> Dict[str, Any]`
Fills missing slots by asking the user for information.

#### `call_calendar_api(operation, parameters, success_message, error_message) -> Dict[str, Any]`
Calls calendar API to perform operations like booking, checking availability, etc.

#### `to_speech(text, voice, speed) -> Dict[str, Any]`
Converts text response to speech using OpenAI TTS.

### Data Structures

#### `IntentType` Enum
```python
class IntentType(Enum):
    GREETING = "greeting"
    APPOINTMENT_REQUEST = "appointment_request"
    APPOINTMENT_CANCELLATION = "appointment_cancellation"
    APPOINTMENT_RESCHEDULE = "appointment_reschedule"
    APPOINTMENT_INQUIRY = "appointment_inquiry"
    CALENDAR_CHECK = "calendar_check"
    AVAILABILITY_REQUEST = "availability_request"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"
```

#### `Slot` Class
```python
@dataclass
class Slot:
    name: str
    value: Optional[str] = None
    required: bool = True
    prompt: Optional[str] = None
    validation_regex: Optional[str] = None
```

#### `Intent` Class
```python
@dataclass
class Intent:
    type: IntentType
    confidence: float
    slots: Dict[str, Slot] = field(default_factory=dict)
    raw_text: str = ""
```

## Default Slots

The agent comes with pre-configured slots for appointment scheduling:

- **date**: Appointment date (YYYY-MM-DD or natural language)
- **time**: Appointment time (HH:MM AM/PM or natural language)
- **duration**: Appointment duration (optional)
- **purpose**: Appointment purpose (optional)
- **name**: User's name
- **phone**: Contact phone number (optional)

## Conversation Flow

1. **User Input** → `process_message()`
2. **Intent Detection** → `handle_intent()` (via OpenAI function-calling)
3. **Slot Extraction** → Fill available slots from message
4. **Slot Validation** → Check if required slots are filled
5. **Next Action** → Determine next step:
   - `fill_slots`: Ask for missing information
   - `call_calendar_api`: Perform calendar operation
   - `to_speech`: Convert response to speech
   - `wait_for_response`: Wait for user input

## Example Conversation

```
User: "Hi, I need to schedule an appointment"
Agent: "Hello! I'd be happy to help you schedule an appointment. What date would you like to schedule it for?"

User: "Tomorrow at 2 PM"
Agent: "Great! I have tomorrow at 2 PM available. What is your name?"

User: "John Smith"
Agent: "Perfect! I've scheduled your appointment for tomorrow at 2 PM, John. Is there anything else you need?"
```

## Integration with Main Application

To integrate the AgentSession with your main FastAPI application:

```python
from fastapi import FastAPI, WebSocket
from agent_session import AgentSession

app = FastAPI()

@app.websocket("/agent/chat")
async def agent_chat(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize agent for this session
    agent = AgentSession(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        client_id=f"ws_{websocket.client.host}"
    )
    
    try:
        while True:
            # Receive user message
            user_message = await websocket.receive_text()
            
            # Process with agent
            result = await agent.process_message(user_message)
            
            # Send response
            await websocket.send_json({
                "response": result["response_message"],
                "next_action": result["next_action"],
                "slots": agent.get_conversation_summary()["filled_slots"]
            })
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
```

## Error Handling

The agent includes comprehensive error handling:

- **API Errors**: Graceful handling of OpenAI API failures
- **Validation Errors**: Slot validation with user-friendly messages
- **Connection Errors**: WebSocket disconnection handling
- **Invalid Input**: Robust parsing of user messages

## Customization

### Adding New Intents

```python
# Add to IntentType enum
class IntentType(Enum):
    # ... existing intents ...
    CUSTOM_INTENT = "custom_intent"

# Update system prompt in _create_system_prompt()
```

### Adding New Slots

```python
# Add to _initialize_default_slots()
self.slots["custom_slot"] = Slot(
    name="custom_slot",
    required=True,
    prompt="What is your custom information?",
    validation_regex=r".+"
)
```

### Custom Calendar API

Replace the simulation methods with actual API calls:

```python
async def _book_appointment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    # Replace with actual calendar API call
    response = await your_calendar_api.book_appointment(parameters)
    return {
        "success": response.success,
        "data": response.data
    }
```

## Testing

Run the example script:

```bash
# Run predefined conversation
python example_usage.py

# Run interactive mode
python example_usage.py interactive
```

## Dependencies

- `openai>=1.3.7`: OpenAI API client
- `asyncio`: Asynchronous programming
- `dataclasses`: Data structure definitions
- `logging`: Logging and debugging
- `re`: Regular expressions for validation

## License

This code is part of the SpeakServe AI project. 