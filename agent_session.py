import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import openai
from datetime import datetime, timedelta
import re

# Configure logging
logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Enumeration of possible user intents"""
    GREETING = "greeting"
    APPOINTMENT_REQUEST = "appointment_request"
    APPOINTMENT_CANCELLATION = "appointment_cancellation"
    APPOINTMENT_RESCHEDULE = "appointment_reschedule"
    APPOINTMENT_INQUIRY = "appointment_inquiry"
    CALENDAR_CHECK = "calendar_check"
    AVAILABILITY_REQUEST = "availability_request"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"

@dataclass
class Slot:
    """Represents a slot that needs to be filled"""
    name: str
    value: Optional[str] = None
    required: bool = True
    prompt: Optional[str] = None
    validation_regex: Optional[str] = None
    
    def is_filled(self) -> bool:
        """Check if the slot has a valid value"""
        if not self.required:
            return True
        return self.value is not None and self.value.strip() != ""
    
    def validate(self, value: str) -> bool:
        """Validate a value against the slot's validation rules"""
        if self.validation_regex:
            return bool(re.match(self.validation_regex, value))
        return True

@dataclass
class Intent:
    """Represents a detected intent with confidence and slots"""
    type: IntentType
    confidence: float
    slots: Dict[str, Slot] = field(default_factory=dict)
    raw_text: str = ""

@dataclass
class CalendarEvent:
    """Represents a calendar event"""
    id: str
    title: str
    start_time: datetime
    end_time: datetime
    description: Optional[str] = None
    location: Optional[str] = None
    attendees: List[str] = field(default_factory=list)

class AgentSession:
    """
    AI Agent that handles conversation flow using OpenAI function-calling.
    Manages intents, slot filling, calendar operations, and text-to-speech.
    """
    
    def __init__(self, openai_api_key: str, client_id: str = None):
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.client_id = client_id or f"client_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history: List[Dict[str, str]] = []
        self.current_intent: Optional[Intent] = None
        self.slots: Dict[str, Slot] = {}
        self.context: Dict[str, Any] = {}
        self.is_active = True
        
        # Initialize default slots for appointment booking
        self._initialize_default_slots()
        
        # Define available functions for OpenAI function-calling
        self.available_functions = [
            {
                "name": "handle_intent",
                "description": "Detect and handle user intent from their message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent_type": {
                            "type": "string",
                            "enum": [intent.value for intent in IntentType],
                            "description": "The detected intent type"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence score for the intent detection"
                        },
                        "extracted_slots": {
                            "type": "object",
                            "description": "Any slots that can be extracted from the message",
                            "additionalProperties": True
                        },
                        "next_action": {
                            "type": "string",
                            "enum": ["fill_slots", "call_calendar_api", "to_speech", "end_conversation"],
                            "description": "The next action to take"
                        },
                        "response_message": {
                            "type": "string",
                            "description": "The response message to send to the user"
                        }
                    },
                    "required": ["intent_type", "confidence", "next_action", "response_message"]
                }
            },
            {
                "name": "fill_slots",
                "description": "Fill missing slots by asking the user for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "missing_slots": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of slot names that need to be filled"
                        },
                        "prompt_message": {
                            "type": "string",
                            "description": "Message to ask the user for missing information"
                        },
                        "next_action": {
                            "type": "string",
                            "enum": ["wait_for_response", "call_calendar_api", "to_speech"],
                            "description": "The next action after filling slots"
                        }
                    },
                    "required": ["missing_slots", "prompt_message", "next_action"]
                }
            },
            {
                "name": "call_calendar_api",
                "description": "Call calendar API to perform operations like booking, checking availability, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["check_availability", "book_appointment", "cancel_appointment", "reschedule_appointment", "list_appointments"],
                            "description": "The calendar operation to perform"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Parameters for the calendar operation",
                            "additionalProperties": True
                        },
                        "success_message": {
                            "type": "string",
                            "description": "Message to send on successful operation"
                        },
                        "error_message": {
                            "type": "string",
                            "description": "Message to send on error"
                        }
                    },
                    "required": ["operation", "parameters", "success_message", "error_message"]
                }
            },
            {
                "name": "to_speech",
                "description": "Convert text response to speech and provide audio output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to convert to speech"
                        },
                        "voice": {
                            "type": "string",
                            "enum": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                            "description": "The voice to use for speech synthesis"
                        },
                        "speed": {
                            "type": "number",
                            "minimum": 0.25,
                            "maximum": 4.0,
                            "description": "Speed of speech (1.0 is normal speed)"
                        }
                    },
                    "required": ["text", "voice"]
                }
            }
        ]
    
    def _initialize_default_slots(self):
        """Initialize default slots for appointment booking"""
        self.slots = {
            "date": Slot(
                name="date",
                required=True,
                prompt="What date would you like to schedule the appointment for?",
                validation_regex=r"^\d{4}-\d{2}-\d{2}$|^(today|tomorrow|next week|this week)$"
            ),
            "time": Slot(
                name="time",
                required=True,
                prompt="What time would you like to schedule the appointment for?",
                validation_regex=r"^\d{1,2}:\d{2}\s*(am|pm)?$|^(morning|afternoon|evening)$"
            ),
            "duration": Slot(
                name="duration",
                required=False,
                prompt="How long should the appointment be?",
                validation_regex=r"^\d+\s*(minutes?|hours?|mins?|hrs?)$"
            ),
            "purpose": Slot(
                name="purpose",
                required=False,
                prompt="What is the purpose of the appointment?",
                validation_regex=r".+"
            ),
            "name": Slot(
                name="name",
                required=True,
                prompt="What is your name?",
                validation_regex=r"^[A-Za-z\s]+$"
            ),
            "phone": Slot(
                name="phone",
                required=False,
                prompt="What is your phone number?",
                validation_regex=r"^\+?[\d\s\-\(\)]+$"
            )
        }
    
    async def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and determine the next action using OpenAI function-calling.
        
        Args:
            user_message: The user's input message
            
        Returns:
            Dict containing the response and next action
        """
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Create system prompt based on current context
            system_prompt = self._create_system_prompt()
            
            # Call OpenAI with function-calling
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.conversation_history[-10:]  # Last 10 messages for context
                ],
                functions=self.available_functions,
                function_call="auto",
                temperature=0.1
            )
            
            # Process the response
            message = response.choices[0].message
            
            if message.function_call:
                # Execute the function call
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                logger.info(f"Executing function: {function_name} with args: {function_args}")
                
                # Execute the appropriate function
                if function_name == "handle_intent":
                    result = await self.handle_intent(**function_args)
                elif function_name == "fill_slots":
                    result = await self.fill_slots(**function_args)
                elif function_name == "call_calendar_api":
                    result = await self.call_calendar_api(**function_args)
                elif function_name == "to_speech":
                    result = await self.to_speech(**function_args)
                else:
                    result = {"error": f"Unknown function: {function_name}"}
                
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": result.get("response_message", "Function executed successfully")
                })
                
                return result
            else:
                # No function call, just return the message
                response_message = message.content or "I'm not sure how to help with that."
                self.conversation_history.append({"role": "assistant", "content": response_message})
                
                return {
                    "response_message": response_message,
                    "next_action": "wait_for_response"
                }
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "error": f"Error processing message: {str(e)}",
                "response_message": "I'm sorry, I encountered an error. Please try again.",
                "next_action": "wait_for_response"
            }
    
    def _create_system_prompt(self) -> str:
        """Create a dynamic system prompt based on current context"""
        prompt = f"""You are an AI assistant for {self.client_id} that helps with appointment scheduling and calendar management.

Current context:
- Active intent: {self.current_intent.type.value if self.current_intent else 'None'}
- Filled slots: {[name for name, slot in self.slots.items() if slot.is_filled()]}
- Missing slots: {[name for name, slot in self.slots.items() if not slot.is_filled()]}
- Conversation history: {len(self.conversation_history)} messages

Your capabilities:
1. handle_intent: Detect user intent and determine next action
2. fill_slots: Ask user for missing information
3. call_calendar_api: Perform calendar operations
4. to_speech: Convert responses to speech

Guidelines:
- Be conversational and helpful
- Always use function-calling to determine next actions
- Fill all required slots before calling calendar API
- Provide clear, actionable responses
- Handle errors gracefully

Available calendar operations:
- check_availability: Check available time slots
- book_appointment: Schedule a new appointment
- cancel_appointment: Cancel an existing appointment
- reschedule_appointment: Change appointment time
- list_appointments: Show user's appointments

Required slots for appointment booking:
- date: Appointment date (YYYY-MM-DD or natural language)
- time: Appointment time (HH:MM AM/PM or natural language)
- name: User's name
- duration: Optional, appointment duration
- purpose: Optional, appointment purpose
- phone: Optional, contact phone number"""
        
        return prompt
    
    async def handle_intent(self, intent_type: str, confidence: float, 
                          extracted_slots: Dict[str, Any], next_action: str, 
                          response_message: str) -> Dict[str, Any]:
        """
        Handle detected user intent and extract slots from the message.
        
        Args:
            intent_type: The detected intent type
            confidence: Confidence score for the intent
            extracted_slots: Any slots that can be extracted from the message
            next_action: The next action to take
            response_message: Response message to send to user
            
        Returns:
            Dict containing the result of intent handling
        """
        try:
            # Update current intent
            self.current_intent = Intent(
                type=IntentType(intent_type),
                confidence=confidence,
                raw_text=response_message
            )
            
            # Fill extracted slots
            for slot_name, value in extracted_slots.items():
                if slot_name in self.slots:
                    if self.slots[slot_name].validate(str(value)):
                        self.slots[slot_name].value = str(value)
                        logger.info(f"Filled slot {slot_name} with value: {value}")
                    else:
                        logger.warning(f"Invalid value for slot {slot_name}: {value}")
            
            # Check if all required slots are filled
            missing_slots = [name for name, slot in self.slots.items() 
                           if not slot.is_filled() and slot.required]
            
            if missing_slots and next_action == "fill_slots":
                # Need to fill more slots
                return await self.fill_slots(
                    missing_slots=missing_slots,
                    prompt_message=response_message,
                    next_action="wait_for_response"
                )
            
            return {
                "intent_type": intent_type,
                "confidence": confidence,
                "filled_slots": extracted_slots,
                "missing_slots": missing_slots,
                "response_message": response_message,
                "next_action": next_action
            }
            
        except Exception as e:
            logger.error(f"Error handling intent: {str(e)}")
            return {
                "error": f"Error handling intent: {str(e)}",
                "response_message": "I'm sorry, I couldn't understand your request. Please try again.",
                "next_action": "wait_for_response"
            }
    
    async def fill_slots(self, missing_slots: List[str], prompt_message: str, 
                        next_action: str) -> Dict[str, Any]:
        """
        Fill missing slots by asking the user for information.
        
        Args:
            missing_slots: List of slot names that need to be filled
            prompt_message: Message to ask the user for missing information
            next_action: The next action after filling slots
            
        Returns:
            Dict containing the result of slot filling
        """
        try:
            # Create a focused prompt for the missing slots
            slot_prompts = []
            for slot_name in missing_slots:
                if slot_name in self.slots:
                    slot = self.slots[slot_name]
                    slot_prompts.append(f"{slot_name}: {slot.prompt}")
            
            enhanced_prompt = f"{prompt_message}\n\nPlease provide: {', '.join(slot_prompts)}"
            
            return {
                "missing_slots": missing_slots,
                "prompt_message": enhanced_prompt,
                "response_message": enhanced_prompt,
                "next_action": next_action,
                "status": "waiting_for_slots"
            }
            
        except Exception as e:
            logger.error(f"Error filling slots: {str(e)}")
            return {
                "error": f"Error filling slots: {str(e)}",
                "response_message": "I'm sorry, I encountered an error. Please try again.",
                "next_action": "wait_for_response"
            }
    
    async def call_calendar_api(self, operation: str, parameters: Dict[str, Any],
                               success_message: str, error_message: str) -> Dict[str, Any]:
        """
        Call calendar API to perform operations.
        
        Args:
            operation: The calendar operation to perform
            parameters: Parameters for the calendar operation
            success_message: Message to send on successful operation
            error_message: Message to send on error
            
        Returns:
            Dict containing the result of the calendar operation
        """
        try:
            logger.info(f"Calling calendar API: {operation} with parameters: {parameters}")
            
            # Simulate calendar API calls (replace with actual API integration)
            if operation == "check_availability":
                result = await self._check_availability(parameters)
            elif operation == "book_appointment":
                result = await self._book_appointment(parameters)
            elif operation == "cancel_appointment":
                result = await self._cancel_appointment(parameters)
            elif operation == "reschedule_appointment":
                result = await self._reschedule_appointment(parameters)
            elif operation == "list_appointments":
                result = await self._list_appointments(parameters)
            else:
                raise ValueError(f"Unknown calendar operation: {operation}")
            
            if result.get("success"):
                return {
                    "operation": operation,
                    "success": True,
                    "response_message": success_message,
                    "data": result.get("data"),
                    "next_action": "to_speech"
                }
            else:
                return {
                    "operation": operation,
                    "success": False,
                    "response_message": error_message,
                    "error": result.get("error"),
                    "next_action": "wait_for_response"
                }
                
        except Exception as e:
            logger.error(f"Error calling calendar API: {str(e)}")
            return {
                "error": f"Error calling calendar API: {str(e)}",
                "response_message": error_message,
                "next_action": "wait_for_response"
            }
    
    async def to_speech(self, text: str, voice: str = "nova", speed: float = 1.0) -> Dict[str, Any]:
        """
        Convert text response to speech.
        
        Args:
            text: The text to convert to speech
            voice: The voice to use for speech synthesis
            speed: Speed of speech (1.0 is normal speed)
            
        Returns:
            Dict containing the speech synthesis result
        """
        try:
            logger.info(f"Converting to speech: {text[:50]}... with voice: {voice}")
            
            # Call OpenAI TTS API
            response = await self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=speed
            )
            
            # Save audio to file (you might want to stream this instead)
            audio_filename = f"speech_{self.client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            response.stream_to_file(audio_filename)
            
            return {
                "text": text,
                "voice": voice,
                "speed": speed,
                "audio_file": audio_filename,
                "response_message": f"Speech generated: {audio_filename}",
                "next_action": "end_conversation"
            }
            
        except Exception as e:
            logger.error(f"Error converting to speech: {str(e)}")
            return {
                "error": f"Error converting to speech: {str(e)}",
                "response_message": "I'm sorry, I couldn't convert the response to speech.",
                "next_action": "wait_for_response"
            }
    
    # Calendar API simulation methods (replace with actual API calls)
    async def _check_availability(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate checking calendar availability"""
        await asyncio.sleep(0.5)  # Simulate API delay
        return {
            "success": True,
            "data": {
                "available_slots": [
                    "2024-01-15 10:00:00",
                    "2024-01-15 14:00:00",
                    "2024-01-16 09:00:00"
                ]
            }
        }
    
    async def _book_appointment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate booking an appointment"""
        await asyncio.sleep(0.5)  # Simulate API delay
        return {
            "success": True,
            "data": {
                "appointment_id": f"apt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "confirmation": "Appointment booked successfully"
            }
        }
    
    async def _cancel_appointment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate canceling an appointment"""
        await asyncio.sleep(0.5)  # Simulate API delay
        return {
            "success": True,
            "data": {
                "confirmation": "Appointment canceled successfully"
            }
        }
    
    async def _reschedule_appointment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate rescheduling an appointment"""
        await asyncio.sleep(0.5)  # Simulate API delay
        return {
            "success": True,
            "data": {
                "confirmation": "Appointment rescheduled successfully"
            }
        }
    
    async def _list_appointments(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate listing appointments"""
        await asyncio.sleep(0.5)  # Simulate API delay
        return {
            "success": True,
            "data": {
                "appointments": [
                    {
                        "id": "apt_001",
                        "title": "Doctor Appointment",
                        "date": "2024-01-15",
                        "time": "10:00 AM"
                    }
                ]
            }
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state"""
        return {
            "client_id": self.client_id,
            "current_intent": self.current_intent.type.value if self.current_intent else None,
            "filled_slots": {name: slot.value for name, slot in self.slots.items() if slot.is_filled()},
            "missing_slots": [name for name, slot in self.slots.items() if not slot.is_filled()],
            "conversation_length": len(self.conversation_history),
            "is_active": self.is_active
        }
    
    def reset_session(self):
        """Reset the session to initial state"""
        self.conversation_history.clear()
        self.current_intent = None
        self._initialize_default_slots()
        self.context.clear()
        self.is_active = True
        logger.info(f"Session reset for client: {self.client_id}") 