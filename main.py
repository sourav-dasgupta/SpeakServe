import json
import logging
import sys
import base64
import audioop
import asyncio
import os
from typing import Dict, Any, Optional
import numpy as np
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Configure logging with environment variable support
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Test log message
logger.info("=== Server Starting ===")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Whisper model with environment variable
whisper_model = os.getenv("WHISPER_MODEL", "small")
logger.info(f"Loading Whisper model: {whisper_model}")
model = whisper.load_model(whisper_model)
logger.info("Whisper model loaded successfully")

# Audio configuration with environment variables
TWILIO_SAMPLE_RATE = 8000
WHISPER_SAMPLE_RATE = 16000
BUFFER_DURATION = float(os.getenv("BUFFER_DURATION", "8.0"))  # Configurable buffer duration
BUFFER_SIZE = int(BUFFER_DURATION * WHISPER_SAMPLE_RATE)
OVERLAP_DURATION = 2.0  # 2 seconds overlap for context
OVERLAP_SIZE = int(OVERLAP_DURATION * WHISPER_SAMPLE_RATE)

logger.info(f"Audio configuration: Buffer duration={BUFFER_DURATION}s, Buffer size={BUFFER_SIZE} samples")

class AudioBuffer:
    def __init__(self, sample_rate=WHISPER_SAMPLE_RATE, buffer_duration=BUFFER_DURATION):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.current_position = 0
        self.is_full = False
        self.last_transcript = ""
        self.full_transcript = ""  # Accumulate full transcript
        self.last_processed_position = 0  # Track where we last processed
        logger.info(f"Initialized audio buffer with size {self.buffer_size} samples ({buffer_duration} seconds)")

    def add_audio(self, audio_chunk):
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0
        chunk_size = len(audio_array)
        logger.info(f"Adding audio chunk of size {chunk_size} to buffer at position {self.current_position}")
        
        # Handle buffer overflow with sliding window
        if self.current_position + chunk_size > self.buffer_size:
            # Shift buffer to make room (sliding window)
            shift_amount = chunk_size
            self.buffer = np.roll(self.buffer, -shift_amount)
            self.buffer[-chunk_size:] = audio_array
            self.current_position = self.buffer_size
            self.last_processed_position = max(0, self.last_processed_position - shift_amount)
        else:
            self.buffer[self.current_position:self.current_position + chunk_size] = audio_array
            self.current_position += chunk_size
            
        fill_percent = (self.current_position / self.buffer_size) * 100
        logger.info(f"Buffer fill: {self.current_position}/{self.buffer_size} samples ({fill_percent:.1f}%)")
        
        if self.current_position >= self.buffer_size:
            self.is_full = True
            logger.info("Buffer is now full")
            return True
        return False

    def get_buffer_for_processing(self):
        """Get the portion of buffer that hasn't been processed yet"""
        if self.current_position <= self.last_processed_position:
            return None
            
        # Get audio from last processed position to current position
        start_pos = self.last_processed_position
        end_pos = self.current_position
        
        # Ensure we have enough audio to process (at least 2 seconds)
        min_samples = 2 * self.sample_rate
        if (end_pos - start_pos) < min_samples:
            return None
            
        return self.buffer[start_pos:end_pos].copy()

    def mark_processed(self, processed_samples):
        """Mark the last N samples as processed"""
        self.last_processed_position = min(
            self.current_position, 
            self.last_processed_position + processed_samples
        )
        logger.info(f"Marked {processed_samples} samples as processed. Last processed position: {self.last_processed_position}")

    def get_full_buffer(self):
        """Get the entire buffer (for final processing)"""
        return self.buffer[:self.current_position].copy()

    def clear_processed(self):
        self.current_position = 0
        self.is_full = False
        self.last_processed_position = 0
        logger.info("Buffer cleared")

async def process_audio_chunk(buffer: AudioBuffer, websocket: WebSocket, force: bool = False, is_connected: bool = True) -> None:
    try:
        if force:
            # For final processing, use the entire buffer
            chunk = buffer.get_full_buffer()
            if len(chunk) < 1600:
                logger.debug(f"Final chunk too small: {len(chunk)} samples")
                return
        else:
            # Get unprocessed portion of buffer
            chunk = buffer.get_buffer_for_processing()
            if chunk is None or len(chunk) < 1600:
                logger.debug(f"No new audio to process or chunk too small: {len(chunk) if chunk is not None else 0} samples")
                return

        # Audio is already normalized in the buffer
        max_amp = np.max(np.abs(chunk))
        mean_amp = np.mean(np.abs(chunk))
        logger.info(f"Processing audio chunk - max amplitude: {max_amp:.3f}, mean amplitude: {mean_amp:.3f}, size: {len(chunk)} samples")
        
        # Check if audio is too quiet
        if max_amp < 0.01:  # Very low amplitude threshold
            logger.info("Audio too quiet, skipping transcription")
            buffer.mark_processed(len(chunk))
            return
            
        logger.info("Starting Whisper transcription...")
        try:
            # Prepare initial prompt with previous context
            initial_prompt = "This is a clear speech recording."
            if buffer.full_transcript:
                initial_prompt = f"Previous context: {buffer.full_transcript}. This is a clear speech recording."
            
            logger.info(f"Using initial prompt: {initial_prompt}")
            
            # Transcribe with Whisper
            result = model.transcribe(
                chunk,
                language="en",
                task="transcribe",
                fp16=False,
                verbose=True,  # Enable verbose output
                temperature=0.0,  # Reduce randomness
                best_of=5,  # Try multiple samples
                beam_size=5,  # Use beam search
                condition_on_previous_text=True,  # Consider previous context
                initial_prompt=initial_prompt,  # Use context-aware prompt
                compression_ratio_threshold=2.4,  # More lenient compression ratio
                logprob_threshold=-1.0,  # More lenient log probability threshold
                no_speech_threshold=0.3  # Much more lenient no-speech threshold
            )
            
            logger.info(f"Whisper result: {result}")
            
            transcript = result["text"].strip()
            if transcript:
                logger.info(f"Generated transcript: {transcript}")
                
                # Accumulate the full transcript
                if buffer.full_transcript:
                    buffer.full_transcript += " " + transcript
                else:
                    buffer.full_transcript = transcript
                
                # Only send if still connected
                if is_connected:
                    try:
                        await websocket.send_json({
                            "event": "transcript",
                            "text": transcript,
                            "full_transcript": buffer.full_transcript
                        })
                        logger.info(f"Transcript sent to client. Full transcript: {buffer.full_transcript}")
                        buffer.last_transcript = transcript
                    except Exception as e:
                        logger.error(f"Error sending transcript: {str(e)}")
                        # Don't raise the exception, just log it
                else:
                    logger.info(f"WebSocket disconnected, transcript not sent: {transcript}")
            else:
                logger.info("No speech detected in audio chunk")
                
            # Mark the processed audio
            buffer.mark_processed(len(chunk))
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            logger.error(traceback.format_exc())
            # Still mark as processed to avoid infinite loops
            buffer.mark_processed(len(chunk))
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")

def twilio_payload_to_pcm(payload: str) -> bytes:
    """Convert Twilio's 8-bit µ-law audio to 16-bit PCM at 16 kHz."""
    try:
        # Decode base64
        audio_data = base64.b64decode(payload)
        logger.debug(f"Decoded audio data size: {len(audio_data)} bytes")
        
        # Convert µ-law to PCM
        pcm_data = audioop.ulaw2lin(audio_data, 2)  # 2 bytes per sample
        logger.debug(f"PCM data size after ulaw2lin: {len(pcm_data)} bytes")
        
        # Upsample from 8kHz to 16kHz
        pcm_data = audioop.ratecv(pcm_data, 2, 1, TWILIO_SAMPLE_RATE, WHISPER_SAMPLE_RATE, None)[0]
        logger.debug(f"PCM data size after upsampling: {len(pcm_data)} bytes")
        
        # Apply audio preprocessing
        # Normalize audio with higher gain
        pcm_data = audioop.mul(pcm_data, 2, 8.0)  # Increased volume gain to 8.0
        
        # Apply noise reduction
        pcm_data = audioop.lin2lin(pcm_data, 2, 2)  # Ensure 16-bit
        pcm_data = audioop.bias(pcm_data, 2, 0)  # Remove DC offset
        
        return pcm_data
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        return b''

@app.websocket("/twilio/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    audio_buffer = AudioBuffer()
    is_connected = True
    last_process_time = 0
    min_process_interval = 4.0  # Process every 4 seconds for longer sentences
    min_buffer_fill = 0.2  # Process when buffer is at least 20% full
    
    try:
        while is_connected:
            try:
                message = await websocket.receive_text()
                logger.info(f"Received WebSocket message: {message}")
                data = json.loads(message)
                event = data.get("event")
                
                if event == "media":
                    media = data.get("media", {})
                    if media.get("track") == "inbound":
                        payload = media.get("payload")
                        if payload:
                            logger.info("Received audio payload")
                            pcm_data = twilio_payload_to_pcm(payload)
                            if pcm_data:
                                buffer_full = audio_buffer.add_audio(pcm_data)
                                current_time = asyncio.get_event_loop().time()
                                buffer_fill = audio_buffer.current_position / audio_buffer.buffer_size
                                
                                # Process if buffer is full or enough time has passed with sufficient audio
                                if buffer_full or ((current_time - last_process_time) >= min_process_interval and buffer_fill >= min_buffer_fill):
                                    logger.info(f"Processing audio chunk (buffer fill: {buffer_fill:.1%})")
                                    await process_audio_chunk(audio_buffer, websocket, is_connected=is_connected)
                                    last_process_time = current_time
                elif event == "stop":
                    logger.info("Received stop event")
                    # Process any remaining audio
                    await process_audio_chunk(audio_buffer, websocket, force=True, is_connected=is_connected)
                    is_connected = False
                    break
                elif event == "start":
                    logger.info("Received start event")
                    await websocket.send_json({"event": "start", "status": "ok"})
                elif event == "mark":
                    logger.info(f"Received mark event: {data.get('mark', {}).get('name')}")
                    await websocket.send_json({"event": "mark", "status": "ok"})
                
            except WebSocketDisconnect:
                logger.warning("WebSocket disconnected")
                is_connected = False
                # Process any remaining audio
                try:
                    await process_audio_chunk(audio_buffer, websocket, force=True, is_connected=False)
                except Exception as e:
                    logger.error(f"Error processing final audio: {str(e)}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        try:
            # Send final message with complete transcript only if still connected
            if is_connected:
                final_message = {
                    "event": "transcript",
                    "text": "Call ended",
                    "full_transcript": audio_buffer.full_transcript
                }
                try:
                    await websocket.send_json(final_message)
                    logger.info(f"Sent call ended message with full transcript: {audio_buffer.full_transcript}")
                except Exception as e:
                    logger.error(f"Error sending call ended message: {str(e)}")
            else:
                logger.info(f"Call ended. Final transcript: {audio_buffer.full_transcript}")
            
            # Clear the buffer
            audio_buffer.clear_processed()
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Twilio Voice WebSocket Server is running"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    
    # Get server configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    
    logger.info(f"Server configuration: Host={host}, Port={port}")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="debug",
        access_log=True,
        ws_ping_interval=20,  # Enable ping every 20 seconds
        ws_ping_timeout=20,   # Wait 20 seconds for pong response
        ws_max_size=1024 * 1024,  # 1MB max message size
        timeout_keep_alive=65,  # Keep connections alive for 65 seconds
        limit_concurrency=1000,  # Increase concurrent connection limit
        backlog=2048  # Increase connection backlog
    ) 