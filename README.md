# SpeakServe AI

A sophisticated AI-powered voice assistant system that combines real-time speech transcription, intelligent conversation management, and appointment scheduling capabilities.

## 🚀 Features

### 🎤 **Real-time Speech Transcription**
- **Whisper AI Integration**: High-quality speech-to-text using OpenAI's Whisper model
- **Sliding Window Processing**: Continuous audio processing without losing context
- **Multi-language Support**: Transcribe speech in multiple languages
- **Context Preservation**: Maintains conversation history for better accuracy

### 🤖 **Intelligent Agent System**
- **OpenAI Function-calling**: Advanced intent recognition and conversation flow
- **Slot Filling**: Smart collection of user information (dates, times, names, etc.)
- **Calendar Integration**: Book, cancel, and manage appointments
- **Text-to-Speech**: Convert responses to natural speech using OpenAI TTS

### 🔧 **Technical Features**
- **WebSocket Support**: Real-time bidirectional communication
- **Twilio Integration**: Phone call transcription and voice processing
- **Configurable Audio Processing**: Adjustable buffer sizes and processing intervals
- **Error Handling**: Robust error recovery and graceful degradation

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- (Optional) Twilio account for phone integration

## 🛠️ Installation

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Waati-AI/Speakserve.git
   cd SpeakServe AI
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```
   This will:
   - Check Python version and dependencies
   - Create `.env` file from `.env.example`
   - Set up necessary directories
   - Guide you through configuration

3. **Configure environment variables**
   ```bash
   # Edit .env file with your API keys
   nano .env
   ```
   
   Required variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Manual Setup

If you prefer manual setup:

1. **Copy environment template**
   ```bash
   cp .env.example .env
   ```

2. **Edit .env file**
   ```bash
   nano .env
   # Add your OpenAI API key and other configuration
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Start the Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:8001` (configurable via environment variables).

### Test the Agent

```bash
# Run predefined conversation demo
python example_usage.py

# Interactive mode for testing
python example_usage.py interactive
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `HOST` | Server host address | `0.0.0.0` |
| `PORT` | Server port | `8001` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `WHISPER_MODEL` | Whisper model size | `small` |
| `BUFFER_DURATION` | Audio buffer duration (seconds) | `8.0` |

## 📚 Documentation

- **[AgentSession Documentation](README_AgentSession.md)**: Complete guide to the AI agent system
- **[API Documentation](docs/api.md)**: WebSocket API reference
- **[Deployment Guide](docs/deployment.md)**: Production deployment instructions

## 🏗️ Architecture

### Core Components

1. **Audio Processing Pipeline**
   - Twilio audio input → PCM conversion → Whisper transcription
   - Sliding window buffer for continuous processing
   - Context-aware transcription with history

2. **AgentSession System**
   - OpenAI function-calling for intent recognition
   - Slot filling and validation
   - Calendar API integration
   - Text-to-speech conversion

3. **WebSocket Communication**
   - Real-time bidirectional messaging
   - Connection state management
   - Error handling and recovery

### Data Flow

```
User Speech → Audio Processing → Transcription → Intent Recognition → Slot Filling → Calendar API → Response → Text-to-Speech
```

## 🔧 Configuration

### Audio Settings

```python
# Buffer configuration
BUFFER_DURATION = 8.0  # seconds
WHISPER_SAMPLE_RATE = 16000  # Hz
TWILIO_SAMPLE_RATE = 8000  # Hz
```

### Agent Settings

```python
# OpenAI configuration
model = "gpt-4"  # or "gpt-3.5-turbo"
temperature = 0.1
max_tokens = 1000
```

## 🧪 Testing

### Unit Tests

```bash
python -m pytest tests/
```

### Integration Tests

```bash
# Test transcription
python test_transcription.py

# Test agent conversation
python example_usage.py interactive
```

## 🚀 Deployment

### Development

```bash
python main.py
```

### Production

```bash
# Using Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001

# Using Docker
docker build -t speakserve-ai .
docker run -p 8001:8001 speakserve-ai
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [README_AgentSession.md](README_AgentSession.md)
- **Issues**: [GitHub Issues](https://github.com/Waati-AI/Speakserve/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Waati-AI/Speakserve/discussions)

## 🗺️ Roadmap

- [ ] Multi-language support for agent conversations
- [ ] Integration with Google Calendar and Outlook
- [ ] Voice emotion detection
- [ ] Advanced analytics and reporting
- [ ] Mobile app integration
- [ ] Custom voice training 