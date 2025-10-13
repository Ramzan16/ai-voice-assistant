# AI Voice Assistant

A modular, asynchronous, and extensible **AI Voice Assistant** built with Python. It uses wake word detection to activate, transcribes speech in real-time, processes queries with Google's Gemini model, and responds with streaming text-to-speech.

> The core design emphasizes a clean separation of concerns, making it easy to swap components or add new capabilities.

---

## Core Features

- **Wake Word Activation** — Listens passively for wake words (e.g., `jarvis`, `assistant`) using `openwakeword` before activating to save resources.
- **Real-time Speech-to-Text** — Once activated, it transcribes user speech in real-time using the `RealtimeSTT` library.
- **Advanced Conversational AI** — Utilizes Google's Gemini large language model through LangChain to understand context, maintain conversation history, and provide intelligent responses.
- **Tool-Using Agent** — The agent, built with LangGraph, can intelligently decide when to use external tools to answer questions.
- **Web Search** — Can perform real-time web searches for up-to-date information using the Tavily search API.
- **Calculator** — Includes a safe, built-in calculator to evaluate mathematical expressions.
- **Streaming Text-to-Speech** — Employs the Kokoro TTS engine to synthesize speech and stream the audio back to the user with minimal latency.
- **Fully Asynchronous** — Built from the ground up with `asyncio` to handle I/O-bound tasks (audio streaming, API calls) efficiently without blocking.
- **Configuration Driven** — All settings, from API keys to model names and log levels, are managed centrally via a Pydantic settings object, configurable through a `.env` file.

---

## Technology Stack

- **Conversational Agent:** LangChain, LangGraph, Google Gemini
- **Speech-to-Text (STT):** RealtimeSTT
- **Text-to-Speech (TTS):** Kokoro
- **Tools:** Tavily (Web Search)
- **Async:** `asyncio`
- **Configuration:** Pydantic
- **Audio I/O:** SoundDevice

---

## Project Architecture

The project is organized into distinct modules, each with a specific responsibility:

```
/src/config      # Manages application settings (Pydantic) and logging (Loguru)
/src/core        # Core abstractions and data structures (SpeechToText, TextToSpeech, Agent, ChatMessage)
/src/services    # Concrete implementations that integrate external STT, TTS, Agent, and tools
/src/orchestrator # VoiceAssistant class: coordinates flow between services (audio in -> audio out)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An appropriate audio backend for your OS. For Debian/Ubuntu-based systems:

```bash
sudo apt-get update && sudo apt-get install libportaudio2 portaudio19-dev
```

### Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/ramzan16-ai-voice-assistant.git
cd ramzan16-ai-voice-assistant
```

2. Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure API Keys:

- Create a `.env` file in the project root. You can copy the template:

```bash
cp .env.example .env
# or on Windows:
# copy .env.example .env
```

- Open the `.env` file and add your API keys:

```
GEMINI_API_KEY="your_google_ai_studio_api_key"
TAVILY_API_KEY="your_tavily_api_key"
# ... any other settings you wish to override
```

5. Run the Assistant:

```bash
python -m assistant.src.main
```

The assistant will initialize and begin listening for the wake word.

---

## Future Roadmap

### 1. Deployment and Developer Experience

- **Containerize with Docker:** Create a `Dockerfile` to package the application with all system and Python dependencies, and a `docker-compose.yml` to simplify running the container, manage environment variables, and map host audio devices.
- **Create a Makefile:** Add a `Makefile` with common targets (`make install`, `make run`, `make test`, `make lint`).
- **Improve .gitignore:** Ensure generated files, logs, virtual environments, and environment files are properly ignored.

### 2. New Features

- **Add New Tools:** Leverage the extensible tool registry to add integrations (e.g., Notion, Spotify) already anticipated in the settings.
- **Stateful System Commands:** Implement assistant-control tools such as `change your voice`, `speak faster`, and `enter silent mode`.

---
