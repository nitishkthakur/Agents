# Devin Agent

A full-featured AI agent chat application built using the `create_deep_agent` function from the `deepagents` langchain package. Features a ChatGPT-like interface with streaming responses, markdown rendering, file upload, web search, and more.

## Features

- Multi-model support (Claude, GPT, Groq models)
- Streaming responses with real-time updates
- Markdown rendering with code highlighting and LaTeX equations
- PDF file upload with text and image extraction
- Web search via Tavily
- Artifact saving to disk
- Conversation export as PDF
- Dark theme professional interface

## Setup

### Prerequisites

- Python 3.11+
- poppler-utils (for PDF to image conversion)

### Installation

1. Install backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

2. Create a `.env` file in the `devin_app` directory with your API keys:

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

Required environment variables:
- `ANTHROPIC_API_KEY` - For Claude models
- `OPENAI_API_KEY` - For GPT models
- `GROQ_API_KEY` - For Groq models
- `TAVILY_API_KEY` - For web search

### Running the Application

1. Start the backend server:

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

2. Open the frontend in a browser:

```bash
cd frontend
# Open index.html in a browser, or serve it with a simple HTTP server:
python -m http.server 3000
```

3. Navigate to `http://localhost:3000` in your browser.

## Configuration

The `config.json` file contains the list of available models. You can modify this file to add or remove models.

## Project Structure

```
devin_app/
├── backend/
│   ├── main.py          # FastAPI backend with agent implementation
│   └── requirements.txt # Python dependencies
├── frontend/
│   ├── index.html       # Main HTML file
│   ├── styles.css       # Dark theme CSS
│   └── app.js           # Frontend JavaScript
├── artifacts/           # Saved artifacts directory
├── config.json          # Model configuration
├── .env.example         # Environment variables template
└── README.md            # This file
```

## API Endpoints

- `GET /models` - Get available models
- `POST /chat` - Send a message (streaming response)
- `POST /upload` - Upload a PDF file
- `POST /download-pdf` - Download conversation as PDF
- `GET /artifacts` - List saved artifacts
- `GET /artifacts/{filename}` - Get artifact content

## License

MIT
