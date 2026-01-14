# Devin Agent - Setup and Running Instructions

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.11 or higher
- pip (Python package manager)
- poppler-utils (required for PDF to image conversion)

To install poppler-utils on Ubuntu/Debian:
```bash
sudo apt-get install poppler-utils
```

On macOS with Homebrew:
```bash
brew install poppler
```

## Installation

1. Navigate to the devin_app directory:
```bash
cd devin_app
```

2. Install the backend dependencies:
```bash
cd backend
pip install -r requirements.txt
cd ..
```

3. Create your environment file with API keys:
```bash
cp .env.example .env
```

4. Edit the `.env` file and add your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

You only need to provide API keys for the models you plan to use. For example, if you only want to use Claude models, you only need the ANTHROPIC_API_KEY.

## Running the Application

### Option 1: Using the startup script (Recommended)

Simply run the startup script from the devin_app directory:
```bash
./start.sh
```

This will:
- Start the backend server on port 8000
- Start the frontend server on port 3000
- Open the application in your default browser (if available)

To stop the servers, press `Ctrl+C` in the terminal.

### Option 2: Manual startup

If you prefer to start the servers manually:

1. Start the backend server:
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

2. In a separate terminal, start the frontend server:
```bash
cd frontend
python -m http.server 3000
```

3. Open your browser and navigate to: http://localhost:3000

## Using the Application

### Chat Interface

- Type your message in the text box at the bottom and press Enter or click the send button
- The assistant will respond with streaming text
- Conversations are maintained across multiple messages

### Model Selection

- Use the dropdown in the sidebar to select different AI models
- Available models include Claude (Anthropic), GPT (OpenAI), and open-source models via Groq
- The selected model will be used for all subsequent messages in the conversation

### Web Search

- Toggle the "Web Search" checkbox to enable/disable web search capabilities
- When enabled, the agent can search the internet for current information

### File Upload

- Click the paperclip icon to upload a PDF file
- The PDF text content will be extracted and sent to the agent
- PDF pages are also converted to images for visual analysis

### Download Conversation

- Click "Download as PDF" in the sidebar to export the current conversation
- The PDF will include all messages from both you and the assistant

### View Artifacts

- Click "View Artifacts" to see files saved by the agent
- The agent can save markdown files, code, and other artifacts during conversations

## Configuration

### Models

The available models are configured in `config.json`. You can modify this file to add or remove models. Each model entry requires:
- `id`: The model identifier in the format `provider:model_name`
- `name`: Display name shown in the dropdown
- `provider`: The provider name (anthropic, openai, or groq)

### Artifacts Directory

Artifacts saved by the agent are stored in the `artifacts/` directory. You can access these files directly from your file system.

## Troubleshooting

### Backend fails to start

- Ensure all dependencies are installed: `pip install -r backend/requirements.txt`
- Check that the required ports (8000, 3000) are not in use
- Verify your `.env` file exists and contains valid API keys

### PDF upload not working

- Ensure poppler-utils is installed for PDF to image conversion
- Check that the file is a valid PDF

### Model errors

- Verify you have the correct API key for the selected model provider
- Check your API key has sufficient credits/quota

## Project Structure

```
devin_app/
├── backend/
│   ├── main.py              # FastAPI backend server
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Main HTML file
│   ├── styles.css           # Dark theme CSS styles
│   └── app.js               # Frontend JavaScript
├── artifacts/               # Saved artifacts directory
├── config.json              # Model configuration
├── .env.example             # Environment variables template
├── .env                     # Your environment variables (create this)
├── instructions.md          # This file
├── start.sh                 # Startup script
└── README.md                # Project overview
```
