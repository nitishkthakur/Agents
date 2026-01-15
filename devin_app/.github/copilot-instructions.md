# Devin Agent - AI Coding Assistant Instructions

## Architecture Overview

This is a **full-stack AI agent chat application** built with FastAPI (backend) and vanilla JavaScript (frontend). The core is powered by LangChain's `create_deep_agent` function from the `deepagents` package, which provides:

- **Multi-model support**: Integrates Anthropic, OpenAI, and Groq models via `langchain.chat_models.init_chat_model()`
- **Streaming architecture**: Uses Server-Sent Events (SSE) for real-time response streaming
- **Agent capabilities**: Equipped with custom tools (web search, artifact management) and built-in filesystem operations via `FilesystemBackend`
- **Stateful conversations**: In-memory conversation storage keyed by `conversation_id` (UUID-based)

### Key Components

```
backend/main.py       → FastAPI server, agent initialization, SSE streaming, tools
frontend/app.js       → Chat UI, SSE consumption, markdown/LaTeX rendering (marked.js, KaTeX)
config.json           → Model definitions (id, name, provider)
artifacts/            → Persistent storage for agent-saved files
```

## Development Workflow

### Starting the Application

**Preferred method**: Use `./start.sh` from the project root. It handles:
- Backend server on port 8000 (uvicorn)
- Frontend server on port 3000 (Python HTTP server)
- Graceful shutdown on Ctrl+C

**Manual alternative** (requires two terminals):
```bash
# Terminal 1: Backend
cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend && python -m http.server 3000
```

### Environment Setup

1. **System dependency**: `poppler-utils` (required for PDF to image conversion)
   - Ubuntu/Debian: `sudo apt-get install poppler-utils`
   - macOS: `brew install poppler`

2. **Python dependencies**: Install from `backend/requirements.txt`

3. **API keys**: Create `.env` in project root with:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   OPENAI_API_KEY=sk-...
   GROQ_API_KEY=gsk_...
   TAVILY_API_KEY=tvly-...
   ```

## Code Patterns & Conventions

### Agent Creation Pattern

Agents are created **per-request** with model-specific initialization:

```python
def create_agent_for_model(model_id: str):
    # Special handling for Ollama models with custom base_url
    if model_id.startswith("ollama:"):
        model = init_chat_model(model=model_id, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    else:
        model = init_chat_model(model=model_id)
    
    # Attach tools and filesystem backend
    agent = create_deep_agent(
        model=model,
        tools=[internet_search, save_artifact, list_artifacts, read_artifact],
        system_prompt=SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=str(ARTIFACTS_DIR), virtual_mode=True)
    )
```

### Streaming Event Handling

The `/chat` endpoint uses `agent.astream_events(version="v2")` with specific event types:

- `on_chat_model_start` → Track LLM call steps
- `on_chat_model_stream` → Stream content chunks (handle both string and list formats for different providers)
- `on_tool_start` / `on_tool_end` → Tool execution notifications
- Send SSE-formatted JSON: `data: {JSON}\n\n`

**Critical**: Handle provider-specific content formats:
- OpenAI/Groq: `chunk.content` is a string
- Anthropic: `chunk.content` is a list of content blocks with `.text` or `dict["text"]`

### Tool Definition Pattern

Tools are Python functions with docstrings (used by the agent for understanding):

```python
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> str:
    """Run a web search using Tavily."""
    # Implementation...
```

### Frontend-Backend Communication

- **API calls**: Fetch API to `http://localhost:8000`
- **Message format**: Send `ChatRequest` with `message`, `model_id`, `conversation_id`, `web_search_enabled`
- **SSE consumption**: `EventSource` not used; custom fetch + text stream parsing
- **Rendering**: `marked.js` for markdown, `highlight.js` for code, `KaTeX` for LaTeX

## File Organization

### Adding New Models

1. Edit `config.json` → Add to `models` array with `id`, `name`, `provider`
2. Model ID format: `{provider}:{model_name}` (e.g., `anthropic:claude-sonnet-4-5-20250929`)
3. No backend code changes needed (dynamic loading via `init_chat_model`)

### Adding New Tools

1. Define function in `backend/main.py` with clear docstring
2. Add to `tools` list in `create_agent_for_model()`
3. Update `SYSTEM_PROMPT` to document tool availability
4. Handle tool events in streaming loop for UI feedback

### Artifact Management

- **Save**: `save_artifact(filename, content)` writes to `artifacts/`
- **Read**: `read_artifact(filename)` reads from `artifacts/`
- **List**: `list_artifacts()` scans `artifacts/`
- **Access**: Frontend can fetch via `/artifacts/{filename}` endpoint

## Testing & Debugging

### Backend Logs

Uvicorn logs to console. Look for:
- Model initialization errors (API key issues)
- Agent execution traces (tool calls, steps)
- PDF processing errors (poppler-utils missing)

### Frontend Debugging

- Console logs track SSE events: `content`, `progress`, `tool_start`, `tool_end`, `error`, `done`
- Message rendering: Check `marked.js` and `KaTeX` initialization
- CORS issues: Backend allows all origins (`*`)

### Common Issues

1. **Streaming hangs**: Check provider-specific content format handling in `on_chat_model_stream`
2. **PDF upload fails**: Verify `poppler-utils` installed and `pdf2image` working
3. **Model not found**: Ensure API key set in `.env` and `config.json` ID matches provider format

## Dependencies

### Backend Core
- `deepagents` → Agent framework (wraps LangChain)
- `langchain-*` → Provider-specific integrations (anthropic, openai, groq)
- `tavily-python` → Web search tool
- `pypdf`, `pdf2image` → PDF text/image extraction

### Frontend Core
- `marked.js` → Markdown rendering
- `highlight.js` → Code syntax highlighting
- `KaTeX` → LaTeX math rendering

## Conversation State

Conversations stored in-memory at `conversations` dict:
```python
conversations[conversation_id] = {
    "messages": [{"role": "user/assistant", "content": "..."}],
    "model_id": "anthropic:claude-sonnet-4-5-20250929"
}
```

**Note**: Server restart clears all conversations. No database persistence.

## Customization Points

1. **System prompt**: Edit `SYSTEM_PROMPT` in `backend/main.py`
2. **Artifact location**: Change `ARTIFACTS_DIR` path
3. **Model list**: Update `config.json` (default model via `default_model` key)
4. **Styling**: Modify `frontend/styles.css` (dark theme)
5. **Max PDF pages**: Adjust `images[:10]` limit in upload handler
