import os
import json
import base64
import asyncio
import uuid
from pathlib import Path
from typing import Optional, Literal, AsyncGenerator
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, CompositeBackend, StateBackend
from tavily import TavilyClient

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
import re

load_dotenv()

app = FastAPI(title="Devin Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config.json"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOADS_DIR = BASE_DIR / "uploads"
EXPORTS_DIR = BASE_DIR / "exports"

ARTIFACTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

conversations: dict = {}


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def get_tavily_client() -> Optional[TavilyClient]:
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        return TavilyClient(api_key=api_key)
    return None


def internet_search(
    query: str,
    max_results: int = 10,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> str:
    """Search the web using Tavily for current information.
    
    Args:
        query: Search query string.
        max_results: Number of results to return (default 10).
        topic: Category to search - general, news, or finance (default general).
        include_raw_content: Include raw HTML content in results (default False).
    """
    client = get_tavily_client()
    if not client:
        return "Web search is not available. TAVILY_API_KEY not configured."
    
    try:
        results = client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Search error: {str(e)}"


def save_artifact(filename: str, content: str) -> str:
    """Save content to an artifact file.
    
    Args:
        filename: Name of the file to save.
        content: Content to write to the file.
    """
    filepath = ARTIFACTS_DIR / filename
    with open(filepath, "w") as f:
        f.write(content)
    return f"Artifact saved to {filepath}"


def list_artifacts() -> str:
    """List all saved artifacts."""
    files = list(ARTIFACTS_DIR.glob("*"))
    if not files:
        return "No artifacts found."
    return "\n".join([f.name for f in files])


def read_artifact(filename: str) -> str:
    """Read content from an artifact file.
    
    Args:
        filename: Name of the file to read.
    """
    filepath = ARTIFACTS_DIR / filename
    if not filepath.exists():
        return f"Artifact {filename} not found."
    with open(filepath, "r") as f:
        return f.read()


SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools.

You can:
1. Search the web for current information using the internet_search tool
2. Save artifacts (markdown files, code, notes) to disk using save_artifact
3. List and read previously saved artifacts
4. Use the built-in filesystem tools for managing context
5. Spawn subagents when needed for specialized tasks - or if parallel execution of multiple tasks is required. 
6. If the user asks you to be careful or critique your task, use a subagent to give you targeted feedback on how to improve your work. Then, improve your answer based on the feedback.
When the user asks you to create or save something, use the save_artifact tool.
When searching for information, use the internet_search tool.

Be helpful, accurate, and thorough in your responses."""


def create_agent_for_model(model_id: str):
    """Create a deep agent with the specified model."""
    # Handle Ollama models - they need base_url configuration
    if model_id.startswith("ollama:"):
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = init_chat_model(
            model=model_id,
            base_url=ollama_base_url
        )
    else:
        model = init_chat_model(model=model_id)
    
    tools = [internet_search, save_artifact, list_artifacts, read_artifact]
    
    filesystem_backend = FilesystemBackend(
        root_dir=str(ARTIFACTS_DIR),
        virtual_mode=True
    )
    
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        backend=filesystem_backend,
    )
    
    return agent


class ChatRequest(BaseModel):
    message: str
    model_id: str
    conversation_id: Optional[str] = None
    web_search_enabled: bool = True


class ConversationMessage(BaseModel):
    role: str
    content: str


class DownloadRequest(BaseModel):
    conversation_id: str


@app.get("/models")
async def get_models():
    """Get available models from config."""
    config = load_config()
    return {
        "models": config["models"],
        "default_model": config["default_model"]
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat request with streaming response."""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    if conversation_id not in conversations:
        conversations[conversation_id] = {
            "messages": [],
            "model_id": request.model_id
        }
    
    conversations[conversation_id]["messages"].append({
        "role": "user",
        "content": request.message
    })
    conversations[conversation_id]["model_id"] = request.model_id
    
    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            agent = create_agent_for_model(request.model_id)
            
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in conversations[conversation_id]["messages"]
            ]
            
            full_response = ""
            llm_call_count = 0
            current_step = ""
            
            # Get recursion limit from config
            config = load_config()
            recursion_limit = config.get("recursion_limit", 150)
            
            async for event in agent.astream_events(
                {"messages": messages},
                version="v2",
                config={"recursion_limit": recursion_limit}
            ):
                kind = event.get("event")
                
                if kind == "on_chat_model_start":
                    llm_call_count += 1
                    model_name = event.get("name", "LLM")
                    step_desc = f"Step {llm_call_count}: Processing with {model_name}"
                    yield f"data: {json.dumps({'type': 'progress', 'step': llm_call_count, 'description': step_desc})}\n\n"
                
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        # Handle string content (OpenAI, Groq, etc.)
                        if isinstance(content, str):
                            full_response += content
                            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                        # Handle list of content blocks (Claude/Anthropic models)
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text:
                                        full_response += text
                                        yield f"data: {json.dumps({'type': 'content', 'content': text})}\n\n"
                                elif hasattr(block, "text"):
                                    # Handle content block objects
                                    text = block.text
                                    if text:
                                        full_response += text
                                        yield f"data: {json.dumps({'type': 'content', 'content': text})}\n\n"
                
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    tool_input = event.get("data", {}).get("input", {})
                    # Extract a brief description of what the tool is doing
                    tool_desc = ""
                    if tool_name == "internet_search" and isinstance(tool_input, dict):
                        query = tool_input.get("query", "")
                        tool_desc = f"Searching: {query[:50]}..." if len(query) > 50 else f"Searching: {query}"
                    elif tool_name == "save_artifact" and isinstance(tool_input, dict):
                        filename = tool_input.get("filename", "")
                        tool_desc = f"Saving artifact: {filename}"
                    elif tool_name == "read_artifact" and isinstance(tool_input, dict):
                        filename = tool_input.get("filename", "")
                        tool_desc = f"Reading artifact: {filename}"
                    elif tool_name == "list_artifacts":
                        tool_desc = "Listing saved artifacts"
                    else:
                        tool_desc = f"Using {tool_name}"
                    
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name, 'description': tool_desc})}\n\n"
                
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "")
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool_name})}\n\n"
                
                elif kind == "on_chain_start":
                    chain_name = event.get("name", "")
                    if chain_name and chain_name not in ["RunnableSequence", "RunnableLambda"]:
                        yield f"data: {json.dumps({'type': 'chain_start', 'name': chain_name})}\n\n"
                
                elif kind == "on_chain_end":
                    chain_name = event.get("name", "")
                    if chain_name and chain_name not in ["RunnableSequence", "RunnableLambda"]:
                        yield f"data: {json.dumps({'type': 'chain_end', 'name': chain_name})}\n\n"
            
            conversations[conversation_id]["messages"].append({
                "role": "assistant",
                "content": full_response
            })
            
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id, 'total_steps': llm_call_count})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF file and extract text and images."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        file_id = str(uuid.uuid4())
        file_path = UPLOADS_DIR / f"{file_id}.pdf"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        text_content = ""
        images_base64 = []
        
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page in reader.pages:
                text_content += page.extract_text() or ""
                text_content += "\n\n"
        except Exception as e:
            text_content = f"Error extracting text: {str(e)}"
        
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(file_path, dpi=150)
            for i, img in enumerate(images[:10]):
                import io
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                images_base64.append({
                    "page": i + 1,
                    "data": img_base64
                })
        except Exception as e:
            pass
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "text_content": text_content.strip(),
            "images": images_base64,
            "page_count": len(images_base64) if images_base64 else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def format_content_for_pdf(content: str, base_style, styles):
    """
    Convert markdown and HTML formatted content to ReportLab story elements.
    Returns a list of Paragraph and Spacer objects.
    """
    story_elements = []
    
    # Convert HTML line breaks to actual newlines for processing
    content = content.replace("<br/>", "\n").replace("<br>", "\n")
    
    # Escape HTML entities - do angle brackets first, then ampersand
    # This prevents double-escaping of existing entities
    content = content.replace("<", "&lt;").replace(">", "&gt;")
    content = content.replace("&", "&amp;")
    
    # Now unescape the tags we want to use for formatting
    content = content.replace("&amp;lt;b&amp;gt;", "<b>").replace("&amp;lt;/b&amp;gt;", "</b>")
    content = content.replace("&amp;lt;i&amp;gt;", "<i>").replace("&amp;lt;/i&amp;gt;", "</i>")
    
    # Split content into lines to process line by line
    lines = content.split('\n')
    
    # Define styles for different heading levels
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=10,
        leftIndent=base_style.leftIndent,
    )
    
    h3_style = ParagraphStyle(
        'H3Style',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=8,
        leftIndent=base_style.leftIndent,
    )
    
    list_style = ParagraphStyle(
        'ListStyle',
        parent=base_style,
        leftIndent=base_style.leftIndent + 20,
        bulletIndent=base_style.leftIndent + 10,
    )
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines but add spacing
        if not line:
            i += 1
            continue
            
        # Handle markdown headings
        if line.startswith('## '):
            heading_text = line[3:].strip()
            story_elements.append(Paragraph(f"<b>{heading_text}</b>", h2_style))
        elif line.startswith('### '):
            heading_text = line[4:].strip()
            story_elements.append(Paragraph(f"<b>{heading_text}</b>", h3_style))
        # Handle list items
        elif line.startswith('- '):
            list_text = line[2:].strip()
            story_elements.append(Paragraph(f"• {list_text}", list_style))
        # Handle horizontal rules
        elif line.strip() == '---':
            story_elements.append(Spacer(1, 6))
        # Regular paragraph
        else:
            # Handle inline markdown formatting
            # Process in order: bold with **, bold with __, italic with *, italic with _
            # Use more specific patterns to avoid conflicts and improve efficiency
            line = re.sub(r'\*\*([^*]+?)\*\*', r'<b>\1</b>', line)
            line = re.sub(r'__([^_]+?)__', r'<b>\1</b>', line)
            # For italic, avoid single underscores that might be part of variable names
            # by requiring word boundaries or spaces
            line = re.sub(r'(?<!\w)\*([^*]+?)\*(?!\w)', r'<i>\1</i>', line)
            line = re.sub(r'(?<!\w)_([^_]+?)_(?!\w)', r'<i>\1</i>', line)
            
            story_elements.append(Paragraph(line, base_style))
        
        i += 1
    
    return story_elements


@app.post("/download-pdf")
async def download_conversation_pdf(request: DownloadRequest):
    """Download conversation as PDF."""
    conversation_id = request.conversation_id
    
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = conversations[conversation_id]
    
    pdf_filename = f"conversation_{conversation_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = EXPORTS_DIR / pdf_filename
    
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    
    user_style = ParagraphStyle(
        'UserStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor='#1a1a2e',
        spaceAfter=12,
        leftIndent=20,
    )
    
    assistant_style = ParagraphStyle(
        'AssistantStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor='#16213e',
        spaceAfter=12,
        leftIndent=20,
    )
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
    )
    
    story = []
    story.append(Paragraph("Conversation Export", title_style))
    story.append(Paragraph(f"Model: {conversation.get('model_id', 'Unknown')}", styles['Normal']))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    for msg in conversation["messages"]:
        role = msg["role"].upper()
        content = msg["content"]
        
        # Add role label
        story.append(Paragraph(f"<b>{role}:</b>", styles['Normal']))
        
        # Use the appropriate style for the role
        base_style = user_style if msg["role"] == "user" else assistant_style
        
        # Format content and add to story
        formatted_elements = format_content_for_pdf(content, base_style, styles)
        story.extend(formatted_elements)
        
        story.append(Spacer(1, 10))
    
    doc.build(story)
    
    return FileResponse(
        path=str(pdf_path),
        filename=pdf_filename,
        media_type="application/pdf"
    )


@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversations[conversation_id]


@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_id in conversations:
        del conversations[conversation_id]
    return {"status": "deleted"}


@app.get("/artifacts")
async def get_artifacts():
    """List all artifacts."""
    files = list(ARTIFACTS_DIR.glob("*"))
    return {
        "artifacts": [
            {
                "name": f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            }
            for f in files if f.is_file()
        ]
    }


@app.get("/artifacts/{filename}")
async def get_artifact(filename: str):
    """Get artifact content."""
    filepath = ARTIFACTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    with open(filepath, "r") as f:
        content = f.read()
    
    return {"filename": filename, "content": content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
