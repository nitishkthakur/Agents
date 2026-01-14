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
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> str:
    """Run a web search using Tavily."""
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
    """Save an artifact file to disk."""
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
    """Read an artifact file from disk."""
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

When the user asks you to create or save something, use the save_artifact tool.
When searching for information, use the internet_search tool.

Be helpful, accurate, and thorough in your responses."""


def create_agent_for_model(model_id: str):
    """Create a deep agent with the specified model."""
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
            
            async for event in agent.astream_events(
                {"messages": messages},
                version="v2"
            ):
                kind = event.get("event")
                
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        if isinstance(content, str):
                            full_response += content
                            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"
                
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "")
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool_name})}\n\n"
            
            conversations[conversation_id]["messages"].append({
                "role": "assistant",
                "content": full_response
            })
            
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
            
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
        content = msg["content"].replace("\n", "<br/>")
        content = content.replace("<", "&lt;").replace(">", "&gt;").replace("<br/>", "<br/>")
        
        if msg["role"] == "user":
            story.append(Paragraph(f"<b>{role}:</b>", styles['Normal']))
            story.append(Paragraph(content, user_style))
        else:
            story.append(Paragraph(f"<b>{role}:</b>", styles['Normal']))
            story.append(Paragraph(content, assistant_style))
        
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
