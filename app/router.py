"""
FastAPI backend.

Endpoints:
  GET  /health            — system health check
  POST /chat              — text chat (non-streaming)
  POST /chat/stream       — text chat (streaming SSE)
  POST /tool              — direct tool invocation
  GET  /memory            — query long-term memory
  POST /memory            — save to long-term memory
  DELETE /memory/{id}     — delete a memory entry
  WS   /ws/audio          — WebSocket streaming audio input/output
  POST /settings/web-search — enable/disable web search
"""
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from app.config import settings
from app.utils.logging import setup_logging


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup; clean up on shutdown."""
    setup_logging(settings.api_log_level.upper())
    logger.info("Starting assistant backend...")

    from app.memory.manager import memory_manager
    await memory_manager.initialize()

    from app.llm.engine import llm_engine
    if not llm_engine.is_loaded():
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, llm_engine.load)
        except FileNotFoundError as e:
            logger.error("LLM model not found: {}", e)

    from app.audio.stt import stt_engine
    if not stt_engine.is_loaded():
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, stt_engine.load)
        except Exception as e:
            logger.error("STT load failed: {}", e)

    from app.audio.tts import tts_engine
    if not tts_engine.is_loaded():
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, tts_engine.load)
        except Exception as e:
            logger.error("TTS load failed: {}", e)

    # ─── MCP Startup ───────────────────────────────────────────────
    if settings.mcp_enabled:
        from app.tools.mcp_client import mcp_registry
        mcp_registry.configure_from_env()
        results = await mcp_registry.connect_all()
        for srv, ok in results.items():
            logger.info("MCP '{}': {}", srv, "connected" if ok else "FAILED")
    # ────────────────────────────────────────────────────────────────

    logger.info("Backend ready.")
    yield

    # ─── MCP Shutdown ───────────────────────────────────────────────
    if settings.mcp_enabled:
        from app.tools.mcp_client import mcp_registry
        await mcp_registry.disconnect_all()
        logger.info("MCP servers disconnected.")
    # ────────────────────────────────────────────────────────────────

    logger.info("Shutting down...")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Local Voice Assistant API",
    description="Fully local, offline-first voice assistant backend.",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS Configuration ───────────────────────────────────────────────────────
# Configure allowed origins for API access
# For production: restrict to specific domains only
# For development: include localhost and common dev ports

ALLOWED_ORIGINS = [
    "http://localhost:3000",      # React dev server
    "http://127.0.0.1:3000",
    "http://localhost:5173",      # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:8080",      # Alternative dev port
    "http://127.0.0.1:8080",
]

# Add production origins from environment variable if set
if hasattr(settings, 'allowed_origins') and settings.allowed_origins:
    ALLOWED_ORIGINS.extend(settings.allowed_origins.split(','))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)


# ─── Models ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)

class ChatResponse(BaseModel):
    response: str
    tool_results: list[dict] = []

class ToolRequest(BaseModel):
    tool: str
    params: dict = {}

class MemorySaveRequest(BaseModel):
    category: str
    content: str

class WebSearchToggle(BaseModel):
    enabled: bool

class MultiAgentToggle(BaseModel):
    enabled: bool

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    from app.llm.engine import llm_engine
    from app.audio.stt import stt_engine
    from app.audio.tts import tts_engine
    from app.audio.wake_word import wake_word_detector

    return {
        "status":       "ok",
        "llm":          "loaded" if llm_engine.is_loaded()         else "not loaded",
        "stt":          "loaded" if stt_engine.is_loaded()         else "not loaded",
        "tts":          "loaded" if tts_engine.is_loaded()         else "not loaded",
        "wake_word":    "loaded" if wake_word_detector.is_loaded() else "failed",
        "web_search":   settings.web_search_enabled,
        "multi_agent":  settings.multi_agent_enabled,
        "workspace":    str(settings.workspace_dir),
    }


# ─── Chat ─────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    from app.pipeline import pipeline
    from app.tools.router import extract_tool_calls

    try:
        response = await pipeline.process_text_input(req.message)
        return ChatResponse(response=response)
    except Exception as e:
        logger.error("/chat error: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Server-Sent Events streaming response."""
    from app.pipeline import pipeline

    async def event_generator():
        tool_results_collected: list[dict] = []
        try:
            async for chunk in pipeline.process_text_input_streaming(req.message):
                # Status tokens like "[Running tool_name...]" — parse tool result
                # signals embedded in the stream and collect them
                if chunk.startswith("[Running ") and chunk.endswith("...]"):
                    tool_name = chunk[9:-4]
                    # Emit a status token for the UI to display
                    data = json.dumps({"token": chunk})
                    yield f"data: {data}\n\n"
                else:
                    data = json.dumps({"token": chunk})
                    yield f"data: {data}\n\n"
 
            # After all tokens are streamed, emit any collected tool results
            # so the UI can render expandable tool result rows.
            # The pipeline stores the last turn's tool results on itself.
            if hasattr(pipeline, "_last_tool_results") and pipeline._last_tool_results:
                yield f"data: {json.dumps({'tool_results': pipeline._last_tool_results})}\n\n"
 
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error("/chat/stream error: {}", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Tool ─────────────────────────────────────────────────────────────────────

@app.post("/tool")
async def invoke_tool(req: ToolRequest):
    """Directly invoke a tool (for testing/automation)."""
    from app.tools.router import ToolRouter
    from app.memory.manager import memory_manager

    router = ToolRouter(memory_manager=memory_manager)
    call = {"tool": req.tool, **req.params}
    result = await router.dispatch(call)
    return result


# ─── Memory ───────────────────────────────────────────────────────────────────

@app.get("/memory")
async def query_memory(q: str = Query(default=""), category: str = Query(default="")):
    from app.memory.manager import memory_manager

    if category:
        memories = await memory_manager.long_term.recall_by_category(category)
    elif q:
        memories = await memory_manager.long_term.recall(q)
    else:
        memories = await memory_manager.long_term.recall_by_category("preference")
        memories += await memory_manager.long_term.recall_by_category("note")

    return {"memories": memories}


@app.post("/memory")
async def save_memory(req: MemorySaveRequest):
    from app.memory.manager import memory_manager

    mem_id = await memory_manager.long_term.save(req.category, req.content)
    return {"id": mem_id, "status": "saved"}


@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: int):
    from app.memory.manager import memory_manager

    ok = await memory_manager.long_term.delete(memory_id)
    return {"deleted": ok}


# ─── Settings ─────────────────────────────────────────────────────────────────

@app.post("/settings/web-search")
async def toggle_web_search(req: WebSearchToggle):
    settings.web_search_enabled = req.enabled
    status = "enabled" if req.enabled else "disabled"
    logger.info("Web search {} via API", status)
    return {"web_search": req.enabled, "status": status}

@app.post("/settings/multi-agent")
async def toggle_multi_agent(req: MultiAgentToggle):
    settings.multi_agent_enabled = req.enabled
    status = "enabled" if req.enabled else "disabled"
    logger.info("Multi-agent mode {} via API", status)
    return {"multi_agent": req.enabled, "status": status}


# ─── Cache Management ─────────────────────────────────────────────────────────

@app.get("/cache/stats")
async def cache_stats():
    """Get response cache statistics."""
    from app.cache import response_cache
    return response_cache.stats()


@app.post("/cache/clear")
async def cache_clear():
    """Clear the entire response cache."""
    from app.cache import response_cache
    count = await response_cache.clear()
    return {"cleared": count, "status": "success"}


@app.post("/cache/prune")
async def cache_prune():
    """Remove expired entries from cache."""
    from app.cache import response_cache
    count = await response_cache.prune_expired()
    return {"pruned": count, "status": "success"}


@app.post("/cache/toggle")
async def cache_toggle(enabled: bool):
    """Enable or disable response caching."""
    from app.cache import response_cache
    if enabled:
        response_cache.enable()
    else:
        response_cache.disable()
    return {"enabled": enabled, "status": "success"}


# ─── WebSocket Audio ──────────────────────────────────────────────────────────

@app.websocket("/ws/audio")
async def websocket_audio(ws: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.

    Protocol (client → server):
      Binary frames: raw 16kHz float32 PCM audio chunks
      Text frames: JSON control messages {"cmd": "stop"}

    Protocol (server → client):
      Text frames: JSON {"type": "transcript"|"response"|"tool_result"|"error", "data": ...}
      Binary frames: raw 24kHz float32 PCM TTS audio (if TTS enabled)
    """
    await ws.accept()
    logger.info("WebSocket audio connection opened")

    from app.audio.vad import VADEngine, SpeechCollector
    from app.audio.stt import stt_engine
    from app.audio.tts import tts_engine
    from app.pipeline import pipeline

    import numpy as np

    local_vad = VADEngine()
    local_vad.load()
    collector = SpeechCollector(local_vad)

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message and message["bytes"]:
                raw = message["bytes"]
                chunk = np.frombuffer(raw, dtype=np.float32)
                utterance = collector.push(chunk)

                if utterance is not None:
                    # Transcribe
                    await ws.send_text(json.dumps({"type": "status", "data": "transcribing"}))
                    user_text = await stt_engine.transcribe_async(utterance)

                    if not user_text.strip():
                        continue

                    await ws.send_text(json.dumps({"type": "transcript", "data": user_text}))

                    # LLM
                    await ws.send_text(json.dumps({"type": "status", "data": "thinking"}))
                    response = await pipeline.process_text_input(user_text)
                    await ws.send_text(json.dumps({"type": "response", "data": response}))

                    # TTS
                    if tts_engine.is_loaded():
                        async for audio_chunk in tts_engine.stream_sentences(response):
                            await ws.send_bytes(audio_chunk.tobytes())

            elif "text" in message and message["text"]:
                try:
                    ctrl = json.loads(message["text"])
                    if ctrl.get("cmd") == "stop":
                        break
                    elif ctrl.get("cmd") == "text":
                        # Text input over WebSocket
                        user_text = ctrl.get("data", "")
                        if user_text:
                            response = await pipeline.process_text_input(user_text)
                            await ws.send_text(json.dumps({"type": "response", "data": response}))
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("WebSocket audio disconnected")
    except Exception as e:
        logger.error("WebSocket error: {}", e)
        try:
            await ws.send_text(json.dumps({"type": "error", "data": str(e)}))
        except Exception:
            pass
