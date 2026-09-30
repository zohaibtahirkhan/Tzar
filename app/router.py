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
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from starlette.websockets import WebSocketClose
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

    # Warm the embedding model so the first search (or needs_rag prefetch)
    # doesn't stall ~15s mid-conversation, then start vault sync.
    from app.tools.obsidian import _get_embed_model
    await asyncio.get_event_loop().run_in_executor(None, _get_embed_model)

    from app.tools.obsidian_sync import start_vault_sync, stop_vault_sync
    await start_vault_sync()

    logger.info("Backend ready.")
    yield

    await stop_vault_sync()

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
    # Tauri v2 webview origins for the packaged desktop app
    "tauri://localhost",          # macOS / Linux
    "http://tauri.localhost",     # Windows (WebView2)
    "https://tauri.localhost",
]

# Add production origins from environment variable if set
if hasattr(settings, 'allowed_origins') and settings.allowed_origins:
    for origin in settings.allowed_origins.split(','):
        origin = origin.strip()
        if not origin:
            continue
        # A wildcard here would let any website the user visits drive this API,
        # which exposes the filesystem and memory tools. Refuse it.
        if origin == "*":
            logger.warning(
                "Ignoring ALLOWED_ORIGINS='*' — a CORS wildcard would let any "
                "site reach this local API and its file/memory tools. "
                "List the exact origins you need instead."
            )
            continue
        ALLOWED_ORIGINS.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1", settings.api_host} | {
    urlsplit(o).hostname for o in ALLOWED_ORIGINS
}


class LocalOnlyMiddleware:
    """
    CORS does not apply to WebSockets, and a DNS-rebound page is same-origin to
    the browser — either way any website could drive the file/memory tools.
    Refuse browser requests from unknown origins and requests for a Host that
    isn't ours. Non-browser clients (curl, scripts) send no Origin and pass.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope["headers"])
            origin = headers.get(b"origin", b"").decode("latin-1")
            host = urlsplit("//" + headers.get(b"host", b"").decode("latin-1")).hostname
            if (origin and origin not in ALLOWED_ORIGINS) or host not in ALLOWED_HOSTS:
                logger.warning("Refused {} from origin={!r} host={!r}", scope["type"], origin, host)
                if scope["type"] == "websocket":
                    return await WebSocketClose(code=1008)(scope, receive, send)
                return await PlainTextResponse("Forbidden origin or host", status_code=403)(scope, receive, send)
        await self.app(scope, receive, send)


app.add_middleware(LocalOnlyMiddleware)   # added last = runs first, before CORS


# ─── Models ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)

class ChatResponse(BaseModel):
    response: str
    tool_results: list[dict] = []
    sources: list[dict] = []      # notes/documents the answer was grounded in

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

    llm_status = "loaded" if llm_engine.is_loaded() else "not loaded"
    if settings.LLM_BACKEND == "ollama":
        # Ollama runs separately and can stop or lack the model — check it for real.
        llm_status = await asyncio.to_thread(llm_engine.ollama_problem) or "loaded"

    return {
        "status":       "ok",
        "llm":          llm_status,
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

    try:
        result = await pipeline.run_turn(req.message)
    except Exception as e:
        logger.exception("/chat error")
        raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(response=result.response, tool_results=result.tool_results, sources=result.sources)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Server-Sent Events. Each event is one JSON object:
      {"token": text}                        status lines and answer text, in order
      {"tool_results": [...], "sources": [...]}   once, after the answer
      {"error": message}
    then the literal `[DONE]`.
    """
    from app.pipeline import pipeline

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    async def event_generator():
        try:
            async for event in pipeline.stream(req.message):
                if event.kind in ("status", "text"):
                    yield sse({"token": event.text})
                elif event.kind == "done":
                    yield sse({"tool_results": event.result.tool_results, "sources": event.result.sources})
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("/chat/stream error")
            yield sse({"error": str(e)})

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

    from app.audio.vad import vad_engine, SpeechCollector
    from app.audio.stt import stt_engine
    from app.audio.tts import tts_engine
    from app.pipeline import pipeline

    import numpy as np

    collector = SpeechCollector(vad_engine)

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message and message["bytes"]:
                raw = message["bytes"]
                chunk = np.frombuffer(raw, dtype=np.float32)
                utterance = collector.push(chunk)   # re-framed to VAD-sized frames inside

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
