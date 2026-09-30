"""
Entry point for the local voice assistant.

Usage:
  # Start the API server (for text/WebSocket access)
  python -m app.main --mode server

  # Start the full voice loop (mic + speaker)
  python -m app.main --mode voice

  # Interactive terminal chat
  python -m app.main --mode terminal

  # Run a single query and exit
  python -m app.main --mode query --query "What files are in my workspace?"
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Make sure project root is on sys.path when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from app.utils.logging import setup_logging
from app.config import settings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local Offline Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["server", "voice", "terminal", "query"],
        default="terminal",
        help="Operating mode (default: terminal)",
    )
    parser.add_argument("--query", type=str, help="Query string for --mode query")
    parser.add_argument("--host", type=str, default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()

async def initialize_mcp():
    """Connect to MCP servers if enabled."""
    if settings.mcp_enabled:
        from app.tools.mcp_client import mcp_registry
        mcp_registry.configure_from_env()
        results = await mcp_registry.connect_all()
        for srv, ok in results.items():
            logger.info("MCP '{}': {}", srv, "connected" if ok else "FAILED")

def load_core_models():
    """Load LLM, STT, and TTS models synchronously at startup."""
    from app.llm.engine import llm_engine
    from app.audio.stt import stt_engine
    from app.audio.tts import tts_engine

    logger.info("Loading core models...")

    try:
        llm_engine.load()   # for Ollama: checks it's running and the model is pulled
    except Exception as e:
        logger.error("LLM load failed: {}", e)

    try:
        stt_engine.load()
    except Exception as e:
        logger.error("STT load failed: {}", e)

    try:
        tts_engine.load()
    except Exception as e:
        logger.error("TTS load failed (non-critical): {}", e)

    # Warm the embedding model too: it takes ~15s cold, and the first note or
    # document search (including the needs_rag prompt prefetch) would otherwise
    # stall on it mid-conversation.
    try:
        from app.tools.obsidian import _get_embed_model
        _get_embed_model()
    except Exception as e:
        logger.error("Embedding model load failed (semantic search degraded): {}", e)

    logger.info("Models loaded.")


def load_voice_models():
    """Additional models needed for voice mode."""
    from app.audio.vad import vad_engine
    from app.audio.wake_word import wake_word_detector

    try:
        vad_engine.load()
    except Exception as e:
        logger.error("VAD load failed: {}", e)

    try:
        wake_word_detector.load()
    except Exception as e:
        logger.error("Wake word load failed: {}", e)


# ─── Mode: Server ─────────────────────────────────────────────────────────────

def run_server(host: str, port: int):
    import uvicorn
    from app.router import app

    logger.info("Starting API server at http://{}:{}", host, port)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=settings.api_log_level,
    )


# ─── Mode: Voice ──────────────────────────────────────────────────────────────

async def run_voice():
    from app.memory.manager import memory_manager
    from app.audio.microphone import microphone
    from app.pipeline import pipeline
    from app.memory.session_store import init_session_db
    
    await memory_manager.initialize()
    await init_session_db()
    await initialize_mcp()
    
    load_core_models()
    load_voice_models()

    from app.tools.obsidian_sync import start_vault_sync, stop_vault_sync
    await start_vault_sync()

    microphone.start()
    try:
        await pipeline.run_voice_loop()
    finally:
        microphone.stop()
        await stop_vault_sync()


# ─── Mode: Terminal ───────────────────────────────────────────────────────────

async def run_terminal():
    from app.memory.manager import memory_manager
    from app.pipeline import pipeline
    from app.memory.session_store import init_session_db
    
    await memory_manager.initialize()
    await init_session_db()
    await initialize_mcp()
    
    load_core_models()

    print("\n" + "═" * 55)
    print("  Local Voice Assistant — Terminal Mode")
    print("  Type your message and press Enter.")
    print("  Commands: 'quit' to exit, 'clear' to reset memory")
    print("  'enable web search' to allow online searches")
    print("═" * 55 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            memory_manager.short_term.clear()
            print("[Memory cleared]")
            continue

        if user_input.lower() == "enable web search":
            from app.tools.web_search import enable_web_search
            print(enable_web_search())
            continue

        if user_input.lower() == "disable web search":
            from app.tools.web_search import disable_web_search
            print(disable_web_search())
            continue

        try:
            print("Assistant: ", end="", flush=True)
            response = await pipeline.process_text_input(user_input)
            print(response)
            print()
        except Exception as e:
            logger.error("Pipeline error: {}", e)
            print(f"[Error: {e}]")


# ─── Mode: Query ──────────────────────────────────────────────────────────────

async def run_query(query: str):
    from app.memory.manager import memory_manager
    from app.pipeline import pipeline
    from app.memory.session_store import init_session_db
    
    await memory_manager.initialize()
    await init_session_db()
    await initialize_mcp()
    
    load_core_models()

    response = await pipeline.process_text_input(query)
    print(f"Assistant: {response}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.log_level)

    if args.mode == "server":
        run_server(args.host, args.port)

    elif args.mode == "voice":
        asyncio.run(run_voice())

    elif args.mode == "terminal":
        asyncio.run(run_terminal())

    elif args.mode == "query":
        if not args.query:
            print("Error: --query is required in query mode.")
            sys.exit(1)
        asyncio.run(run_query(args.query))


if __name__ == "__main__":
    main()
