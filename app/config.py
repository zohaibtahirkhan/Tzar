"""
Central configuration for the local voice assistant.
All settings are overridable via environment variables or a .env file.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ─── Paths ────────────────────────────────────────────────────────────────
    base_dir: Path = BASE_DIR
    models_dir: Path = BASE_DIR / "models"
    data_dir: Path = BASE_DIR / "data"
    workspace_dir: Path = BASE_DIR / "AssistantWorkspace"
    memory_db: Path = BASE_DIR / "data" / "memory.db"
    conversations_dir: Path = BASE_DIR / "data" / "conversations"
    cache_dir: Path = BASE_DIR / "data" / "cache"
    log_dir: Path = BASE_DIR / "data" / "logs"

    # ─── LLM ──────────────────────────────────────────────────────────────────
    llm_model_path: Path = BASE_DIR / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
    llm_context_length: int = 4096
    llm_threads: int = 6          # leave 2 threads for OS / audio
    llm_max_tokens: int = 400
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_repeat_penalty: float = 1.1
    llm_stream: bool = True
    llm_model: str = 'qwen2.5:3b'
    LLM_BACKEND: str = "llamacpp"              # "ollama" or "llamacpp"
    LLM_OLLAMA_HOST: str = "http://localhost:11434"

    # ─── STT ──────────────────────────────────────────────────────────────────
    stt_model: str = "small"       # small supports Urdu + English
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str | None = None  # None = auto-detect
    stt_beam_size: int = 5
    stt_vad_filter: bool = True

    # ─── VAD ──────────────────────────────────────────────────────────────────
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 700
    vad_sample_rate: int = 16000
    vad_chunk_size: int = 512     # samples per chunk (32ms @ 16kHz)

    # ─── Wake Word ────────────────────────────────────────────────────────────
    wake_word_model: str = "hey_jarvis"   # closest built-in; custom model added later
    wake_word_threshold: float = 0.5
    wake_word_phrase: str = "Hey Jarvis"

    # ─── TTS ──────────────────────────────────────────────────────────────────
    tts_voice: str = "af_heart"    # Kokoro voice id
    tts_speed: float = 1.0
    tts_sample_rate: int = 24000

    # ─── Audio I/O ────────────────────────────────────────────────────────────
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_chunk_ms: int = 32       # milliseconds per read chunk
    audio_input_device: int | None = None   # None = system default
    audio_output_device: int | None = None

    # ─── Memory ───────────────────────────────────────────────────────────────
    memory_short_term_limit: int = 6   # last N conversation turns kept in context
    memory_long_term_enabled: bool = True

    # ─── Web Search ───────────────────────────────────────────────────────────
    web_search_enabled: bool = False    # disabled by default; user must request
    web_search_max_results: int = 5
    
    # ─── Obsidian ─────────────────────────────────────────────────────────────────
    obsidian_vault_path: Path = "/home/zohaib/Documents/notes"   # change to your vault path
    obsidian_daily_notes_folder: str = "Daily"
    obsidian_ai_notes_folder: str = "AI Notes"
    obsidian_conversations_folder: str = "Conversations"
    obsidian_ideas_folder: str = "Ideas"
    obsidian_projects_folder: str = "Projects"
    obsidian_embed_model: str = "all-MiniLM-L6-v2"   # for semantic search
    obsidian_vector_db: Path = BASE_DIR / "data" / "obsidian_vectors.db"

    # ─── Filesystem Sandbox ───────────────────────────────────────────────────
    allowed_extensions: list[str] = [".txt", ".md", ".json", ".py", ".js", ".csv", ".yaml", ".toml", ".pdf"]

    # ─── API ──────────────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_log_level: str = "info"

    # ─── Performance targets (ms) ─────────────────────────────────────────────
    target_wake_latency_ms: int = 200
    target_stt_latency_ms: int = 1000
    target_llm_first_token_ms: int = 1500
    target_tts_start_ms: int = 500
    target_full_response_ms: int = 4000
    
    # ─── RAG — Local Document Index ─────────────────────────────────
    rag_documents_dir: Path = BASE_DIR / "Documents"   # default ingest folder
    rag_chunk_words: int = 200
    rag_chunk_overlap: int = 40

    # ─── Knowledge Graph  ────────────────────────────────────────────
    kg_db: Path = BASE_DIR / "data" / "knowledge_graph.db"
    kg_auto_extract: bool = True     # auto-extract entities on obsidian_create_note

    # ─── MCP  ────────────────────────────────────────────────────────
    mcp_enabled: bool = False        # set True when you add servers
    mcp_servers: str = "[]"          # JSON array of server configs (see mcp_client.py)
    
    # ─── Memory Scoring  ────────────────────────────────────────────
    memory_score_weight_importance: float = 0.45
    memory_score_weight_recency:    float = 0.30
    memory_score_weight_confidence: float = 0.15
    memory_score_weight_frequency:  float = 0.10
    memory_recency_half_life_days:  float = 14.0
    memory_prune_threshold:         float = 0.08
    memory_prune_min_age_days:      int   = 7

    # ─── Projects ──────────────────────────────────────────────────
    projects_db: Path = BASE_DIR / "data" / "projects.db"
    project_auto_detect: bool = True   # detect "continue X" in queries

    # ─── Skill Auto-Learning ───────────────────────────────────────
    skill_learning_enabled:   bool = True
    skill_pattern_threshold:  int  = 3     # times seen before proposing
    skill_min_sequence_len:   int  = 2
 
    # ─── Multi-Agent Orchestrator ───────────────────────────────────────   
    multi_agent_enabled: bool = False
    
    multi_agent_experimental: bool = False   # enables old 5-agent orchestrator
    goal_tracking_enabled:    bool = True    # persist multi-step goals to DB
    critic_enabled:           bool = True    # enable Critic layer in Coordinator

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

# Ensure required directories exist
for _dir in [
    settings.models_dir,
    settings.data_dir,
    settings.workspace_dir,
    settings.conversations_dir,
    settings.cache_dir,
    settings.log_dir,
]:
    _dir.mkdir(parents=True, exist_ok=True)
