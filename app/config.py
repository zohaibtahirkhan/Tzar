"""
Central configuration for the local voice assistant.
All settings are overridable via environment variables or a .env file.
Platform-aware configuration for Windows, macOS, and Linux support.
"""
import os
import platform
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

# Platform detection
SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"

# Platform-specific defaults
# AUDIO_BACKEND values: "auto" | "sounddevice" | "pyaudio" — consumed by
# app/audio/microphone.py. "auto" prefers sounddevice, which ships prebuilt
# PortAudio wheels on every platform, and falls back to pyaudio.
if IS_WINDOWS:
    # Windows default paths
    DEFAULT_OBSIDIAN_PATH = Path(os.path.expanduser("~/Documents/Obsidian Vault"))
    PATH_SEPARATOR = "\\"
    AUDIO_BACKEND = "sounddevice"  # pyaudio needs a toolchain Windows rarely has
else:
    # Unix-like systems (macOS/Linux)
    DEFAULT_OBSIDIAN_PATH = Path(os.path.expanduser("~/Documents/notes"))
    PATH_SEPARATOR = "/"
    AUDIO_BACKEND = "auto"

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ─── Platform Information ────────────────────────────────────────────────
    platform_system: str = SYSTEM
    is_windows: bool = IS_WINDOWS
    is_macos: bool = IS_MACOS
    is_linux: bool = IS_LINUX
    path_separator: str = PATH_SEPARATOR
    audio_backend: str = AUDIO_BACKEND
    
    # ─── Paths (Platform-aware) ──────────────────────────────────────────────
    base_dir: Path = BASE_DIR
    models_dir: Path = BASE_DIR / "models"
    data_dir: Path = BASE_DIR / "data"
    workspace_dir: Path = BASE_DIR / "AssistantWorkspace"
    memory_db: Path = BASE_DIR / "data" / "memory.db"
    log_dir: Path = BASE_DIR / "data" / "logs"

    # ─── LLM ──────────────────────────────────────────────────────────────────
    llm_model_path: Path = BASE_DIR / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
    llm_context_length: int = 8192  # Increased for Ollama 7B model
    llm_threads: int = 6          # leave 2 threads for OS / audio
    llm_max_tokens: int = 512     # Increased back to reasonable value
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_repeat_penalty: float = 1.1
    llm_n_gpu_layers: int = 0     # llamacpp only: 0 = CPU, -1 = all layers on GPU (Metal/CUDA)
    llm_model: str = 'qwen2.5:3b'
    LLM_BACKEND: str = "ollama"              # "ollama" or "llamacpp"
    LLM_OLLAMA_HOST: str = "http://localhost:11434"

    # ─── STT ──────────────────────────────────────────────────────────────────
    stt_model: str = "small"       # small supports Urdu + English
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str | None = None  # None = auto-detect (blank in .env means the same)
    stt_beam_size: int = 5
    stt_vad_filter: bool = True

    # ─── VAD ──────────────────────────────────────────────────────────────────
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 700
    vad_sample_rate: int = 16000
    # Give up listening this long after a wake word if no utterance arrives, so
    # a false trigger can't strand the assistant in listening mode.
    wake_listen_timeout_s: float = 10.0

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
    obsidian_vault_path: Path = DEFAULT_OBSIDIAN_PATH   # Platform-specific default
    obsidian_daily_notes_folder: str = "Daily"
    obsidian_ai_notes_folder: str = "AI Notes"
    obsidian_ideas_folder: str = "Ideas"
    obsidian_projects_folder: str = "Projects"
    obsidian_embed_model: str = "all-MiniLM-L6-v2"   # for semantic search
    obsidian_vector_db: Path = BASE_DIR / "data" / "obsidian_vectors.db"
    # Vault sync: keep the search index and knowledge graph current when notes
    # are edited in Obsidian itself, not only through the assistant.
    obsidian_watch_enabled: bool = True
    obsidian_watch_debounce_s: float = 3.0    # quiet period before a changed note is re-indexed
    obsidian_backfill_on_start: bool = True   # reconcile the whole vault once at startup

    # ─── Filesystem Sandbox ───────────────────────────────────────────────────
    allowed_extensions: list[str] = [".txt", ".md", ".json", ".py", ".js", ".csv", ".yaml", ".toml", ".pdf"]

    # ─── API ──────────────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_log_level: str = "info"
    allowed_origins: str = ""  # Comma-separated additional origins for CORS

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
 
    # ─── Coordinator (multi-step tasks) ────────────────────────────────
    multi_agent_enabled: bool = False
    
    goal_tracking_enabled:    bool = True    # persist multi-step goals to DB
    critic_enabled:           bool = True    # enable Critic layer in Coordinator

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # A blank line in .env ("AUDIO_INPUT_DEVICE=") means "use the default",
    # not an unparseable int.
    @field_validator("stt_language", "audio_input_device", "audio_output_device", mode="before")
    @classmethod
    def _blank_is_default(cls, value):
        return value or None

    @field_validator("*", mode="after")
    @classmethod
    def _expand_home(cls, value):
        # "~/notes" from .env would otherwise become a literal "~" directory.
        return value.expanduser() if isinstance(value, Path) else value


settings = Settings()


def get_platform_optimizations() -> dict:
    """Return platform-specific optimization recommendations."""
    import multiprocessing
    
    optimizations = {
        "platform": SYSTEM,
        "cpu_cores": multiprocessing.cpu_count(),
        "recommended_threads": max(2, multiprocessing.cpu_count() - 2),
        "audio_backend": AUDIO_BACKEND,
    }
    
    if IS_WINDOWS:
        optimizations.update({
            "llm_backend": "ollama",  # Recommended for Windows
            "audio_advice": "Use sounddevice backend for better compatibility",
            "path_advice": "Use raw strings for paths: r'C:\\Users\\...'",
            "performance_tips": [
                "Use Ollama instead of llama.cpp for easier setup",
                "Run PowerShell as Administrator for build operations",
                "Use Windows Terminal instead of cmd.exe",
            ]
        })
    elif IS_MACOS:
        optimizations.update({
            "llm_backend": "ollama" if platform.machine() == "arm64" else "llamacpp",
            "metal_support": platform.machine() == "arm64",
            "performance_tips": [
                "Enable Metal acceleration for Apple Silicon",
                "Use CPU pinning for better performance",
            ]
        })
    else:  # Linux
        optimizations.update({
            "llm_backend": "llamacpp",
            "performance_tips": [
                "Consider CUDA/ROCm for GPU acceleration",
                "Use CPU affinity for multi-core systems",
                "Adjust OMP_NUM_THREADS for OpenMP optimizations",
            ]
        })
    
    return optimizations


def format_path_for_platform(path: str) -> str:
    """Convert path separators for the current platform."""
    if IS_WINDOWS:
        # Convert forward slashes to backslashes for Windows
        return path.replace("/", "\\")
    else:
        # Ensure forward slashes for Unix-like systems
        return path.replace("\\", "/")


# Ensure required directories exist
for _dir in [
    settings.models_dir,
    settings.data_dir,
    settings.workspace_dir,
    settings.log_dir,
]:
    _dir.mkdir(parents=True, exist_ok=True)
