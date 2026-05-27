# Local Offline Voice Assistant — Second Brain Edition

A fully local, privacy-first AI voice assistant with Obsidian integration and a Knowledge Graph engine. Runs entirely on-device. Nothing leaves your machine unless you explicitly enable web search.

---

## Hardware Target

| Component | Spec |
|---|---|
| CPU | Intel i5-1235U |
| RAM | 16 GB |
| GPU | Intel Integrated (CPU-only inference) |
| OS | Ubuntu 24.04 |

---

## Architecture

```
Mic
 ↓
Silero VAD  (end-of-utterance detection)
 ↓
OpenWakeWord  ("Hey Zohaib")
 ↓
Faster-Whisper STT
 ↓
Qwen2.5 3B Instruct  (llama.cpp, streaming)
 ↓
Tool Router ─┬── Filesystem Sandbox
             ├── Long-Term Memory (SQLite FTS5)
             ├── Obsidian Vault (notes, daily log, semantic search)
             ├── Knowledge Graph (entities, relations, pathfinding)
             └── Web Search (opt-in, DuckDuckGo)
 ↓
Kokoro TTS  (sentence-by-sentence streaming)
 ↓
Speaker
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Qwen2.5-3B-Instruct (Q4_K_M GGUF via llama.cpp) |
| STT | Faster-Whisper (small model, CPU int8) |
| TTS | Kokoro TTS |
| VAD | Silero VAD |
| Wake Word | OpenWakeWord |
| Backend | FastAPI + asyncio |
| Memory | SQLite with FTS5 (aiosqlite) |
| Obsidian Search | sentence-transformers (all-MiniLM-L6-v2) + SQLite vectors |
| Knowledge Graph | SQLite adjacency table + label propagation clustering |
| Web Search | DuckDuckGo (disabled by default) |

---

## Quick Start

### 1. Run setup

```bash
chmod +x setup.sh
./setup.sh
```

Downloads the LLM model (~2 GB), installs system packages, builds llama.cpp, and creates the Python virtual environment.

### 2. Activate the environment

```bash
source venv/bin/activate
```

### 3. Choose a mode

**Terminal chat** (no microphone needed):
```bash
python -m app.main --mode terminal
```

**Voice mode** (microphone + speaker):
```bash
python -m app.main --mode voice
```

**API server** (for integrations):
```bash
python -m app.main --mode server
```

**Single query and exit:**
```bash
python -m app.main --mode query --query "List my workspace files"
```

---

## Directory Structure

```
assistant/
├── app/
│   ├── main.py              # Entry point and mode selector
│   ├── config.py            # All settings (overridable via .env)
│   ├── router.py            # FastAPI endpoints
│   ├── pipeline.py          # Core listen → think → speak orchestrator
│   ├── audio/
│   │   ├── microphone.py    # Continuous mic capture
│   │   ├── vad.py           # Silero VAD + speech collector
│   │   ├── wake_word.py     # OpenWakeWord detector
│   │   ├── stt.py           # Faster-Whisper transcription
│   │   └── tts.py           # Kokoro TTS + audio player (speed param, no global mutation)
│   ├── llm/
│   │   └── engine.py        # llama.cpp streaming wrapper (async-safe lock)
│   ├── memory/
│   │   └── manager.py       # Short-term buffer + SQLite long-term (query-aware recall)
│   ├── tools/
│   │   ├── router.py        # Tool dispatcher and validator
│   │   ├── filesystem.py    # Sandboxed file operations
│   │   ├── web_search.py    # DuckDuckGo search (opt-in)
│   │   ├── obsidian.py      # Vault integration: notes, search, graph awareness
│   │   └── knowledge_graph.py  # NEW: entity/relation graph, pathfinding, clustering
│   ├── prompts/
│   │   └── templates.py     # System prompt (with conversation history injection)
│   └── utils/
│       ├── logging.py       # Loguru setup
│       └── timing.py        # Latency measurement
├── models/                  # Place .gguf model files here
├── data/
│   ├── memory.db            # SQLite long-term memory
│   ├── knowledge_graph.db   # NEW: entity and relation store
│   ├── obsidian_vectors.db  # Semantic search index for vault
│   ├── conversations/       # Conversation logs
│   ├── cache/               # Temp cache
│   └── logs/                # Rotating daily logs
├── AssistantWorkspace/      # Sandboxed file workspace
├── setup.sh
├── requirements.txt
├── .env.example
└── README.md
```

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

```bash
# LLM
LLM_THREADS=6                 # CPU threads for inference (leave 2 for OS/audio)
LLM_CONTEXT_LENGTH=4096
LLM_MAX_TOKENS=512
LLM_TEMPERATURE=0.7

# STT
STT_MODEL=small               # tiny / base / small / medium
STT_LANGUAGE=                 # blank = auto-detect (supports Urdu + English)

# TTS
TTS_VOICE=af_heart            # Kokoro voice ID
TTS_SPEED=1.0

# Wake word
WAKE_WORD_MODEL=hey_jarvis    # built-in; swap for custom .onnx
WAKE_WORD_PHRASE=Hey Zohaib
WAKE_WORD_THRESHOLD=0.5

# Memory
MEMORY_SHORT_TERM_LIMIT=20    # last N turns kept in context

# Obsidian
OBSIDIAN_VAULT_PATH=/home/zohaib/Documents/notes
OBSIDIAN_EMBED_MODEL=all-MiniLM-L6-v2

# Web search (disabled by default)
WEB_SEARCH_ENABLED=false
WEB_SEARCH_MAX_RESULTS=5
```

---

## Memory System

### Short-term
The last 20 conversation turns are kept in RAM and injected into every prompt so the assistant remembers what was said earlier in the session.

### Long-term
SQLite database with FTS5 full-text search. Memories are recalled by semantic relevance to the current query (not just category), so the right memories surface at the right time.

```
You: Remember that I prefer responses in Urdu when possible.
Assistant: [saves to long-term memory, category=preference]

You: What do you know about my preferences?
Assistant: [recalls relevant memories and summarises]
```

---

## Obsidian Integration

The assistant reads and writes your Obsidian vault directly. Supported operations:

| Voice command | Tool invoked |
|---|---|
| "Create a note about transformers" | `obsidian_create_note` |
| "Remember this idea / capture this thought" | `obsidian_capture_idea` |
| "Log this to my daily note" | `obsidian_append_daily` |
| "What did I write about attention?" | `obsidian_search` (semantic) |
| "Search my notes for transformer" | `obsidian_keyword_search` |
| "What's related to [[Transformers]]?" | `obsidian_get_related` |
| "Continue working on project X" | `obsidian_get_project` |
| "Good morning" | `obsidian_morning_briefing` |
| "List my vault" | `obsidian_list_vault` |
| "Read note titled X" | `obsidian_read_note` |
| "Reindex my vault" | `obsidian_reindex` |

### Semantic search
Notes are embedded using `all-MiniLM-L6-v2` and stored as float32 vectors in SQLite. Search is cosine similarity. Falls back to keyword search if the embedding model is not installed.

### Knowledge graph auto-indexing
Every time a note is created, its `[[wikilinks]]` are automatically extracted and added to the knowledge graph as `links_to` edges. No manual step required.

---

## Knowledge Graph

The knowledge graph is a persistent SQLite store of **concepts** (nodes) and **typed relations** (edges). It grows automatically as you create notes and can be queried by voice.

### Database schema

```sql
kg_nodes  (id, name, note_title, created_at)
kg_edges  (id, source, target, relation, weight, created_at)
```

### Voice commands

| Voice command | Tool invoked |
|---|---|
| "How does attention connect to memory?" | `kg_path` |
| "What's related to transformers in my graph?" | `kg_neighbors` |
| "Show me my knowledge graph stats" | `kg_summary` |
| "What are my orphan / isolated concepts?" | `kg_orphans` |
| "What topic clusters do I have?" | `kg_clusters` |
| "How did my thinking on X evolve?" | `kg_timeline` |

### Relation types

The LLM uses these when auto-extracting relations from notes:

- `is_part_of` — hierarchical containment
- `analogous_to` — structural similarity
- `causes` — causal link
- `contradicts` — opposing ideas
- `enables` — one concept makes another possible
- `relates_to` — general association (default)
- `links_to` — explicit Obsidian wikilink

### Graph-expanded retrieval

When you search your notes semantically, the top results are used as seed nodes and the graph is walked 1–2 hops to surface connected concepts you may not have searched for directly. This is the core second-brain behaviour: finding non-obvious connections.

### Pathfinding example

```
You: How does attention connect to working memory?
Assistant: Path: [[attention mechanism]] —[is_part_of]→ [[transformer]]
           —[analogous_to]→ [[working memory]]
```

---

## Tool System

The LLM never executes actions directly. It requests tools via structured JSON, which is validated and dispatched by the tool router.

### Request format (LLM internal)

```json
{
  "thought": "user wants to search their notes",
  "tool": "obsidian_search",
  "tool_params": {"query": "attention mechanisms"},
  "response": null,
  "speech": {"pace": 1.0, "clause_pause_ms": 120, "tone": "focused"}
}
```

### Full tool registry

| Tool | Parameters | Description |
|---|---|---|
| `read_file` | `path` | Read a workspace file |
| `write_file` | `path`, `content` | Write or overwrite a file |
| `append_file` | `path`, `content` | Append to a file |
| `list_directory` | `path` (opt) | List workspace directory |
| `delete_file` | `path` | Delete a file |
| `web_search` | `query` | DuckDuckGo search (opt-in) |
| `save_memory` | `category`, `content` | Save to long-term memory |
| `recall_memory` | `query` | Recall relevant memories |
| `obsidian_create_note` | `title`, `content`, `folder`, `tags`, `related` | Create a vault note |
| `obsidian_append_daily` | `content`, `section` | Append to today's daily note |
| `obsidian_capture_idea` | `raw_thought`, `structured` | Save a voice-captured idea |
| `obsidian_search` | `query` | Semantic search across vault |
| `obsidian_keyword_search` | `query` | Full-text keyword search |
| `obsidian_get_related` | `note_title` | Backlink graph for a note |
| `obsidian_get_project` | `project_name` | Load all notes in a project folder |
| `obsidian_morning_briefing` | — | Daily summary: tasks, ideas, recent notes |
| `obsidian_reindex` | — | Re-embed entire vault |
| `obsidian_read_note` | `title` | Read note contents |
| `obsidian_list_vault` | `folder` (opt) | List vault structure |
| `kg_summary` | — | Knowledge graph stats |
| `kg_path` | `source`, `target` | Shortest path between two concepts |
| `kg_neighbors` | `node`, `depth` | Connected concepts up to N hops |
| `kg_orphans` | — | Isolated concepts with no connections |
| `kg_clusters` | — | Community detection (label propagation) |
| `kg_timeline` | `node`, `after_date`, `before_date` | How connections evolved over time |
| `kg_add` | `title`, `entities`, `relations` | Manually add nodes and edges |

### Tool loop

The pipeline supports up to 4 chained tool calls per turn. Each tool call has a 30-second timeout — a hanging web search cannot block the pipeline indefinitely.

---

## Sandbox Security

File operations are restricted to `AssistantWorkspace/`. Path traversal is blocked at the resolver level:

```python
real_path = os.path.realpath(user_path)
if not real_path.startswith(str(ALLOWED_ROOT)):
    raise PermissionError
```

Allowed file types: `.txt`, `.md`, `.json`, `.py`, `.js`, `.csv`, `.yaml`, `.toml`, `.pdf`

The LLM cannot execute shell commands, access system directories, install packages, or browse autonomously.

---

## Web Search

Disabled by default. Enable for the current session:

```
You: enable web search
You: search the web for latest Python 3.13 release notes
```

Or enable permanently in `.env`:
```bash
WEB_SEARCH_ENABLED=true
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | System status |
| POST | `/chat` | Text chat |
| POST | `/chat/stream` | Streaming SSE text chat |
| POST | `/tool` | Direct tool invocation |
| GET | `/memory` | Query memories |
| POST | `/memory` | Save a memory |
| DELETE | `/memory/{id}` | Delete a memory |
| WS | `/ws/audio` | Real-time audio streaming |
| POST | `/settings/web-search` | Toggle web search |

### Example: text chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What files are in my workspace?"}'
```

### Example: streaming chat

```bash
curl -N http://localhost:8000/chat/stream \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarise my recent notes"}'
```

---

## Performance Targets

| Stage | Target |
|---|---|
| Wake word detection | < 200 ms |
| STT transcription | < 1 s |
| LLM first token | < 1.5 s |
| TTS first audio | < 500 ms |
| Full response | < 4 s |

---

## Custom Wake Word

OpenWakeWord supports custom model training. To use "Hey Zohaib" instead of the built-in fallback:

1. Record ~150 positive examples of "Hey Zohaib"
2. Follow the [OpenWakeWord training guide](https://github.com/dscripka/openWakeWord/blob/main/docs/training.md)
3. Export the `.onnx` model to `models/hey_zohaib.onnx`
4. Set in `.env`: `WAKE_WORD_MODEL=models/hey_zohaib.onnx`

---

## Bug Fixes Applied

The following issues from the original codebase have been resolved:

**Async lock in LLM engine** — `run_in_executor` was not awaited inside the async lock, causing concurrent requests to race. Fixed by properly awaiting the executor call.

**TTS global settings mutation** — `speak()` was mutating `settings.tts_speed` directly, which is not thread-safe. TTS now accepts a `speed` parameter and the global settings object is never modified at runtime.

**`asyncio.get_event_loop()` inside executor threads** — deprecated in Python 3.10+ and raises errors in 3.12+. All async functions in `obsidian.py` now capture the loop with `asyncio.get_running_loop()` before entering executor threads.

**Wake word state machine** — the `awaiting_speech` flag was set to `True` before the wake word check, causing utterance collection to start on the same chunk that triggered the wake word. The flag is now only set on confirmed detection.

**Conversation history dropped from prompt** — `build_system_prompt()` accepted a `conversation_history` parameter but never used it. Short-term memory context is now correctly injected into every prompt.

**Memory recall not query-aware** — `get_context()` always recalled memories by category (preference/note) regardless of what the user asked. It now passes the current user query to `format_for_context()` so relevant memories surface.

**Tool call timeout** — tool dispatch had no timeout guard. A hanging `web_search` could block the pipeline indefinitely. Every tool call is now wrapped in `asyncio.wait_for()` with a 30-second limit.

---

## Troubleshooting

**Model not found** — place `qwen2.5-3b-instruct-q4_k_m.gguf` in the `models/` directory, or run `setup.sh` to download it automatically.

**No audio input** — run `arecord -l` to list devices. Set `AUDIO_INPUT_DEVICE=<index>` in `.env`.

**High latency** — reduce `LLM_THREADS` if audio stutters. Try `STT_MODEL=tiny` for faster (less accurate) transcription.

**Wake word not triggering** — lower `WAKE_WORD_THRESHOLD` to `0.3` in `.env`.

**Semantic search returns nothing** — run `obsidian_reindex` once to index existing vault notes. New notes are indexed automatically on creation.

**Knowledge graph empty** — the graph builds automatically as you create notes. Seed it faster by saying "reindex my vault" (embeds notes) followed by asking the assistant to read and index a few key notes.

---

## Roadmap

- GUI (web or desktop)
- RAG over local documents (PDF, EPUB)
- Calendar integration
- Vision model for image understanding
- Mobile companion app
- Custom "Hey Zohaib" wake word model
- Graph-expanded retrieval (semantic search + graph hop combined)
- Periodic cluster reports in morning briefing