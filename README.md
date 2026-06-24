# Tzar — Local AI Assistant

A fully local, privacy-first AI assistant with voice I/O, Obsidian integration, Knowledge Graph, RAG over local documents, multi-agent orchestration, and a self-improving skills system. Runs entirely on-device. Nothing leaves your machine unless you explicitly enable web search.

Named after **حافظ** — the keeper, the one who preserves.

---

## Hardware Target

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | i5 / Ryzen 5, 4 cores | i7 / Ryzen 7, 8+ cores |
| RAM | 8 GB | 16–32 GB |
| GPU | None (CPU-only) | NVIDIA 8GB+ VRAM or Apple Silicon |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 / macOS 14+ |
| Disk | 10 GB free | 30 GB free |

Not sure what model your hardware can run? See [System Profiler](#system-profiler).

---

## Architecture

```
Voice Input (mic)
    ↓
Wake Word Detector (openWakeWord)
    ↓
VAD (Silero)          ← detects end of speech
    ↓
STT (faster-whisper)  ← Whisper small, multilingual
    ↓
Intent Classifier     ← 0ms, rule-based, no LLM call
    ↓
┌─────────────────────────────────────┐
│  CHAT    → direct LLM               │
│  MEMORY  → hot memory recall + LLM  │
│  TOOL    → tool router + LLM        │
│  RESEARCH→ Research Agent           │
│  PLANNING→ Planner + tool loop      │
└─────────────────────────────────────┘
    ↓
[Optional] Multi-Agent Orchestrator
    ↓
FastAPI Backend (port 8000)
    ↓
TTS (Kokoro)          ← streamed sentence-by-sentence
    ↓
Audio Output (speaker)
```

---

## Quick Start

### 1. Install dependencies

```bash
# Python deps
pip install -r requirements.txt

# System deps (Ubuntu 24.04)
sudo apt install libwebkit2gtk-4.1-dev libssl-dev \
  libayatana-appindicator3-dev librsvg2-dev portaudio19-dev

# Ollama (recommended LLM backend)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set your Obsidian vault path, model choice, etc.
```

Minimum required settings:
```bash
OBSIDIAN_VAULT_PATH=/home/yourname/Documents/ObsidianVault
LLM_BACKEND=ollama
LLM_MODEL=qwen2.5:7b
STT_MODEL=small
```

### 3. Run

```bash
# API server (for UI or automation)
python -m app.main --mode server

# Voice loop (mic + speaker)
python -m app.main --mode voice

# Terminal chat
python -m app.main --mode terminal

# Single query and exit
python -m app.main --mode query --query "What's in my daily note?"
```

### 4. Desktop UI

```bash
cd ui
npm install
npm run dev          # browser at http://localhost:3000 (backend must be running)
npm run tauri:dev    # native Tauri window (requires Rust)
```

---

## System Profiler

Tzar can analyse your hardware and recommend the best model configuration:

```bash
python -m app.system_profiler
```

Or say: **"What model should I use?"** / **"Analyse my hardware"**

The profiler detects RAM, VRAM, CPU cores, AVX support, and Ollama availability, then outputs:

- The best LLM to use (Qwen2.5, Llama 3.1, Mistral, Phi, Gemma, DeepSeek-R1)
- Quantisation tier (Q3 / Q4 / Q5 based on available RAM)
- Whether GPU offload is viable (`n_gpu_layers`)
- Whisper model tier (tiny / base / small / medium / large-v3)
- A ready-to-paste `.env` snippet

---

## Intelligence Layers

Tzar is built in phases of increasing capability. All are opt-in and backward-compatible.

### Intent Classifier (`app/intent.py`)

Classifies every input before any LLM is called. Zero latency — pure regex, no model.

| Intent | Example | Route |
|--------|---------|-------|
| CHAT | "what is RAG?" | Direct LLM |
| MEMORY | "remember I prefer dark mode" | Hot memory + LLM |
| TOOL | "create a note about transformers" | Tool router |
| RESEARCH | "research latest llama.cpp updates" | Research Agent |
| PLANNING | "find the file and then create a note" | Planner + tool loop |

### Research Agent (`app/agents/researcher.py`)

Multi-source web research pipeline. Triggered automatically for RESEARCH intent.

```
Goal → LLM generates 3-5 search queries → concurrent DuckDuckGo searches
     → top pages fetched and extracted → LLM synthesises findings
     → structured markdown report → auto-saved to Obsidian vault (Research/)
     → short spoken summary returned
```

### Local Document RAG (`app/tools/rag/`)

Ingest your own files and query them with natural language.

Supported formats: **PDF, DOCX, EPUB, Markdown, TXT**

Retrieval: **BM25 keyword + MiniLM semantic similarity, fused with Reciprocal Rank Fusion (RRF)**

```bash
# Via chat
"ingest document /path/to/contract.pdf"
"what did the Databricks contract say about Genie?"

# Via UI
# Open the Documents panel → paste path → Ingest
```

Documents are stored in `doc_vectors` table alongside Obsidian note embeddings. `unified_search` queries both at once.

### Knowledge Graph (`app/tools/knowledge_graph.py`)

SQLite-backed directed property graph. Auto-extracts entities from every note saved.

```
obsidian_create_note() → entity extraction (CamelCase, acronyms, quoted phrases)
                       → kg_extract_and_index() → graph update
```

Tools:

| Tool | Description |
|------|-------------|
| `kg_summary` | Node/edge counts, top connected entities |
| `kg_neighbors` | BFS neighbours of a node (configurable depth) |
| `kg_path` | Shortest path between two nodes |
| `kg_clusters` | Connected clusters via Union-Find |
| `kg_timeline` | When a concept first appeared |
| `kg_expand` | Semantic graph expansion — vector search + graph walk |

### MCP Integration (`app/tools/mcp_client.py`)

Connects to any MCP server (HTTP/SSE or stdio). Tools are discovered dynamically and auto-registered.

```bash
# .env
MCP_SERVERS='[
  {"name": "gmail",    "transport": "http", "url": "http://localhost:3001/mcp"},
  {"name": "calendar", "transport": "http", "url": "http://localhost:3002/mcp"},
  {"name": "github",   "transport": "stdio", "command": "npx @modelcontextprotocol/server-github"}
]'
MULTI_AGENT_ENABLED=false   # set true to enable
```

### Memory Scoring (`app/memory/scoring.py`)

Every memory gets four scores computed on write:

| Score | How | Weight |
|-------|-----|--------|
| Importance | Rule-based: preferences/names → 0.9, casual → 0.1 | 45% |
| Recency | Exponential decay, 14-day half-life | 30% |
| Confidence | Hedge detection ("I think maybe") | 15% |
| Frequency | Log-normalised recall count (N_MAX=50) | 10% |

Memories are ranked by composite score before LLM context injection. `memory_prune` removes anything below 0.08 older than 7 days.

### Project Continuity (`app/memory/projects.py`)

The "feels like Jarvis" feature. Tracks active projects and loads their full context on demand.

```
"Continue the AI assistant project."
    ↓
project_switch("AI Assistant")
    ↓ (parallel)
  ├── Obsidian project folder notes
  ├── Open tasks (- [ ] items extracted)
  ├── KG entity neighbours
  └── Semantic memory matches
    ↓
Full context injected into every subsequent system prompt
```

Auto-detected from phrases like: *"continue X"*, *"switch to X"*, *"work on X"*, *"resume X"*.

### Skill Auto-Learning (`app/memory/skill_learner.py`)

Observes every successful tool sequence. At 3 occurrences, proposes it as a named skill.

```
Task completes → sequence normalised (noise removed, adjacent deduped)
             → hash + count stored
             → count ≥ 3? → proposal generated
             → "I noticed you always do X → Y → Z. Save as skill?"
             → yes → skill_create() / no → candidate cleared
```

### Desktop UI (`ui/`)

React + Tauri frontend. Connects to the FastAPI backend.

| Panel | What it does |
|-------|-------------|
| Chat | SSE streaming, voice input (WebSocket), expandable tool results, stop button |
| Memory | Cards with scoring bars, add/delete/search/prune, category filter |
| Graph | D3 force-directed KG, click to expand, path finder, zoom/pan |
| Documents | Ingest files/dirs, doc-only or unified search, remove from index |
| Skills | Detected patterns, proposal status |
| Projects | Create/switch projects, click to load context into chat |
| System | Hardware profiler, model recommendation, copy `.env` snippet |
| Settings | LLM/STT/TTS status, web search toggle, multi-agent toggle, MCP status |

```bash
cd ui
npm install
npm run dev        # http://localhost:3000 (no Rust needed)
npm run tauri:dev  # native window
npm run tauri:build # .AppImage / .deb / .dmg / .exe
```

### Multi-Agent Layer (`app/agents/orchestrator.py`)

Five agents coordinating through a shared `AgentContext`. Opt-in via `.env`.

```
User text
    ↓
PlannerAgent      → maps intent to AgentTasks
    ↓ (parallel)
ResearcherAgent   → handles RESEARCH tasks
ExecutorAgent     → runs TOOL tasks in dependency order
    ↓
Response synthesis (LLM)
    ↓ (background, non-blocking)
MemoryCuratorAgent → scores memories, logs to skill learner, prunes every 50 turns
SkillBuilderAgent  → checks for pending skill proposals, appends notification
```

Enable:
```bash
# .env
MULTI_AGENT_ENABLED=true
```

---

## API Reference

All endpoints on `http://127.0.0.1:8000`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | LLM/STT/TTS status, web search flag, workspace path |
| `POST` | `/chat` | Text chat (blocking, returns full response) |
| `POST` | `/chat/stream` | Text chat (SSE streaming, `data: {"token": "..."}`) |
| `POST` | `/tool` | Direct tool invocation `{"tool": "name", "param": "val"}` |
| `GET` | `/memory` | Query memories (`?q=search&category=preference`) |
| `POST` | `/memory` | Save memory `{"category": "note", "content": "..."}` |
| `DELETE` | `/memory/{id}` | Delete a memory by ID |
| `POST` | `/settings/web-search` | Toggle web search `{"enabled": true}` |
| `WS` | `/ws/audio` | WebSocket: binary float32 PCM in, JSON + PCM out |

### WebSocket protocol

**Client → Server**
- Binary frames: 16kHz float32 PCM audio chunks (512 samples)
- Text frame: `{"cmd": "stop"}` or `{"cmd": "text", "data": "message"}`

**Server → Client**
- `{"type": "status", "data": "transcribing" | "thinking"}`
- `{"type": "transcript", "data": "what the user said"}`
- `{"type": "response", "data": "assistant reply"}`
- `{"type": "error", "data": "error message"}`
- Binary frames: 24kHz float32 PCM TTS audio

---

## Tool Reference

All tools callable via `POST /tool` or from the LLM.

### File System
| Tool | Key params | Description |
|------|-----------|-------------|
| `read_file` | `path` | Read a file from the workspace |
| `write_file` | `path`, `content` | Write a file |
| `append_file` | `path`, `content` | Append to a file |
| `list_directory` | `path` | List directory contents |
| `delete_file` | `path` | Delete a file |

### Web
| Tool | Key params | Description |
|------|-----------|-------------|
| `web_search` | `query` | DuckDuckGo search (requires web search enabled) |

### Obsidian
| Tool | Key params | Description |
|------|-----------|-------------|
| `obsidian_create_note` | `title`, `content`, `folder?` | Create a note |
| `obsidian_append_daily` | `content` | Append to today's daily note |
| `obsidian_capture_idea` | `idea` | Save to Ideas folder |
| `obsidian_semantic_search` | `query`, `top_k?` | MiniLM semantic search |
| `obsidian_keyword_search` | `query` | BM25 keyword search |
| `obsidian_read_note` | `title` | Read a specific note |
| `obsidian_list_vault` | — | List all notes |
| `obsidian_get_related` | `title` | Find related notes |
| `obsidian_morning_briefing` | — | Daily summary |
| `obsidian_reindex` | — | Rebuild vector index |

### RAG Documents
| Tool | Key params | Description |
|------|-----------|-------------|
| `doc_search` | `query`, `top_k?`, `source_filter?` | Hybrid search over ingested docs |
| `unified_search` | `query`, `top_k?` | Search docs + Obsidian notes together |
| `ingest_document` | `path` | Index a single file |
| `ingest_directory` | `directory`, `recursive?` | Bulk index a folder |
| `list_documents` | — | List all indexed documents |
| `remove_document` | `path` | Remove a document from the index |

### Knowledge Graph
| Tool | Key params | Description |
|------|-----------|-------------|
| `kg_summary` | — | Graph stats (nodes, edges, top connected) |
| `kg_neighbors` | `node`, `depth?` | BFS neighbours |
| `kg_path` | `source`, `target` | Shortest path between nodes |
| `kg_clusters` | `min_size?` | Find connected clusters |
| `kg_timeline` | `node` | When a concept appeared and how it evolved |
| `kg_orphans` | — | Nodes with no connections |
| `kg_expand` | `query` | Semantic vector search + graph walk |
| `kg_add_text` | `text`, `source_note?` | Auto-extract and add entities from text |

### MCP
| Tool | Key params | Description |
|------|-----------|-------------|
| `mcp_status` | — | Connected servers and their tools |
| `mcp_list_tools` | — | All registered MCP tool names |
| `mcp_{server}_{tool}` | varies | Dynamically registered MCP tools |

### Memory
| Tool | Key params | Description |
|------|-----------|-------------|
| `memory_scores` | — | Scoring statistics across all memories |
| `memory_prune` | — | Remove low-score stale memories |
| `memory_write` | `target`, `content` | Write to hot memory (curator-gated) |
| `memory_read` | `target?` | Read hot memory |
| `memory_remove` | `target`, `substring` | Remove from hot memory |
| `save_memory` | `category`, `content` | Save to long-term memory |
| `recall_memory` | `query` | Recall from long-term memory |

### Projects
| Tool | Key params | Description |
|------|-----------|-------------|
| `project_list` | — | All projects with status and last opened |
| `project_new` | `name`, `description?` | Create a project |
| `project_switch` | `name` | Load full project context |
| `project_update` | `name`, `description?`, `status?` | Update project |
| `project_archive` | `name` | Mark as archived |
| `project_status` | — | Currently active project |

### Skills
| Tool | Key params | Description |
|------|-----------|-------------|
| `skill_load` | `name` | Load a saved skill |
| `skill_create` | `name`, `description`, `content` | Create a skill |
| `skill_update` | `name`, `old_text`, `new_text` | Edit a skill |
| `skill_rewrite` | `name`, `description`, `content` | Full rewrite |
| `skill_delete` | `name` | Delete a skill |
| `skill_learning_stats` | — | Auto-learning status and detected patterns |

### Sessions
| Tool | Key params | Description |
|------|-----------|-------------|
| `session_search` | `query` | Search past conversation sessions |
| `session_list` | — | List recent sessions |

### System (new)
| Tool | Key params | Description |
|------|-----------|-------------|
| `system_profile` | — | Full hardware analysis + model recommendation |

---

## Configuration Reference (`.env`)

```bash
# ── Paths ──────────────────────────────────────────────────────────────────
OBSIDIAN_VAULT_PATH=/home/yourname/Documents/ObsidianVault

# ── LLM ────────────────────────────────────────────────────────────────────
LLM_BACKEND=ollama                      # ollama | llamacpp
LLM_MODEL=qwen2.5:7b                    # Ollama model tag
LLM_MODEL_PATH=models/qwen2.5-7b.gguf  # llama.cpp path (if backend=llamacpp)
LLM_N_GPU_LAYERS=0                      # 0=CPU only, -1=all GPU, N=partial
LLM_CONTEXT_LENGTH=8192
LLM_THREADS=6
LLM_TEMPERATURE=0.7
LLM_OLLAMA_HOST=http://localhost:11434

# ── STT ────────────────────────────────────────────────────────────────────
STT_MODEL=small                         # tiny|base|small|medium|large-v3
STT_DEVICE=cpu                          # cpu|cuda
STT_LANGUAGE=                           # blank = auto-detect (supports Urdu)

# ── TTS ────────────────────────────────────────────────────────────────────
TTS_VOICE=af_heart                      # Kokoro voice ID

# ── Wake Word ──────────────────────────────────────────────────────────────
WAKE_WORD_PHRASE=Hey Jarvis

# ── Web Search ─────────────────────────────────────────────────────────────
WEB_SEARCH_ENABLED=false                # off by default

# ── RAG ───────────────────────────────────────────────────────────
RAG_DOCUMENTS_DIR=/home/yourname/Documents

# ── MCP ───────────────────────────────────────────────────────────
MCP_ENABLED=false
MCP_SERVERS=[]

# ── Projects ──────────────────────────────────────────────────────
PROJECT_AUTO_DETECT=true

# ── Skill Learning ────────────────────────────────────────────────
SKILL_LEARNING_ENABLED=true
SKILL_PATTERN_THRESHOLD=3

# ── Multi-Agent ──────────────────────────────────────────────────
MULTI_AGENT_ENABLED=false
```

---

## Directory Structure

```
tzar/
├── app/
│   ├── main.py                 # Entry point (server/voice/terminal/query modes)
│   ├── config.py               # All settings, pydantic-settings
│   ├── router.py               # FastAPI app and all endpoints
│   ├── planner.py              # Task planner (LLM call → structured plan)
│   ├── intent.py               # Intent classifier (no LLM)
│   ├── system_profiler.py      # Hardware detection + model recommender
│   │
│   ├── agents/
│   │   ├── researcher.py       # Multi-source research agent
│   │   └── orchestrator.py     # Multi-agent coordinator
│   │
│   ├── audio/
│   │   ├── stt.py              # faster-whisper STT
│   │   ├── tts.py              # Kokoro TTS (streamed)
│   │   ├── vad.py              # Silero VAD + speech collector
│   │   ├── microphone.py       # PyAudio input stream
│   │   └── wake_word.py        # openWakeWord detector
│   │
│   ├── llm/
│   │   └── engine.py           # Ollama + llama-cpp-python backend
│   │
│   ├── memory/
│   │   ├── manager.py          # Short-term + long-term memory coordinator
│   │   ├── hot_memory.py       # In-context structured memory
│   │   ├── scoring.py          # Importance/recency/confidence/frequency
│   │   ├── projects.py         # Project registry + context loading
│   │   ├── skill_learner.py    # Pattern detection + skill proposals
│   │   ├── skills.py           # Named skill CRUD
│   │   ├── curator.py          # Memory quality gating (LLM-based)
│   │   └── session_store.py    # SQLite session archive
│   │
│   ├── tools/
│   │   ├── router.py           # Tool dispatch + validation
│   │   ├── filesystem.py       # File I/O (sandboxed to workspace)
│   │   ├── web_search.py       # DuckDuckGo search
│   │   ├── obsidian.py         # Full Obsidian vault integration + vectors
│   │   ├── knowledge_graph.py  # SQLite KG
│   │   ├── mcp_client.py       # MCP HTTP/stdio client
│   │   └── rag/
│   │       ├── ingestor.py     # PDF/DOCX/EPUB chunking + embedding
│   │       └── search.py       # BM25 + semantic + RRF hybrid retrieval
│   │
│   └── prompts/
│       └── templates.py        # System prompt builder
│
├── ui/                         # React + Tauri desktop UI
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts              # Typed FastAPI client
│   │   ├── styles.css
│   │   ├── stores/useStore.ts  # Zustand global state
│   │   └── components/
│   │       ├── ChatPanel.tsx
│   │       ├── MemoryPanel.tsx
│   │       ├── GraphPanel.tsx
│   │       ├── DocsPanel.tsx
│   │       ├── SystemPanel.tsx
│   │       ├── SecondaryPanels.tsx  # Skills, Projects, Settings
│   │       └── Sidebar.tsx
│   └── src-tauri/              # Tauri shell (Rust)
│
├── models/                     # GGUF model files (if using llama.cpp)
├── data/
│   ├── memory.db               # SQLite: hot memory + scored memories + skill candidates
│   ├── obsidian_vectors.db     # SQLite: note embeddings + doc embeddings
│   ├── knowledge_graph.db      # SQLite: KG nodes + edges
│   ├── projects.db             # SQLite: project registry
│   └── conversations/          # Session archives (JSON)
├── tests/
│   ├── test_08_intent_research.py      # 58 tests
│   ├── test_09_rag_kg_mcp.py           # 64 tests
│   ├── test_10_scoring_projects_skills.py # 83 tests
│   └── test_11_multiagent_profiler.py  # 49 tests
├── AssistantWorkspace/         # Sandboxed file I/O directory
├── requirements.txt
└── .env
```

---

## Running Tests

```bash
pytest tests/ -v
# 254 tests, all offline — no LLM, no network, no Obsidian vault required
```

---

## Model Recommendations

| RAM | Best model | Quant | Backend |
|-----|-----------|-------|---------|
| 4 GB | Qwen2.5-1.5B | Q4_K_M | Ollama |
| 8 GB | Qwen2.5-7B or Llama-3.1-8B | Q4_K_M | Ollama |
| 16 GB | Qwen2.5-14B or DeepSeek-R1-14B | Q4_K_M | Ollama |
| 32 GB+ | Qwen2.5-32B | Q4_K_M | Ollama |
| Apple Silicon | Any of the above | — | Ollama (Metal) |

For GPU offload, run the system profiler — it calculates the correct `LLM_N_GPU_LAYERS` for your VRAM automatically.

---

## Supported Languages

STT (Whisper) supports 99 languages including **English and Urdu** simultaneously with auto-detection. Set `STT_LANGUAGE=ur` to force Urdu, or leave blank for automatic.

---

## Privacy

- All processing is local. No data is sent anywhere.
- Web search (DuckDuckGo) is **off by default** and must be explicitly enabled.
- MCP server connections are opt-in and require explicit configuration.
- The system profiler runs only on your machine — hardware data is never transmitted.