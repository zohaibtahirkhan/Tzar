# Local Offline Voice Assistant — Second Brain Edition

A fully local, privacy-first AI voice assistant with Obsidian integration, Knowledge Graph engine, self-improving skills, and a curator-gated memory system. Runs entirely on-device. Nothing leaves your machine unless you explicitly enable web search.

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
Silero VAD          (end-of-utterance detection)
 ↓
OpenWakeWord        ("Hey Zohaib")
 ↓
Faster-Whisper STT
 ↓
needs_planning()?
 ├─ yes → Planner LLM call → Plan injected into prompt
 └─ no  → skip
 ↓
Conversation Agent  (structured JSON, no thought leakage, confidence score)
 ↓
Tool Router ─┬── Filesystem Sandbox
             ├── Long-Term Memory (SQLite FTS5)
             ├── Hot Memory (MEMORY.md + USER.md — always-on curated facts)
             │    └── memory_write → Curator Agent → store / reject / consolidate
             ├── Skills (procedural memory — SKILL.md files, self-improving)
             ├── Session Archive (full conversation FTS5 search)
             ├── Obsidian Vault (notes, daily log, semantic search)
             ├── Knowledge Graph (entities, relations, pathfinding, clustering)
             └── Web Search (opt-in, DuckDuckGo)
 ↓
Kokoro TTS  (sentence-by-sentence streaming, respects interruptible flag)
 ↓
Speaker
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Qwen2.5-3B-Instruct (Q4_K_M GGUF via llama.cpp) or any Ollama model |
| STT | Faster-Whisper (small model, CPU int8) |
| TTS | Kokoro TTS |
| VAD | Silero VAD |
| Wake Word | OpenWakeWord |
| Backend | FastAPI + asyncio |
| Memory (hot) | MEMORY.md + USER.md (curated, always-on, char-limited) |
| Memory (cold) | SQLite with FTS5 (aiosqlite) |
| Session Archive | SQLite FTS5 (every conversation turn, searchable) |
| Skills | SKILL.md files (procedural memory, self-improving) |
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
│   ├── main.py                  # Entry point and mode selector
│   ├── config.py                # All settings (overridable via .env)
│   ├── router.py                # FastAPI endpoints
│   ├── pipeline.py              # Core listen → think → speak orchestrator
│   ├── planner.py               # Task planner — step-by-step plan for complex tasks
│   ├── audio/
│   │   ├── microphone.py
│   │   ├── vad.py               # Silero VAD + speech collector
│   │   ├── wake_word.py         # OpenWakeWord detector
│   │   ├── stt.py               # Faster-Whisper transcription
│   │   └── tts.py               # Kokoro TTS + audio player (speed param, no global mutation)
│   ├── llm/
│   │   └── engine.py            # llama.cpp or Ollama streaming wrapper (async-safe lock)
│   ├── memory/
│   │   ├── manager.py           # Short-term buffer + SQLite long-term (query-aware recall)
│   │   ├── hot_memory.py        # MEMORY.md + USER.md (always-on curated facts)
│   │   ├── curator.py           # Memory Curator Agent (store / reject / consolidate)
│   │   ├── skills.py            # Self-improving skills (procedural memory)
│   │   └── session_store.py     # Full conversation archive + FTS5 search
│   ├── tools/
│   │   ├── router.py            # Tool dispatcher and validator
│   │   ├── filesystem.py        # Sandboxed file operations
│   │   ├── web_search.py        # DuckDuckGo search (opt-in)
│   │   ├── obsidian.py          # Vault integration: notes, search, graph awareness
│   │   └── knowledge_graph.py   # Entity/relation graph, pathfinding, clustering
│   ├── prompts/
│   │   └── templates.py         # System prompt (hot memory + skills + plan injected)
│   └── utils/
│       ├── logging.py
│       └── timing.py
├── tests/
│   ├── conftest.py              # Shared fixtures, mock LLM helpers
│   ├── test_01_tool_selection.py  # 30 tests — tool routing, JSON parsing
│   ├── test_02_memory.py          # 20 tests — hot memory, curator, SQLite FTS5
│   ├── test_03_obsidian.py        # 13 tests — note creation, search, related, projects
│   ├── test_04_planning.py        # 21 tests — complexity classifier, planner, sequences
│   ├── test_05_voice_speech.py    # 23 tests — speech fields, interrupts, confidence
│   ├── test_06_session_kg.py      # 16 tests — session archive, knowledge graph
│   └── test_07_skills.py          # 15 tests — skill CRUD, index, prompt injection
├── data/
│   ├── memory.db                # SQLite long-term memory (cold archive)
│   ├── sessions.db              # SQLite session archive (full conversation FTS5)
│   ├── knowledge_graph.db       # Entity and relation store
│   ├── obsidian_vectors.db      # Semantic search index for vault
│   ├── MEMORY.md                # Agent hot memory (env facts, conventions, lessons)
│   ├── USER.md                  # User profile (preferences, communication style)
│   ├── skills/                  # Self-improving SKILL.md files
│   │   └── <skill-name>/
│   │       └── SKILL.md
│   ├── conversations/
│   ├── cache/
│   └── logs/
├── models/                      # Place .gguf model files here
├── AssistantWorkspace/          # Sandboxed file workspace
├── pytest.ini
├── requirements.txt
├── requirements-test.txt
├── setup.sh
└── README.md
```

---

## Running Tests

### Install test dependencies

```bash
pip install -r requirements-test.txt
```

### Run all tests (no models, no hardware needed)

```bash
pytest
```

### Run a specific suite

```bash
pytest tests/test_01_tool_selection.py   # Tool routing
pytest tests/test_02_memory.py           # Memory precision
pytest tests/test_03_obsidian.py         # Obsidian coherence
pytest tests/test_04_planning.py         # Planner
pytest tests/test_05_voice_speech.py     # Speech behaviour
pytest tests/test_06_session_kg.py       # Session store + Knowledge Graph
pytest tests/test_07_skills.py           # Skills system
```

### Verbose output with timing

```bash
pytest -v --tb=short
```

### Run only fast tests (exclude slow/integration)

```bash
pytest -m "not slow and not integration"
```

---

## Test Coverage

| Suite | Tests | What it measures |
|---|---|---|
| 01 Tool Selection | 30 | Correct tool chosen (or no tool); JSON robustness; validation |
| 02 Memory | 20 | Store vs reject precision; hot memory CRUD; curator decisions; SQLite FTS5 |
| 03 Obsidian | 13 | Note creation, reading, keyword search, related notes, project coherence |
| 04 Planning | 21 | Complexity classifier; plan step order; required tools; pipeline integration |
| 05 Voice/Speech | 23 | Pace/pause/tone per input type; interruptibility flag; confidence field |
| 06 Session + KG | 16 | Session archive FTS5; pathfinding; clustering; orphan detection; temporal |
| 07 Skills | 15 | Create/load/update/rewrite/delete; index in prompt; slug sanitization |
| **Total** | **138** | |

All 138 tests run offline with no models, no microphone, no Obsidian vault required. LLM calls are mocked — tests inject pre-built JSON responses via `make_llm_response()`.

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

```bash
# LLM
LLM_MODEL_PATH=models/qwen2.5-3b-instruct-q4_k_m.gguf
LLM_THREADS=6
LLM_CONTEXT_LENGTH=4096
LLM_MAX_TOKENS=512
LLM_TEMPERATURE=0.7

# STT
STT_MODEL=small               # tiny / base / small / medium
STT_LANGUAGE=                 # blank = auto-detect (supports Urdu + English)

# TTS
TTS_VOICE=af_heart
TTS_SPEED=1.0

# Wake word
WAKE_WORD_PHRASE=Hey Zohaib
WAKE_WORD_THRESHOLD=0.5

# Memory
MEMORY_SHORT_TERM_LIMIT=20

# Obsidian
OBSIDIAN_VAULT_PATH=/home/zohaib/Documents/notes

# Web search (disabled by default)
WEB_SEARCH_ENABLED=false
```

---

## LLM Response Format

Every LLM response is structured JSON. No prose outside JSON is ever output.

```json
{
  "tool": null,
  "tool_params": null,
  "response": "What to say to the user.",
  "confidence": 0.92,
  "interruptible": true,
  "speech": {
    "pace": 1.0,
    "clause_pause_ms": 120,
    "tone": "neutral"
  }
}
```

The `thought` field has been removed to prevent chain-of-thought and system prompt leakage.

### Speech field reference

| Field | Values | Notes |
|---|---|---|
| `pace` | 0.75–1.15 | 0.75 = reading log files; 1.15 = short confirmations |
| `clause_pause_ms` | 50–350 | 50 = "Done."; 350 = reading a list item by item |
| `tone` | neutral / informative / warm / focused / urgent | Guides TTS voice energy |
| `confidence` | 0.0–1.0 | Below 0.6 → prefer tool use or acknowledge uncertainty |
| `interruptible` | true / false | false = critical confirmations (delete, save, alarms) |

---

## Memory System

### Hot memory (always-on)

Two character-limited files injected into every prompt:

- `data/MEMORY.md` — agent memory: environment facts, project conventions, tool quirks, lessons learned (2,200 char limit)
- `data/USER.md` — user profile: name, preferences, communication style, things to avoid (1,375 char limit)

The Curator Agent evaluates every proposed memory before writing. It stores durable facts, rejects ephemeral ones, and consolidates updates.

```
You: I prefer Urdu for casual chat.
Curator: → store [user] "Prefers Urdu for casual chat."

You: I had chicken for lunch.
Curator: → reject (ephemeral meal detail)

You: I watched a movie yesterday.
Curator: → reject (one-off activity)
```

### Cold memory (query-aware recall)

SQLite FTS5 full-text search. Recalled by relevance to the current query, not just by category.

### Session archive

Every conversation turn is logged to `data/sessions.db`. Searchable by voice:

```
You: What did I say about Snowflake last week?
→ session_search("Snowflake")
```

---

## Skills (Procedural Memory)

The assistant creates reusable workflow procedures from experience. After completing a multi-tool task, it may save the approach as a skill:

```
You: Find the Snowflake release notes, create a note, and link it to Data Engineering.
[3 tool calls later — task complete]
Assistant: [saves skill "research-and-document" to data/skills/]

Next time:
You: Research the dbt release and document it.
Assistant: [loads skill "research-and-document"] → follows the saved procedure
```

Skills are plain `SKILL.md` files you can read and edit directly.

---

## Obsidian Integration

| Voice command | Tool |
|---|---|
| "Create a note about RAG" | `obsidian_create_note` |
| "Remember this idea" | `obsidian_capture_idea` |
| "Log this to my daily note" | `obsidian_append_daily` |
| "What did I write about attention?" | `obsidian_search` (semantic) |
| "Search my notes for transformer" | `obsidian_keyword_search` |
| "What's related to [[Transformers]]?" | `obsidian_get_related` |
| "Continue working on project X" | `obsidian_get_project` |
| "Good morning" | `obsidian_morning_briefing` |
| "Read note titled X" | `obsidian_read_note` |
| "Reindex my vault" | `obsidian_reindex` |

Every note created is automatically indexed into the Knowledge Graph via its `[[wikilinks]]`.

---

## Knowledge Graph

Persistent SQLite store of concepts (nodes) and typed relations (edges).

```sql
kg_nodes  (id, name, note_title, created_at)
kg_edges  (id, source, target, relation, weight, created_at)
```

| Voice command | Tool |
|---|---|
| "How does attention connect to memory?" | `kg_path` |
| "What's related to transformers in my graph?" | `kg_neighbors` |
| "Show me my knowledge graph stats" | `kg_summary` |
| "What are my orphan / isolated concepts?" | `kg_orphans` |
| "What topic clusters do I have?" | `kg_clusters` |
| "How did my thinking on X evolve?" | `kg_timeline` |

---

## Planner

For complex multi-step requests, a planner LLM call runs before the main assistant:

```
You: Find the latest Snowflake release, create a note, link it to Data Engineering, add to daily.

Planner produces:
  Goal: Research and document Snowflake release.
  Step 1: Search web for latest Snowflake release notes
  Step 2: Summarize findings
  Step 3: Create Obsidian note titled "Snowflake <version>"
  Step 4: Add knowledge graph links to Data Engineering
  Step 5: Append summary to daily note
  Tools needed: web_search, obsidian_create_note, kg_add, obsidian_append_daily

Main assistant executes steps in order.
```

Simple inputs (greetings, math, short questions) skip the planner entirely for latency.

---

## Tool System

The LLM never executes actions directly. It outputs structured JSON; the tool router validates and dispatches.

### Full tool registry

| Tool | Parameters |
|---|---|
| `read_file` | `path` |
| `write_file` | `path`, `content` |
| `append_file` | `path`, `content` |
| `list_directory` | `path` (opt) |
| `delete_file` | `path` |
| `web_search` | `query` |
| `save_memory` | `category`, `content` |
| `recall_memory` | `query` |
| `memory_write` | `target`, `content` → routed through Curator |
| `memory_remove` | `target`, `substring` |
| `memory_replace` | `target`, `old_substring`, `new_content` |
| `memory_read` | `target` |
| `obsidian_create_note` | `title`, `content`, `folder`, `tags`, `related` |
| `obsidian_append_daily` | `content`, `section` |
| `obsidian_capture_idea` | `raw_thought` |
| `obsidian_search` | `query` |
| `obsidian_keyword_search` | `query` |
| `obsidian_get_related` | `note_title` |
| `obsidian_get_project` | `project_name` |
| `obsidian_morning_briefing` | — |
| `obsidian_reindex` | — |
| `obsidian_read_note` | `title` |
| `obsidian_list_vault` | `folder` (opt) |
| `kg_summary` | — |
| `kg_path` | `source`, `target` |
| `kg_neighbors` | `node`, `depth` |
| `kg_orphans` | — |
| `kg_clusters` | — |
| `kg_timeline` | `node`, `after_date`, `before_date` |
| `kg_add` | `title`, `entities`, `relations` |
| `skill_load` | `name` |
| `skill_create` | `name`, `description`, `content`, `category` |
| `skill_update` | `name`, `old_text`, `new_text` |
| `skill_rewrite` | `name`, `description`, `content` |
| `skill_delete` | `name` |
| `session_search` | `query` |
| `session_list` | — |

Tool loop: up to 4 chained calls per turn. Each call has a 30-second timeout.

---

## Sandbox Security

File operations restricted to `AssistantWorkspace/`. Path traversal blocked at the resolver. Allowed extensions: `.txt .md .json .py .js .csv .yaml .toml .pdf`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | System status |
| POST | `/chat` | Text chat |
| POST | `/chat/stream` | Streaming SSE |
| POST | `/tool` | Direct tool invocation |
| GET | `/memory` | Query memories |
| POST | `/memory` | Save a memory |
| DELETE | `/memory/{id}` | Delete a memory |
| WS | `/ws/audio` | Real-time audio |
| POST | `/settings/web-search` | Toggle web search |

---

## Using Ollama Instead of llama.cpp

Change in `app/config.py`:
```python
LLM_BACKEND: str = "ollama"
LLM_MODEL: str = "qwen2.5:3b"
LLM_OLLAMA_HOST: str = "http://localhost:11434"
```

Swap models with zero code changes:
```bash
# In .env
LLM_MODEL=qwen2.5:7b        # better reasoning, fits in 16 GB RAM
LLM_MODEL=qwen2.5:1.5b      # faster, smaller
LLM_MODEL=aya:8b             # strong Urdu + English
LLM_MODEL=qwen2.5-coder:7b  # code-heavy sessions
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

## Bug Fixes Applied

**Async lock in LLM engine** — `run_in_executor` was not awaited inside the async lock. Fixed with `await loop.run_in_executor(...)`.

**TTS global settings mutation** — `speak()` mutated `settings.tts_speed` directly (not thread-safe). TTS now accepts a `speed` parameter.

**`asyncio.get_event_loop()` inside executor threads** — deprecated in Python 3.10+. All `obsidian.py` functions now use `asyncio.get_running_loop()`.

**Wake word state machine** — `awaiting_speech` was set to `True` before the wake word check, causing collection to start on the triggering chunk. Now only set on confirmed detection.

**Conversation history dropped from prompt** — `build_system_prompt()` accepted `conversation_history` but never used it. Now correctly injected.

**Memory recall not query-aware** — `get_context()` ignored the user query when recalling memories. Now passes `user_text` to `format_for_context()`.

**Tool call timeout** — no timeout guard on tool dispatch. Now wrapped in `asyncio.wait_for()` with 30-second limit.

**Chain-of-thought leakage** — `thought` field in JSON output exposed internal reasoning and system prompt fragments. Field removed entirely.

**Memory pollution** — assistant wrote to memory freely, accumulating noise over time. All `memory_write` calls now route through the Curator Agent (store / reject / consolidate).

**No planner for complex tasks** — multi-step requests had no execution plan, causing skipped steps. Planner layer now injects an ordered plan for complex inputs.

---

## Troubleshooting

**Model not found** — place `.gguf` file in `models/`, or run `setup.sh`.

**No audio input** — run `arecord -l`. Set `AUDIO_INPUT_DEVICE=<index>` in `.env`.

**High latency** — reduce `LLM_THREADS`. Try `STT_MODEL=tiny`.

**Wake word not triggering** — lower `WAKE_WORD_THRESHOLD` to `0.3`.

**Semantic search returns nothing** — run `obsidian_reindex` once to index existing notes.

**Tests failing with import errors** — ensure you are in the project root with `venv` active. Run `pip install -r requirements-test.txt`.

---

## Roadmap

- Decision model layer (intent classifier before planner)
- Labeled evaluation dataset (100–200 interactions with correct tool / memory / speech labels)
- GUI (web or desktop)
- RAG over local documents (PDF, EPUB)
- Calendar integration
- Vision model for image understanding
- Custom "Hey Zohaib" wake word model (.onnx training)
- Graph-expanded retrieval (semantic search + graph hop combined)
- Periodic cluster reports in morning briefing
- Mobile companion app