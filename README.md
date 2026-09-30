# Tzar — Local AI Assistant

A local, privacy-first voice assistant built around your Obsidian vault. It listens for a wake word, answers with a local LLM, speaks back, and can search, write and link your notes and documents. Answers from your notes and documents name their source.

Everything runs on your machine. Nothing goes online unless you turn on web search.

Named after **ضار** — the keeper, the one who preserves.

---

## Features

- **Voice loop**: wake word → speech-to-text → LLM → text-to-speech, streamed clause by clause. Saying the wake word again interrupts the answer.
- **Obsidian second brain**: create notes, log to the daily note, capture ideas, and search the vault semantically or by keyword. Edits you make in Obsidian are re-indexed automatically.
- **Cited answers over local documents**: ingest PDF, DOCX, EPUB, Markdown, TXT and RST files. Search is hybrid (BM25 + embeddings, fused with Reciprocal Rank Fusion). Answers name the note or document they came from.
- **Knowledge graph**: entities and links are extracted from your notes into a queryable graph.
- **Memory**: short-term conversation, scored long-term memory, and a small curated profile (`data/USER.md`, `data/MEMORY.md`).
- **Projects, goals and skills**: "continue project X" loads that project's context; multi-step plans are tracked as goals; tool sequences you repeat are proposed as reusable skills.
- **Desktop UI**: React + Tauri app with chat, memory, knowledge graph, documents, projects and system panels.
- **Extensible**: connect MCP servers and their tools are registered automatically.

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Ubuntu 22.04+ (the only tested platform, see [Platforms](#platforms)) | Ubuntu 24.04 |
| Python | 3.11 | 3.11 |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB+ |
| GPU | none | NVIDIA 8 GB+ VRAM (used by Ollama) |
| Disk | 10 GB free | 20 GB free |

Not sure which model your machine can run? Run the [System Profiler](#system-profiler).

---

## Quick Start

### 1. Install

```bash
# System packages (Ubuntu). build-essential/cmake are needed because
# requirements.txt builds llama-cpp-python for the optional llama.cpp backend.
sudo apt install python3.11 python3.11-venv build-essential cmake \
  portaudio19-dev libsndfile1 ffmpeg espeak-ng

# Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Ollama, the default LLM backend
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b          # or qwen2.5:3b on 8 GB RAM
```

On first run the assistant downloads its speech and embedding models (Whisper, Kokoro, openWakeWord, MiniLM, about 1 GB in total). After that it works offline.

### 2. Configure

```bash
cp .env.example .env
```

At minimum, set these in `.env`:

```bash
OBSIDIAN_VAULT_PATH=~/Documents/MyVault   # your vault; created if it doesn't exist
LLM_MODEL=qwen2.5:7b                      # must match the model you pulled
```

### 3. Run

```bash
python -m app.main --mode voice      # voice loop: say "Hey Jarvis", then speak
python -m app.main --mode terminal   # text chat in the terminal
python -m app.main --mode server     # HTTP API on 127.0.0.1:8000 (needed for the UI)
python -m app.main --mode query --query "What's in my daily note?"
```

If Ollama isn't running or the model isn't pulled, the assistant says so and tells you the command to fix it. `GET /health` reports the same.

### 4. Desktop UI (optional)

Start the server first, then:

```bash
cd ui
npm install
npm run dev          # browser at http://localhost:3000
npm run tauri:dev    # native window (needs Rust and the Tauri system packages, see ui/README.md)
npm run tauri:build  # installable app in ui/src-tauri/target/release/bundle/
```

---

## How a request is handled

```
Mic ─► wake word (openWakeWord) ─► VAD (Silero) ─► STT (faster-whisper)
                                                        │
                                                        ▼
                                   Capability classifier (regex, no LLM)
                                                        │
   Answered without the LLM when possible: ◄────────────┤
   "where did that come from?", cached answer,          │
   "good morning" briefing, "remember that…",           │
   "continue project X", "continue" (resume a goal)     │
                                                        ▼
   Multi-step request + MULTI_AGENT_ENABLED ─► Coordinator (plan → run tools → critic → answer)
   Research request (web search on)         ─► Research Agent
   Everything else                          ─► LLM + tool loop (up to 4 tool calls)
                                                        │
                                                        ▼
                         Answer + source attribution ─► TTS (Kokoro), spoken as it streams
```

The classifier sets flags such as `needs_tools`, `needs_rag`, `needs_memory`, `needs_research` and `needs_planning` (`app/intent.py`). They decide which shortcuts are tried, whether notes and documents are searched before the LLM is called, and whether a plan is made.

The LLM replies in JSON: either a tool call or a spoken `response` with speech hints (pace, pauses, tone, whether it can be interrupted). Tool calls are validated and dispatched by `app/tools/router.py`; the model never runs anything directly.

---

## Features in detail

### Obsidian vault (`app/tools/obsidian.py`, `app/tools/obsidian_sync.py`)

Notes the assistant writes go in `AI Notes/`, `Daily/`, `Ideas/` and `Research/` (the folder names are configurable). Every note is embedded for semantic search, and its `[[wikilinks]]` and entities go into the knowledge graph.

Vault sync keeps the index current when you edit notes in Obsidian itself:

```
note saved in Obsidian ─► file watcher ─► 3 s debounce ─► re-embed if changed ─► update knowledge graph
startup                ─► backfill: index new/changed notes, drop deleted ones
```

Background sync never calls the LLM, so typing in Obsidian doesn't compete with your conversation.

### Local documents (`app/tools/rag/`)

```
"ingest document ~/Downloads/contract.pdf"
"what does the contract say about termination?"
```

`ingest_document` accepts any path on your machine, so you don't have to copy files into a special folder first. `ingest_directory` with no argument ingests `RAG_DOCUMENTS_DIR`. Chunks are stored in `data/obsidian_vectors.db`, and `unified_search` searches documents and notes together.

**Citations** (`app/tools/rag/citations.py`): search tools record what they retrieved. The code (not the LLM) appends "That's from *X*." unless the answer already names its source. "Where did that come from?" is answered from the previous turn's sources without calling the LLM. `/chat` and `/chat/stream` also return the sources as data, and the UI shows them as chips.

### Knowledge graph (`app/tools/knowledge_graph.py`)

A SQLite property graph in `data/knowledge_graph.db`. Creating a note extracts entities from it (plus an LLM triple-extraction pass). You can ask for neighbours, shortest paths, clusters, orphans, or how a concept evolved over time.

### Memory (`app/memory/`)

| Layer | Where | What |
|---|---|---|
| Conversation | in process | The last `MEMORY_SHORT_TERM_LIMIT` turns |
| Long-term | `data/memory.db` | Facts and preferences ("remember that…"), ranked by a composite score |
| Profile | `data/USER.md`, `data/MEMORY.md` | Short curated notes about you and the environment, injected into every prompt |
| Sessions | `data/sessions.db` | Full-text archive of past conversations |

Long-term memories are scored for importance (45%), recency (30%, 14-day half-life), confidence (15%) and recall frequency (10%). `memory_prune` removes memories scoring below 0.08 that are older than 7 days.

### Projects, goals and skills

- **Projects** (`app/memory/projects.py`): "continue X", "switch to X", "work on X" or "resume X" loads the project's vault folder, open tasks, related graph entities and memories into the prompt.
- **Goals** (`app/planning/goal_tracker.py`): multi-step plans are saved to `data/goals.db`. Saying "continue" or "what's next" resumes the latest unfinished one.
- **Skills** (`app/memory/skills.py`, `app/memory/skill_learner.py`): skills are Markdown procedures in `data/skills/`. After a tool sequence succeeds `SKILL_PATTERN_THRESHOLD` times (default 3), the assistant offers to save it as a skill.

### Coordinator (`app/agents/coordinator.py`)

Off by default (`MULTI_AGENT_ENABLED=false`). When on, multi-step requests go to a Coordinator:

1. The LLM writes a JSON plan of tool steps.
2. Each step runs through the tool router.
3. If `CRITIC_ENABLED`, a critic checks each result. An unsatisfied verdict is logged; the step isn't retried.
4. The LLM writes the spoken answer from the step results.

In the background it then prunes low-scoring memories and logs the tool sequence for skill learning. If the Coordinator fails or times out, the normal tool loop answers instead.

### Research agent (`app/agents/researcher.py`)

Needs web search turned on. For research questions it plans up to 4 DuckDuckGo queries, runs them concurrently, fetches the top 2 pages, and writes a report to `Research/` in your vault. It then speaks a short summary.

### MCP servers (`app/tools/mcp_client.py`)

```bash
# .env
MCP_ENABLED=true
MCP_SERVERS='[
  {"name": "github", "transport": "stdio", "command": "npx @modelcontextprotocol/server-github"},
  {"name": "notes",  "transport": "http",  "url": "http://localhost:3001/mcp"}
]'
```

Each server tool is registered as `mcp_<server>_<tool>`.

### System profiler (`app/system_profiler.py`)

```bash
python -m app.system_profiler
```

Or ask "what model should I use?". It detects RAM, VRAM, CPU cores, AVX support and Ollama. It then recommends an LLM and quantisation, a Whisper size and GPU offload settings, and prints a ready-to-paste `.env` snippet.

---

## Security model

The API is meant for **your machine only** and has no authentication.

- The server binds to `127.0.0.1`. Docker publishes it on `127.0.0.1` too.
- Browser requests are accepted only from the UI's origins (the Vite dev server and the Tauri app) and only for local host names. This covers WebSockets and blocks DNS-rebinding attacks. Clients that send no `Origin` header (curl, scripts) are allowed. To add an origin, set `ALLOWED_ORIGINS`; a `*` wildcard is refused.
- File tools (`read_file`, `write_file`, `delete_file`, …) are sandboxed to `AssistantWorkspace/`, and note tools to your vault. Paths that escape via `..` or symlinks are rejected.
- `ingest_document` reads any path you give it (see [Local documents](#local-documents-apptoolsrag)).
- Web search is off until you say "enable web search", call `POST /settings/web-search` or set `WEB_SEARCH_ENABLED=true`. MCP servers are off until configured.

Don't expose port 8000 to a network you don't trust.

---

## API reference

Base URL: `http://127.0.0.1:8000`

| Method | Path | Body / query | Returns |
|---|---|---|---|
| `GET` | `/health` | — | Status of LLM (for Ollama, a fix-it message if it's down or the model is missing), STT, TTS and wake word; web search and multi-agent flags; workspace path |
| `POST` | `/chat` | `{"message": "..."}` | `{"response", "tool_results", "sources"}` |
| `POST` | `/chat/stream` | `{"message": "..."}` | Server-sent events, see below |
| `POST` | `/tool` | `{"tool": "name", "params": {...}}` | `{"tool", "status": "ok"\|"error", "result"}` |
| `GET` | `/memory` | `?q=text` or `?category=preference` | `{"memories": [...]}` |
| `POST` | `/memory` | `{"category": "note", "content": "..."}` | `{"id", "status"}` |
| `DELETE` | `/memory/{id}` | — | `{"deleted": bool}` |
| `POST` | `/settings/web-search` | `{"enabled": true}` | new state |
| `POST` | `/settings/multi-agent` | `{"enabled": true}` | new state |
| `GET` | `/cache/stats` | — | response-cache statistics |
| `POST` | `/cache/clear`, `/cache/prune` | — | number of entries removed |
| `POST` | `/cache/toggle?enabled=false` | — | new state |
| `WS` | `/ws/audio` | see below | see below |

**`/chat/stream` events.** Each event is `data: <json>`:
- `{"token": "..."}`: status lines (`[Running doc_search...]`) and answer text, in order
- `{"tool_results": [...], "sources": [...]}`: once, after the answer
- `{"error": "..."}`: on failure
- then the literal `data: [DONE]`

**`/ws/audio` protocol.**
- Client → server: binary frames of 16 kHz mono float32 PCM; text frames `{"cmd": "text", "data": "..."}` or `{"cmd": "stop"}`
- Server → client: JSON text frames `{"type": "status"|"transcript"|"response"|"error", "data": ...}`, and binary frames of 24 kHz float32 PCM speech

---

## Tool reference

Tools can be called by the LLM or directly with `POST /tool`. Parameters marked `?` are optional.

| Area | Tool | Parameters |
|---|---|---|
| Files (sandboxed) | `read_file` | `path` |
| | `write_file` / `append_file` | `path`, `content` |
| | `list_directory` | `path?` |
| | `delete_file` | `path` |
| Web | `web_search` | `query` (web search must be enabled) |
| | `browser_action` | `task?` or `url?` (needs `pip install browser-use langchain-ollama playwright`) |
| Obsidian | `obsidian_create_note` | `title`, `content`, `folder?`, `tags?`, `related?` |
| | `obsidian_append_daily` | `content`, `section?` |
| | `obsidian_capture_idea` | `raw_thought`, `structured?` |
| | `obsidian_search` (semantic) | `query`, `top_k?` |
| | `obsidian_keyword_search` | `query` |
| | `obsidian_read_note` | `title` |
| | `obsidian_list_vault` | `folder?` |
| | `obsidian_get_related` | `note_title` |
| | `obsidian_get_project` | `project_name` |
| | `obsidian_morning_briefing` | — |
| | `obsidian_reindex` | — |
| Documents | `ingest_document` | `path`, `project?` |
| | `ingest_directory` | `directory?`, `recursive?`, `project?` |
| | `doc_search` | `query`, `top_k?`, `source_filter?` |
| | `unified_search` | `query`, `top_k?` |
| | `list_documents` | `project?` |
| | `remove_document` | `path` (a full path or a title) |
| Knowledge graph | `kg_summary`, `kg_orphans` | — |
| | `kg_neighbors` | `node`, `depth?` |
| | `kg_path` | `source`, `target`, `max_hops?` |
| | `kg_clusters` | `min_size?` |
| | `kg_timeline` | `node` |
| | `kg_expand` | `query`, `top_k?` |
| | `kg_add` | `title`, `entities`, `relations` |
| | `kg_add_text` | `text`, `source_note?` |
| Long-term memory | `save_memory` | `category`, `content` |
| | `recall_memory` | `query` |
| | `memory_scores` | — |
| | `memory_prune` | `dry_run?` |
| Profile memory | `memory_read` | `target?` (`user`, `memory` or `both`) |
| | `memory_write` | `target`, `content` |
| | `memory_replace` | `target`, `old_substring`, `new_content` |
| | `memory_remove` | `target`, `substring` |
| Sessions | `session_search` | `query` |
| | `session_list` | — |
| Projects | `project_list`, `project_status` | — |
| | `project_new` | `name`, `description?` |
| | `project_switch`, `project_archive` | `name` |
| | `project_update` | `name`, `description?`, `status?` |
| Goals | `goal_list` | — |
| | `goal_status`, `goal_abandon` | `goal_id?` (defaults to the active goal) |
| Skills | `skill_load`, `skill_delete` | `name` |
| | `skill_create`, `skill_rewrite` | `name`, `description`, `content` |
| | `skill_update` | `name`, `old_text`, `new_text` |
| | `skill_learning_stats` | — |
| | `confirm_skill_proposal` | `accepted?` |
| MCP | `mcp_status`, `mcp_list_tools` | — |
| | `mcp_<server>_<tool>` | depends on the server |
| System | `system_profile` | — |

---

## Configuration

All settings live in `app/config.py` and can be overridden in `.env` (names are case-insensitive). `.env.example` lists the common ones. The most useful:

| Setting | Default | Notes |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `llamacpp` |
| `LLM_MODEL` | `qwen2.5:3b` | Ollama model name |
| `LLM_OLLAMA_HOST` | `http://localhost:11434` | |
| `LLM_CONTEXT_LENGTH` | `8192` | Context window requested from the model |
| `LLM_MODEL_PATH` | `models/qwen2.5-3b-instruct-q4_k_m.gguf` | llama.cpp backend only |
| `LLM_N_GPU_LAYERS` | `0` | llama.cpp only: `-1` = all layers on GPU |
| `STT_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `STT_LANGUAGE` | blank | blank = auto-detect; e.g. `ur` forces Urdu |
| `WAKE_WORD_MODEL` | `hey_jarvis` | A built-in openWakeWord model, or a path to a custom one |
| `WAKE_WORD_PHRASE` | `Hey Jarvis` | Only the text shown in prompts; it doesn't change what is detected |
| `TTS_VOICE` | `af_heart` | Kokoro voice ID |
| `OBSIDIAN_VAULT_PATH` | `~/Documents/notes` | `~` is expanded |
| `OBSIDIAN_WATCH_ENABLED` | `true` | Re-index notes edited in Obsidian |
| `RAG_DOCUMENTS_DIR` | `./Documents` | Default folder for `ingest_directory` |
| `WEB_SEARCH_ENABLED` | `false` | |
| `MULTI_AGENT_ENABLED` | `false` | Route multi-step requests to the Coordinator |
| `CRITIC_ENABLED` | `true` | Coordinator's per-step check |
| `MCP_ENABLED`, `MCP_SERVERS` | `false`, `[]` | See [MCP servers](#mcp-servers-apptoolsmcp_clientpy) |
| `API_HOST`, `API_PORT` | `127.0.0.1`, `8000` | |
| `ALLOWED_ORIGINS` | blank | Extra browser origins, comma-separated |

### Choosing a model

| RAM | Model (Q4_K_M) |
|---|---|
| 8 GB | `qwen2.5:3b` |
| 16 GB | `qwen2.5:7b` or `llama3.1:8b` |
| 32 GB+ | `qwen2.5:14b` |

Smaller models follow the JSON reply format less reliably. The assistant recovers from malformed replies, but tool use is noticeably better from 7B up.

---

## Project layout

```
app/
├── main.py              entry point (voice / terminal / server / query)
├── config.py            all settings
├── router.py            FastAPI app, endpoints, origin/host guard
├── pipeline.py          one conversation turn: shortcuts, LLM + tool loop, voice loop
├── intent.py            capability classifier (regex)
├── planner.py           step planner for multi-step requests
├── cache.py             response cache
├── system_profiler.py   hardware detection + model recommendation
├── agents/              coordinator.py, researcher.py
├── audio/               microphone, wake_word, vad, stt, tts
├── llm/engine.py        Ollama and llama.cpp backends
├── memory/              manager, hot_memory, scoring, curator, projects, skills, skill_learner, session_store
├── planning/            goal_tracker.py
├── prompts/             templates.py (system prompt)
├── tools/               router (dispatch + validation), filesystem, obsidian, obsidian_sync,
│                        knowledge_graph, kg_triple_extraction, web_search, mcp_client,
│                        rag/ (ingestor, search, citations)
└── utils/               logging
ui/                      React + Tauri desktop app (see ui/README.md)
tests/                   offline test suite; eval/ holds the routing cases
data/                    runtime state (databases, profile, skills, logs), git-ignored
AssistantWorkspace/      sandbox for the file tools
models/                  GGUF files for the llama.cpp backend
```

---

## Development

```bash
pip install -r requirements-ci.txt   # the slim set the tests need
pytest -q                            # ~550 tests, offline: no models, network or vault
```

CI (`.github/workflows/ci.yml`) runs the wiring checks, the routing evaluation and then the full suite on every push and pull request.

The UI's **Test Runner** panel sends real queries to a running backend and checks the answers, as an end-to-end smoke test.

---

## Docker

```bash
docker compose up -d                                # API server + Ollama
docker compose exec ollama ollama pull qwen2.5:3b
```

Both services are published on `127.0.0.1` only. Set `OBSIDIAN_VAULT_PATH` in your shell or `.env` to mount your vault. The image runs the API server; voice mode needs your microphone, so run it natively.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "I can't reach Ollama at …" | `ollama serve` (or `systemctl start ollama`) |
| "The model X isn't downloaded" | `ollama pull X`, or set `LLM_MODEL` to a model from `ollama list` |
| Wake word never triggers | Say "Hey Jarvis". Lower `WAKE_WORD_THRESHOLD` (e.g. `0.3`), or set `AUDIO_INPUT_DEVICE` |
| No audio output | Set `AUDIO_OUTPUT_DEVICE`. On Linux check PulseAudio/PipeWire and that you're in the `audio` group |
| `pip install` fails building `llama-cpp-python` | Install `build-essential cmake`. If you only use Ollama, you can remove that line from `requirements.txt` |
| UI shows "Forbidden origin or host" | Add the UI's origin to `ALLOWED_ORIGINS` |

---

## Platforms

Tested on **Ubuntu (x86_64)**.

The code has macOS and Windows paths (sounddevice audio, the Ollama backend, Metal offload for llama.cpp), but they haven't been tested yet. If you try it, please open an issue with what worked and what didn't.

---

## Privacy

- Conversations, notes, memories and documents stay on your machine, in `data/` and your vault.
- The only network access is:
  - downloading models on first run (Hugging Face, openWakeWord, Ollama)
  - web search and the research agent, when you enable them
  - MCP servers you configure
  - `browser_action`, if you install and use it
- The system profiler only reads local hardware information.

---

## License

Copyright 2026 Muhammad Zohaib Tahir ([@zohaibtahirkhan](https://github.com/zohaibtahirkhan)).
Licensed under the [Apache License 2.0](LICENSE).
