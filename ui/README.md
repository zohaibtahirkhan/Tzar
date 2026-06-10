# Tzar — Desktop UI

React + Tauri frontend for the Tzar assistant backend.

## Prerequisites

```bash
# Node
node >= 18

# Rust (for Tauri shell only)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Tauri CLI
npm install -g @tauri-apps/cli

# Linux system deps (skip on macOS/Windows)
sudo apt install libwebkit2gtk-4.0-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
```

## Development (browser only — no Rust needed)

Start the FastAPI backend first:
```bash
cd ..
python -m app.main --mode server
```

Then start Vite:
```bash
cd ui
npm install
npm run dev
# Open http://localhost:3000
```

The Vite dev server proxies `/chat`, `/memory`, `/health`, `/tool`,
`/settings` to `http://127.0.0.1:8000` automatically.
WebSocket connects to `ws://127.0.0.1:8000/ws/audio` directly.

## Development (full Tauri desktop app)

```bash
cd ui
npm install
npm run tauri:dev
```

Tauri opens a native window running the Vite dev server.
Hot-reload works — edit React files and the window updates instantly.

## Production build

```bash
cd ui
npm run tauri:build
```

Output: `ui/src-tauri/target/release/bundle/`
- Linux: `.AppImage` and `.deb`
- macOS: `.dmg`
- Windows: `.msi` and `.exe`

## Architecture

```
ui/
├── src/
│   ├── api.ts                  — typed FastAPI client + SSE streaming
│   ├── stores/useStore.ts      — Zustand global state
│   ├── App.tsx                 — shell + panel routing
│   ├── styles.css              — full stylesheet (dark terminal aesthetic)
│   └── components/
│       ├── Sidebar.tsx         — nav, status dot, project indicator
│       ├── ChatPanel.tsx       — SSE streaming chat + WebSocket audio
│       ├── MemoryPanel.tsx     — memory cards with scoring bars
│       ├── GraphPanel.tsx      — D3 force-directed knowledge graph
│       └── SecondaryPanels.tsx — Skills, Projects, Settings
├── src-tauri/
│   ├── src/main.rs             — Tauri shell (system tray, window)
│   ├── tauri.conf.json         — window config (1100×740, min 800×600)
│   └── Cargo.toml
├── vite.config.ts              — dev proxy to FastAPI
├── package.json
└── tsconfig.json

Backend endpoints used:
  GET  /health            → status dots in sidebar + settings
  POST /chat              → (unused — we use /chat/stream)
  POST /chat/stream       → SSE streaming in ChatPanel
  WS   /ws/audio          → voice input in ChatPanel
  GET  /memory            → MemoryPanel cards
  POST /memory            → save from UI
  DELETE /memory/:id      → delete card
  POST /tool              → all panel data (KG, skills, projects, MCP)
  POST /settings/web-search → toggle in SettingsPanel
```

## Panels

| Panel    | Key data source              | Main interaction |
|----------|------------------------------|-----------------|
| Chat     | `/chat/stream` SSE           | Type or speak   |
| Memory   | `GET /memory` + `memory_scores` tool | Browse, delete, prune |
| Graph    | `kg_summary` + `kg_clusters` + `kg_neighbors` tools | Click node to expand |
| Skills   | `skill_learning_stats` tool  | View patterns   |
| Projects | `project_list` + `project_switch` tools | Click to load context |
| Settings | `GET /health` + `mcp_status` tool | Toggle web search |

## Window behaviour

- Close button hides to system tray (does not quit)
- Click tray icon → show window
- Tray menu: Show / Hide / Quit
- Min size: 800×600
- Default: 1100×740, centered
