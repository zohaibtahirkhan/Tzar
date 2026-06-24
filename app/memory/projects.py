"""
app/memory/projects.py

Project Continuity. The "killer feature."

Tracks active projects and auto-loads their full context on demand:
  notes, memories, graph nodes, open tasks, recent conversations.

"Continue the AI assistant project."
    ↓
project_switch("AI Assistant")
    ↓
  Load Obsidian project folder notes
  Load memory tags matching project name
  Fetch KG cluster for project entities
  Pull open tasks (- [ ] items across project notes)
  Inject all of it into the system prompt as context
    ↓
Response feels like the assistant has been thinking about it for weeks.

Schema (data/projects.db):

    projects
    ┌─────────────────────────────────────────────────────┐
    │ id           INTEGER PRIMARY KEY                     │
    │ name         TEXT UNIQUE                             │
    │ slug         TEXT UNIQUE  (normalised key)           │
    │ description  TEXT                                    │
    │ vault_folder TEXT  (Obsidian subfolder under Projects/)│
    │ memory_tags  TEXT  (JSON list of tags)               │
    │ status       TEXT  (active / paused / archived)      │
    │ last_opened  TEXT  (ISO datetime)                    │
    │ created_at   TEXT                                    │
    └─────────────────────────────────────────────────────┘

Tools registered in TOOL_REGISTRY:
    project_list      — list all projects + last opened
    project_new       — create a project
    project_switch    — load a project's context
    project_update    — update description/status
    project_archive   — mark as archived
    project_status    — show current active project
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from app.config import settings


# ─── DB ───────────────────────────────────────────────────────────────────────

_PROJECT_DB = settings.data_dir / "projects.db"

# Module-level active project state
_active_project: Optional["ProjectContext"] = None


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_PROJECT_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT UNIQUE NOT NULL,
            slug         TEXT UNIQUE NOT NULL,
            description  TEXT DEFAULT '',
            vault_folder TEXT DEFAULT '',
            memory_tags  TEXT DEFAULT '[]',
            status       TEXT DEFAULT 'active',
            last_opened  TEXT,
            created_at   TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_proj_slug ON projects(slug)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_proj_status ON projects(status)")
    conn.commit()
    return conn


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class ProjectContext:
    """Full loaded context for an active project."""
    name: str
    slug: str
    description: str
    vault_folder: str

    # Loaded content
    notes: list[dict] = field(default_factory=list)       # [{title, snippet}]
    open_tasks: list[str] = field(default_factory=list)   # unchecked - [ ] items
    memories: list[str] = field(default_factory=list)     # relevant memory snippets
    graph_entities: list[str] = field(default_factory=list)
    loaded_at: str = ""

    def to_context_string(self) -> str:
        """Format as system-prompt context block."""
        parts = [f"ACTIVE PROJECT: {self.name}"]
        
        if self.description:
            parts.append(f"Description: {self.description}")

        if self.open_tasks:
            parts.append(f"\nOpen tasks ({len(self.open_tasks)}):")
            for t in self.open_tasks[:8]:
                parts.append(f"  ☐ {t}")

        if self.notes:
            parts.append(f"\nRecent notes ({len(self.notes)}):")
            for n in self.notes[:5]:
                parts.append(f"  [[{n['title']}]]: {n['snippet']}")

        if self.memories:
            parts.append(f"\nRelevant memories:")
            for m in self.memories[:5]:
                parts.append(f"  • {m}")

        if self.graph_entities:
            parts.append(f"\nKnown concepts: {', '.join(self.graph_entities[:15])}")

        parts.append(f"\n(Project context loaded at {self.loaded_at})")
        return "\n".join(parts)

    def to_spoken_summary(self) -> str:
        """Short spoken confirmation for TTS."""
        task_str  = f"{len(self.open_tasks)} open tasks" if self.open_tasks else "no open tasks"
        note_str  = f"{len(self.notes)} notes" if self.notes else "no notes"
        mem_str   = f"{len(self.memories)} related memories" if self.memories else "no related memories"
        return (
            f"Switched to project {self.name}. "
            f"I've loaded {note_str}, {task_str}, and {mem_str}. "
            f"What would you like to do?"
        )


# ─── CRUD ─────────────────────────────────────────────────────────────────────

async def project_new(
    name: str,
    description: str = "",
    vault_folder: str = "",
    memory_tags: list[str] = None,
) -> str:
    """
    Create a new project entry.

    Args:
        name:         Project display name (e.g. "AI Assistant")
        description:  Short description
        vault_folder: Obsidian subfolder under Projects/ (defaults to slug)
        memory_tags:  Additional tags to match memories against
    """
    slug         = _slugify(name)
    vault_folder = vault_folder or slug
    tags         = json.dumps(memory_tags or [])

    loop = asyncio.get_running_loop()

    def _create():
        conn = _get_db()
        try:
            conn.execute(
                "INSERT INTO projects (name, slug, description, vault_folder, memory_tags) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, slug, description, vault_folder, tags),
            )
            conn.commit()
            return f"Project '{name}' created (vault: Projects/{vault_folder})."
        except sqlite3.IntegrityError:
            return f"Project '{name}' already exists."
        finally:
            conn.close()

    return await loop.run_in_executor(None, _create)


async def project_list() -> str:
    """List all projects with status and last opened time."""
    loop = asyncio.get_running_loop()

    def _list():
        conn = _get_db()
        try:
            return conn.execute(
                "SELECT name, status, description, last_opened FROM projects ORDER BY last_opened DESC NULLS LAST"
            ).fetchall()
        finally:
            conn.close()

    rows = await loop.run_in_executor(None, _list)

    if not rows:
        return "No projects yet. Say 'create project <name>' to get started."

    lines = [f"Projects ({len(rows)}):"]
    for r in rows:
        last = r["last_opened"][:10] if r["last_opened"] else "never"
        status_icon = {"active": "●", "paused": "◐", "archived": "○"}.get(r["status"], "?")
        desc = f" — {r['description'][:50]}" if r["description"] else ""
        lines.append(f"  {status_icon} {r['name']}{desc} (last: {last})")

    if _active_project:
        lines.append(f"\nCurrently active: {_active_project.name}")

    return "\n".join(lines)


async def project_update(
    name: str,
    description: str = None,
    status: str = None,
) -> str:
    """Update a project's description or status."""
    slug = _slugify(name)
    loop = asyncio.get_running_loop()

    def _update():
        conn = _get_db()
        try:
            exists = conn.execute("SELECT 1 FROM projects WHERE slug=?", (slug,)).fetchone()
            if not exists:
                return 0
            
            if description is not None:
                conn.execute("UPDATE projects SET description=? WHERE slug=?", (description, slug))
            if status is not None:
                conn.execute("UPDATE projects SET status=? WHERE slug=?", (status, slug))
            conn.commit()
            return 1
        finally:
            conn.close()

    changed = await loop.run_in_executor(None, _update)
    
    if changed:
        # Refresh active context if the currently active project was updated
        if _active_project and _active_project.slug == slug:
            if description is not None:
                _active_project.description = description
        return f"Project '{name}' updated."
        
    return f"Project '{name}' not found."


async def project_archive(name: str) -> str:
    """Mark a project as archived."""
    result = await project_update(name, status="archived")
    if "updated" in result and _active_project and _slugify(name) == _active_project.slug:
        clear_active_project()
        return f"{result} Active project cleared."
    return result


async def project_status() -> str:
    """Show the currently active project."""
    if _active_project is None:
        return "No project is currently active. Say 'continue <project name>' to load one."
    return f"Active project: {_active_project.name}\n" + _active_project.to_context_string()


# ─── Context loading ──────────────────────────────────────────────────────────

async def project_switch(name: str) -> str:
    """
    Load a project's full context into memory.
    This is the main entry point — called when the user says
    "continue the X project" or "switch to X project".

    Returns a spoken summary + loads context into _active_project.
    """
    global _active_project
    slug = _slugify(name)

    # 1. Look up project in DB
    loop = asyncio.get_running_loop()

    def _find():
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM projects WHERE slug=? OR name LIKE ?",
                (slug, f"%{name}%"),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE projects SET last_opened=datetime('now') WHERE id=?",
                    (row["id"],)
                )
                conn.commit()
            return dict(row) if row else None
        finally:
            conn.close()

    proj_row = await loop.run_in_executor(None, _find)

    if not proj_row:
        return f"I don't have a project called '{name}'. Would you like me to create one?"

    ctx = ProjectContext(
        name         = proj_row["name"],
        slug         = proj_row["slug"],
        description  = proj_row.get("description", ""),
        vault_folder = proj_row.get("vault_folder", slug),
        loaded_at    = datetime.now().strftime("%H:%M"),
    )

    # 2. Load in parallel: notes, tasks, memories, graph
    await asyncio.gather(
        _load_notes(ctx),
        _load_memories(ctx, proj_row.get("memory_tags", "[]")),
        _load_graph_entities(ctx),
    )

    _active_project = ctx
    logger.info(
        "Project '{}' loaded: {} notes, {} tasks, {} memories, {} entities",
        ctx.name, len(ctx.notes), len(ctx.open_tasks),
        len(ctx.memories), len(ctx.graph_entities),
    )
    return ctx.to_spoken_summary()


async def _load_notes(ctx: ProjectContext) -> None:
    """Load notes from the project's Obsidian vault folder."""
    try:
        from app.tools.obsidian import obsidian_get_project_context
        raw = await obsidian_get_project_context(ctx.name)

        # Parse "=== Title ===\n<content>" blocks
        blocks = re.split(r"=== (.+?) ===\n", raw)
        notes = []
        open_tasks = []

        if not blocks or len(blocks) < 3:
            logger.warning("No notes parsed from vault folder '{}'", ctx.vault_folder)

        for i in range(1, len(blocks) - 1, 2):
            title   = blocks[i].strip()
            content = blocks[i + 1].strip()

            # Extract first non-heading line as snippet
            snippet_lines = [
                l for l in content.splitlines()
                if l.strip() and not l.startswith("#") and not l.startswith("---")
            ]
            snippet = snippet_lines[0][:100] if snippet_lines else ""

            notes.append({"title": title, "snippet": snippet})

            # Extract open tasks
            tasks = re.findall(r"- \[ \] (.+)", content)
            for t in tasks:
                open_tasks.append(f"{t} [{title}]")

        ctx.notes      = notes
        ctx.open_tasks = open_tasks[:20]  # cap at 20 tasks

    except Exception as e:
        logger.debug("Could not load project notes: {}", e)


async def _load_memories(ctx: ProjectContext, tags_json: str) -> None:
    """Load memories tagged with this project or matching its name."""
    try:
        from app.tools.obsidian import obsidian_semantic_search

        tags = json.loads(tags_json) if tags_json else []
        query = f"{ctx.name} {' '.join(tags)}".strip()
        
        # Semantic search for project-related notes
        result = await obsidian_semantic_search(query, top_k=5)
        lines  = [
            l for l in result.splitlines()
            if l.strip() and not l.startswith("Semantically")
        ]
        ctx.memories = lines[:8]

    except Exception as e:
        logger.debug("Could not load project memories: {}", e)


async def _load_graph_entities(ctx: ProjectContext) -> None:
    """Load knowledge graph entities related to this project."""
    try:
        from app.tools.knowledge_graph import kg_get_neighbors
        result = await kg_get_neighbors(ctx.name, depth=1)
        entities = re.findall(r"→ (.+?) \(", result)
        
        if not entities:
            logger.debug("No KG entities parsed for project '{}'", ctx.name)
            
        ctx.graph_entities = entities[:20]
    except Exception as e:
        logger.debug("Could not load KG entities for project: {}", e)


# ─── Context injection ────────────────────────────────────────────────────────

def get_active_project_context() -> str:
    """
    Return the active project's context string for injection into the
    system prompt. Called by the pipeline on every turn.

    Returns "" if no project is active (no overhead).
    """
    if _active_project is None:
        return ""
    return "\n\n" + _active_project.to_context_string()


def get_active_project_name() -> Optional[str]:
    """Return the name of the currently active project, or None."""
    return _active_project.name if _active_project else None


def clear_active_project() -> str:
    """Deactivate the current project."""
    global _active_project
    if _active_project:
        name = _active_project.name
        _active_project = None
        return f"Project '{name}' deactivated."
    return "No active project."


# ─── Intent detection helper ─────────────────────────────────────────────────

_PROJECT_NAME_PATTERN = re.compile(
    r"\b(switch to|load|open|resume|continue|work on)\s+"
    r"(?:the\s+)?(.+?)(?:\s+project)?\s*[.!]?\s*$",
    re.IGNORECASE,
)

# Fast-reject: queries starting with these verbs are NEVER project switches
_NON_PROJECT_PREFIX = re.compile(
    r"^(search|find|look|list|show|get|fetch|read|write|create|add|enable|disable|"
    r"what|who|when|where|how|why|tell|give|check|open browser|navigate|go to|"
    r"extract|capture|append|delete|remove|ingest)\b",
    re.IGNORECASE,
)


def extract_project_name_from_query(text: str) -> Optional[str]:
    """
    Extract a project name only from explicit project-management commands.
    Returns None for all other queries including searches.

    Examples:
        "continue the AI assistant project" → "AI assistant"
        "switch to workers welfare" → "workers welfare"
        "resume Databricks work" → "Databricks work"
    """
    stripped = text.strip()

    # Fast reject to avoid matching "search for...", "find...", etc.
    if _NON_PROJECT_PREFIX.match(stripped):
        return None

    m = _PROJECT_NAME_PATTERN.search(stripped)
    if m:
        candidate = m.group(2).strip().rstrip('.!').strip()
        # Filter out obviously wrong matches
        if len(candidate) > 2 and candidate.lower() not in ("my", "the", "a", "an", "it", "this", "that"):
            return candidate
    return None