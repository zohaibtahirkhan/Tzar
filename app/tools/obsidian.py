"""
Obsidian vault integration.

Covers:
- Smart note creation (auto-title, tags, backlinks, folder routing)
- Voice-to-thought capture
- Daily note logging
- Semantic search via sentence-transformers + SQLite VSS
- Knowledge graph awareness (backlink traversal)
- Project context persistence
- Morning briefing
"""
import asyncio
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import aiofiles
import numpy as np
from loguru import logger

from app.config import settings
from app.tools.rag.citations import Source, record_sources

# Public alias — monkeypatched by tests via:
#   monkeypatch.setattr(obs, "VAULT_PATH", tmp_vault)
VAULT_PATH: Path = settings.obsidian_vault_path


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _vault() -> Path:
    import app.tools.obsidian as _self
    p = _self.VAULT_PATH
    p.mkdir(parents=True, exist_ok=True)
    return p


def _vault_subpath(*parts: str) -> Path:
    """
    Resolve a vault-relative path from tool params, refusing anything that
    escapes the vault — folder and project names are LLM output. Same sandbox
    rule as the workspace filesystem tools.
    """
    from app.tools.filesystem import _safe_path
    return _safe_path("/".join(p for p in parts if p), root=_vault().resolve())


def _folder(name: str) -> Path:
    p = _vault_subpath(name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize_title(title: str) -> str:
    """Make a string safe as a filename."""
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    title = title.strip().strip(".")
    return title[:100] or "Untitled"


def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _note_path(folder: str, title: str) -> Path:
    return _folder(folder) / f"{_sanitize_title(title)}.md"


def _extract_wikilinks(content: str) -> list[str]:
    """Find all [[Link]] references in markdown."""
    return re.findall(r"\[\[([^\]]+)\]\]", content)


def _extract_tags(content: str) -> list[str]:
    """Find all #tag references."""
    return re.findall(r"#(\w[\w/-]*)", content)


# ─── 1. Smart Note Creation ───────────────────────────────────────────────────

async def obsidian_create_note(
    title: str,
    content: str,
    folder: str = "",
    tags: list[str] = None,
    related: list[str] = None,
) -> str:
    """
    Create a well-structured Obsidian note with YAML frontmatter,
    tags, backlinks, and proper folder routing.
    """
    folder = folder or settings.obsidian_ai_notes_folder
    tags = tags or []
    related = related or []

    now = _now_str()
    tag_str = " ".join(f"#{t}" for t in tags) if tags else ""
    backlinks = "\n".join(f"[[{r}]]" for r in related) if related else ""

    frontmatter = f"""---
title: {title}
created: {now}
tags: [{", ".join(tags)}]
---
"""
    body = f"""# {title}

{content}
"""
    if tag_str:
        body += f"\n{tag_str}\n"
    if backlinks:
        body += f"\n## Related\n{backlinks}\n"

    path = _note_path(folder, title)

    # Don't overwrite — append with timestamp if exists
    appended = path.exists()
    if appended:
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(f"\n\n---\n*Updated {now}*\n\n{content}\n")
        logger.info("Obsidian: appended to existing note '{}'", title)
    else:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(frontmatter + body)
        logger.info("Obsidian: created note '{}' in {}", title, folder)

    # Index for semantic search. Read the file back so an appended update is
    # indexed against the note's full current text, not just the new fragment —
    # otherwise updates stay invisible to obsidian_search forever.
    async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as f:
        full_text = await f.read()
    await _index_note(str(path), title, full_text)

    # Auto-extract backlinks as graph edges
    from app.tools.knowledge_graph import kg_add_from_note
    wikilinks = _extract_wikilinks(content)
    if wikilinks:
        relations = [(title, "links_to", link) for link in wikilinks]
        await kg_add_from_note(title, [title] + wikilinks, relations)
    
    if settings.kg_auto_extract:
        from app.tools.knowledge_graph import kg_extract_and_index
        await kg_extract_and_index(body, source_note=title)

    if appended:
        return f"Updated existing note: {title}"
    return f"Note created: {title} (in {folder}/)"


# ─── 2. Daily Note ────────────────────────────────────────────────────────────

async def obsidian_append_daily(content: str, section: str = "Log") -> str:
    """
    Append a line to today's daily note. Creates it if it doesn't exist.
    """
    today = _today_str()
    path = _note_path(settings.obsidian_daily_notes_folder, today)
    now = _now_str()

    if not path.exists():
        template = f"""---
date: {today}
tags: [daily]
---

# {today}

## Log

## Ideas

## Tasks

"""
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(template)

    # Append under the right section
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        text = await f.read()

    entry = f"- {now}: {content}\n"

    if f"## {section}" in text:
        text = text.replace(f"## {section}\n", f"## {section}\n{entry}")
    else:
        text += f"\n## {section}\n{entry}"

    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(text)

    await _index_note(str(path), today, text)
    logger.info("Obsidian: daily note updated '{}'", today)
    return f"Logged to daily note ({today}): {content[:60]}"


# ─── 3. Voice-to-Thought Capture ─────────────────────────────────────────────

async def obsidian_capture_idea(raw_thought: str, structured: str = "") -> str:
    """
    Save a raw thought/idea. structured is the LLM-cleaned version.
    Goes into the Ideas folder with today's date.
    """
    today = _today_str()
    path = _note_path(settings.obsidian_ideas_folder, f"Ideas {today}")
    now = _now_str()

    content_to_write = structured if structured else raw_thought
    entry = f"\n## {now}\n\n{content_to_write}\n\n*Raw: {raw_thought}*\n"

    if path.exists():
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(entry)
    else:
        header = f"---\ndate: {today}\ntags: [ideas]\n---\n\n# Ideas — {today}\n"
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(header + entry)

    # Index the whole day's file, not just this entry — the embedding is keyed
    # by path, so indexing the fragment would overwrite and hide earlier ideas.
    async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as f:
        full_text = await f.read()
    await _index_note(str(path), f"Ideas {today}", full_text)
    logger.info("Obsidian: idea captured")
    return f"Idea captured in Ideas/{today}"


# ─── 4. Semantic Search ───────────────────────────────────────────────────────

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer(settings.obsidian_embed_model)
            logger.info("Embedding model loaded: {}", settings.obsidian_embed_model)
        except ImportError:
            logger.warning("sentence-transformers not installed. Semantic search unavailable.")
    return _embed_model


def _get_vector_db() -> sqlite3.Connection:
    db_path = str(settings.obsidian_vector_db)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS note_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            title TEXT,
            content_hash TEXT,
            embedding BLOB,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON note_vectors(path)")
    conn.commit()
    return conn


async def _index_note(path: str, title: str, content: str) -> bool:
    """
    Index a note's embedding for semantic search.
    Returns True if the note was (re)embedded, False if its content was
    unchanged since the last index or no embedding model is available.
    """
    model = _get_embed_model()
    if model is None:
        return False

    content_hash = hashlib.md5(content.encode()).hexdigest()

    loop = asyncio.get_running_loop()

    def _do_index() -> bool:
        conn = _get_vector_db()
        try:
            # Skip if already indexed with same content
            row = conn.execute(
                "SELECT content_hash FROM note_vectors WHERE path = ?", (path,)
            ).fetchone()
            if row and row[0] == content_hash:
                return False

            # Embed first 512 words for speed
            snippet = " ".join(content.split()[:512])
            embedding = model.encode(snippet, normalize_embeddings=True)
            blob = embedding.astype(np.float32).tobytes()

            conn.execute(
                """INSERT OR REPLACE INTO note_vectors (path, title, content_hash, embedding)
                   VALUES (?, ?, ?, ?)""",
                (path, title, content_hash, blob),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    return await loop.run_in_executor(None, _do_index)


async def _remove_note_from_index(path: str) -> bool:
    """Drop a note's embedding (the file was deleted or moved). True if a row was removed."""
    loop = asyncio.get_running_loop()

    def _do_remove() -> bool:
        conn = _get_vector_db()
        try:
            cur = conn.execute("DELETE FROM note_vectors WHERE path = ?", (path,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    return await loop.run_in_executor(None, _do_remove)


async def obsidian_semantic_search(query: str, top_k: int = 5) -> str:
    """
    Search all indexed notes semantically.
    Falls back to keyword search if no embedding model.
    """
    model = _get_embed_model()

    if model is None:
        # Keyword fallback
        return await obsidian_keyword_search(query)

    loop = asyncio.get_running_loop()

    def _do_search():
        query_vec = model.encode(query, normalize_embeddings=True).astype(np.float32)
        conn = _get_vector_db()
        rows = conn.execute(
            "SELECT path, title, embedding FROM note_vectors"
        ).fetchall()
        conn.close()

        if not rows:
            return []

        results = []
        for path, title, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            score = float(np.dot(query_vec, vec))
            results.append((score, title, path))

        results.sort(reverse=True)
        return results[:top_k]

    results = await loop.run_in_executor(None, _do_search)

    if not results:
        return "No notes found matching that query."

    lines, sources = [], []
    for score, title, path in results:
        rel = Path(path).relative_to(_vault()) if Path(path).is_relative_to(_vault()) else Path(path).name
        lines.append(f"[[{title}]] ({rel}) — relevance: {score:.2f}")
        sources.append(Source(title, str(rel), "note"))
    record_sources(sources)

    return "Semantically related notes:\n" + "\n".join(lines)


async def obsidian_keyword_search(query: str) -> str:
    """Full-text keyword search across all vault markdown files."""
    vault = _vault()
    results = []
    query_lower = query.lower()

    loop = asyncio.get_running_loop()

    def _search():
        found = []
        for md_file in vault.rglob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
                if query_lower in text.lower():
                    # Find the matching line for context
                    for line in text.splitlines():
                        if query_lower in line.lower():
                            found.append((md_file.stem, str(md_file.relative_to(vault)), line.strip()[:100]))
                            break
            except Exception:
                pass
        return found[:10]

    results = await loop.run_in_executor(None, _search)

    if not results:
        return f"No notes found containing '{query}'."

    record_sources(Source(title, path, "note") for title, path, _ in results)
    lines = [f"[[{title}]] ({path}): ...{snippet}..." for title, path, snippet in results]
    return "Notes matching '{}':\n".format(query) + "\n".join(lines)


# ─── 5. Knowledge Graph Awareness ─────────────────────────────────────────────

async def obsidian_get_related(note_title: str) -> str:
    """
    Find notes that link to or are linked from the given note.
    Returns a summary of the local knowledge graph.
    """
    vault = _vault()
    target_lower = note_title.lower()

    loop = asyncio.get_running_loop()

    def _find():
        links_to = []       # notes this note links to
        linked_from = []    # notes that link TO this note

        target_file = None
        for md in vault.rglob("*.md"):
            if md.stem.lower() == target_lower:
                target_file = md
                break

        for md in vault.rglob("*.md"):
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
                wikilinks = _extract_wikilinks(text)

                # This note links to others
                if md == target_file:
                    links_to = wikilinks

                # Other notes link to this note
                if any(l.lower() == target_lower for l in wikilinks) and md != target_file:
                    linked_from.append(md.stem)
            except Exception:
                pass

        return links_to, linked_from, target_file is not None

    links_to, linked_from, found = await loop.run_in_executor(None, _find)

    if not found:
        return f"Note '[[{note_title}]]' not found in vault."

    parts = [f"Knowledge graph for [[{note_title}]]:"]
    if links_to:
        parts.append(f"Links to: {', '.join(f'[[{l}]]' for l in links_to[:10])}")
    if linked_from:
        parts.append(f"Linked from: {', '.join(f'[[{l}]]' for l in linked_from[:10])}")
    if not links_to and not linked_from:
        parts.append("No connections found — this note is isolated.")

    return "\n".join(parts)


# ─── 6. Project Context ───────────────────────────────────────────────────────

async def obsidian_get_project_context(project_name: str) -> str:
    """
    Load all notes in a project folder and summarize their content.
    Used for 'continue working on X' type requests.
    """
    vault = _vault()
    project_folder = _vault_subpath(settings.obsidian_projects_folder, project_name)

    if not project_folder.exists():
        # Try a normalised search. Projects are stored on disk under their slug
        # ("ai-assistant") but are usually referred to by display name
        # ("AI Assistant"), so treat -, _ and space as equivalent.
        def _norm(value: str) -> str:
            return value.lower().replace("-", " ").replace("_", " ").strip()

        projects = list((vault / settings.obsidian_projects_folder).iterdir()) if (vault / settings.obsidian_projects_folder).exists() else []
        match = next((p for p in projects if _norm(p.name) == _norm(project_name)), None)
        if match:
            project_folder = match
        else:
            return f"Project '{project_name}' not found. Available projects: {', '.join(p.name for p in projects) or 'none'}"

    loop = asyncio.get_running_loop()

    def _load():
        notes = []
        for md in sorted(project_folder.rglob("*.md")):
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
                notes.append(f"=== {md.stem} ===\n{text[:800]}")
            except Exception:
                pass
        return notes

    notes = await loop.run_in_executor(None, _load)

    if not notes:
        return f"Project '{project_name}' folder is empty."

    combined = "\n\n".join(notes[:8])   # cap at 8 notes to stay within context
    return f"Project context for '{project_name}':\n\n{combined}"


# ─── 7. Morning Briefing ──────────────────────────────────────────────────────

async def obsidian_morning_briefing() -> str:
    """
    Generate a morning briefing from:
    - Today's daily note (if exists)
    - Yesterday's unfinished tasks
    - Recent ideas (last 3 days)
    - Recent notes (last 7 days)
    """
    vault = _vault()
    today = date.today()
    parts = []

    loop = asyncio.get_running_loop()

    def _gather():
        recent_notes = []
        recent_ideas = []
        unfinished_tasks = []

        cutoff_notes = 7
        cutoff_ideas = 3

        for md in vault.rglob("*.md"):
            try:
                mtime = datetime.fromtimestamp(md.stat().st_mtime).date()
                days_old = (today - mtime).days

                text = md.read_text(encoding="utf-8", errors="replace")

                if days_old <= cutoff_notes:
                    title = md.stem
                    snippet = next((l for l in text.splitlines() if l.strip() and not l.startswith("#") and not l.startswith("---")), "")
                    recent_notes.append(f"[[{title}]] ({days_old}d ago): {snippet[:80]}")

                if days_old <= cutoff_ideas and settings.obsidian_ideas_folder in str(md):
                    recent_ideas.append(f"[[{md.stem}]]")

                # Find unchecked tasks: - [ ]
                tasks = [l.strip() for l in text.splitlines() if l.strip().startswith("- [ ]")]
                if tasks:
                    unfinished_tasks.extend(f"{md.stem}: {t}" for t in tasks[:3])

            except Exception:
                pass

        return recent_notes[:6], recent_ideas[:5], unfinished_tasks[:5]

    recent_notes, recent_ideas, tasks = await loop.run_in_executor(None, _gather)

    parts.append(f"Good morning. Here is your briefing for {today.strftime('%A, %B %d')}.")

    if tasks:
        parts.append(f"You have {len(tasks)} unfinished tasks: " + "; ".join(t.split(": ", 1)[-1].replace("- [ ]", "").strip() for t in tasks[:3]) + ".")

    if recent_ideas:
        parts.append(f"You captured {len(recent_ideas)} idea{'s' if len(recent_ideas) > 1 else ''} recently.")

    if recent_notes:
        parts.append(f"Recently active notes: {', '.join(n.split(':')[0] for n in recent_notes[:4])}.")

    if not tasks and not recent_ideas and not recent_notes:
        parts.append("Your vault looks quiet. Nothing recent to report.")

    return " ".join(parts)


# ─── 8. Re-index entire vault ─────────────────────────────────────────────────

async def obsidian_reindex_vault() -> str:
    """
    Reconcile the whole vault with the search index and knowledge graph:
    index new/changed notes, prune entries for deleted notes.
    """
    from app.tools.obsidian_sync import backfill_vault
    return str(await backfill_vault())


# ─── 9. Read a note ───────────────────────────────────────────────────────────

async def obsidian_read_note(title: str) -> str:
    """Read the content of a note by title (case-insensitive search)."""
    vault = _vault()
    title_lower = title.lower()

    loop = asyncio.get_running_loop()

    def _find():
        for md in vault.rglob("*.md"):
            if md.stem.lower() == title_lower:
                return md.read_text(encoding="utf-8", errors="replace")
        return None

    content = await loop.run_in_executor(None, _find)

    if content is None:
        return f"Note '[[{title}]]' not found in vault."

    return content[:3000]  # cap for LLM context


# ─── 10. List vault structure ─────────────────────────────────────────────────

async def obsidian_list_vault(folder: str = "") -> str:
    """List notes in the vault or a subfolder."""
    vault = _vault()
    _vault_subpath(folder)          # traversal guard; keep the unresolved path for relative_to()
    target = vault / folder

    if not target.exists():
        return f"Folder '{folder}' not found in vault."

    loop = asyncio.get_running_loop()

    def _list():
        entries = []
        for md in sorted(target.rglob("*.md")):
            rel = md.relative_to(vault)
            mtime = datetime.fromtimestamp(md.stat().st_mtime).strftime("%Y-%m-%d")
            entries.append(f"[[{md.stem}]] ({rel.parent}) — {mtime}")
        return entries

    entries = await loop.run_in_executor(None, _list)

    if not entries:
        return "Vault is empty." if not folder else f"No notes in '{folder}'."

    return f"Vault contains {len(entries)} notes:\n" + "\n".join(entries[:30])