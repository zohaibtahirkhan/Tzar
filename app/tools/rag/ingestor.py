"""
app/tools/rag/ingestor.py

Document ingestion pipeline for Phase 3 — Local Document RAG.

Supports: PDF, DOCX, EPUB, Markdown, plain text.
Chunks content, embeds with the same MiniLM model already used for
Obsidian notes, and stores in a dedicated doc_vectors table in the
same SQLite database — so obsidian_semantic_search and doc_search
can work over a shared embedding space.

Schema (in obsidian_vectors.db):

    doc_vectors
    ┌─────────────────────────────────────────────────────┐
    │ id          INTEGER PRIMARY KEY AUTOINCREMENT        │
    │ source_path TEXT     — original file path            │
    │ title       TEXT     — filename stem                 │
    │ chunk_index INTEGER  — 0-based chunk number          │
    │ content     TEXT     — raw chunk text (for display)  │
    │ content_hash TEXT    — MD5 of chunk (skip re-embed)  │
    │ embedding   BLOB     — float32 numpy bytes           │
    │ indexed_at  TIMESTAMP                                │
    └─────────────────────────────────────────────────────┘

Integration:
    - Call ingest_document(path) for individual files
    - Call ingest_directory(directory) to bulk-ingest a folder
    - doc_search(query) is registered in TOOL_REGISTRY as "doc_search"
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Iterator

import numpy as np
from loguru import logger

from app.config import settings


# ─── Chunking ─────────────────────────────────────────────────────────────────

CHUNK_WORDS   = 200     # target words per chunk
CHUNK_OVERLAP = 40      # words of overlap between consecutive chunks


def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping word-window chunks.
    Respects paragraph boundaries when possible.
    """
    # Split into paragraphs first; keep chunks paragraph-aligned
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()

        # If adding this paragraph would exceed limit, flush first
        if current_words and len(current_words) + len(para_words) > CHUNK_WORDS:
            chunks.append(" ".join(current_words))
            # Keep overlap from end of current chunk
            current_words = current_words[-CHUNK_OVERLAP:]

        current_words.extend(para_words)

        # If current buffer is large enough on its own, flush it
        while len(current_words) >= CHUNK_WORDS:
            chunks.append(" ".join(current_words[:CHUNK_WORDS]))
            current_words = current_words[CHUNK_WORDS - CHUNK_OVERLAP:]

    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if len(c.split()) >= 10]   # drop tiny trailing chunks


# ─── Extractors (one per format) ─────────────────────────────────────────────

def _extract_pdf(path: Path) -> str:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        logger.error("pdfplumber not installed — run: pip install pdfplumber")
        return ""
    except Exception as e:
        logger.error("PDF extraction failed for {}: {}", path.name, e)
        return ""


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        logger.error("python-docx not installed — run: pip install python-docx")
        return ""
    except Exception as e:
        logger.error("DOCX extraction failed for {}: {}", path.name, e)
        return ""


def _extract_epub(path: Path) -> str:
    try:
        import ebooklib
        from ebooklib import epub
        from html.parser import HTMLParser

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self._parts: list[str] = []
            def handle_data(self, data):
                self._parts.append(data)
            def get_text(self):
                return " ".join(self._parts)

        book = epub.read_epub(str(path))
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            s = _Stripper()
            s.feed(item.get_body_content().decode("utf-8", errors="replace"))
            parts.append(s.get_text())
        return "\n\n".join(parts)
    except ImportError:
        logger.error("ebooklib not installed — run: pip install ebooklib")
        return ""
    except Exception as e:
        logger.error("EPUB extraction failed for {}: {}", path.name, e)
        return ""


def _extract_markdown(path: Path) -> str:
    """Strip YAML frontmatter and Obsidian wikilinks, return plain text."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        # Remove YAML frontmatter
        raw = re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL)
        # Remove wikilinks but keep label
        raw = re.sub(r"\[\[([^\]]+)\]\]", r"\1", raw)
        # Remove markdown bold/italic
        raw = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", raw)
        # Remove heading markers
        raw = re.sub(r"^#{1,6}\s+", "", raw, flags=re.MULTILINE)
        return raw
    except Exception as e:
        logger.error("Markdown extraction failed for {}: {}", path.name, e)
        return ""


_EXTRACTORS = {
    ".pdf":  _extract_pdf,
    ".docx": _extract_docx,
    ".epub": _extract_epub,
    ".md":   _extract_markdown,
    ".txt":  lambda p: p.read_text(encoding="utf-8", errors="replace"),
    ".rst":  lambda p: p.read_text(encoding="utf-8", errors="replace"),
}


def _extract(path: Path) -> str:
    ext = path.suffix.lower()
    fn = _EXTRACTORS.get(ext)
    if fn is None:
        logger.warning("No extractor for {}", ext)
        return ""
    return fn(path)


# ─── Vector DB ────────────────────────────────────────────────────────────────

def _get_doc_db() -> sqlite3.Connection:
    """Return a connection to the shared vector DB with doc_vectors table."""
    db_path = str(settings.obsidian_vector_db)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doc_vectors (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path  TEXT,
            title        TEXT,
            chunk_index  INTEGER,
            content      TEXT,
            content_hash TEXT,
            embedding    BLOB,
            indexed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_path, chunk_index)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_path ON doc_vectors(source_path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_title ON doc_vectors(title)")
    conn.commit()
    return conn


# ─── Embedding helper ─────────────────────────────────────────────────────────

def _get_embed_model():
    """Reuse the same model loader from obsidian.py to avoid double-loading."""
    from app.tools.obsidian import _get_embed_model as _obsidian_embed
    return _obsidian_embed()


# ─── Core ingestion ───────────────────────────────────────────────────────────

def _ingest_sync(path: Path) -> tuple[int, int]:
    """
    Blocking ingestion. Returns (chunks_added, chunks_skipped).
    Run in executor — never call from async code directly.
    """
    model = _get_embed_model()
    if model is None:
        logger.error("No embedding model — cannot ingest documents")
        return 0, 0

    text = _extract(path)
    if not text.strip():
        logger.warning("No text extracted from {}", path.name)
        return 0, 0

    chunks = _chunk_text(text)
    if not chunks:
        return 0, 0

    title = path.stem
    conn = _get_doc_db()
    added = skipped = 0

    for idx, chunk in enumerate(chunks):
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()

        # Check if this chunk is already current
        row = conn.execute(
            "SELECT content_hash FROM doc_vectors WHERE source_path=? AND chunk_index=?",
            (str(path), idx),
        ).fetchone()
        if row and row[0] == chunk_hash:
            skipped += 1
            continue

        vec = model.encode(chunk, normalize_embeddings=True).astype(np.float32).tobytes()
        conn.execute(
            """INSERT OR REPLACE INTO doc_vectors
               (source_path, title, chunk_index, content, content_hash, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(path), title, idx, chunk, chunk_hash, vec),
        )
        added += 1

    # Remove stale chunks (file was re-indexed with fewer chunks)
    conn.execute(
        "DELETE FROM doc_vectors WHERE source_path=? AND chunk_index>=?",
        (str(path), len(chunks)),
    )
    conn.commit()
    conn.close()

    return added, skipped


async def ingest_document(path: str | Path) -> str:
    """
    Ingest a single document into the vector store.
    Returns a status string.
    """
    path = Path(path)
    if not path.exists():
        return f"File not found: {path}"
    if path.suffix.lower() not in _EXTRACTORS:
        return f"Unsupported format: {path.suffix} (supported: {list(_EXTRACTORS)})"

    loop = asyncio.get_running_loop()
    added, skipped = await loop.run_in_executor(None, _ingest_sync, path)

    if added == 0 and skipped == 0:
        return f"Nothing ingested from {path.name} — extraction returned no text."

    return (
        f"Ingested '{path.name}': {added} chunk(s) added"
        + (f", {skipped} unchanged" if skipped else "")
        + "."
    )


async def ingest_directory(directory: str | Path, recursive: bool = True) -> str:
    """
    Ingest all supported documents in a directory.
    Returns a summary string.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return f"Not a directory: {directory}"

    pattern = "**/*" if recursive else "*"
    files = [
        f for f in directory.glob(pattern)
        if f.is_file() and f.suffix.lower() in _EXTRACTORS
    ]

    if not files:
        return f"No supported documents found in {directory}."

    total_added = total_skipped = 0
    errors = 0

    for f in files:
        loop = asyncio.get_running_loop()
        try:
            added, skipped = await loop.run_in_executor(None, _ingest_sync, f)
            total_added   += added
            total_skipped += skipped
        except Exception as e:
            logger.error("Failed to ingest {}: {}", f.name, e)
            errors += 1

    return (
        f"Directory ingestion complete: {len(files)} files, "
        f"{total_added} chunks added, {total_skipped} unchanged"
        + (f", {errors} errors" if errors else "")
        + "."
    )


async def list_ingested_documents() -> str:
    """List all documents currently in the doc_vectors table."""
    loop = asyncio.get_running_loop()

    def _list():
        conn = _get_doc_db()
        rows = conn.execute("""
            SELECT source_path, title, COUNT(*) as chunks, MAX(indexed_at) as last_indexed
            FROM doc_vectors
            GROUP BY source_path
            ORDER BY last_indexed DESC
        """).fetchall()
        conn.close()
        return rows

    rows = await loop.run_in_executor(None, _list)

    if not rows:
        return "No documents ingested yet. Use 'ingest document <path>' to add files."

    lines = []
    for source_path, title, chunks, last_indexed in rows:
        suffix = Path(source_path).suffix.upper().lstrip(".")
        lines.append(f"  [{suffix}] {title} — {chunks} chunks (indexed {last_indexed[:10]})")

    return f"Ingested documents ({len(rows)}):\n" + "\n".join(lines)


async def remove_document(path: str | Path) -> str:
    """Remove a document's chunks from the vector store."""
    path = Path(path)
    loop = asyncio.get_running_loop()

    def _remove():
        conn = _get_doc_db()
        cur = conn.execute("DELETE FROM doc_vectors WHERE source_path=?", (str(path),))
        count = cur.rowcount
        conn.commit()
        conn.close()
        return count

    removed = await loop.run_in_executor(None, _remove)
    if removed:
        return f"Removed {removed} chunk(s) for '{path.name}' from the document index."
    return f"'{path.name}' was not found in the document index."
