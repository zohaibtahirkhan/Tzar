"""
app/tools/rag/search.py

Hybrid document search

Combines:
  1. Semantic similarity (MiniLM cosine similarity over doc_vectors)
  2. BM25 keyword matching (over stored chunk content)

Fusion strategy: Reciprocal Rank Fusion (RRF).
  score(d) = Σ 1 / (k + rank_i(d))  for each ranker i
  k = 60 (standard RRF constant)

This gives better results than either ranker alone, especially for
queries that mix specific terms ("Genie") with semantic intent
("what did the contract say about").

Tool name:  doc_search
Parameters: query (str), top_k (int, default 5), source_filter (str, optional)

Integration — add to tools/router.py:

    from app.tools.rag.search import doc_search
    TOOL_REGISTRY["doc_search"] = doc_search
    TOOL_SCHEMA["doc_search"] = ["query"]
"""

from __future__ import annotations

import asyncio
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from app.config import settings


# ─── BM25 (pure Python, no extra deps) ───────────────────────────────────────

class _BM25:
    """
    Minimal BM25 implementation over a list of tokenised documents.
    k1=1.5, b=0.75 (standard defaults).
    """
    K1 = 1.5
    B  = 0.75

    def __init__(self, corpus: list[list[str]]):
        self.corpus = corpus
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n, 1)
        self.df: dict[str, int] = defaultdict(int)
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1

    def score(self, query_terms: list[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        doc_len = len(doc)
        score = 0.0
        tf_map: dict[str, int] = defaultdict(int)
        for t in doc:
            tf_map[t] += 1

        for term in query_terms:
            if term not in self.df:
                continue
            tf = tf_map.get(term, 0)
            idf = math.log(
                (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1
            )
            numerator = tf * (self.K1 + 1)
            denominator = tf + self.K1 * (1 - self.B + self.B * doc_len / self.avgdl)
            score += idf * numerator / denominator
        return score


def _tokenise(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


# ─── RRF fusion ───────────────────────────────────────────────────────────────

def _rrf_fuse(
    semantic_ranking: list[int],
    bm25_ranking: list[int],
    k: int = 60,
) -> list[tuple[float, int]]:
    """
    Reciprocal Rank Fusion over two ranked lists of doc indices.
    Returns [(rrf_score, doc_idx), ...] sorted descending.
    """
    scores: dict[int, float] = defaultdict(float)
    for rank, idx in enumerate(semantic_ranking):
        scores[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_ranking):
        scores[idx] += 1.0 / (k + rank + 1)

    return sorted(((score, idx) for idx, score in scores.items()), reverse=True)


# ─── Core search ──────────────────────────────────────────────────────────────

def _search_sync(
    query: str,
    top_k: int,
    source_filter: Optional[str],
) -> list[dict]:
    """
    Blocking hybrid search. Returns list of result dicts:
    {title, source_path, chunk_index, content, score}
    """
    from app.tools.obsidian import _get_embed_model

    model = _get_embed_model()
    db_path = str(settings.obsidian_vector_db)
    conn = sqlite3.connect(db_path)

    # Ensure table exists (in case doc search is called before any ingestion)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doc_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT, title TEXT, chunk_index INTEGER,
            content TEXT, content_hash TEXT, embedding BLOB,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_path, chunk_index)
        )
    """)
    conn.commit()

    # Fetch all rows (or filtered by source)
    if source_filter:
        rows = conn.execute(
            "SELECT rowid, source_path, title, chunk_index, content, embedding "
            "FROM doc_vectors WHERE source_path LIKE ? OR title LIKE ?",
            (f"%{source_filter}%", f"%{source_filter}%"),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT rowid, source_path, title, chunk_index, content, embedding "
            "FROM doc_vectors"
        ).fetchall()
    conn.close()

    if not rows:
        return []

    rowids       = [r[0] for r in rows]
    source_paths = [r[1] for r in rows]
    titles       = [r[2] for r in rows]
    chunk_idxs   = [r[3] for r in rows]
    contents     = [r[4] for r in rows]
    embeddings   = [r[5] for r in rows]

    # ── Semantic ranking ──────────────────────────────────────────────────────
    semantic_ranking: list[int] = list(range(len(rows)))  # fallback: unsorted

    if model is not None and embeddings[0] is not None:
        q_vec = model.encode(query, normalize_embeddings=True).astype(np.float32)
        sem_scores = []
        for i, blob in enumerate(embeddings):
            try:
                vec = np.frombuffer(blob, dtype=np.float32)
                sem_scores.append((float(np.dot(q_vec, vec)), i))
            except Exception:
                sem_scores.append((0.0, i))
        sem_scores.sort(reverse=True)
        semantic_ranking = [i for _, i in sem_scores]

    # ── BM25 ranking ──────────────────────────────────────────────────────────
    corpus = [_tokenise(c) for c in contents]
    bm25 = _BM25(corpus)
    query_terms = _tokenise(query)
    bm25_scores = [(bm25.score(query_terms, i), i) for i in range(len(rows))]
    bm25_scores.sort(reverse=True)
    bm25_ranking = [i for _, i in bm25_scores]

    # ── RRF fusion ────────────────────────────────────────────────────────────
    fused = _rrf_fuse(semantic_ranking, bm25_ranking)

    results = []
    for score, idx in fused[:top_k]:
        results.append({
            "title":       titles[idx],
            "source_path": source_paths[idx],
            "chunk_index": chunk_idxs[idx],
            "content":     contents[idx],
            "score":       round(score, 4),
        })

    return results


def _format_results(results: list[dict], query: str) -> str:
    """Format results for LLM consumption + TTS-friendly display."""
    if not results:
        return f"No documents found matching '{query}'."

    parts = [f"Found {len(results)} relevant passage(s) for '{query}':\n"]

    for i, r in enumerate(results, 1):
        filename = Path(r["source_path"]).name
        snippet = r["content"][:400].replace("\n", " ").strip()
        if len(r["content"]) > 400:
            snippet += "…"
        parts.append(
            f"[{i}] {r['title']} ({filename}, chunk {r['chunk_index']})\n"
            f"     {snippet}\n"
        )

    return "\n".join(parts)


# ─── Public tool function ─────────────────────────────────────────────────────

async def doc_search(
    query: str,
    top_k: int = 5,
    source_filter: Optional[str] = None,
) -> str:
    """
    Hybrid search over ingested local documents.

    Args:
        query:         Natural-language question or keyword phrase
        top_k:         Number of results to return (default 5)
        source_filter: Optional filename/path substring to restrict search

    Returns:
        Formatted string with matched passages, for LLM context injection.

    Voice example:
        "What did the Databricks contract say about Genie?"
        → doc_search(query="Databricks contract Genie") → top passages
    """
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None, _search_sync, query, top_k, source_filter
    )
    return _format_results(results, query)


# ─── Unified search: Obsidian notes + local docs ─────────────────────────────

async def unified_search(query: str, top_k: int = 5) -> str:
    """
    Search across both Obsidian notes AND ingested local documents.
    Results are interleaved by relevance.

    Tool name: unified_search
    """
    from app.tools.obsidian import obsidian_semantic_search

    # Run both searches concurrently
    note_task = asyncio.create_task(obsidian_semantic_search(query, top_k=top_k))
    doc_task  = asyncio.create_task(doc_search(query, top_k=top_k))

    note_result, doc_result = await asyncio.gather(note_task, doc_task)

    parts = []
    if "No notes found" not in note_result:
        parts.append("── Obsidian Notes ──\n" + note_result)
    if "No documents found" not in doc_result:
        parts.append("── Local Documents ──\n" + doc_result)

    if not parts:
        return f"Nothing found matching '{query}' in notes or documents."

    return "\n\n".join(parts)
