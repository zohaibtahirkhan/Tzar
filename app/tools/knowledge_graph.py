"""
app/tools/knowledge_graph.py

Knowledge Graph

SQLite-backed directed property graph. Every node is an entity
(concept, person, project, tool). Every edge is a typed relation.

Combines with the embedding layer for semantic graph expansion:
  query → vector search → graph neighbours → expanded context

Schema (data/knowledge_graph.db):

    nodes
    ┌─────────────────────────────────────────────────────┐
    │ id         INTEGER PRIMARY KEY                       │
    │ name       TEXT UNIQUE                               │
    │ label      TEXT  (concept / person / project / tool) │
    │ summary    TEXT  (optional short description)        │
    │ created_at TIMESTAMP                                 │
    │ updated_at TIMESTAMP                                 │
    └─────────────────────────────────────────────────────┘

    edges
    ┌─────────────────────────────────────────────────────┐
    │ id          INTEGER PRIMARY KEY                      │
    │ source_id   INTEGER → nodes.id                       │
    │ target_id   INTEGER → nodes.id                       │
    │ relation    TEXT  (links_to / related_to / part_of   │
    │                    uses / implements / learned_from) │
    │ weight      REAL  (default 1.0)                      │
    │ source_note TEXT  (note that established this edge)  │
    │ created_at  TIMESTAMP                                │
    │ UNIQUE(source_id, target_id, relation)               │
    └─────────────────────────────────────────────────────┘

All public functions are async and registered in TOOL_REGISTRY.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import settings


# ─── DB path ─────────────────────────────────────────────────────────────────

_KG_DB = settings.data_dir / "knowledge_graph.db"


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_KG_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT UNIQUE NOT NULL COLLATE NOCASE,
            label      TEXT DEFAULT 'concept',
            summary    TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id   INTEGER NOT NULL REFERENCES nodes(id),
            target_id   INTEGER NOT NULL REFERENCES nodes(id),
            relation    TEXT NOT NULL DEFAULT 'related_to',
            weight      REAL DEFAULT 1.0,
            source_note TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now')),
            UNIQUE(source_id, target_id, relation)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_src ON edges(source_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_tgt ON edges(target_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_node_name ON nodes(name)")
    conn.commit()
    return conn


# ─── Low-level helpers ────────────────────────────────────────────────────────

def _upsert_node(conn: sqlite3.Connection, name: str, label: str = "concept") -> int:
    """Insert or update a node, return its id."""
    conn.execute(
        """INSERT INTO nodes (name, label) VALUES (?, ?)
           ON CONFLICT(name) DO UPDATE SET updated_at=datetime('now')""",
        (name.strip(), label),
    )
    row = conn.execute("SELECT id FROM nodes WHERE name=?", (name.strip(),)).fetchone()
    return row["id"]


def _upsert_edge(
    conn: sqlite3.Connection,
    source_id: int,
    target_id: int,
    relation: str,
    weight: float = 1.0,
    source_note: str = "",
) -> None:
    conn.execute(
        """INSERT INTO edges (source_id, target_id, relation, weight, source_note)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(source_id, target_id, relation)
           DO UPDATE SET weight=weight+0.1, source_note=excluded.source_note""",
        (source_id, target_id, relation, weight, source_note),
    )


# ─── Entity extraction ────────────────────────────────────────────────────────

# Simple rule-based extractor (no spaCy dependency).
# Catches CamelCase terms, acronyms, quoted phrases, and
# a list of common AI/tech concept patterns.

_TECH_CONCEPTS = re.compile(
    r"\b("
    r"RAG|LLM|LLMs|API|APIs|MCP|TTS|STT|VAD|GPU|CPU|RAM|"
    r"Transformer[s]?|Attention|Embedding[s]?|Vector[s]?|"
    r"Knowledge Graph|Fine.?tun(?:ing|ed)|RLHF|LoRA|QLoRA|"
    r"[A-Z][a-z]+(?:[A-Z][a-z]+)+|"   # CamelCase: LlamaIndex, OpenAI
    r"[A-Z]{2,}(?:\d+)?(?:\.\d+)?"    # Acronyms: GPT4, Qwen2, BERT
    r")\b",
    re.UNICODE,
)

_STOP_ENTITIES = {
    "A", "AN", "THE", "I", "OK", "MY", "IN", "OF", "TO",
    "STT", "TTS", "VAD",  # keep these as concepts but not people
}


def extract_entities(text: str) -> list[str]:
    """
    Extract likely entities/concepts from text.
    Returns a deduplicated list in order of first appearance.
    """
    found: list[str] = []
    seen: set[str] = set()

    # Tech patterns
    for m in _TECH_CONCEPTS.finditer(text):
        ent = m.group().strip()
        key = ent.lower()
        if key not in seen and ent not in _STOP_ENTITIES:
            seen.add(key)
            found.append(ent)

    # Quoted phrases (often proper nouns / named things)
    for m in re.finditer(r'"([^"]{3,40})"', text):
        ent = m.group(1).strip()
        key = ent.lower()
        if key not in seen:
            seen.add(key)
            found.append(ent)

    return found


def infer_relations(entities: list[str], source_note: str) -> list[tuple[str, str, str]]:
    """
    Infer simple co-occurrence relations between entities.
    Returns list of (source, relation, target) triples.
    """
    relations = []
    for i, a in enumerate(entities):
        for b in entities[i + 1:]:
            relations.append((a, "related_to", b))
    return relations


# ─── Public tool functions ────────────────────────────────────────────────────

async def kg_add_from_note(
    title: str,
    entities: list[str],
    relations: list[tuple[str, str, str]],
) -> str:
    """
    Add entities and relations derived from a note to the graph.
    Called automatically by obsidian_create_note.

    Args:
        title:     Note title (becomes a node itself)
        entities:  List of entity names to add as nodes
        relations: List of (source, relation, target) triples
    """
    loop = asyncio.get_running_loop()

    def _add():
        conn = _get_db()
        added_nodes = 0
        added_edges = 0

        # Add the note itself as a node
        _upsert_node(conn, title, label="note")
        added_nodes += 1

        # Add all entities
        for ent in entities:
            if ent and ent != title:
                _upsert_node(conn, ent, label="concept")
                added_nodes += 1

        # Add relations
        for src, rel, tgt in relations:
            src_id = conn.execute("SELECT id FROM nodes WHERE name=?", (src,)).fetchone()
            tgt_id = conn.execute("SELECT id FROM nodes WHERE name=?", (tgt,)).fetchone()
            if src_id and tgt_id:
                _upsert_edge(conn, src_id["id"], tgt_id["id"], rel, source_note=title)
                added_edges += 1

        conn.commit()
        conn.close()
        return added_nodes, added_edges

    added_nodes, added_edges = await loop.run_in_executor(None, _add)
    return f"KG: +{added_nodes} nodes, +{added_edges} edges (from '{title}')"


async def kg_add_from_text(text: str, source_note: str = "") -> str:
    """
    Auto-extract entities from raw text and add them to the graph.
    Used by the auto-extraction hook in obsidian_create_note.

    This is the 'active KG' function — call it whenever a note is saved.
    """
    entities = extract_entities(text)
    if not entities:
        return "KG: no entities extracted"

    relations = infer_relations(entities, source_note)
    return await kg_add_from_note(source_note or "auto", entities, relations)


async def kg_get_neighbors(node: str, depth: int = 1) -> str:
    """
    Return all nodes connected to `node` within `depth` hops.
    Useful for expanding search context.

    Args:
        node:  Entity name (case-insensitive)
        depth: BFS depth (1 = direct neighbours, 2 = neighbours of neighbours)
    """
    loop = asyncio.get_running_loop()

    def _bfs():
        conn = _get_db()
        start = conn.execute(
            "SELECT id, name, label FROM nodes WHERE name=?", (node,)
        ).fetchone()
        if not start:
            conn.close()
            return None, []

        visited: set[int] = {start["id"]}
        queue: deque[tuple[int, int]] = deque([(start["id"], 0)])
        results: list[dict] = []

        while queue:
            node_id, d = queue.popleft()
            if d >= depth:
                continue
            rows = conn.execute("""
                SELECT n.id, n.name, n.label, e.relation, e.weight
                FROM edges e
                JOIN nodes n ON (
                    CASE WHEN e.source_id=? THEN e.target_id ELSE e.source_id END = n.id
                )
                WHERE (e.source_id=? OR e.target_id=?) AND n.id NOT IN ({})
            """.format(",".join("?" * len(visited))),
                [node_id, node_id, node_id, *visited],
            ).fetchall()

            for r in rows:
                visited.add(r["id"])
                results.append({
                    "name":     r["name"],
                    "label":    r["label"],
                    "relation": r["relation"],
                    "depth":    d + 1,
                })
                queue.append((r["id"], d + 1))

        conn.close()
        return start["name"], results

    start_name, neighbours = await loop.run_in_executor(None, _bfs)

    if start_name is None:
        return f"Node '{node}' not found in knowledge graph."

    if not neighbours:
        return f"'{node}' has no connections in the knowledge graph yet."

    lines = [f"Knowledge graph neighbours of '{node}':"]
    by_depth: dict[int, list] = defaultdict(list)
    for n in neighbours:
        by_depth[n["depth"]].append(n)

    for d in sorted(by_depth):
        label = "Direct connections" if d == 1 else f"Depth-{d} connections"
        lines.append(f"\n{label}:")
        for n in by_depth[d]:
            lines.append(f"  {n['relation']} → {n['name']} ({n['label']})")

    return "\n".join(lines)


async def kg_find_path(source: str, target: str) -> str:
    """
    Find the shortest path between two nodes using BFS.

    Args:
        source: Starting entity name
        target: Target entity name
    """
    loop = asyncio.get_running_loop()

    def _bfs_path():
        conn = _get_db()

        def _get_id(name: str) -> Optional[int]:
            row = conn.execute("SELECT id FROM nodes WHERE name=?", (name,)).fetchone()
            return row["id"] if row else None

        src_id = _get_id(source)
        tgt_id = _get_id(target)

        if src_id is None:
            return None, f"Node '{source}' not found."
        if tgt_id is None:
            return None, f"Node '{target}' not found."

        # BFS over adjacency
        prev: dict[int, tuple[int, str]] = {}
        queue: deque[int] = deque([src_id])
        visited = {src_id}

        while queue:
            cur = queue.popleft()
            if cur == tgt_id:
                break
            rows = conn.execute("""
                SELECT
                    CASE WHEN source_id=? THEN target_id ELSE source_id END as next_id,
                    relation
                FROM edges
                WHERE source_id=? OR target_id=?
            """, [cur, cur, cur]).fetchall()

            for r in rows:
                nxt = r["next_id"]
                if nxt not in visited:
                    visited.add(nxt)
                    prev[nxt] = (cur, r["relation"])
                    queue.append(nxt)

        if tgt_id not in prev:
            conn.close()
            return None, f"No path found between '{source}' and '{target}'."

        # Reconstruct path
        path: list[tuple[int, str]] = []
        cur = tgt_id
        while cur != src_id:
            p_id, rel = prev[cur]
            path.append((cur, rel))
            cur = p_id
        path.reverse()

        # Resolve names while connection is still open
        id_to_name: dict[int, str] = {}
        all_ids = {src_id, tgt_id} | {nid for nid, _ in path}
        for nid in all_ids:
            row = conn.execute("SELECT name FROM nodes WHERE id=?", (nid,)).fetchone()
            id_to_name[nid] = row["name"] if row else str(nid)

        conn.close()

        steps = [id_to_name[src_id]]
        for nid, rel in path:
            steps.append(f"—[{rel}]→ {id_to_name[nid]}")
        return steps, None

    path, error = await loop.run_in_executor(None, _bfs_path)

    if error:
        return error
    return "Path: " + " ".join(path)


async def kg_find_orphans() -> str:
    """Find nodes with no edges — isolated concepts in the graph."""
    loop = asyncio.get_running_loop()

    def _find():
        conn = _get_db()
        rows = conn.execute("""
            SELECT n.name, n.label, n.created_at
            FROM nodes n
            LEFT JOIN edges e ON (e.source_id=n.id OR e.target_id=n.id)
            WHERE e.id IS NULL
            ORDER BY n.created_at DESC
            LIMIT 20
        """).fetchall()
        conn.close()
        return rows

    rows = await loop.run_in_executor(None, _find)

    if not rows:
        return "No orphan nodes — all entities are connected."

    names = [f"{r['name']} ({r['label']})" for r in rows]
    return f"Orphan nodes ({len(names)}):\n" + "\n".join(f"  - {n}" for n in names)


async def kg_find_clusters(min_size: int = 3) -> str:
    """
    Find connected clusters of related nodes using Union-Find.
    Returns clusters larger than `min_size`.
    """
    loop = asyncio.get_running_loop()

    def _cluster():
        conn = _get_db()
        nodes = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM nodes").fetchall()}
        edges = conn.execute("SELECT source_id, target_id FROM edges").fetchall()
        conn.close()

        # Union-Find
        parent = {nid: nid for nid in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for e in edges:
            if e["source_id"] in parent and e["target_id"] in parent:
                union(e["source_id"], e["target_id"])

        clusters: dict[int, list[str]] = defaultdict(list)
        for nid, name in nodes.items():
            clusters[find(nid)].append(name)

        return [c for c in clusters.values() if len(c) >= min_size]

    clusters = await loop.run_in_executor(None, _cluster)

    if not clusters:
        return "No clusters found (graph may be sparsely connected)."

    clusters.sort(key=len, reverse=True)
    lines = [f"Found {len(clusters)} cluster(s) with ≥{min_size} nodes:\n"]
    for i, c in enumerate(clusters[:8], 1):
        sample = ", ".join(c[:6])
        suffix = f" (+{len(c)-6} more)" if len(c) > 6 else ""
        lines.append(f"  Cluster {i} ({len(c)} nodes): {sample}{suffix}")
    return "\n".join(lines)


async def kg_temporal_query(node: str) -> str:
    """
    Show the timeline of when a node and its edges were created.
    Useful for 'when did I start learning about X'.

    Args:
        node: Entity name to query
    """
    loop = asyncio.get_running_loop()

    def _query():
        conn = _get_db()
        n = conn.execute("SELECT id, name, label, created_at FROM nodes WHERE name=?", (node,)).fetchone()
        if not n:
            conn.close()
            return None, []

        edges = conn.execute("""
            SELECT
                CASE WHEN e.source_id=? THEN n2.name ELSE n1.name END as partner,
                e.relation, e.source_note, e.created_at, e.weight
            FROM edges e
            JOIN nodes n1 ON e.source_id=n1.id
            JOIN nodes n2 ON e.target_id=n2.id
            WHERE e.source_id=? OR e.target_id=?
            ORDER BY e.created_at
        """, [n["id"], n["id"], n["id"]]).fetchall()
        conn.close()
        return n, edges

    node_row, edges = await loop.run_in_executor(None, _query)

    if node_row is None:
        return f"'{node}' not found in knowledge graph."

    lines = [
        f"Timeline for '{node_row['name']}' ({node_row['label']}):",
        f"  First seen: {node_row['created_at'][:10]}",
    ]

    if edges:
        lines.append(f"\n  Connections ({len(edges)}):")
        for e in edges:
            note = f" [from: {e['source_note']}]" if e["source_note"] else ""
            lines.append(
                f"  {e['created_at'][:10]} — {e['relation']} → {e['partner']}{note}"
            )
    else:
        lines.append("\n  No connections recorded yet.")

    return "\n".join(lines)


async def kg_graph_summary() -> str:
    """High-level stats about the knowledge graph."""
    loop = asyncio.get_running_loop()

    def _stats():
        conn = _get_db()
        n_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

        labels = conn.execute(
            "SELECT label, COUNT(*) as c FROM nodes GROUP BY label ORDER BY c DESC"
        ).fetchall()

        relations = conn.execute(
            "SELECT relation, COUNT(*) as c FROM edges GROUP BY relation ORDER BY c DESC LIMIT 5"
        ).fetchall()

        most_connected = conn.execute("""
            SELECT n.name, COUNT(*) as c
            FROM nodes n
            JOIN edges e ON (e.source_id=n.id OR e.target_id=n.id)
            GROUP BY n.id ORDER BY c DESC LIMIT 5
        """).fetchall()

        conn.close()
        return n_nodes, n_edges, labels, relations, most_connected

    n_nodes, n_edges, labels, relations, most_connected = await loop.run_in_executor(None, _stats)

    lines = [
        f"Knowledge Graph — {n_nodes} nodes, {n_edges} edges",
        "",
        "Node types:",
    ]
    for r in labels:
        lines.append(f"  {r['label']}: {r['c']}")

    lines.append("\nTop relation types:")
    for r in relations:
        lines.append(f"  {r['relation']}: {r['c']}")

    if most_connected:
        lines.append("\nMost connected nodes:")
        for r in most_connected:
            lines.append(f"  {r['name']} ({r['c']} edges)")

    return "\n".join(lines)


async def kg_semantic_expand(query: str, top_k: int = 5) -> str:
    """
    Semantic graph expansion.

    1. Find semantically similar notes via embedding search
    2. For each match, fetch its graph neighbours
    3. Return the expanded entity set

    This gives much richer context than vector search alone.

    Example:
        query = "What have I learned about RAG?"
        → finds notes about RAG
        → expands to: Vector DB, Embeddings, Hybrid Search, Chunking
        → returns all related concepts
    """
    from app.tools.obsidian import obsidian_semantic_search, _get_embed_model
    import numpy as np

    # Step 1: semantic search for relevant notes
    note_results = await obsidian_semantic_search(query, top_k=top_k)

    # Step 2: extract note titles from results
    titles = re.findall(r"\[\[([^\]]+)\]\]", note_results)
    if not titles:
        return f"No relevant notes found for '{query}'."

    # Step 3: expand through graph neighbours
    loop = asyncio.get_running_loop()
    all_entities: set[str] = set(titles)

    def _expand(title: str) -> list[str]:
        conn = _get_db()
        row = conn.execute("SELECT id FROM nodes WHERE name=?", (title,)).fetchone()
        if not row:
            conn.close()
            return []
        neighbours = conn.execute("""
            SELECT n.name FROM edges e
            JOIN nodes n ON (
                CASE WHEN e.source_id=? THEN e.target_id ELSE e.source_id END = n.id
            )
            WHERE e.source_id=? OR e.target_id=?
            LIMIT 10
        """, [row["id"], row["id"], row["id"]]).fetchall()
        conn.close()
        return [n["name"] for n in neighbours]

    for title in titles[:3]:   # expand top 3 matches
        neighbours = await loop.run_in_executor(None, _expand, title)
        all_entities.update(neighbours)

    entities_str = ", ".join(sorted(all_entities)[:20])
    return (
        f"Semantic graph expansion for '{query}':\n"
        f"Related notes: {', '.join(titles)}\n"
        f"Connected concepts: {entities_str}"
    )