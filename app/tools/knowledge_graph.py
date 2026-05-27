"""
Knowledge Graph for Second Brain.

Stores entities and typed relations extracted from notes/conversations
in a SQLite adjacency table. Supports:
- Adding nodes and edges
- Pathfinding between concepts
- Finding orphan (isolated) nodes
- Community/cluster detection
- Temporal queries (how did thinking evolve)
- Graph-expanded retrieval (semantic search + graph hop)
"""
import asyncio
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import settings

KG_DB_PATH = settings.data_dir / "knowledge_graph.db"


# ─── Schema ───────────────────────────────────────────────────────────────────

def _get_kg_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(KG_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kg_nodes (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            note_title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS kg_edges (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            source   TEXT NOT NULL,
            target   TEXT NOT NULL,
            relation TEXT NOT NULL DEFAULT 'relates_to',
            weight   REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source, target, relation)
        );

        CREATE INDEX IF NOT EXISTS idx_source ON kg_edges(source);
        CREATE INDEX IF NOT EXISTS idx_target ON kg_edges(target);
    """)
    conn.commit()
    return conn


# ─── Core graph ops ───────────────────────────────────────────────────────────

def _add_node(conn: sqlite3.Connection, name: str, note_title: str = "") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO kg_nodes (name, note_title) VALUES (?, ?)",
        (name.strip().lower(), note_title)
    )


def _add_edge(conn: sqlite3.Connection, source: str, target: str,
              relation: str = "relates_to", weight: float = 1.0) -> None:
    source = source.strip().lower()
    target = target.strip().lower()
    if source == target:
        return
    _add_node(conn, source)
    _add_node(conn, target)
    conn.execute(
        """INSERT INTO kg_edges (source, target, relation, weight)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(source, target, relation) DO UPDATE SET weight = weight + 0.1""",
        (source, target, relation, weight)
    )


# ─── Public async API ─────────────────────────────────────────────────────────

async def kg_add_from_note(title: str, entities: list[str], relations: list[tuple]) -> str:
    """
    Add entities and relations extracted from a note.
    entities: ["attention mechanism", "transformer", "memory"]
    relations: [("attention mechanism", "is_part_of", "transformer"),
                ("attention mechanism", "analogous_to", "memory")]
    """
    loop = asyncio.get_running_loop()

    def _write():
        conn = _get_kg_conn()
        for e in entities:
            _add_node(conn, e, note_title=title)
        for rel in relations:
            if len(rel) == 3:
                _add_edge(conn, rel[0], rel[1], rel[2])
            elif len(rel) == 2:
                _add_edge(conn, rel[0], rel[1])
        conn.commit()
        conn.close()

    await loop.run_in_executor(None, _write)
    return f"Graph updated: {len(entities)} entities, {len(relations)} relations from '{title}'"


async def kg_find_path(source: str, target: str, max_hops: int = 4) -> str:
    """
    BFS pathfinding between two concepts.
    Returns the shortest path as a human-readable string.
    """
    source = source.strip().lower()
    target = target.strip().lower()
    loop = asyncio.get_running_loop()

    def _search():
        conn = _get_kg_conn()
        # Build adjacency list
        rows = conn.execute("SELECT source, target, relation FROM kg_edges").fetchall()
        conn.close()

        adj: dict[str, list[tuple]] = {}
        for row in rows:
            s, t, r = row["source"], row["target"], row["relation"]
            adj.setdefault(s, []).append((t, r))
            adj.setdefault(t, []).append((s, r))  # undirected

        # BFS
        from collections import deque
        queue = deque([(source, [source], [])])
        visited = {source}

        while queue:
            node, path, edges = queue.popleft()
            if node == target:
                return path, edges
            if len(path) > max_hops:
                continue
            for neighbor, relation in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], edges + [relation]))
        return None, None

    path, edges = await loop.run_in_executor(None, _search)

    if path is None:
        return f"No path found between '{source}' and '{target}' within {max_hops} hops."

    parts = []
    for i, node in enumerate(path):
        parts.append(f"[[{node}]]")
        if i < len(edges):
            parts.append(f"—[{edges[i]}]→")
    return "Path: " + " ".join(parts)


async def kg_get_neighbors(node: str, depth: int = 1) -> str:
    """Get all nodes connected to a concept, up to N hops."""
    node = node.strip().lower()
    loop = asyncio.get_running_loop()

    def _find():
        conn = _get_kg_conn()
        rows = conn.execute("SELECT source, target, relation FROM kg_edges").fetchall()
        conn.close()

        adj: dict[str, list[tuple]] = {}
        for row in rows:
            s, t, r = row["source"], row["target"], row["relation"]
            adj.setdefault(s, []).append((t, r))
            adj.setdefault(t, []).append((s, r))

        visited = {node}
        frontier = {node}
        result = []

        for _ in range(depth):
            next_frontier = set()
            for n in frontier:
                for neighbor, relation in adj.get(n, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                        result.append((n, relation, neighbor))
            frontier = next_frontier

        return result

    edges = await loop.run_in_executor(None, _find)

    if not edges:
        return f"No connections found for '[[{node}]]'."

    lines = [f"Connections for [[{node}]]:"]
    for s, r, t in edges:
        lines.append(f"  [[{s}]] —[{r}]→ [[{t}]]")
    return "\n".join(lines)


async def kg_find_orphans() -> str:
    """Find nodes with no connections — candidates for linking."""
    loop = asyncio.get_running_loop()

    def _find():
        conn = _get_kg_conn()
        all_nodes = {r["name"] for r in conn.execute("SELECT name FROM kg_nodes").fetchall()}
        connected = set()
        for row in conn.execute("SELECT source, target FROM kg_edges").fetchall():
            connected.add(row["source"])
            connected.add(row["target"])
        conn.close()
        return list(all_nodes - connected)

    orphans = await loop.run_in_executor(None, _find)

    if not orphans:
        return "No orphan nodes — everything is connected."
    return f"Isolated concepts ({len(orphans)}): " + ", ".join(f"[[{o}]]" for o in orphans[:20])


async def kg_find_clusters() -> str:
    """
    Simple community detection using label propagation.
    Returns cluster summaries.
    """
    loop = asyncio.get_running_loop()

    def _cluster():
        conn = _get_kg_conn()
        nodes = [r["name"] for r in conn.execute("SELECT name FROM kg_nodes").fetchall()]
        edges = [(r["source"], r["target"]) for r in conn.execute("SELECT source, target FROM kg_edges").fetchall()]
        conn.close()

        if not nodes:
            return {}

        # Label propagation
        labels = {n: i for i, n in enumerate(nodes)}
        adj: dict[str, list[str]] = {}
        for s, t in edges:
            adj.setdefault(s, []).append(t)
            adj.setdefault(t, []).append(s)

        import random
        for _ in range(10):
            for node in random.sample(nodes, len(nodes)):
                neighbors = adj.get(node, [])
                if not neighbors:
                    continue
                neighbor_labels = [labels[n] for n in neighbors if n in labels]
                if neighbor_labels:
                    labels[node] = max(set(neighbor_labels), key=neighbor_labels.count)

        # Group by label
        clusters: dict[int, list[str]] = {}
        for node, label in labels.items():
            clusters.setdefault(label, []).append(node)
        return clusters

    clusters = await loop.run_in_executor(None, _cluster)

    if not clusters:
        return "Graph is empty."

    lines = [f"Found {len(clusters)} concept clusters:"]
    for i, (label, members) in enumerate(sorted(clusters.items(), key=lambda x: -len(x[1]))):
        if len(members) < 2:
            continue
        lines.append(f"  Cluster {i+1} ({len(members)} concepts): {', '.join(f'[[{m}]]' for m in members[:8])}")
    return "\n".join(lines)


async def kg_temporal_query(node: str, before_date: str = "", after_date: str = "") -> str:
    """
    Show how connections to a concept evolved over time.
    before_date / after_date: "YYYY-MM-DD"
    """
    node = node.strip().lower()
    loop = asyncio.get_running_loop()

    def _query():
        conn = _get_kg_conn()
        q = "SELECT source, target, relation, created_at FROM kg_edges WHERE source=? OR target=?"
        params = [node, node]
        if after_date:
            q += " AND created_at >= ?"
            params.append(after_date)
        if before_date:
            q += " AND created_at <= ?"
            params.append(before_date)
        q += " ORDER BY created_at"
        rows = conn.execute(q, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    rows = await loop.run_in_executor(None, _query)

    if not rows:
        return f"No temporal data found for '[[{node}]]'."

    lines = [f"Timeline for [[{node}]]:"]
    for r in rows:
        other = r["target"] if r["source"] == node else r["source"]
        lines.append(f"  {r['created_at'][:10]}: —[{r['relation']}]→ [[{other}]]")
    return "\n".join(lines)


async def kg_graph_summary() -> str:
    """High-level stats about the knowledge graph."""
    loop = asyncio.get_running_loop()

    def _stats():
        conn = _get_kg_conn()
        n_nodes = conn.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
        n_edges = conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
        top = conn.execute("""
            SELECT source, COUNT(*) as c FROM kg_edges GROUP BY source
            UNION
            SELECT target, COUNT(*) as c FROM kg_edges GROUP BY target
            ORDER BY c DESC LIMIT 5
        """).fetchall()
        conn.close()
        return n_nodes, n_edges, [(r[0], r[1]) for r in top]

    n_nodes, n_edges, top = await loop.run_in_executor(None, _stats)
    hub_str = ", ".join(f"[[{n}]] ({c})" for n, c in top)
    return (
        f"Knowledge graph: {n_nodes} concepts, {n_edges} relations. "
        f"Most connected: {hub_str or 'none yet'}."
    )


async def kg_extract_and_index(note_title: str, content: str, llm_generate_fn) -> str:
    """
    Call the LLM to extract entities+relations from a note,
    then index them into the graph.
    llm_generate_fn: async callable(messages, system_prompt) -> str
    """
    system = (
        "Extract entities and relations from the text. "
        "Respond ONLY with valid JSON, no prose:\n"
        '{"entities": ["concept1", "concept2"], '
        '"relations": [["concept1", "relation_type", "concept2"]]}\n'
        "relation_type examples: is_part_of, analogous_to, causes, contradicts, enables, relates_to"
    )
    messages = [{"role": "user", "content": f"Note title: {note_title}\n\n{content[:1500]}"}]

    try:
        raw = await llm_generate_fn(messages, system)
        import re, json
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return "Could not extract graph data from note."
        data = json.loads(match.group())
        entities = data.get("entities", [])
        relations = data.get("relations", [])
        return await kg_add_from_note(note_title, entities, relations)
    except Exception as e:
        logger.error("kg_extract_and_index failed: {}", e)
        return f"Graph indexing failed: {e}"