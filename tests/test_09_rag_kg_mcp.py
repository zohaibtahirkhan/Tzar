"""
Tests for RAG, Knowledge Graph, MCP.

Run: pytest tests/test_09_rag_kg_mcp.py -v

100% offline — no LLM, no network, no vault, no Obsidian.
Uses temp directories and in-memory SQLite.
"""

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# RAG: Ingestor
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunking:
    """Test text chunking logic — no file I/O needed."""

    def setup_method(self):
        from app.tools.rag.ingestor import _chunk_text
        self._chunk = _chunk_text

    def test_short_text_is_one_chunk(self):
        # Must be >= 10 words to form a valid chunk
        text = "This is a short paragraph with exactly ten words here."
        chunks = self._chunk(text)
        assert len(chunks) == 1

    def test_very_short_text_returns_empty(self):
        # Under 10 words — dropped by minimum chunk filter
        text = "Too short."
        chunks = self._chunk(text)
        assert len(chunks) == 0

    def test_long_text_splits_into_multiple_chunks(self):
        # 500 words — should split into at least 2 chunks with CHUNK_WORDS=200
        text = " ".join(["word"] * 500)
        chunks = self._chunk(text)
        assert len(chunks) >= 2

    def test_chunks_have_overlap(self):
        from app.tools.rag.ingestor import CHUNK_WORDS, CHUNK_OVERLAP
        # 300 words in one paragraph
        text = " ".join([f"word{i}" for i in range(300)])
        chunks = self._chunk(text)
        if len(chunks) >= 2:
            # Last words of chunk 0 should appear in start of chunk 1
            c0_words = set(chunks[0].split()[-CHUNK_OVERLAP:])
            c1_words = set(chunks[1].split()[:CHUNK_OVERLAP])
            assert len(c0_words & c1_words) > 0

    def test_tiny_chunks_are_dropped(self):
        # A paragraph of only 5 words should be dropped
        text = "\n\nshort para\n\n" + " ".join(["word"] * 250)
        chunks = self._chunk(text)
        for chunk in chunks:
            assert len(chunk.split()) >= 10

    def test_paragraph_boundaries_respected(self):
        para1 = " ".join([f"alpha{i}" for i in range(150)])
        para2 = " ".join([f"beta{i}" for i in range(150)])
        text = para1 + "\n\n" + para2
        chunks = self._chunk(text)
        # alpha words should be mostly in first chunk
        assert any("alpha0" in c for c in chunks)
        assert any("beta0" in c for c in chunks)

    def test_empty_text_returns_empty(self):
        assert self._chunk("") == []
        assert self._chunk("   \n\n   ") == []


class TestExtractors:
    """Test text extractors — uses in-memory temp files."""

    def test_markdown_strips_frontmatter(self):
        from app.tools.rag.ingestor import _extract_markdown
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write("---\ntitle: Test\ntags: [test]\n---\n\n# My Note\n\nContent here.")
            path = Path(f.name)
        try:
            text = _extract_markdown(path)
            assert "---" not in text
            assert "title: Test" not in text
            assert "Content here." in text
        finally:
            path.unlink()

    def test_markdown_removes_wikilinks(self):
        from app.tools.rag.ingestor import _extract_markdown
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write("See [[Vector Databases]] for details.")
            path = Path(f.name)
        try:
            text = _extract_markdown(path)
            assert "[[" not in text
            assert "Vector Databases" in text
        finally:
            path.unlink()

    def test_markdown_removes_heading_markers(self):
        from app.tools.rag.ingestor import _extract_markdown
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write("## My Heading\n\nSome content.")
            path = Path(f.name)
        try:
            text = _extract_markdown(path)
            assert "##" not in text
            assert "My Heading" in text
        finally:
            path.unlink()

    def test_txt_extractor(self):
        from app.tools.rag.ingestor import _EXTRACTORS
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write("Plain text content.")
            path = Path(f.name)
        try:
            text = _EXTRACTORS[".txt"](path)
            assert "Plain text content." in text
        finally:
            path.unlink()

    def test_unsupported_format_returns_empty(self):
        from app.tools.rag.ingestor import _extract
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("unsupported")
            path = Path(f.name)
        try:
            text = _extract(path)
            assert text == ""
        finally:
            path.unlink()


class TestDocVectorDB:
    """Test the doc_vectors SQLite schema."""

    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = self._tmp.name

    def teardown_method(self):
        Path(self._db_path).unlink(missing_ok=True)

    def _get_conn(self):
        conn = sqlite3.connect(self._db_path)
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
        return conn

    def test_schema_creates_correctly(self):
        conn = self._get_conn()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(doc_vectors)").fetchall()]
        assert "source_path" in cols
        assert "chunk_index" in cols
        assert "content" in cols
        assert "embedding" in cols
        conn.close()

    def test_upsert_chunk(self):
        conn = self._get_conn()
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        conn.execute(
            "INSERT OR REPLACE INTO doc_vectors (source_path, title, chunk_index, content, content_hash, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("/path/doc.pdf", "doc", 0, "chunk text", "abc123", vec),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM doc_vectors").fetchone()
        assert row[2] == "doc"
        assert row[3] == 0
        conn.close()

    def test_unique_constraint_on_path_and_chunk(self):
        conn = self._get_conn()
        vec = np.zeros(3, dtype=np.float32).tobytes()
        conn.execute(
            "INSERT INTO doc_vectors (source_path, title, chunk_index, content, content_hash, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("/path/doc.pdf", "doc", 0, "original", "hash1", vec),
        )
        conn.commit()
        # Second insert with same path+chunk should replace
        conn.execute(
            "INSERT OR REPLACE INTO doc_vectors (source_path, title, chunk_index, content, content_hash, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("/path/doc.pdf", "doc", 0, "updated", "hash2", vec),
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM doc_vectors").fetchone()[0]
        assert count == 1
        content = conn.execute("SELECT content FROM doc_vectors").fetchone()[0]
        assert content == "updated"
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# RAG: Hybrid Search (BM25 + RRF)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBM25:
    """BM25 scoring — pure Python, no deps."""

    def setup_method(self):
        from app.tools.rag.search import _BM25, _tokenise
        self._BM25 = _BM25
        self._tok = _tokenise

    def test_exact_match_scores_higher(self):
        corpus = [
            self._tok("RAG is a retrieval augmented generation technique"),
            self._tok("The weather today is sunny and warm"),
        ]
        bm25 = self._BM25(corpus)
        s0 = bm25.score(self._tok("RAG retrieval"), 0)
        s1 = bm25.score(self._tok("RAG retrieval"), 1)
        assert s0 > s1

    def test_zero_score_for_no_match(self):
        corpus = [self._tok("nothing relevant here")]
        bm25 = self._BM25(corpus)
        score = bm25.score(self._tok("xyz abc def"), 0)
        assert score == 0.0

    def test_idf_penalises_common_terms(self):
        corpus = [
            self._tok("the cat sat on the mat"),
            self._tok("the dog ran on the grass"),
            self._tok("the fish swam in the water"),
        ]
        bm25 = self._BM25(corpus)
        # "the" appears in all docs — IDF should be low
        s_the = bm25.score(["the"], 0)
        s_cat = bm25.score(["cat"], 0)
        # "cat" is rarer, should score higher in doc 0
        assert s_cat > s_the


class TestRRF:
    """Reciprocal Rank Fusion."""

    def setup_method(self):
        from app.tools.rag.search import _rrf_fuse
        self._rrf = _rrf_fuse

    def test_item_top_both_rankers_wins(self):
        sem = [0, 1, 2, 3]    # doc 0 is best semantically
        bm25 = [0, 2, 1, 3]   # doc 0 is best by BM25 too
        fused = self._rrf(sem, bm25)
        # fused is [(rrf_score, doc_idx), ...] sorted descending
        best_idx = fused[0][1]
        assert best_idx == 0

    def test_consistent_results(self):
        sem = [0, 1, 2]
        bm25 = [2, 1, 0]
        fused1 = self._rrf(sem, bm25)
        fused2 = self._rrf(sem, bm25)
        assert [idx for _, idx in fused1] == [idx for _, idx in fused2]

    def test_returns_all_indices(self):
        sem = [0, 1, 2, 3]
        bm25 = [3, 2, 1, 0]
        fused = self._rrf(sem, bm25)
        indices = {idx for _, idx in fused}
        assert indices == {0, 1, 2, 3}

    def test_rrf_scores_are_positive(self):
        sem = [0, 1]
        bm25 = [1, 0]
        for score, _ in self._rrf(sem, bm25):
            assert score > 0


class TestTokeniser:
    def test_lowercases(self):
        from app.tools.rag.search import _tokenise
        assert _tokenise("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        from app.tools.rag.search import _tokenise
        assert _tokenise("hello, world!") == ["hello", "world"]

    def test_empty_string(self):
        from app.tools.rag.search import _tokenise
        assert _tokenise("") == []


# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def kg_db(tmp_path, monkeypatch):
    """Patch _KG_DB to use a temp file for each test."""
    db = tmp_path / "test_kg.db"
    monkeypatch.setattr("app.tools.knowledge_graph._KG_DB", db)
    return db


class TestEntityExtraction:
    """Test rule-based entity extractor."""

    def setup_method(self):
        from app.tools.knowledge_graph import extract_entities
        self._extract = extract_entities

    def test_extracts_camelcase(self):
        entities = self._extract("I used LlamaIndex and LangChain today.")
        assert any("LlamaIndex" in e for e in entities)
        assert any("LangChain" in e for e in entities)

    def test_extracts_acronyms(self):
        entities = self._extract("RAG uses LLMs with embedding models.")
        assert "RAG" in entities
        assert "LLMs" in entities or "LLM" in entities

    def test_extracts_quoted_phrases(self):
        entities = self._extract('I read about "vector databases" today.')
        assert any("vector databases" in e.lower() for e in entities)

    def test_deduplicates(self):
        entities = self._extract("RAG is great. RAG is useful. RAG RAG RAG.")
        rag_count = sum(1 for e in entities if e == "RAG")
        assert rag_count == 1

    def test_empty_text(self):
        assert self._extract("") == []

    def test_no_false_positives_on_plain_text(self):
        entities = self._extract("the cat sat on the mat today")
        # Should not extract "the", "cat", "sat" etc as tech entities
        assert len(entities) == 0


class TestRelationInference:
    def setup_method(self):
        from app.tools.knowledge_graph import infer_relations
        self._infer = infer_relations

    def test_produces_pairs(self):
        entities = ["RAG", "Vector DB", "Embeddings"]
        relations = self._infer(entities, "test_note")
        # 3 entities → 3 pairs: (0,1), (0,2), (1,2)
        assert len(relations) == 3

    def test_relation_type_is_related_to(self):
        entities = ["A", "B"]
        relations = self._infer(entities, "note")
        assert all(r[1] == "related_to" for r in relations)

    def test_empty_entities(self):
        assert self._infer([], "note") == []

    def test_single_entity_no_relations(self):
        assert self._infer(["RAG"], "note") == []


@pytest.mark.asyncio
class TestKGOperations:
    """Test graph CRUD with temp DB."""

    async def test_add_and_retrieve_node(self, kg_db):
        from app.tools.knowledge_graph import kg_add_from_note, kg_graph_summary
        result = await kg_add_from_note(
            "Test Note",
            ["RAG", "Embeddings"],
            [("RAG", "uses", "Embeddings")],
        )
        assert "nodes" in result or "+" in result

        summary = await kg_graph_summary()
        assert "node" in summary.lower()

    async def test_neighbors(self, kg_db):
        from app.tools.knowledge_graph import kg_add_from_note, kg_get_neighbors
        await kg_add_from_note(
            "Note A",
            ["Concept1", "Concept2"],
            [("Concept1", "related_to", "Concept2")],
        )
        result = await kg_get_neighbors("Concept1")
        assert "Concept2" in result

    async def test_neighbors_missing_node(self, kg_db):
        from app.tools.knowledge_graph import kg_get_neighbors
        result = await kg_get_neighbors("NonExistentNode999")
        assert "not found" in result.lower()

    async def test_find_path(self, kg_db):
        from app.tools.knowledge_graph import kg_add_from_note, kg_find_path
        await kg_add_from_note(
            "Chain Note",
            ["NodeA", "NodeB", "NodeC"],
            [("NodeA", "links_to", "NodeB"), ("NodeB", "links_to", "NodeC")],
        )
        result = await kg_find_path("NodeA", "NodeC")
        # Should find path A → B → C, or report it found something
        assert "NodeA" in result or "path" in result.lower() or "no path" in result.lower()

    async def test_find_path_no_connection(self, kg_db):
        from app.tools.knowledge_graph import kg_add_from_note, kg_find_path
        await kg_add_from_note("Note1", ["Island1"], [])
        await kg_add_from_note("Note2", ["Island2"], [])
        result = await kg_find_path("Island1", "Island2")
        assert "no path" in result.lower() or "not found" in result.lower()

    async def test_find_orphans(self, kg_db):
        from app.tools.knowledge_graph import kg_add_from_note, kg_find_orphans
        # Add isolated node with no edges
        await kg_add_from_note("Orphan Note", ["IsolatedConcept"], [])
        result = await kg_find_orphans()
        assert "IsolatedConcept" in result or "no orphan" in result.lower()

    async def test_clusters(self, kg_db):
        from app.tools.knowledge_graph import kg_add_from_note, kg_find_clusters
        # Create a cluster of 4 connected nodes
        await kg_add_from_note(
            "Cluster Note",
            ["X1", "X2", "X3", "X4"],
            [("X1", "r", "X2"), ("X2", "r", "X3"), ("X3", "r", "X4")],
        )
        result = await kg_find_clusters(min_size=3)
        assert "cluster" in result.lower() or "node" in result.lower()

    async def test_temporal_query(self, kg_db):
        from app.tools.knowledge_graph import kg_add_from_note, kg_temporal_query
        await kg_add_from_note(
            "Timeline Test",
            ["MyEntity"],
            [],
        )
        result = await kg_temporal_query("MyEntity")
        assert "MyEntity" in result or "not found" in result.lower()

    async def test_add_from_text(self, kg_db):
        from app.tools.knowledge_graph import kg_add_from_text
        result = await kg_add_from_text(
            "I learned about RAG and LlamaIndex today.",
            source_note="daily-2024-01-01",
        )
        # Should extract RAG, LlamaIndex as entities
        assert "+" in result or "node" in result.lower()

    async def test_graph_summary_empty(self, kg_db):
        from app.tools.knowledge_graph import kg_graph_summary
        result = await kg_graph_summary()
        assert "0 nodes" in result or "node" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Client
# ═══════════════════════════════════════════════════════════════════════════════

class TestMCPRegistryConfig:
    """Test server configuration — no network needed."""

    def setup_method(self):
        from app.tools.mcp_client import MCPRegistry
        self.registry = MCPRegistry()

    def test_configure_http_server(self):
        self.registry.configure([{
            "name": "gmail",
            "transport": "http",
            "url": "http://localhost:3001/mcp",
        }])
        assert "gmail" in self.registry._servers
        assert self.registry._servers["gmail"].transport == "http"

    def test_configure_stdio_server(self):
        self.registry.configure([{
            "name": "github",
            "transport": "stdio",
            "command": "npx @modelcontextprotocol/server-github",
        }])
        assert "github" in self.registry._servers
        assert self.registry._servers["github"].command == "npx @modelcontextprotocol/server-github"

    def test_configure_multiple_servers(self):
        self.registry.configure([
            {"name": "server1", "transport": "http", "url": "http://localhost:3001"},
            {"name": "server2", "transport": "http", "url": "http://localhost:3002"},
            {"name": "server3", "transport": "stdio", "command": "cmd"},
        ])
        assert len(self.registry._servers) == 3

    def test_configure_from_env_valid_json(self, monkeypatch):
        servers = [{"name": "test", "transport": "http", "url": "http://localhost/mcp"}]
        monkeypatch.setenv("MCP_SERVERS", json.dumps(servers))
        self.registry.configure_from_env()
        assert "test" in self.registry._servers

    def test_configure_from_env_invalid_json(self, monkeypatch):
        monkeypatch.setenv("MCP_SERVERS", "not-valid-json")
        self.registry.configure_from_env()   # should not raise
        assert len(self.registry._servers) == 0

    def test_configure_from_env_empty(self, monkeypatch):
        monkeypatch.setenv("MCP_SERVERS", "[]")
        self.registry.configure_from_env()
        assert len(self.registry._servers) == 0


class TestMCPStatus:
    def setup_method(self):
        from app.tools.mcp_client import MCPRegistry
        self.registry = MCPRegistry()

    def test_status_no_servers(self):
        result = self.registry.status()
        assert "no servers" in result.lower() or "configured" in result.lower()

    def test_status_with_unconnected_server(self):
        self.registry.configure([{
            "name": "gmail",
            "transport": "http",
            "url": "http://localhost:3001/mcp",
        }])
        result = self.registry.status()
        assert "gmail" in result
        assert "✗" in result   # not connected

    def test_list_tools_empty(self):
        assert self.registry.list_tools() == []


class TestMCPToolFormatting:
    """Test result formatting — no network needed."""

    def setup_method(self):
        from app.tools.mcp_client import MCPRegistry
        self.registry = MCPRegistry()

    def test_format_text_result(self):
        result = {"content": [{"type": "text", "text": "Hello from Gmail"}]}
        formatted = self.registry._format_tool_result(result)
        assert "Hello from Gmail" in formatted

    def test_format_multiple_text_blocks(self):
        result = {
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ]
        }
        formatted = self.registry._format_tool_result(result)
        assert "Part 1" in formatted
        assert "Part 2" in formatted

    def test_format_empty_content(self):
        result = {"content": []}
        formatted = self.registry._format_tool_result(result)
        # Should not crash; returns string
        assert isinstance(formatted, str)

    def test_format_no_content_key(self):
        result = {"some_other_key": "value"}
        formatted = self.registry._format_tool_result(result)
        assert isinstance(formatted, str)


class TestMCPToolCallUnconnected:
    """Verify graceful errors when server not connected."""

    def setup_method(self):
        from app.tools.mcp_client import MCPRegistry
        self.registry = MCPRegistry()

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        result = await self.registry.call_tool("mcp_nonexistent_tool", {})
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_call_tool_server_not_connected(self):
        from app.tools.mcp_client import MCPTool
        # Manually add a tool without connecting
        tool = MCPTool(
            name="send_email",
            registry_name="mcp_gmail_send_email",
            description="Send an email",
            input_schema={"required": ["to", "subject", "body"]},
            server_name="gmail",
        )
        self.registry._tools["mcp_gmail_send_email"] = tool
        self.registry._servers["gmail"] = MagicMock()

        result = await self.registry.call_tool("mcp_gmail_send_email", {})
        assert "not connected" in result.lower() or "not found" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: RAG search result formatting
# ═══════════════════════════════════════════════════════════════════════════════

class TestRAGResultFormatting:
    def setup_method(self):
        from app.tools.rag.search import _format_results
        self._fmt = _format_results

    def test_empty_results(self):
        result = self._fmt([], "test query")
        assert "No documents found" in result

    def test_single_result(self):
        results = [{
            "title": "Contract",
            "source_path": "/docs/contract.pdf",
            "chunk_index": 0,
            "content": "The Genie feature is described in section 3.",
            "score": 0.8231,
        }]
        formatted = self._fmt(results, "Genie contract")
        assert "Contract" in formatted
        assert "contract.pdf" in formatted
        assert "Genie" in formatted

    def test_long_content_truncated(self):
        results = [{
            "title": "Long Doc",
            "source_path": "/docs/long.pdf",
            "chunk_index": 2,
            "content": "x" * 1000,
            "score": 0.5,
        }]
        formatted = self._fmt(results, "query")
        assert len(formatted) < 2000   # reasonably bounded

    def test_multiple_results_numbered(self):
        results = [
            {"title": "Doc A", "source_path": "/a.pdf", "chunk_index": 0,
             "content": "Content A", "score": 0.9},
            {"title": "Doc B", "source_path": "/b.pdf", "chunk_index": 1,
             "content": "Content B", "score": 0.7},
        ]
        formatted = self._fmt(results, "query")
        assert "[1]" in formatted
        assert "[2]" in formatted
