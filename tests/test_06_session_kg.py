"""
Suite 6 — Session Store & Knowledge Graph Tests

Session store: all turns are archived and searchable by FTS5.
Knowledge graph: entities, typed relations, pathfinding, clustering.
"""
import pytest
from unittest.mock import AsyncMock, patch


# ─── 6.1  Session store ───────────────────────────────────────────────────────

class TestSessionStore:

    @pytest.fixture
    def fresh_session_db(self, tmp_path, monkeypatch):
        import app.memory.session_store as ss
        db_path = tmp_path / "sessions.db"
        monkeypatch.setattr(ss, "SESSION_DB", db_path)
        return db_path

    @pytest.mark.asyncio
    async def test_init_creates_db(self, fresh_session_db):
        from app.memory.session_store import init_session_db
        await init_session_db()
        assert fresh_session_db.exists()

    @pytest.mark.asyncio
    async def test_log_and_search(self, fresh_session_db):
        from app.memory.session_store import init_session_db, log_turn, session_search
        await init_session_db()
        await log_turn("user", "My favourite database is Snowflake.")
        await log_turn("assistant", "Got it, I will remember that.")
        result = await session_search("Snowflake")
        assert "Snowflake" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self, fresh_session_db):
        from app.memory.session_store import init_session_db, session_search
        await init_session_db()
        result = await session_search("quantum entanglement banana")
        assert "no" in result.lower() or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_session_list(self, fresh_session_db):
        from app.memory.session_store import init_session_db, log_turn, session_list
        await init_session_db()
        await log_turn("user", "Hello.")
        await log_turn("assistant", "Hi!")
        result = await session_list()
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.asyncio
    async def test_empty_content_not_logged(self, fresh_session_db):
        """Empty turns should be silently skipped."""
        from app.memory.session_store import init_session_db, log_turn, session_search
        await init_session_db()
        await log_turn("user", "")        # should not raise
        await log_turn("assistant", "   ")  # whitespace only
        result = await session_search("anything")
        assert "no" in result.lower() or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_multiple_sessions_searchable(self, fresh_session_db, monkeypatch):
        """Turns from different session IDs should all be searchable."""
        import app.memory.session_store as ss
        from app.memory.session_store import init_session_db, log_turn, session_search

        await init_session_db()

        monkeypatch.setattr(ss, "_CURRENT_SESSION", "20240101_090000")
        await log_turn("user", "Tell me about RAG.")
        await log_turn("assistant", "RAG uses a retriever and a language model.")

        monkeypatch.setattr(ss, "_CURRENT_SESSION", "20240102_090000")
        await log_turn("user", "Tell me about embeddings.")
        await log_turn("assistant", "Embeddings map text to dense vectors.")

        rag_result = await session_search("RAG")
        emb_result = await session_search("embeddings")

        assert "RAG" in rag_result or "retriever" in rag_result
        assert "embedding" in emb_result.lower()


# ─── 6.2  Knowledge graph ─────────────────────────────────────────────────────

class TestKnowledgeGraph:

    @pytest.fixture(autouse=True)
    def patch_kg_db(self, tmp_path, monkeypatch):
        import app.tools.knowledge_graph as kg
        monkeypatch.setattr(kg, "KG_DB_PATH", tmp_path / "kg.db")

    @pytest.mark.asyncio
    async def test_add_entities_and_relations(self):
        from app.tools.knowledge_graph import kg_add_from_note, kg_graph_summary
        result = await kg_add_from_note(
            title="RAG Overview",
            entities=["rag", "retriever", "language model"],
            relations=[
                ("rag", "uses", "retriever"),
                ("rag", "uses", "language model"),
            ],
        )
        assert "updated" in result.lower() or "graph" in result.lower()

    @pytest.mark.asyncio
    async def test_graph_summary_counts(self):
        from app.tools.knowledge_graph import kg_add_from_note, kg_graph_summary
        await kg_add_from_note(
            title="Transformers",
            entities=["transformer", "attention", "embedding"],
            relations=[("attention", "is_part_of", "transformer")],
        )
        summary = await kg_graph_summary()
        assert "concept" in summary.lower() or "node" in summary.lower() or any(c.isdigit() for c in summary)

    @pytest.mark.asyncio
    async def test_pathfinding_direct_connection(self):
        from app.tools.knowledge_graph import kg_add_from_note, kg_find_path
        await kg_add_from_note(
            title="RAG",
            entities=["rag", "embeddings"],
            relations=[("rag", "requires", "embeddings")],
        )
        path = await kg_find_path("rag", "embeddings")
        assert "rag" in path.lower() and "embeddings" in path.lower()

    @pytest.mark.asyncio
    async def test_pathfinding_no_connection(self):
        from app.tools.knowledge_graph import kg_find_path
        result = await kg_find_path("quantum_physics", "chocolate_cake", max_hops=2)
        assert "no path" in result.lower() or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_get_neighbors(self):
        from app.tools.knowledge_graph import kg_add_from_note, kg_get_neighbors
        await kg_add_from_note(
            title="Transformers",
            entities=["transformer", "attention", "ffn"],
            relations=[
                ("attention", "is_part_of", "transformer"),
                ("ffn", "is_part_of", "transformer"),
            ],
        )
        result = await kg_get_neighbors("transformer", depth=1)
        assert "attention" in result.lower() or "ffn" in result.lower()

    @pytest.mark.asyncio
    async def test_find_orphans(self):
        from app.tools.knowledge_graph import kg_add_from_note, kg_find_orphans
        # Add node with no edges
        from app.tools.knowledge_graph import _get_kg_conn, _add_node
        import asyncio
        loop = asyncio.get_event_loop()
        def _add():
            conn = _get_kg_conn()
            _add_node(conn, "isolated_concept")
            conn.commit()
            conn.close()
        await loop.run_in_executor(None, _add)

        result = await kg_find_orphans()
        assert "isolated_concept" in result or "isolated" in result.lower()

    @pytest.mark.asyncio
    async def test_clustering(self):
        from app.tools.knowledge_graph import kg_add_from_note, kg_find_clusters
        await kg_add_from_note(
            title="ML",
            entities=["transformer", "attention", "bert", "gpt"],
            relations=[
                ("attention", "is_part_of", "transformer"),
                ("bert", "uses", "transformer"),
                ("gpt", "uses", "transformer"),
            ],
        )
        result = await kg_find_clusters()
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.asyncio
    async def test_self_loop_ignored(self):
        """An edge from a concept to itself should not be added."""
        from app.tools.knowledge_graph import kg_add_from_note, kg_get_neighbors
        await kg_add_from_note(
            title="Self",
            entities=["concept"],
            relations=[("concept", "relates_to", "concept")],
        )
        result = await kg_get_neighbors("concept")
        # Should say no connections or empty — not loop to itself
        assert "concept" not in result.lower() or "no connections" in result.lower()

    @pytest.mark.asyncio
    async def test_temporal_query(self):
        from app.tools.knowledge_graph import kg_add_from_note, kg_temporal_query
        await kg_add_from_note(
            title="Test",
            entities=["alpha", "beta"],
            relations=[("alpha", "relates_to", "beta")],
        )
        result = await kg_temporal_query("alpha")
        assert "alpha" in result.lower() or "timeline" in result.lower()

    @pytest.mark.asyncio
    async def test_extract_and_index_with_mock_llm(self, tmp_path, monkeypatch):
        """kg_extract_and_index should call LLM and store results."""
        import app.tools.knowledge_graph as kg
        monkeypatch.setattr(kg, "KG_DB_PATH", tmp_path / "kg2.db")

        from app.tools.knowledge_graph import kg_extract_and_index

        mock_llm = AsyncMock(return_value='{"entities": ["rag", "embeddings"], "relations": [["rag", "requires", "embeddings"]]}')
        result = await kg_extract_and_index(
            note_title="RAG Overview",
            content="RAG requires embeddings for retrieval.",
            llm_generate_fn=mock_llm,
        )
        assert "updated" in result.lower() or "graph" in result.lower()
        mock_llm.assert_called_once()
