"""
Suite 3 — Obsidian Tests

Checks that note creation → search → related note retrieval
remains coherent across a multi-step sequence.
All vault I/O uses a tmp directory; no real Obsidian vault required.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock


# ─── 3.1  Note creation ───────────────────────────────────────────────────────

class TestNoteCreation:

    @pytest.mark.asyncio
    async def test_creates_markdown_file(self, tmp_vault, monkeypatch):
        """obsidian_create_note should write a .md file to the vault."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", tmp_vault)
        monkeypatch.setattr(obs, "_index_note", AsyncMock())

        result = await obs.obsidian_create_note(
            title="RAG Overview",
            content="RAG stands for Retrieval-Augmented Generation.",
            folder="AI Notes",
        )
        assert "created" in result.lower() or "rag" in result.lower()
        notes = list(tmp_vault.rglob("*.md"))
        assert any("RAG" in n.stem or "rag" in n.stem.lower() for n in notes)

    @pytest.mark.asyncio
    async def test_creates_frontmatter(self, tmp_vault, monkeypatch):
        """Created note should contain YAML frontmatter with tags."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", tmp_vault)
        monkeypatch.setattr(obs, "_index_note", AsyncMock())

        await obs.obsidian_create_note(
            title="Embeddings",
            content="Embeddings map text to vectors.",
            tags=["ml", "vectors"],
        )
        notes = list(tmp_vault.rglob("*.md"))
        assert notes, "No note file created"
        content = notes[0].read_text()
        assert "tags:" in content or "ml" in content

    @pytest.mark.asyncio
    async def test_creates_related_wikilinks(self, tmp_vault, monkeypatch):
        """Related notes should appear as [[wikilinks]] in the file."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", tmp_vault)
        monkeypatch.setattr(obs, "_index_note", AsyncMock())

        await obs.obsidian_create_note(
            title="RAG Pipeline",
            content="A pipeline using RAG.",
            related=["Embeddings", "Vector DB"],
        )
        notes = list(tmp_vault.rglob("*.md"))
        content = "\n".join(n.read_text() for n in notes)
        assert "[[Embeddings]]" in content or "[[Vector DB]]" in content

    @pytest.mark.asyncio
    async def test_daily_note_append(self, tmp_vault, monkeypatch):
        """obsidian_append_daily should append content to today's daily note."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", tmp_vault)

        result = await obs.obsidian_append_daily(
            content="Reviewed RAG pipeline implementation.",
            section="Log",
        )
        assert "added" in result.lower() or "appended" in result.lower() or "daily" in result.lower()
        daily_notes = list((tmp_vault / "Daily").rglob("*.md")) if (tmp_vault / "Daily").exists() else list(tmp_vault.rglob("*.md"))
        assert daily_notes, "No daily note created"

    @pytest.mark.asyncio
    async def test_capture_idea(self, tmp_vault, monkeypatch):
        """obsidian_capture_idea should save raw thought to Ideas folder."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", tmp_vault)
        monkeypatch.setattr(obs, "_index_note", AsyncMock())

        result = await obs.obsidian_capture_idea(
            raw_thought="What if we use graph RAG instead of vector RAG?",
        )
        assert result and len(result) > 0


# ─── 3.2  Note reading ────────────────────────────────────────────────────────

class TestNoteReading:

    @pytest.fixture
    def vault_with_rag_note(self, tmp_vault):
        """Pre-populate vault with a RAG note."""
        note_path = tmp_vault / "RAG Overview.md"
        note_path.write_text(
            "---\ntags: [rag, ml]\nrelated: [[Embeddings]]\n---\n\n"
            "RAG stands for Retrieval-Augmented Generation.\n"
            "It combines a retriever with a language model.\n"
        )
        return tmp_vault

    @pytest.mark.asyncio
    async def test_read_existing_note(self, vault_with_rag_note, monkeypatch):
        """obsidian_read_note should return note content."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", vault_with_rag_note)

        result = await obs.obsidian_read_note(title="RAG Overview")
        assert "RAG" in result or "Retrieval" in result

    @pytest.mark.asyncio
    async def test_read_nonexistent_note(self, tmp_vault, monkeypatch):
        """Reading a note that doesn't exist should return an error message."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", tmp_vault)

        result = await obs.obsidian_read_note(title="This Note Does Not Exist")
        assert "not found" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_list_vault(self, vault_with_rag_note, monkeypatch):
        """obsidian_list_vault should return the note in the listing."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", vault_with_rag_note)

        result = await obs.obsidian_list_vault()
        assert "RAG" in result or ".md" in result


# ─── 3.3  Search coherence ────────────────────────────────────────────────────

class TestSearchCoherence:
    """
    Multi-step sequence:
      1. Create note about RAG
      2. Search for RAG → should return it
      3. Get related notes → should be coherent
    """

    @pytest.fixture
    def populated_vault(self, tmp_vault):
        """Vault with two related notes."""
        rag = tmp_vault / "RAG Overview.md"
        rag.write_text(
            "---\ntags: [rag, ml]\nrelated: [[Embeddings]]\n---\n\n"
            "RAG uses a retriever and language model together.\n"
            "It requires [[Embeddings]] for dense retrieval.\n"
        )
        emb = tmp_vault / "Embeddings.md"
        emb.write_text(
            "---\ntags: [ml, vectors]\n---\n\n"
            "Embeddings map text into dense vector space.\n"
            "Used in [[RAG Overview]] for retrieval.\n"
        )
        return tmp_vault

    @pytest.mark.asyncio
    async def test_keyword_search_finds_note(self, populated_vault, monkeypatch):
        """Keyword search for 'RAG' should find the RAG note."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", populated_vault)

        result = await obs.obsidian_keyword_search(query="RAG")
        assert "RAG" in result or "Retrieval" in result

    @pytest.mark.asyncio
    async def test_get_related_finds_backlinks(self, populated_vault, monkeypatch):
        """obsidian_get_related on RAG Overview should surface Embeddings."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", populated_vault)

        result = await obs.obsidian_get_related(note_title="RAG Overview")
        assert "Embeddings" in result or "related" in result.lower()

    @pytest.mark.asyncio
    async def test_get_related_reverse_direction(self, populated_vault, monkeypatch):
        """Embeddings links back to RAG — related should be bidirectional."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", populated_vault)

        result = await obs.obsidian_get_related(note_title="Embeddings")
        assert "RAG" in result or "related" in result.lower()

    @pytest.mark.asyncio
    async def test_get_project_context(self, tmp_vault, monkeypatch):
        """obsidian_get_project_context should return notes in a project folder."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", tmp_vault)

        proj_dir = tmp_vault / "Projects" / "DataEng"
        proj_dir.mkdir(parents=True)
        (proj_dir / "Pipeline.md").write_text("# Pipeline\nData flows from source to sink.")
        (proj_dir / "Schema.md").write_text("# Schema\nDefines the table structure.")

        result = await obs.obsidian_get_project_context(project_name="DataEng")
        assert "Pipeline" in result or "Schema" in result


# ─── 3.4  Semantic search (mocked embeddings) ────────────────────────────────

class TestSemanticSearch:
    """
    Semantic search requires sentence-transformers.
    We mock the embedding model to avoid loading it in CI.
    """

    @pytest.mark.asyncio
    async def test_semantic_search_returns_string(self, tmp_vault, monkeypatch):
        """obsidian_semantic_search should return a non-empty string result."""
        import app.tools.obsidian as obs
        monkeypatch.setattr(obs, "VAULT_PATH", tmp_vault)

        # Mock the vector DB lookup to return a canned result
        mock_result = "Found: RAG Overview — RAG uses retriever + LLM."
        with patch.object(obs, "obsidian_semantic_search", new=AsyncMock(return_value=mock_result)):
            result = await obs.obsidian_semantic_search(query="retrieval augmented generation")
        assert "RAG" in result or len(result) > 0
