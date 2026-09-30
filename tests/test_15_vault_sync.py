"""
Vault sync — the search index and knowledge graph must follow the vault when
notes are edited outside the assistant.

Everything here runs against a temp vault, a temp vector DB, and a stub
embedding model. The watchdog observer itself (OS file events) is not
exercised — the debounce/dispatch logic is driven directly instead, so these
tests are deterministic and need no filesystem-event timing.
"""
import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

import app.tools.obsidian as obs
from app.tools import obsidian_sync as sync
from app.tools.obsidian_sync import BackfillReport, VaultWatcher, backfill_vault, should_track, sync_note


class _StubEmbedder:
    """Deterministic stand-in for SentenceTransformer that counts calls."""
    def __init__(self):
        self.calls = 0

    def encode(self, text, normalize_embeddings=True):
        self.calls += 1
        return np.ones(8, dtype=np.float32)


@pytest.fixture
def vault(tmp_vault, tmp_path, monkeypatch):
    """Isolated vault + vector DB + stub embedder, wired the way the module reads them."""
    monkeypatch.setattr(obs, "VAULT_PATH", tmp_vault)
    monkeypatch.setattr(obs.settings, "obsidian_vector_db", tmp_path / "vectors.db")
    stub = _StubEmbedder()
    monkeypatch.setattr(obs, "_embed_model", stub)
    monkeypatch.setattr(sync.settings, "kg_auto_extract", False)
    return _Vault(tmp_vault, stub)


class _Vault:
    """A vault path plus its stub embedder, so tests can assert on embed calls."""
    def __init__(self, path: Path, embedder: _StubEmbedder):
        self.path = path
        self.embedder = embedder

    def __truediv__(self, other):
        return self.path / other


def _index_rows(vault) -> dict[str, str]:
    conn = sqlite3.connect(str(obs.settings.obsidian_vector_db))
    try:
        return {p: h for p, h in conn.execute("SELECT path, content_hash FROM note_vectors")}
    except sqlite3.OperationalError:
        return {}   # nothing has created the table yet — the index is empty
    finally:
        conn.close()


# ─── should_track ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel,expected", [
    ("Notes/Kafka.md", True),
    ("deep/nested/Note.md", True),
    ("Notes/image.png", False),
    ("Notes/data.txt", False),
    (".obsidian/workspace.md", False),
    (".trash/Old Note.md", False),
    ("Notes/.hidden/Secret.md", False),
    (".git/README.md", False),
])
def test_should_track(rel, expected):
    assert should_track(Path("/vault") / rel) is expected


# ─── sync_note ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_new_note_is_indexed(vault):
    note = vault / "Kafka.md"
    note.write_text("# Kafka\nStreaming platform.", encoding="utf-8")

    assert await sync_note(note) == "indexed"
    assert str(note) in _index_rows(vault)
    assert vault.embedder.calls == 1


@pytest.mark.asyncio
async def test_unchanged_note_is_not_reembedded(vault):
    note = vault / "Kafka.md"
    note.write_text("# Kafka\nStreaming platform.", encoding="utf-8")
    await sync_note(note)

    assert await sync_note(note) == "unchanged"
    assert vault.embedder.calls == 1, "unchanged content must be hash-skipped, not re-embedded"


@pytest.mark.asyncio
async def test_edited_note_is_reindexed_with_new_hash(vault):
    note = vault / "Kafka.md"
    note.write_text("# Kafka\nv1", encoding="utf-8")
    await sync_note(note)
    old_hash = _index_rows(vault)[str(note)]

    note.write_text("# Kafka\nv2 — edited in Obsidian", encoding="utf-8")
    assert await sync_note(note) == "indexed"
    assert _index_rows(vault)[str(note)] != old_hash
    assert vault.embedder.calls == 2


@pytest.mark.asyncio
async def test_deleted_note_is_removed_from_index(vault):
    note = vault / "Kafka.md"
    note.write_text("# Kafka", encoding="utf-8")
    await sync_note(note)
    note.unlink()

    assert await sync_note(note) == "removed"
    assert str(note) not in _index_rows(vault)


@pytest.mark.asyncio
async def test_untracked_path_is_skipped(vault):
    cfg = vault / ".obsidian" / "workspace.md"
    cfg.parent.mkdir()
    cfg.write_text("{}", encoding="utf-8")

    assert await sync_note(cfg) == "skipped"
    assert not _index_rows(vault)


@pytest.mark.asyncio
async def test_kg_extraction_runs_without_llm_and_only_on_change(vault, monkeypatch):
    """Background sync must never hit the LLM, and must not redo KG work for unchanged notes."""
    monkeypatch.setattr(sync.settings, "kg_auto_extract", True)
    note = vault / "Kafka.md"
    note.write_text("# Kafka\nApache Kafka streams events.", encoding="utf-8")

    with patch("app.tools.knowledge_graph.kg_extract_and_index", new=AsyncMock(return_value="ok")) as kg:
        await sync_note(note)
        kg.assert_awaited_once()
        assert kg.await_args.kwargs.get("use_llm") is False

        await sync_note(note)              # unchanged
        assert kg.await_count == 1


# ─── backfill_vault ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_backfill_indexes_prunes_and_is_idempotent(vault):
    for name in ("A", "B", "C"):
        (vault / f"{name}.md").write_text(f"# {name}", encoding="utf-8")
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "ignored.md").write_text("x", encoding="utf-8")

    # A stale index row for a note that no longer exists on disk.
    gone = vault / "Deleted.md"
    gone.write_text("# gone", encoding="utf-8")
    await sync_note(gone)
    gone.unlink()

    first = await backfill_vault()
    assert (first.indexed, first.unchanged, first.pruned, first.errors) == (3, 0, 1, 0)
    assert set(Path(p).stem for p in _index_rows(vault)) == {"A", "B", "C"}

    second = await backfill_vault()
    assert (second.indexed, second.unchanged, second.pruned) == (0, 3, 0)
    assert vault.embedder.calls == 4, "second backfill must not re-embed anything"


@pytest.mark.asyncio
async def test_backfill_does_not_touch_index_rows_outside_the_vault(vault, tmp_path):
    """Pruning is scoped to this vault's paths — other rows are someone else's."""
    foreign = tmp_path / "elsewhere" / "Other.md"
    foreign.parent.mkdir()
    foreign.write_text("# other", encoding="utf-8")
    await obs._index_note(str(foreign), "Other", "# other")

    await backfill_vault()
    assert str(foreign) in _index_rows(vault)


@pytest.mark.asyncio
async def test_reindex_tool_delegates_to_backfill(vault):
    (vault / "One.md").write_text("# One", encoding="utf-8")
    result = await obs.obsidian_reindex_vault()
    assert result == str(BackfillReport(indexed=1))


# ─── VaultWatcher debounce ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_burst_of_saves_collapses_to_one_sync(vault, monkeypatch):
    """Obsidian autosaves while typing; one note going quiet must sync once."""
    monkeypatch.setattr(sync.settings, "obsidian_watch_debounce_s", 0.05)
    loop = asyncio.get_running_loop()
    watcher = VaultWatcher(loop)
    note = vault / "Typing.md"
    note.write_text("draft", encoding="utf-8")

    with patch("app.tools.obsidian_sync.sync_note", new=AsyncMock(return_value="indexed")) as mocked:
        for _ in range(5):
            watcher._schedule(str(note))
            await asyncio.sleep(0.01)          # inside the debounce window
        await asyncio.sleep(0.15)              # past the window
        await asyncio.gather(*watcher._tasks)

        assert mocked.await_count == 1
        assert mocked.await_args.args[0] == note

    watcher.stop()


@pytest.mark.asyncio
async def test_watcher_ignores_untracked_paths(vault, monkeypatch):
    monkeypatch.setattr(sync.settings, "obsidian_watch_debounce_s", 0.01)
    watcher = VaultWatcher(asyncio.get_running_loop())

    with patch("app.tools.obsidian_sync.sync_note", new=AsyncMock()) as mocked:
        watcher._schedule(str(vault / ".obsidian" / "workspace.md"))
        watcher._schedule(str(vault / "photo.png"))
        await asyncio.sleep(0.05)
        assert mocked.await_count == 0

    watcher.stop()


def test_watcher_start_fails_cleanly_when_vault_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "VAULT_PATH", tmp_path / "does-not-exist")
    # _vault() creates the dir on demand; simulate a genuinely unusable path instead.
    monkeypatch.setattr(obs, "_vault", lambda: tmp_path / "does-not-exist")
    watcher = VaultWatcher(asyncio.new_event_loop())
    assert watcher.start() is False
