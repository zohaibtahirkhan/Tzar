"""
Vault sync — keeps the semantic index and knowledge graph in step with the
Obsidian vault when notes are changed *outside* the assistant.

Two mechanisms, sharing one code path (sync_note / remove_note):

  backfill_vault()   walks the vault once, indexes anything new or changed,
                     and drops index rows whose file no longer exists.
                     Cheap on repeat runs: unchanged notes are hash-skipped.

  VaultWatcher       a watchdog observer that reacts to create / modify /
                     move / delete events, debounced per path so Obsidian's
                     autosave-while-typing collapses to one re-index.

Background work never calls the LLM. Embeddings are re-computed and the
regex knowledge-graph pass runs, but the LLM triple-extraction pass is left
to the interactive note-creation path where its cost is per request rather
than per keystroke.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import settings

# Folders Obsidian itself owns; nothing in them is a user note.
_IGNORED_DIR_NAMES = {".obsidian", ".trash", ".git"}


# ─── Core operations ──────────────────────────────────────────────────────────

def should_track(path: Path) -> bool:
    """A user note is a .md file not inside a hidden or Obsidian-internal folder."""
    if path.suffix.lower() != ".md":
        return False
    return not any(part in _IGNORED_DIR_NAMES or part.startswith(".") for part in path.parts[:-1])


async def sync_note(path: Path) -> str:
    """
    Bring one note's index entry and KG entities up to date.
    Returns "indexed", "unchanged", "removed" (file gone), or "skipped".
    """
    path = Path(path)
    if not should_track(path):
        return "skipped"
    if not path.exists():
        return await remove_note(path)

    from app.tools.obsidian import _index_note

    loop = asyncio.get_running_loop()
    try:
        content = await loop.run_in_executor(
            None, lambda: path.read_text(encoding="utf-8", errors="replace")
        )
    except OSError as exc:
        logger.warning("Vault sync: cannot read {}: {}", path.name, exc)
        return "skipped"

    changed = await _index_note(str(path), path.stem, content)
    if not changed:
        return "unchanged"

    if settings.kg_auto_extract and content.strip():
        try:
            from app.tools.knowledge_graph import kg_extract_and_index
            await kg_extract_and_index(content, source_note=path.stem, use_llm=False)
        except Exception as exc:
            logger.warning("Vault sync: KG extraction failed for {}: {}", path.name, exc)

    logger.info("Vault sync: indexed '{}'", path.stem)
    return "indexed"


async def remove_note(path: Path) -> str:
    """Drop a deleted or moved-away note from the search index."""
    from app.tools.obsidian import _remove_note_from_index

    removed = await _remove_note_from_index(str(Path(path)))
    if removed:
        logger.info("Vault sync: removed '{}' from index", Path(path).stem)
    return "removed" if removed else "skipped"


@dataclass
class BackfillReport:
    indexed: int = 0
    unchanged: int = 0
    pruned: int = 0
    errors: int = 0

    def __str__(self) -> str:
        return (
            f"Vault backfill: {self.indexed} indexed, {self.unchanged} unchanged, "
            f"{self.pruned} stale entries pruned, {self.errors} errors."
        )


async def backfill_vault() -> BackfillReport:
    """
    Reconcile the whole vault with the index: index new/changed notes and
    prune index rows whose files are gone.
    """
    from app.tools.obsidian import _vault, _get_vector_db

    vault = _vault()
    report = BackfillReport()
    loop = asyncio.get_running_loop()

    files = await loop.run_in_executor(
        None, lambda: [p for p in vault.rglob("*.md") if should_track(p)]
    )
    present = {str(p) for p in files}

    for md in files:
        try:
            outcome = await sync_note(md)
        except Exception as exc:
            report.errors += 1
            logger.warning("Vault backfill: failed on {}: {}", md, exc)
            continue
        if outcome == "indexed":
            report.indexed += 1
        elif outcome == "unchanged":
            report.unchanged += 1

    def _prune() -> int:
        conn = _get_vector_db()
        try:
            rows = conn.execute("SELECT path FROM note_vectors").fetchall()
            vault_prefix = str(vault)
            stale = [
                r[0] for r in rows
                if r[0].startswith(vault_prefix) and r[0] not in present
            ]
            for p in stale:
                conn.execute("DELETE FROM note_vectors WHERE path = ?", (p,))
            conn.commit()
            return len(stale)
        finally:
            conn.close()

    report.pruned = await loop.run_in_executor(None, _prune)
    logger.info(str(report))
    return report


# ─── Watcher ──────────────────────────────────────────────────────────────────

class VaultWatcher:
    """
    Watches the vault directory and re-syncs notes as they change.

    watchdog delivers events on its own thread; each one is handed to the
    asyncio loop, where a per-path debounce timer waits for the note to go
    quiet before sync_note() runs. A move is a remove of the old path plus a
    sync of the new one.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._observer = None
        self._pending: dict[str, asyncio.TimerHandle] = {}
        self._tasks: set[asyncio.Task] = set()
        self._lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            logger.warning(
                "watchdog not installed — vault edits made in Obsidian will not be "
                "indexed until the next restart or 'obsidian_reindex'. "
                "Install with: pip install watchdog"
            )
            return False

        from app.tools.obsidian import _vault
        vault = _vault()
        if not vault.is_dir():
            logger.warning("Vault sync: vault path does not exist: {}", vault)
            return False

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    watcher._schedule(event.src_path)

            def on_modified(self, event):
                if not event.is_directory:
                    watcher._schedule(event.src_path)

            def on_deleted(self, event):
                if not event.is_directory:
                    watcher._schedule(event.src_path)   # sync_note → remove_note when missing

            def on_moved(self, event):
                if not event.is_directory:
                    watcher._schedule(event.src_path)
                    watcher._schedule(event.dest_path)

        self._observer = Observer()
        self._observer.schedule(_Handler(), str(vault), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        logger.info("Vault sync: watching {} (debounce {}s)", vault, settings.obsidian_watch_debounce_s)
        return True

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        with self._lock:
            for handle in self._pending.values():
                handle.cancel()
            self._pending.clear()

    # ── event plumbing ───────────────────────────────────────────────────────

    def _schedule(self, raw_path: str) -> None:
        """Called on the watchdog thread: (re)arm this path's debounce timer on the loop."""
        path = Path(raw_path)
        if not should_track(path):
            return
        self._loop.call_soon_threadsafe(self._arm, str(path))

    def _arm(self, key: str) -> None:
        """On the loop thread: reset the timer so a burst of saves fires once."""
        with self._lock:
            existing = self._pending.pop(key, None)
            if existing:
                existing.cancel()
            self._pending[key] = self._loop.call_later(
                settings.obsidian_watch_debounce_s, self._fire, key
            )

    def _fire(self, key: str) -> None:
        with self._lock:
            self._pending.pop(key, None)
        task = self._loop.create_task(self._run(key))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, key: str) -> None:
        try:
            await sync_note(Path(key))
        except Exception as exc:
            logger.warning("Vault sync: failed for {}: {}", key, exc)


# ─── Module-level lifecycle helpers (used by server lifespan and voice mode) ──

_watcher: Optional[VaultWatcher] = None
_backfill_task: Optional[asyncio.Task] = None


async def start_vault_sync() -> None:
    """Kick off the startup backfill (in the background) and the watcher."""
    global _watcher, _backfill_task

    if settings.obsidian_backfill_on_start:
        _backfill_task = asyncio.create_task(backfill_vault())
        _backfill_task.add_done_callback(_log_backfill_failure)

    if settings.obsidian_watch_enabled:
        _watcher = VaultWatcher(asyncio.get_running_loop())
        _watcher.start()


def _log_backfill_failure(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    if exc := task.exception():
        logger.error("Vault backfill failed: {}", exc)


async def stop_vault_sync() -> None:
    global _watcher, _backfill_task
    if _watcher is not None:
        _watcher.stop()
        _watcher = None
    if _backfill_task is not None and not _backfill_task.done():
        _backfill_task.cancel()
        try:
            await _backfill_task
        except (asyncio.CancelledError, Exception):
            pass
    _backfill_task = None
