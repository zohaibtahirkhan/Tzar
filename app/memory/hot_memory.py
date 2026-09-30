"""
Dual-target hot memory

MEMORY.md  = agent's personal notes: environment, conventions, lessons learned
USER.md    = user profile: preferences, communication style, pet peeves

Both are injected into every prompt as a frozen snapshot.
The LLM manages them via memory_write / memory_remove tools.
Character limits force compression — quality over quantity.
"""
import re
import hashlib
from pathlib import Path
from datetime import datetime

from loguru import logger

from app.config import settings
from app.memory.scoring import score_memory, on_memory_recalled, rank_memories_by_score

MEMORY_PATH = settings.data_dir / "MEMORY.md"
USER_PATH   = settings.data_dir / "USER.md"

MEMORY_CHAR_LIMIT = 1200   # ~400 tokens (reduced from 2200)
USER_CHAR_LIMIT   = 800    # ~280 tokens (reduced from 1375)

SEPARATOR = "\n§\n"   # entry delimiter — same as Hermes


# ─── File init ────────────────────────────────────────────────────────────────

def _ensure_files() -> None:
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text(
            "# Agent Memory\n"
            "# Auto-managed. One fact per entry, separated by §\n\n",
            encoding="utf-8",
        )
    if not USER_PATH.exists():
        USER_PATH.write_text(
            "# User Profile\n"
            "# Auto-managed. One fact per entry, separated by §\n\n",
            encoding="utf-8",
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _read_entries(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # Strip the header comment lines, then split on separator
    body = re.sub(r"^#.*\n", "", text, flags=re.MULTILINE).strip()
    entries = [e.strip() for e in body.split("§") if e.strip()]
    return entries


def _write_entries(path: Path, entries: list[str]) -> None:
    header = "# Agent Memory\n# Auto-managed.\n\n" if path == MEMORY_PATH else "# User Profile\n# Auto-managed.\n\n"
    path.write_text(header + SEPARATOR.join(entries) + "\n", encoding="utf-8")


def _char_count(entries: list[str]) -> int:
    return sum(len(e) for e in entries) + len(SEPARATOR) * max(0, len(entries) - 1)

def _get_memory_id(content: str) -> str:
    """Generate a deterministic ID based on content hash."""
    return hashlib.md5(content.strip().encode()).hexdigest()

# ─── Public API ───────────────────────────────────────────────────────────────

def hot_memory_add(target: str, content: str) -> str:
    """
    Add a new entry.
    target: "memory" or "user"
    Returns status string for the LLM.
    """
    _ensure_files()
    path  = MEMORY_PATH if target == "memory" else USER_PATH
    limit = MEMORY_CHAR_LIMIT if target == "memory" else USER_CHAR_LIMIT

    entries = _read_entries(path)

    # Duplicate check
    if any(content.strip() == e for e in entries):
        return f"[{target}] No duplicate added — entry already exists."

    new_total = _char_count(entries) + len(content) + len(SEPARATOR)
    if new_total > limit:
        used = _char_count(entries)
        return (
            f"[{target}] Memory at {used}/{limit} chars. "
            f"Adding this entry ({len(content)} chars) would exceed the limit. "
            f"Use memory_remove to clear space first.\n"
            f"Current entries:\n" + "\n---\n".join(entries)
        )

    entries.append(content.strip())
    _write_entries(path, entries)

    memory_id = _get_memory_id(content)
    score_memory(memory_id=memory_id, content=content)
    used = _char_count(entries)
    logger.info("hot_memory add [{}] ({}/{}): {}", target, used, limit, content[:60])
    return f"[{target}] Saved ({used}/{limit} chars)."


def hot_memory_remove(target: str, substring: str) -> str:
    """
    Remove the entry that contains `substring`.
    Fails if 0 or >1 matches — forces precision.
    """
    _ensure_files()
    path = MEMORY_PATH if target == "memory" else USER_PATH

    entries  = _read_entries(path)
    matches  = [i for i, e in enumerate(entries) if substring.lower() in e.lower()]

    if not matches:
        return f"[{target}] No entry found containing: '{substring}'"
    if len(matches) > 1:
        previews = "\n".join(f"  {i}: {entries[i][:80]}" for i in matches)
        return f"[{target}] Multiple matches — be more specific:\n{previews}"

    removed = entries.pop(matches[0])
    _write_entries(path, entries)
    logger.info("hot_memory remove [{}]: {}", target, removed[:60])
    return f"[{target}] Removed: '{removed[:80]}'"


def hot_memory_replace(target: str, old_substring: str, new_content: str) -> str:
    """
    Find the entry containing `old_substring`, replace it with `new_content`.
    """
    _ensure_files()
    path  = MEMORY_PATH if target == "memory" else USER_PATH
    limit = MEMORY_CHAR_LIMIT if target == "memory" else USER_CHAR_LIMIT

    entries = _read_entries(path)
    matches = [i for i, e in enumerate(entries) if old_substring.lower() in e.lower()]

    if not matches:
        return f"[{target}] No entry found containing: '{old_substring}'"
    if len(matches) > 1:
        previews = "\n".join(f"  {i}: {entries[i][:80]}" for i in matches)
        return f"[{target}] Multiple matches — be more specific:\n{previews}"

    entries[matches[0]] = new_content.strip()
    new_total = _char_count(entries)
    if new_total > limit:
        entries[matches[0]] = old_substring  # rollback
        return f"[{target}] Replacement would exceed {limit} char limit. Shorten the new content."

    _write_entries(path, entries)
    return f"[{target}] Replaced."


def hot_memory_read(target: str = "both") -> str:
    """Return formatted content of one or both files (for LLM inspection)."""
    _ensure_files()
    out = []

    if target in ("memory", "both"):
        entries = _read_entries(MEMORY_PATH)
        used    = _char_count(entries)
        pct     = int(used / MEMORY_CHAR_LIMIT * 100)
        out.append(
            f"══ AGENT MEMORY [{pct}% — {used}/{MEMORY_CHAR_LIMIT} chars] ══\n"
            + ("\n§\n".join(entries) if entries else "(empty)")
        )

    if target in ("user", "both"):
        entries = _read_entries(USER_PATH)
        used    = _char_count(entries)
        pct     = int(used / USER_CHAR_LIMIT * 100)
        out.append(
            f"══ USER PROFILE [{pct}% — {used}/{USER_CHAR_LIMIT} chars] ══\n"
            + ("\n§\n".join(entries) if entries else "(empty)")
        )

    return "\n\n".join(out)


def hot_memory_for_prompt() -> str:
    """
    Returns the frozen snapshot injected into every system prompt.
    Called once at the start of each pipeline turn.
    """
    _ensure_files()
    sections = []

    for path, label, limit in [
        (MEMORY_PATH, "AGENT MEMORY", MEMORY_CHAR_LIMIT),
        (USER_PATH,   "USER PROFILE", USER_CHAR_LIMIT),
    ]:
        entries = _read_entries(path)
        if not entries:
            continue

        # Generate IDs for all current entries
        entry_ids = [_get_memory_id(e) for e in entries]

        # Update access frequency
        for mid in entry_ids:
            on_memory_recalled(mid)

        # Sort by composite score (highest relevance first)
        ranked = rank_memories_by_score(entry_ids)

        # Create a map for sorting
        id_to_entry = {_get_memory_id(e): e for e in entries}

        # ranked is list[tuple[float, str]] — unpack score and id
        sorted_entries = [
            id_to_entry[mid]
            for _score, mid in ranked
            if mid in id_to_entry
        ]

        # If ranking returned nothing useful, fall back to original order
        if not sorted_entries:
            sorted_entries = entries

        used = _char_count(sorted_entries)
        pct  = int(used / limit * 100)
        body = "\n§\n".join(sorted_entries)
        sections.append(
            f"══════════════════════════════════════\n"
            f"{label} [{pct}% — {used}/{limit} chars]\n"
            f"══════════════════════════════════════\n"
            f"{body}"
        )

    return "\n\n".join(sections) if sections else ""