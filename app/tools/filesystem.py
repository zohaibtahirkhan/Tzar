"""
Sandboxed filesystem tool.
All paths are resolved relative to AssistantWorkspace and validated
before any operation is performed.
"""
import os
import asyncio
from pathlib import Path

import aiofiles
from loguru import logger

from app.config import settings

ALLOWED_ROOT = settings.workspace_dir.resolve()
ALLOWED_EXTENSIONS = set(settings.allowed_extensions)


# ─── Path Validation ──────────────────────────────────────────────────────────

def _safe_path(relative_path: str, root: Path = ALLOWED_ROOT) -> Path:
    """
    Resolve a user-supplied relative path to an absolute path inside `root`
    (the workspace by default; the Obsidian vault reuses it). Raises
    PermissionError if the resolved path escapes the sandbox.
    
    Security Features:
    - Prevents path traversal attacks (../, ../../, etc.)
    - Blocks absolute paths that escape workspace
    - Resolves symlinks and validates final location
    - Strips null bytes and control characters
    """
    # Security: Remove null bytes and control characters
    relative_path = "".join(c for c in relative_path if c.isprintable() and c != '\0')
    
    # Strip leading slashes (either kind) to force relative resolution
    relative_path = relative_path.lstrip("/\\")
    
    # Resolve to absolute path (follows symlinks)
    candidate = (root / relative_path).resolve()

    # Critical check: Ensure resolved path is within the sandbox root
    if not candidate.is_relative_to(root):
        raise PermissionError(
            f"Path traversal denied: '{relative_path}' resolves outside {root}."
        )
    
    return candidate


def _check_extension(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_EXTENSIONS and path.suffix != "":
        raise PermissionError(
            f"File type '{path.suffix}' is not allowed. "
            f"Permitted types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


# ─── Tool Implementations ─────────────────────────────────────────────────────

async def tool_read_file(path: str) -> str:
    safe = _safe_path(path)
    _check_extension(safe)
    if not safe.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not safe.is_file():
        raise IsADirectoryError(f"'{path}' is a directory, not a file.")

    if safe.suffix.lower() == ".pdf":
        # pdfplumber rather than the pdftotext binary — Poppler is not present
        # by default on Windows or macOS.
        from app.tools.rag.ingestor import _extract_pdf

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, _extract_pdf, safe)
        return text.strip() or "[PDF has no extractable text]"


    async with aiofiles.open(safe, "r", encoding="utf-8", errors="replace") as f:
        content = await f.read()

    logger.info("read_file: {}", safe)
    return content


async def tool_write_file(path: str, content: str) -> str:
    safe = _safe_path(path)
    _check_extension(safe)
    safe.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(safe, "w", encoding="utf-8") as f:
        await f.write(content)

    logger.info("write_file: {} ({} bytes)", safe, len(content))
    return f"File written successfully: {path}"


async def tool_append_file(path: str, content: str) -> str:
    safe = _safe_path(path)
    _check_extension(safe)
    safe.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(safe, "a", encoding="utf-8") as f:
        await f.write(content)

    logger.info("append_file: {} (+{} bytes)", safe, len(content))
    return f"Content appended to: {path}"


async def tool_list_directory(path: str = ".") -> str:
    safe = _safe_path(path)
    if not safe.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not safe.is_dir():
        raise NotADirectoryError(f"'{path}' is not a directory.")

    entries = []
    for item in sorted(safe.iterdir()):
        rel = item.relative_to(ALLOWED_ROOT)
        if item.is_dir():
            entries.append(f"[DIR]  {rel}/")
        else:
            size = item.stat().st_size
            entries.append(f"[FILE] {rel}  ({size} bytes)")

    if not entries:
        return f"Directory '{path}' is empty."

    logger.info("list_directory: {}", safe)
    return "\n".join(entries)


async def tool_delete_file(path: str) -> str:
    safe = _safe_path(path)
    _check_extension(safe)
    if not safe.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not safe.is_file():
        raise IsADirectoryError(f"'{path}' is a directory. Use a specific delete command.")

    safe.unlink()
    logger.warning("delete_file: {}", safe)
    return f"File deleted: {path}"
