"""
app/tools/rag/citations.py

Source tracking for retrieval-grounded answers.

Search tools (doc_search, obsidian_search, unified_search) run through the
tool router and return plain strings for the LLM. That loses the one thing a
user needs to trust the answer: which document it came from. This module is
the side channel that keeps it.

    sources = start_source_collection()      # pipeline, once per request
    ...
    record_sources([...])                    # search tools, as they retrieve
    ...
    spoken += attribution_sentence(sources, spoken)   # pipeline, before TTS

The collector is a ContextVar holding a plain list, so tools invoked from a
child task (asyncio.wait_for, gather) still append to the request's list.
"""
from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

MAX_SPOKEN_SOURCES = 3

_collector: ContextVar[Optional[list["Source"]]] = ContextVar("retrieval_sources", default=None)

# "Where did that come from?" — matched as plain substrings of the lowercased text.
_SOURCE_QUESTION_PHRASES = (
    "where did that come from", "where does that come from", "where is that from",
    "where did this come from", "where is this from", "where did it come from",
    "where did you get that", "where did you find that", "where did you read that",
    "your source", "the source for that", "the source of that",
    "which document", "which note", "which file", "cite that", "cite your",
    "how do you know that", "how do you know this",
)
_SOURCE_QUESTION_EXCLUDE = ("source code",)

_NO_ANSWER_RE = re.compile(
    r"\b((could|did)(n't| not) find|no (relevant )?(information|results|notes|documents|mention)"
    r"|don't have (any|that)|nothing (about|on|matching)|not (found|mentioned))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Source:
    title: str
    location: str        # filename for documents, vault-relative path for notes
    kind: str            # "document" | "note"

    @property
    def spoken_name(self) -> str:
        return (self.title or Path(self.location).stem).strip()

    def as_dict(self) -> dict:
        # Title is guaranteed non-empty so consumers never need a fallback.
        return {**asdict(self), "title": self.spoken_name}


# ─── Collection ──────────────────────────────────────────────────────────────

def start_source_collection() -> list[Source]:
    """Begin a fresh per-request source list and return it."""
    bucket: list[Source] = []
    _collector.set(bucket)
    return bucket


def record_sources(sources: Iterable[Source]) -> None:
    """Append retrieved sources to the active request, de-duplicated by document."""
    bucket = _collector.get()
    if bucket is None:
        return
    seen = {(s.kind, s.location) for s in bucket}
    for src in sources:
        key = (src.kind, src.location)
        if key not in seen:
            seen.add(key)
            bucket.append(src)


# ─── Spoken attribution ──────────────────────────────────────────────────────

def _mentions(lowered_spoken: str, source: Source) -> bool:
    candidates = (source.spoken_name, Path(source.location).name)
    return any(c.lower() in lowered_spoken for c in candidates if c)


def _spoken_list(items: list[str]) -> str:
    """'A', 'A and B', 'A, B, and C', 'A, B, and C and 2 others'."""
    extra = len(items) - MAX_SPOKEN_SOURCES
    items = items[:MAX_SPOKEN_SOURCES]
    if len(items) == 1:
        joined = items[0]
    elif len(items) == 2:
        joined = f"{items[0]} and {items[1]}"
    else:
        joined = ", ".join(items[:-1]) + f", and {items[-1]}"
    if extra > 0:
        joined += f" and {extra} other{'s' if extra > 1 else ''}"
    return joined


def attribution_sentence(sources: list[Source], spoken: str) -> str:
    """
    Return a short spoken sentence naming where the answer came from, or ""
    when the answer already names a source, nothing was retrieved, or the
    answer says it found nothing.
    """
    if not sources or not spoken or _NO_ANSWER_RE.search(spoken):
        return ""
    lowered = spoken.lower()
    if any(_mentions(lowered, s) for s in sources):
        return ""
    return f" That's from {_spoken_list([s.spoken_name for s in sources])}."


def is_source_question(text: str) -> bool:
    lowered = text.lower()
    if any(x in lowered for x in _SOURCE_QUESTION_EXCLUDE):
        return False
    return any(p in lowered for p in _SOURCE_QUESTION_PHRASES)


def describe_sources(sources: list[Source]) -> str:
    """Full spoken answer to 'where did that come from?'."""
    if not sources:
        return (
            "My last answer didn't come from your notes or documents — "
            "it was from general knowledge, so treat it with some caution."
        )
    parts = [
        f"your note {s.spoken_name}" if s.kind == "note"
        else f"the document {s.spoken_name}, file {Path(s.location).name}"
        for s in sources
    ]
    return f"That came from {_spoken_list(parts)}."
