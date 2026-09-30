"""
Intent Classifier — Multi-Label Capabilities

Replaces single-label IntentType with a Capabilities dataclass that
independently sets flags for each required capability.

A single query can simultaneously need tools, memory, research, and planning —
the old single-winner classification silently dropped those signals.

Architecture:
    User text
        ↓
    classify(user_text) → Capabilities
        ↓
    caps.needs_research → Research Agent
    caps.needs_planning → Planner
    caps.needs_tools    → Tool Router
    caps.needs_memory   → Memory recall
    caps.needs_rag      → RAG / doc search
    caps.is_complex()   → Multi-step orchestration

Backward compat:
    classify_intent(text) → IntentType  (shim, returns caps.primary as enum)
    needs_planning(text)  → bool        (shim, unchanged signature)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum

from loguru import logger


# ─── Capabilities dataclass ───────────────────────────────────────────────────

@dataclass
class Capabilities:
    needs_tools:    bool = False
    needs_memory:   bool = False
    needs_research: bool = False
    needs_planning: bool = False
    needs_rag:      bool = False   # local docs / notes / vault search

    @property
    def primary(self) -> str:
        """Single-label summary for routing — most specific capability wins."""
        if self.needs_planning:  return "planning"
        if self.needs_research:  return "research"
        if self.needs_rag:       return "rag"
        if self.needs_tools:     return "tool"
        if self.needs_memory:    return "memory"
        return "chat"

    def is_complex(self) -> bool:
        """
        True when the query genuinely needs multi-step orchestration.
 
        Single-tool queries with a secondary signal (e.g. search notes = tool+rag)
        are NOT complex — they go through the fast pipeline path.
 
        Truly complex = planning involved, OR research+tools (requires web+save),
        OR 3+ flags set simultaneously.
        """
        flags = [
            self.needs_tools,
            self.needs_memory,
            self.needs_research,
            self.needs_planning,
            self.needs_rag,
        ]
        flag_count = sum(flags)
 
        # 3+ flags always means complex
        if flag_count >= 3:
            return True
 
        # Planning + anything else = complex
        if self.needs_planning and flag_count >= 2:
            return True
 
        # Research + tools = needs web search then save = complex
        if self.needs_research and self.needs_tools:
            return True
 
        # Everything else (tool+rag, tool+memory, rag alone, etc.) = simple
        return False

    def as_dict(self) -> dict:
        return {
            "needs_tools":    self.needs_tools,
            "needs_memory":   self.needs_memory,
            "needs_research": self.needs_research,
            "needs_planning": self.needs_planning,
            "needs_rag":      self.needs_rag,
            "primary":        self.primary,
            "is_complex":     self.is_complex(),
        }


# ─── Backward-compat enum (kept so existing isinstance checks still work) ─────

class IntentType(Enum):
    CHAT     = "chat"
    MEMORY   = "memory"
    TOOL     = "tool"
    RESEARCH = "research"
    PLANNING = "planning"
    RAG      = "rag"


# ─── Signal tables ────────────────────────────────────────────────────────────
# Each entry: (pattern_or_substring, is_regex)
# All compiled at module load — zero per-call overhead.

_CHAT_SIGNALS: list[tuple[str, bool]] = [
    (r"^(hi|hello|hey|good morning|good evening|good night|morning|evening)[\s!,.]*$", True),
    (r"^(thanks|thank you|thx|ty|ok|okay|sure|yes|no|yep|nope|got it|cool|nice|great)[\s!,.]*$", True),
    (r"^what (is|are|was|were) ", True),
    (r"^who (is|was) ", True),
    (r"^(explain|describe|define|tell me about|how does|what does) ", True),
    (r"^(can you|could you|please) (explain|describe|tell me|clarify)", True),
    (r"^[\d\s\+\-\*\/\(\)\^\.]+[\?\s]*$", True),
    (r"what('s| is) \d", True),
    ("what can you do", False),
    ("what are your capabilities", False),
    ("how are you", False),
]

_MEMORY_SIGNALS: list[tuple[str, bool]] = [
    ("what do you know about me", False),
    ("what did i say", False),
    ("do you remember", False),
    ("recall", False),
    ("what did we talk about", False),
    ("look up in memory", False),
    ("search my sessions", False),
    ("past conversation", False),
    ("session history", False),
    ("remember that", False),
    ("remember this", False),
    ("save to memory", False),
    ("add to memory", False),
    ("note that i", False),
    ("update my profile", False),
    ("i prefer", False),
    ("my name is", False),
    ("what have i learned about", False),
    ("what do i know about", False),
    ("what are my projects", False),
    ("what projects", False),
    ("what are my preferences", False),      # "What are my preferences?"
    ("what do i like", False),               # common variant
    ("what do i prefer", False),             # another variant
    ("what are my settings", False),
    ("what have i told you about me", False),
    ("tell me about myself", False),
    ("what do you know about my preferences", False),
    ("what do i dislike", False), 
    ("what are my goals", False),
    ("what projects do i have", False),
]

_RESEARCH_SIGNALS: list[tuple[str, bool]] = [
    ("research ", False),
    ("investigate ", False),
    ("look into ", False),
    ("find out about ", False),
    ("deep dive into", False),
    (r"compare .{3,} (vs|versus|and) .{3,}", True),
    (r"what('s| is| are) the latest ", True),
    (r"what('s| is) new (in|with|about) ", True),
    ("recent updates to", False),
    ("latest news about", False),
    ("how does .* compare", False),
]

_RAG_SIGNALS: list[tuple[str, bool]] = [
    ("what did the", False),           # "what did the contract say about..."
    ("what does the", False),          # "what does the spec say..."
    ("according to my", False),
    ("search my documents", False),
    ("search documents", False),
    ("search local files", False),
    ("what documents", False),
    ("list documents", False),
    ("ingest document", False),
    ("ingest this", False),
    ("add this document", False),
    (r"\b(previous|last|earlier|prior)\s+(version|draft|doc|document|note|contract|report)", True),
    ("read my", False),                # "read my contract", "read my notes"
    ("compare it to", False),
    ("compare to the previous", False),
    ("compare with the last", False),
    ("from the document", False),
    ("in the document", False),
    ("from my notes", False),
    ("in my notes", False),
    ("search the vault", False),
    ("search my notes", False),
    ("search notes", False),
    ("unified search", False),
]

_TOOL_SIGNALS: list[tuple[str, bool]] = [
    ("create a note", False),
    ("create note", False),
    ("write a note", False),
    ("capture this", False),
    ("capture idea", False),
    ("add to daily", False),
    ("log this", False),
    ("log to daily", False),
    ("read note", False),
    ("open note", False),
    ("list my notes", False),
    ("list vault", False),
    ("morning briefing", False),
    ("read the file", False),
    ("read file", False),
    ("write to file", False),
    ("save to file", False),
    ("list files", False),
    ("list directory", False),
    ("show me the files", False),
    ("knowledge graph", False),
    ("graph stats", False),
    ("find the path", False),
    ("graph clusters", False),
    ("search the web for", False),
    ("search online for", False),
    ("google ", False),
    ("look up ", False),
    ("load skill", False),
    ("save this skill", False),
    ("create a skill", False),
    ("graph expand", False),
    ("semantic expand", False),
    ("mcp status", False),
    ("connect mcp", False),
    ("project list", False),
    ("list my projects", False),
    ("show projects", False),
    ("project status", False),
    ("active project", False),
    ("create project", False),
    ("new project", False),
    ("archive project", False),
    ("memory scores", False),
    ("prune memories", False),
    ("skill learning", False),
    ("skill stats", False),
    ("system profile", False),
    ("what model", False),
    ("suggest a model", False),
    ("recommend a model", False),
    ("hardware profile", False),
    ("check my system", False),
    ("analyse my hardware", False),
    ("analyze my hardware", False),
    ("is my computer good enough", False),
    # Browser / computer use
    ("open browser", False),
    ("go to", False),
    ("click on", False),
    ("fill in", False),
    ("browse to", False),
    ("navigate to", False),
    ("use the browser", False),
    ("use browser", False),
]

_PLANNING_SIGNALS: list[tuple[str, bool]] = [
    (" then ", False),
    (" after that", False),
    (" and then ", False),
    (" and also ", False),
    (" followed by ", False),
    # Two action verbs joined by "and"
    (r"\b(create|write|find|search|read|save|add|log|build|analyze|organize|organise|compare|summarize|summarise|ingest)\b.{3,40}\band\b.{3,40}\b(create|write|add|save|link|log|update|search|note|graph|daily|append|compare|summarize|ingest)\b", True),
    ("set up ", False),
    ("organize ", False),
    ("organise ", False),
    ("go through all", False),
    ("process all", False),
    ("summarize all", False),
    ("summarise all", False),
    ("for each ", False),
    ("figure out ", False),
    ("analyse my ", False),
    ("analyze my ", False),
    ("write a summary", False),
    ("write a report", False),
    ("continue ", False),
    ("resume ", False),
    ("work on ", False),
]


# ─── Compile all signals at module load ──────────────────────────────────────

def _compile(raw: list[tuple[str, bool]]) -> list[tuple[object, bool]]:
    return [
        (re.compile(p) if is_re else p, is_re)
        for p, is_re in raw
    ]

_CHAT_COMPILED     = _compile(_CHAT_SIGNALS)
_MEMORY_COMPILED   = _compile(_MEMORY_SIGNALS)
_RESEARCH_COMPILED = _compile(_RESEARCH_SIGNALS)
_RAG_COMPILED      = _compile(_RAG_SIGNALS)
_TOOL_COMPILED     = _compile(_TOOL_SIGNALS)
_PLANNING_COMPILED = _compile(_PLANNING_SIGNALS)


def _matches(text: str, signals: list) -> bool:
    lower = text.lower().strip()
    for pattern, is_regex in signals:
        if is_regex:
            if re.search(pattern, lower):
                return True
        else:
            if pattern in lower:
                return True
    return False


# ─── Cross-signal boosters ────────────────────────────────────────────────────
# Patterns that imply COMBINATIONS of capabilities when they appear.
# Each entry: (pattern, is_regex, flags_to_set)

_CROSS_SIGNALS: list[tuple[str, bool, dict]] = [
    # "read my X and compare to previous" → RAG + planning
    (r"\bread\b.{3,30}\b(compare|versus|vs)\b", True,
     {"needs_rag": True, "needs_planning": True}),

    # "read X, summarize, create note" → RAG + planning + tools
    (r"\b(read|summarize|summarise)\b.{3,30}\b(create|save|write)\b", True,
     {"needs_rag": True, "needs_planning": True, "needs_tools": True}),

    # "research X and save" → research + tools
    (r"\bresearch\b.{3,50}\b(save|create|note|write|add)\b", True,
     {"needs_research": True, "needs_tools": True}),

    # "remember [anything about a doc/note]" → memory + RAG
    ("remember what the", False, {"needs_memory": True, "needs_rag": True}),

    # "find and summarize" → research or RAG + planning
    (r"\b(find|search)\b.{3,30}\bsummar", True,
     {"needs_planning": True}),
]

_CROSS_COMPILED = [
    (re.compile(p) if is_re else p, is_re, flags)
    for p, is_re, flags in _CROSS_SIGNALS
]


# ─── Public classifier ────────────────────────────────────────────────────────

def classify(user_text: str) -> Capabilities:
    """
    Multi-label classification. Each signal set fires independently.
    Returns a Capabilities object with all matching flags set.
    """
    t0   = time.perf_counter()
    text = user_text.strip()
    lower = text.lower()
    caps  = Capabilities()

    # No length short-circuit: it used to return CHAT for anything under four
    # words, which silently swallowed real commands ("create note",
    # "morning briefing", "prune memories"). Greetings need no special case —
    # they match no signal and fall through to CHAT anyway.

    # ── Independent flag setting — ALL can fire ───────────────────────────────
    if _matches(text, _RESEARCH_COMPILED):  caps.needs_research = True
    if _matches(text, _PLANNING_COMPILED):  caps.needs_planning = True
    if _matches(text, _MEMORY_COMPILED):    caps.needs_memory   = True
    if _matches(text, _TOOL_COMPILED):      caps.needs_tools    = True
    if _matches(text, _RAG_COMPILED):       caps.needs_rag      = True

    # ── Cross-signal boosting — compound patterns set multiple flags ──────────
    for pattern, is_re, flags in _CROSS_COMPILED:
        hit = re.search(pattern, lower) if is_re else (pattern in lower)
        if hit:
            for k, v in flags.items():
                setattr(caps, k, v)

    _log(caps, user_text, t0)
    return caps


def _log(caps: Capabilities, text: str, t0: float) -> None:
    elapsed_us = (time.perf_counter() - t0) * 1_000_000
    flags = [k for k, v in caps.as_dict().items() if isinstance(v, bool) and v]
    label = caps.primary.upper() + (f" [{', '.join(flags)}]" if flags else "")
    logger.debug("Caps: {} ({:.0f}µs) — '{}'", label, elapsed_us, text[:60])


# ─── Backward-compat shims ────────────────────────────────────────────────────

def classify_intent(user_text: str) -> "IntentType":
    """
    Drop-in for old classify_intent(). Returns IntentType enum.
    Pipeline code that does `intent == IntentType.PLANNING` still works.
    """
    caps = classify(user_text)
    primary = caps.primary
    try:
        return IntentType(primary)
    except ValueError:
        return IntentType.CHAT


def needs_planning(user_text: str) -> bool:
    """Drop-in for old planner.needs_planning()."""
    caps = classify(user_text)
    return caps.needs_planning or caps.needs_research