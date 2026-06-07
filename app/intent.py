"""
Intent Classifier

Replaces the binary needs_planning() heuristic with a proper
5-way classification that routes each query to the right engine
before any LLM is called.

Classification is purely rule-based (no LLM call) so it adds < 1 ms
to every request.

Architecture:
    User text
        ↓
    classify_intent()          ← this module
        ↓
    CHAT     → direct LLM (no planner, no tools)
    MEMORY   → hot_memory recall + direct LLM
    TOOL     → tool router + LLM
    RESEARCH → Research Agent
    PLANNING → Planner LLM call → tool-executing LLM

Integration point:  app/pipeline.py  process_text_input()
    Replace the single needs_planning() branch with:

        intent = classify_intent(user_text)
        if intent == IntentType.RESEARCH:
            return await research_agent.run(user_text)
        elif intent == IntentType.PLANNING:
            plan = await make_plan(user_text, llm_engine.generate)
            ...
        else:
            # CHAT / TOOL / MEMORY — skip planner entirely
            ...
"""
from __future__ import annotations

import re
import time
from enum import Enum
from dataclasses import dataclass

from loguru import logger


# ─── Intent enum ─────────────────────────────────────────────────────────────

class IntentType(Enum):
    """
    CHAT     — pure conversation, no tools needed
               "what is 2+2"  /  "hello"  /  "explain transformers"
    MEMORY   — reading or writing hot/cold memory or session history
               "remember that I prefer urdu"  /  "what did I say about X"
    TOOL     — single tool action, no planning needed
               "create a note about RAG"  /  "search my notes for attention"
    RESEARCH — multi-source web research + report generation
               "research the latest llama.cpp updates"
    PLANNING — complex multi-step task that needs the planner
               "find the snowflake release, create a note, add to daily"
    """
    CHAT     = "chat"
    MEMORY   = "memory"
    TOOL     = "tool"
    RESEARCH = "research"
    PLANNING = "planning"


# ─── Signal tables ────────────────────────────────────────────────────────────

# Matched in order — first match wins within each category.
# Each entry: (pattern_or_substring, is_regex)

_CHAT_SIGNALS: list[tuple[str, bool]] = [
    # Greetings / pleasantries
    (r"^(hi|hello|hey|good morning|good evening|good night|morning|evening)[\s!,.]*$", True),
    (r"^(thanks|thank you|thx|ty|ok|okay|sure|yes|no|yep|nope|got it|cool|nice|great)[\s!,.]*$", True),
    # Pure factual / definitional questions that need no tools
    (r"^what (is|are|was|were) ", True),
    (r"^who (is|was) ", True),
    (r"^(explain|describe|define|tell me about|how does|what does) ", True),
    (r"^(can you|could you|please) (explain|describe|tell me|clarify)", True),
    # Math
    (r"^[\d\s\+\-\*\/\(\)\^\.]+[\?\s]*$", True),
    (r"what('s| is) \d", True),
    # Meta / assistant questions
    ("what can you do", False),
    ("what are your capabilities", False),
    ("how are you", False),
]

_MEMORY_SIGNALS: list[tuple[str, bool]] = [
    # Reading memory
    ("what do you know about me", False),
    ("what did i say", False),
    ("do you remember", False),
    ("recall", False),
    ("what did we talk about", False),
    ("look up in memory", False),
    ("search my sessions", False),
    ("past conversation", False),
    ("session history", False),
    # Writing memory
    ("remember that", False),
    ("remember this", False),
    ("save to memory", False),
    ("add to memory", False),
    ("note that i", False),
    ("update my profile", False),
    ("i prefer", False),
    ("my name is", False),
    ("what have i learned about", False),   # → kg_expand
    ("what do i know about", False),        # → kg_expand
    ("what are my projects", False),
    ("what projects", False),
]

_RESEARCH_SIGNALS: list[tuple[str, bool]] = [
    # Explicit research intent
    ("research ", False),
    ("investigate ", False),
    ("look into ", False),
    ("find out about ", False),
    ("deep dive into", False),
    # Multi-source comparison / analysis of external topics
    (r"compare .{3,} (vs|versus|and) .{3,}", True),
    (r"what('s| is| are) the latest ", True),
    (r"what('s| is) new (in|with|about) ", True),
    ("recent updates to", False),
    ("latest news about", False),
    ("how does .* compare", False),
    # Research + document combos
    ("research .* and (create|write|save|document)", False),
    ("find .* and (summarize|summarise|write a report)", False),
]

_TOOL_SIGNALS: list[tuple[str, bool]] = [
    # Obsidian note operations
    ("create a note", False),
    ("create note", False),
    ("write a note", False),
    ("capture this", False),
    ("capture idea", False),
    ("add to daily", False),
    ("log this", False),
    ("log to daily", False),
    ("search my notes", False),
    ("search notes", False),
    ("search the vault", False),
    ("read note", False),
    ("open note", False),
    ("list my notes", False),
    ("list vault", False),
    ("morning briefing", False),
    ("good morning", False),
    # File operations
    ("read the file", False),
    ("read file", False),
    ("write to file", False),
    ("save to file", False),
    ("list files", False),
    ("list directory", False),
    ("show me the files", False),
    # Knowledge graph
    ("knowledge graph", False),
    ("graph stats", False),
    ("how does .* connect", False),
    ("find the path", False),
    ("graph clusters", False),
    # Web search (single)
    ("search the web for", False),
    ("search online for", False),
    ("google ", False),
    ("look up ", False),
    # Skills
    ("load skill", False),
    ("save this skill", False),
    ("create a skill", False),
    ("ingest document", False),
    ("ingest this", False),
    ("add this document", False),
    ("search my documents", False),
    ("search documents", False),
    ("search local files", False),
    ("what documents", False),
    ("list documents", False),
    ("what did the", False),           # doc_search trigger: "what did the contract say..."
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
    ("what can my", False),
    ("hardware profile", False),
    ("check my system", False),
    ("analyse my hardware", False),
    ("analyze my hardware", False),
    ("is my computer good enough", False),
]

_PLANNING_SIGNALS: list[tuple[str, bool]] = [
    # Explicit sequencing language
    (" then ", False),
    (" after that", False),
    (" and then ", False),
    (" and also ", False),
    (" followed by ", False),
    # "X and Y" where both X and Y are actions (two verbs joined by "and")
    (r"\b(create|write|find|search|read|save|add|log|build|analyze|organize|organise)\b.{3,30}\band\b.{3,30}\b(create|write|add|save|link|log|update|search|note|graph|daily|append)\b", True),
    # Organisational complexity
    ("set up ", False),
    ("organize ", False),
    ("organise ", False),
    ("go through all", False),
    ("process all", False),
    ("summarize all", False),
    ("summarise all", False),
    ("for each ", False),
    ("every ", False),
    ("all the ", False),
    # Compound tasks
    ("figure out ", False),
    ("analyze and ", False),
    ("analyse and ", False),
    ("continue ", False),       # "continue the X project"
    ("resume ", False),         # "resume work on X"
    ("switch to ", False),      # "switch to X project"
    ("work on ", False),        # "work on X"
]


# ─── Classifier ───────────────────────────────────────────────────────────────

def _matches(text: str, signals: list[tuple[str, bool]]) -> bool:
    """Return True if any signal matches the lowercased text."""
    lower = text.lower().strip()
    for pattern, is_regex in signals:
        if is_regex:
            if re.search(pattern, lower):
                return True
        else:
            if pattern in lower:
                return True
    return False


def classify_intent(user_text: str) -> IntentType:
    """
    Classify the user's intent without calling the LLM.
    Returns an IntentType enum value.

    Ordering matters — more specific checks win over general ones.

    Priority (highest → lowest):
        RESEARCH  > PLANNING  > MEMORY  > TOOL  > CHAT
    """
    t0 = time.perf_counter()
    text = user_text.strip()
    lower = text.lower()

    # ── Very short inputs → CHAT immediately ──────────────────────────────
    if len(lower.split()) <= 3:
        result = IntentType.CHAT
        _log(result, user_text, t0)
        return result

    # ── RESEARCH — highest specificity, check first ───────────────────────
    if _matches(text, _RESEARCH_SIGNALS):
        result = IntentType.RESEARCH
        _log(result, user_text, t0)
        return result

    # ── PLANNING — multi-step sequences (before TOOL — compound actions win) ─
    if _matches(text, _PLANNING_SIGNALS):
        result = IntentType.PLANNING
        _log(result, user_text, t0)
        return result

    # ── MEMORY — reading or writing personal memory ───────────────────────
    if _matches(text, _MEMORY_SIGNALS):
        result = IntentType.MEMORY
        _log(result, user_text, t0)
        return result

    # ── TOOL — single well-defined tool action ────────────────────────────
    if _matches(text, _TOOL_SIGNALS):
        result = IntentType.TOOL
        _log(result, user_text, t0)
        return result

    # ── CHAT — pure conversation signal or fallback ───────────────────────
    if _matches(text, _CHAT_SIGNALS):
        result = IntentType.CHAT
        _log(result, user_text, t0)
        return result

    # ── Default: CHAT (safest fallback — won't spin up planner) ──────────
    result = IntentType.CHAT
    _log(result, user_text, t0, fallback=True)
    return result


def _log(result: IntentType, text: str, t0: float, fallback: bool = False) -> None:
    elapsed_us = (time.perf_counter() - t0) * 1_000_000
    label = f"{result.value.upper()}" + (" [fallback]" if fallback else "")
    logger.debug(
        "Intent: {} ({:.0f}µs) — '{}'",
        label,
        elapsed_us,
        text[:60],
    )


# ─── Backward-compat shim ────────────────────────────────────────────────────

def needs_planning(user_text: str) -> bool:
    """
    Drop-in replacement for the old planner.needs_planning().
    Returns True for PLANNING and RESEARCH intents.
    Keeps existing code working while we wire in the new classifier.
    """
    intent = classify_intent(user_text)
    return intent in (IntentType.PLANNING, IntentType.RESEARCH)
