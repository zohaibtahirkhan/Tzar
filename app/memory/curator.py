"""
Memory Curator Agent — prevents memory pollution.

The Conversation Agent proposes memory candidates.
The Curator decides: store, reject, or consolidate.

This separation means the main LLM never writes to memory directly.
Every candidate passes through a focused evaluation prompt first.
"""
from loguru import logger


# ─── Evaluation prompt ────────────────────────────────────────────────────────

CURATOR_SYSTEM = """You are a memory curator for a personal AI assistant.
You receive a proposed memory and decide what to do with it.
Respond ONLY in JSON. No prose.

DECISION OPTIONS:
{
  "action": "store",
  "target": "user",
  "content": "cleaned, compressed version of the fact"
}

{
  "action": "store",
  "target": "memory",
  "content": "cleaned, compressed version of the fact"
}

{
  "action": "reject",
  "reason": "why this is not worth storing"
}

{
  "action": "consolidate",
  "target": "user",
  "old_substring": "phrase from existing entry to replace",
  "new_content": "merged, updated version"
}

RULES:
- "user" target = name, preferences, communication style, skill level, things to avoid.
- "memory" target = environment facts, project conventions, tool behavior, lessons learned.
- REJECT if: too specific to one conversation, already obvious, unlikely to matter in future, redundant.
- REJECT examples: "user asked about Python today", "user said thanks", "it is Tuesday".
- STORE examples: "user prefers Urdu for casual chat", "vault uses ISO date format in filenames", "user dislikes bullet points in voice responses".
- CONSOLIDATE when the fact updates or refines something already in memory (prefer over duplicate store).
- Keep content dense: one sentence, no fluff, no pronouns where avoidable.
"""


def _parse_curator_json(raw: str) -> dict:
    from app.llm.engine import parse_json_object
    return parse_json_object(raw) or {"action": "reject", "reason": "curator parse error"}


# ─── Public API ───────────────────────────────────────────────────────────────

async def evaluate_memory_candidate(
    candidate: str,
    existing_memory: str,
    llm_generate_fn,
) -> dict:
    """
    Ask the curator LLM whether a proposed memory should be stored.

    candidate:       the raw proposed fact from the conversation agent
    existing_memory: current hot_memory_for_prompt() snapshot
    llm_generate_fn: async callable(messages, system_prompt) -> str

    Returns a decision dict: {action, target?, content?, reason?, ...}
    """
    user_msg = (
        f"PROPOSED MEMORY:\n{candidate}\n\n"
        f"EXISTING MEMORY SNAPSHOT:\n{existing_memory}\n\n"
        "Decide: store, reject, or consolidate."
    )

    messages = [{"role": "user", "content": user_msg}]

    try:
        raw = await llm_generate_fn(messages, CURATOR_SYSTEM)
        decision = _parse_curator_json(raw)
        logger.info(
            "Curator decision for '{}...': action={} {}",
            candidate[:50],
            decision.get("action"),
            decision.get("reason", decision.get("target", "")),
        )
        return decision
    except Exception as e:
        logger.error("Curator evaluation failed: {}", e)
        return {"action": "reject", "reason": f"curator error: {e}"}


async def process_memory_candidate(
    candidate: str,
    llm_generate_fn,
    hot_memory_module,
) -> str:
    """
    Full pipeline: evaluate then execute the curator's decision.
    Returns a status string (for debug logging — never shown to user).

    hot_memory_module: the app.memory.hot_memory module (passed to avoid circular imports)
    """
    existing = hot_memory_module.hot_memory_for_prompt()
    decision = await evaluate_memory_candidate(candidate, existing, llm_generate_fn)
    action   = decision.get("action", "reject")

    if action == "store":
        target  = decision.get("target", "memory")
        content = decision.get("content", candidate)
        result  = hot_memory_module.hot_memory_add(target, content)
        logger.info("Curator stored [{}]: {}", target, content[:60])
        return result

    elif action == "consolidate":
        target      = decision.get("target", "memory")
        old_sub     = decision.get("old_substring", "")
        new_content = decision.get("new_content", candidate)
        if old_sub:
            result = hot_memory_module.hot_memory_replace(target, old_sub, new_content)
        else:
            result = hot_memory_module.hot_memory_add(target, new_content)
        logger.info("Curator consolidated [{}]: {}", target, new_content[:60])
        return result

    else:
        reason = decision.get("reason", "no reason given")
        logger.debug("Curator rejected: {}", reason)
        return f"rejected: {reason}"