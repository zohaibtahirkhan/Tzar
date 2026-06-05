"""
Task Planner — produces a step-by-step plan before tool execution.

Only activated for requests that need 2+ tools.
Simple / conversational turns skip it entirely for latency.

Architecture:
  User input
    ↓
  Intent Classifier  (app/intent.py — fast, no LLM)
    ↓
  CHAT/MEMORY/TOOL → skip planner, direct to pipeline
  PLANNING         → Planner LLM call → {goal, steps[], required_tools[]}
  RESEARCH         → Research Agent   (app/agents/researcher.py)
    ↓
  Pipeline executes steps in order, with plan in context
"""
import json
import re
from dataclasses import dataclass, field
from loguru import logger

# ─── Complexity classifier ────────────────────────────────────────────────────
# needs_planning() is now a thin shim over the Intent Classifier.
# The real logic lives in app/intent.py.

def needs_planning(user_text: str) -> bool:
    """
    Returns True if the request needs the Planner or Research Agent.
    Delegates to the Intent Classifier — no LLM call.
    """
    from app.intent import classify_intent, IntentType
    intent = classify_intent(user_text)
    return intent in (IntentType.PLANNING, IntentType.RESEARCH)


# ─── Planner prompt ───────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a task planner for a local voice assistant.
Given a user request, produce a minimal execution plan.
Respond ONLY in JSON. No prose.

FORMAT:
{
  "goal": "one sentence describing the end goal",
  "complexity": "low" | "medium" | "high",
  "steps": [
    "Step 1: what to do",
    "Step 2: what to do"
  ],
  "required_tools": ["tool_name_1", "tool_name_2"],
  "can_answer_directly": false
}

RULES:
- If the request can be answered from memory/knowledge with no tools, set can_answer_directly: true and steps: [].
- Keep steps minimal — don't over-plan. 2-4 steps is usually right.
- required_tools should only list tools actually needed, from this set:
  read_file, write_file, append_file, list_directory, delete_file,
  web_search, save_memory, recall_memory, memory_write, memory_remove,
  obsidian_create_note, obsidian_append_daily, obsidian_capture_idea,
  obsidian_search, obsidian_keyword_search, obsidian_get_related,
  obsidian_get_project, obsidian_morning_briefing, obsidian_read_note,
  obsidian_list_vault, obsidian_reindex,
  kg_summary, kg_path, kg_neighbors, kg_orphans, kg_clusters, kg_timeline, kg_add,
  skill_load, skill_create, skill_update, skill_delete,
  session_search, session_list,
  research_agent (special: triggers multi-source web research + vault report)
- complexity: low = 1 tool, medium = 2-3 tools, high = 4+ tools or unclear requirements.
- Use research_agent when the user wants to research an external topic with multiple sources.
  Do NOT use research_agent for simple web_search queries.
"""


@dataclass
class Plan:
    goal: str = ""
    complexity: str = "low"
    steps: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    can_answer_directly: bool = False

    def to_context_string(self) -> str:
        """Format plan as context injected into the conversation agent's prompt."""
        if self.can_answer_directly:
            return ""
        lines = [
            f"TASK PLAN:",
            f"Goal: {self.goal}",
            f"Complexity: {self.complexity}",
        ]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  Step {i}: {step}")
        if self.required_tools:
            lines.append(f"Tools needed: {', '.join(self.required_tools)}")
        lines.append("Execute this plan. Follow the steps in order.")
        return "\n".join(lines)


def _parse_plan_json(raw: str) -> Plan:
    clean = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return Plan(
                goal=data.get("goal", ""),
                complexity=data.get("complexity", "low"),
                steps=data.get("steps", []),
                required_tools=data.get("required_tools", []),
                can_answer_directly=data.get("can_answer_directly", False),
            )
        except json.JSONDecodeError:
            pass
    return Plan()  # empty plan = skip planning, proceed normally


# ─── Public API ───────────────────────────────────────────────────────────────

async def make_plan(user_text: str, llm_generate_fn) -> Plan:
    """
    Generate a plan for a complex request.
    llm_generate_fn: async callable(messages, system_prompt) -> str
    """
    messages = [{"role": "user", "content": user_text}]
    try:
        raw  = await llm_generate_fn(messages, PLANNER_SYSTEM)
        plan = _parse_plan_json(raw)
        logger.info(
            "Plan: goal='{}' complexity={} steps={} tools={}",
            plan.goal[:60], plan.complexity, len(plan.steps), plan.required_tools,
        )
        return plan
    except Exception as e:
        logger.error("Planner failed: {}", e)
        return Plan()