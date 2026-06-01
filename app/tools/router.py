"""
Tool router: validates every LLM tool request and dispatches to the
correct implementation. The LLM never executes actions directly.

Flow:
    LLM output
        ↓
    extract_tool_calls()
        ↓
    dispatch_tool()
        ↓
    Validation layer (sandbox, permission checks)
        ↓
    Tool execution
        ↓
    Result returned to LLM as tool_result message
"""
import json
import re
from typing import Any

from loguru import logger

from app.tools.filesystem import (
    tool_read_file,
    tool_write_file,
    tool_append_file,
    tool_list_directory,
    tool_delete_file,
)
from app.tools.web_search import tool_web_search
from app.tools.obsidian import (
    obsidian_create_note,
    obsidian_append_daily,
    obsidian_capture_idea,
    obsidian_semantic_search,
    obsidian_keyword_search,
    obsidian_get_related,
    obsidian_get_project_context,
    obsidian_morning_briefing,
    obsidian_reindex_vault,
    obsidian_read_note,
    obsidian_list_vault,
)
from app.tools.knowledge_graph import (
    kg_add_from_note,
    kg_find_path,
    kg_get_neighbors,
    kg_find_orphans,
    kg_find_clusters,
    kg_temporal_query,
    kg_graph_summary,
)
from app.memory.hot_memory import (
    hot_memory_add,
    hot_memory_remove,
    hot_memory_replace,
    hot_memory_read,
)
from app.memory.skills import (
    skill_load,
    skill_create,
    skill_update,
    skill_rewrite,
    skill_delete,
)
from app.memory.session_store import session_search, session_list

# ─── Tool Registry ────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Any] = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "append_file": tool_append_file,
    "list_directory": tool_list_directory,
    "delete_file": tool_delete_file,
    "web_search": tool_web_search,
    "obsidian_create_note":       obsidian_create_note,
    "obsidian_append_daily":      obsidian_append_daily,
    "obsidian_capture_idea":      obsidian_capture_idea,
    "obsidian_search":            obsidian_semantic_search,
    "obsidian_keyword_search":    obsidian_keyword_search,
    "obsidian_get_related":       obsidian_get_related,
    "obsidian_get_project":       obsidian_get_project_context,
    "obsidian_morning_briefing":  obsidian_morning_briefing,
    "obsidian_reindex":           obsidian_reindex_vault,
    "obsidian_read_note":         obsidian_read_note,
    "obsidian_list_vault":        obsidian_list_vault,
    "kg_add":           kg_add_from_note,
    "kg_path":          kg_find_path,
    "kg_neighbors":     kg_get_neighbors,
    "kg_orphans":       kg_find_orphans,
    "kg_clusters":      kg_find_clusters,
    "kg_timeline":      kg_temporal_query,
    "kg_summary":       kg_graph_summary,
    "memory_write":   None,  
    "memory_remove":  None,
    "memory_replace": None,
    "memory_read":    None,
    "skill_load":    None,  
    "skill_create":  None,
    "skill_update":  None,
    "skill_rewrite": None,
    "skill_delete":  None,
    "session_search": None,   
    "session_list":   None,
    # memory tools are handled inline by ToolRouter (need memory_manager reference)
}

# Required parameters per tool
TOOL_SCHEMA: dict[str, list[str]] = {
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "append_file": ["path", "content"],
    "list_directory": [],
    "delete_file": ["path"],
    "web_search": ["query"],
    "save_memory": ["category", "content"],
    "recall_memory": ["query"],
    "obsidian_create_note":      ["title", "content"],
    "obsidian_append_daily":     ["content"],
    "obsidian_capture_idea":     ["raw_thought"],
    "obsidian_search":           ["query"],
    "obsidian_keyword_search":   ["query"],
    "obsidian_get_related":      ["note_title"],
    "obsidian_get_project":      ["project_name"],
    "obsidian_morning_briefing": [],
    "obsidian_reindex":          [],
    "obsidian_read_note":        ["title"],
    "obsidian_list_vault":       [],
    "kg_add":       ["title", "entities", "relations"],
    "kg_path":      ["source", "target"],
    "kg_neighbors": ["node"],
    "kg_orphans":   [],
    "kg_clusters":  [],
    "kg_timeline":  ["node"],
    "kg_summary":   [],
    "memory_write":   ["target", "content"],
    "memory_remove":  ["target", "substring"],
    "memory_replace": ["target", "old_substring", "new_content"],
    "memory_read":    [],
    "skill_load":    ["name"],
    "skill_create":  ["name", "description", "content"],
    "skill_update":  ["name", "old_text", "new_text"],
    "skill_rewrite": ["name", "description", "content"],
    "skill_delete":  ["name"],
    "session_search": ["query"],
    "session_list":   [],
}

DESTRUCTIVE_TOOLS = {"delete_file", "write_file"}


# ─── Parser ───────────────────────────────────────────────────────────────────

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> list[dict]:
    """
    Find all <tool_call>{...}</tool_call> blocks in LLM output
    and return the parsed JSON payloads.
    """
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        raw = match.group(1)
        try:
            payload = json.loads(raw)
            calls.append(payload)
        except json.JSONDecodeError as e:
            logger.warning("Could not parse tool_call JSON: {} | error: {}", raw, e)
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove all tool_call blocks from a string (for clean TTS output)."""
    return TOOL_CALL_PATTERN.sub("", text).strip()


# ─── Dispatcher ───────────────────────────────────────────────────────────────

class ToolRouter:
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager

    def _validate(self, tool_name: str, params: dict) -> None:
        if tool_name not in TOOL_SCHEMA:
            raise ValueError(f"Unknown tool: '{tool_name}'")
        required = TOOL_SCHEMA[tool_name]
        missing = [r for r in required if r not in params]
        if missing:
            raise ValueError(f"Tool '{tool_name}' is missing parameters: {missing}")

    async def dispatch(self, tool_call: dict) -> dict:
        """
        Validate and execute a single tool call.
        Returns {"tool": name, "status": "ok"|"error", "result": str}
        """
        tool_name = tool_call.get("tool", "")
        params = {k: v for k, v in tool_call.items() if k != "tool"}

        try:
            self._validate(tool_name, params)
        except ValueError as e:
            logger.warning("Tool validation failed: {}", e)
            return {"tool": tool_name, "status": "error", "result": str(e)}

        logger.info("Dispatching tool: {} with params: {}", tool_name, list(params.keys()))

        try:
            # Memory tools (need manager reference)
            if tool_name == "save_memory":
                if self.memory_manager is None:
                    return {"tool": tool_name, "status": "error", "result": "Memory manager not available."}
                mem_id = await self.memory_manager.long_term.save(
                    category=params["category"],
                    content=params["content"],
                )
                return {"tool": tool_name, "status": "ok", "result": f"Memory saved with ID {mem_id}."}

            elif tool_name == "recall_memory":
                if self.memory_manager is None:
                    return {"tool": tool_name, "status": "error", "result": "Memory manager not available."}
                memories = await self.memory_manager.long_term.recall(params["query"])
                if not memories:
                    return {"tool": tool_name, "status": "ok", "result": "No matching memories found."}
                lines = [f"[{m['category']}] {m['content']}" for m in memories]
                return {"tool": tool_name, "status": "ok", "result": "\n".join(lines)}
            
            elif tool_name == "memory_write":
                # Route through curator to prevent memory pollution
                candidate = f"[{params['target']}] {params['content']}"
                from app.memory.curator import process_memory_candidate
                from app.llm.engine import llm_engine
                import app.memory.hot_memory as _hm
                result = await process_memory_candidate(
                    candidate,
                    llm_engine.generate,
                    _hm,
                )
                return {"tool": tool_name, "status": "ok", "result": result}

            elif tool_name == "memory_remove":
                result = hot_memory_remove(params["target"], params["substring"])
                return {"tool": tool_name, "status": "ok", "result": result}

            elif tool_name == "memory_replace":
                result = hot_memory_replace(params["target"], params["old_substring"], params["new_content"])
                return {"tool": tool_name, "status": "ok", "result": result}

            elif tool_name == "memory_read":
                target = params.get("target", "both")
                result = hot_memory_read(target)
                return {"tool": tool_name, "status": "ok", "result": result}
            
            elif tool_name == "skill_load":
                result = skill_load(params["name"])
                return {"tool": tool_name, "status": "ok", "result": result}

            elif tool_name == "skill_create":
                result = skill_create(
                    params["name"],
                    params["description"],
                    params["content"],
                    params.get("category", "general"),
                )
                return {"tool": tool_name, "status": "ok", "result": result}

            elif tool_name == "skill_update":
                result = skill_update(params["name"], params["old_text"], params["new_text"])
                return {"tool": tool_name, "status": "ok", "result": result}

            elif tool_name == "skill_rewrite":
                result = skill_rewrite(params["name"], params["description"], params["content"])
                return {"tool": tool_name, "status": "ok", "result": result}

            elif tool_name == "skill_delete":
                result = skill_delete(params["name"])
                return {"tool": tool_name, "status": "ok", "result": result}

            elif tool_name == "session_search":
                result = await session_search(params["query"])
                return {"tool": tool_name, "status": "ok", "result": result}

            elif tool_name == "session_list":
                result = await session_list()
                return {"tool": tool_name, "status": "ok", "result": result}
            
            # Registered async tools
            fn = TOOL_REGISTRY[tool_name]
            result = await fn(**params)
            return {"tool": tool_name, "status": "ok", "result": result}

        except PermissionError as e:
            logger.error("Permission denied in tool {}: {}", tool_name, e)
            return {"tool": tool_name, "status": "error", "result": f"Permission denied: {e}"}
        except FileNotFoundError as e:
            return {"tool": tool_name, "status": "error", "result": f"File not found: {e}"}
        except Exception as e:
            logger.error("Tool {} raised exception: {}", tool_name, e)
            return {"tool": tool_name, "status": "error", "result": f"Tool error: {e}"}

    async def process_llm_output(self, llm_text: str) -> tuple[str, list[dict]]:
        """
        Parse tool calls from LLM output, execute them, and return:
        (clean_text_without_tool_blocks, list_of_tool_results)
        """
        tool_calls = extract_tool_calls(llm_text)
        clean_text = strip_tool_calls(llm_text)
        results = []
        for call in tool_calls:
            result = await self.dispatch(call)
            results.append(result)
        return clean_text, results
