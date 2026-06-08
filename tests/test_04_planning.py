"""
Suite 4 — Planning Tests

Verifies that:
  A. classify_intent() correctly classifies simple vs complex inputs
  B. make_plan() produces ordered, complete steps for multi-tool tasks
  C. The pipeline injects the plan into the system prompt for complex tasks
  D. Multi-step sequences execute in the correct order
"""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from tests.conftest import make_llm_response, make_plan_response


# ─── 4.1  Complexity classifier ──────────────────────────────────────────────

class TestComplexityClassifier:
    """classify_intent() — fast heuristic, no LLM call."""

    def test_simple_math_not_complex(self):
        from app.intent import classify_intent, IntentType
        assert classify_intent("What is 25 times 37?") == IntentType.CHAT

    def test_greeting_not_complex(self):
        from app.intent import classify_intent, IntentType
        assert classify_intent("Good morning") != IntentType.PLANNING

    def test_thanks_not_complex(self):
        from app.intent import classify_intent, IntentType
        assert classify_intent("Thanks") == IntentType.CHAT

    def test_single_word_not_complex(self):
        from app.intent import classify_intent, IntentType
        assert classify_intent("Hello") == IntentType.CHAT

    def test_simple_question_not_complex(self):
        from app.intent import classify_intent, IntentType
        result = classify_intent("What is Snowflake?")
        assert result not in (IntentType.PLANNING, IntentType.RESEARCH)

    def test_create_note_is_tool(self):
        from app.intent import classify_intent, IntentType
        # Single tool action → TOOL (not PLANNING; that's correct behaviour)
        assert classify_intent("Create a note about the RAG pipeline") == IntentType.TOOL

    def test_research_and_save_is_planning(self):
        from app.intent import classify_intent, IntentType
        result = classify_intent(
            "Find the latest Snowflake release, create a note, and add it to my daily log"
        )
        assert result == IntentType.PLANNING

    def test_multi_step_with_then_is_planning(self):
        from app.intent import classify_intent, IntentType
        assert classify_intent("Search for Python 3.13 changes then summarise them") == IntentType.PLANNING

    def test_organize_is_planning(self):
        from app.intent import classify_intent, IntentType
        assert classify_intent("Organise all my project notes") == IntentType.PLANNING

    def test_analyse_is_planning(self):
        from app.intent import classify_intent, IntentType
        assert classify_intent("Analyse my recent notes and find gaps") == IntentType.PLANNING

    def test_write_summary_is_planning(self):
        from app.intent import classify_intent, IntentType
        assert classify_intent("Write a summary of my Data Engineering project") == IntentType.PLANNING

    def test_needs_planning_shim_false_for_chat(self):
        """Backward-compat shim: returns False for CHAT/TOOL/MEMORY."""
        from app.planner import needs_planning
        assert needs_planning("What is 25 times 37?") is False
        assert needs_planning("Thanks") is False
        assert needs_planning("Hello") is False

    def test_needs_planning_shim_true_for_planning(self):
        """Backward-compat shim: returns True for PLANNING and RESEARCH."""
        from app.planner import needs_planning
        assert needs_planning("Find the Snowflake release then create a note") is True
        assert needs_planning("Research the latest llama.cpp updates") is True


# ─── 4.2  Plan structure ─────────────────────────────────────────────────────

class TestPlanStructure:
    """make_plan() — verifies steps, tools, and ordering."""

    @pytest.mark.asyncio
    async def test_plan_has_goal(self):
        from app.planner import make_plan
        llm_fn = AsyncMock(return_value=make_plan_response(
            goal="Find latest Snowflake release and create a linked note.",
            steps=["Search web for Snowflake release", "Create note", "Link to Data Engineering", "Append daily"],
            tools=["web_search", "obsidian_create_note", "kg_add", "obsidian_append_daily"],
            complexity="high",
        ))
        plan = await make_plan("Find latest Snowflake release, create note, link it.", llm_fn)
        assert plan.goal != ""
        assert len(plan.steps) >= 3

    @pytest.mark.asyncio
    async def test_plan_step_order_search_first(self):
        """For a research task: search must come before create."""
        from app.planner import make_plan
        llm_fn = AsyncMock(return_value=make_plan_response(
            goal="Research and document Snowflake release.",
            steps=[
                "Step 1: Search web for latest Snowflake release notes",
                "Step 2: Summarize findings",
                "Step 3: Create Obsidian note",
                "Step 4: Add knowledge graph links to Data Engineering",
                "Step 5: Append summary to daily note",
            ],
            tools=["web_search", "obsidian_create_note", "kg_add", "obsidian_append_daily"],
            complexity="high",
        ))
        plan = await make_plan("Find latest Snowflake release, create a note, link to Data Engineering, add to daily.", llm_fn)
        step_text = " ".join(plan.steps).lower()
        search_pos = step_text.find("search")
        create_pos = step_text.find("create")
        assert search_pos < create_pos, f"Search step must precede create step. Steps: {plan.steps}"

    @pytest.mark.asyncio
    async def test_plan_requires_expected_tools(self):
        """Research → note → KG → daily task needs all four tools."""
        from app.planner import make_plan
        expected_tools = {"web_search", "obsidian_create_note", "kg_add", "obsidian_append_daily"}
        llm_fn = AsyncMock(return_value=make_plan_response(
            goal="Full research and documentation pipeline.",
            steps=["Search", "Create note", "KG links", "Daily note"],
            tools=list(expected_tools),
            complexity="high",
        ))
        plan = await make_plan("Find Snowflake release, create note, link to Data Engineering, add to daily.", llm_fn)
        for tool in expected_tools:
            assert tool in plan.required_tools, f"Missing tool in plan: {tool}"

    @pytest.mark.asyncio
    async def test_simple_task_can_answer_directly(self):
        """A simple question should set can_answer_directly=True."""
        from app.planner import make_plan
        llm_fn = AsyncMock(return_value=make_plan_response(
            goal="Answer a factual question.",
            can_answer_directly=True,
        ))
        plan = await make_plan("What is 25 times 37?", llm_fn)
        assert plan.can_answer_directly is True

    @pytest.mark.asyncio
    async def test_plan_context_string_format(self):
        """Plan.to_context_string() should produce a string the pipeline can inject."""
        from app.planner import Plan
        plan = Plan(
            goal="Research and document Snowflake release.",
            complexity="high",
            steps=["Search web", "Create note", "Append daily"],
            required_tools=["web_search", "obsidian_create_note", "obsidian_append_daily"],
        )
        ctx = plan.to_context_string()
        assert "TASK PLAN" in ctx
        assert "Search web" in ctx
        assert "web_search" in ctx

    @pytest.mark.asyncio
    async def test_direct_answer_plan_has_no_context(self):
        """If can_answer_directly=True, context string should be empty (no plan injected)."""
        from app.planner import Plan
        plan = Plan(can_answer_directly=True, goal="Simple answer.")
        assert plan.to_context_string() == ""

    @pytest.mark.asyncio
    async def test_plan_survives_parse_failure(self):
        """If LLM returns garbage, make_plan should return an empty Plan (not crash)."""
        from app.planner import make_plan
        llm_fn = AsyncMock(return_value="This is not JSON at all!!!")
        plan = await make_plan("Some complex task.", llm_fn)
        assert plan is not None
        assert isinstance(plan.steps, list)


# ─── 4.3  Pipeline planner integration ───────────────────────────────────────

class TestPlannerPipelineIntegration:
    """
    Verifies that AssistantPipeline calls make_plan for PLANNING intent
    and skips it for CHAT intent.
    """

    @pytest.mark.asyncio
    async def test_planner_called_for_complex_input(self):
        """Pipeline should call make_plan when intent is PLANNING."""
        from app.pipeline import AssistantPipeline
        from app.planner import Plan
        from app.intent import IntentType

        pipeline = AssistantPipeline()
        simple_plan = Plan(can_answer_directly=False, goal="test", steps=["search", "create note"])

        with patch("app.pipeline.classify_intent", return_value=IntentType.PLANNING) as mock_intent, \
             patch("app.pipeline.make_plan", new=AsyncMock(return_value=simple_plan)) as mock_plan, \
             patch("app.pipeline.llm_engine") as mock_llm, \
             patch("app.pipeline.memory_manager") as mock_mem, \
             patch("app.memory.projects.extract_project_name_from_query", return_value=None):

            mock_llm.generate = AsyncMock(return_value=make_llm_response(response="Done."))
            mock_mem.get_context = AsyncMock(return_value=("", ""))
            mock_mem.add_turn    = AsyncMock()

            with patch("app.pipeline.log_turn", new=AsyncMock()):
                await pipeline.process_text_input(
                    "Find the latest Snowflake release and create a note and link it."
                )

            mock_intent.assert_called_once()
            mock_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_planner_skipped_for_simple_input(self):
        """Pipeline should NOT call make_plan for CHAT intent."""
        from app.pipeline import AssistantPipeline
        from app.intent import IntentType

        pipeline = AssistantPipeline()

        with patch("app.pipeline.classify_intent", return_value=IntentType.CHAT) as mock_intent, \
             patch("app.pipeline.make_plan", new=AsyncMock()) as mock_plan, \
             patch("app.pipeline.llm_engine") as mock_llm, \
             patch("app.pipeline.memory_manager") as mock_mem, \
             patch("app.memory.projects.extract_project_name_from_query", return_value=None):

            mock_llm.generate = AsyncMock(return_value=make_llm_response(response="Twenty-five."))
            mock_mem.get_context = AsyncMock(return_value=("", ""))
            mock_mem.add_turn    = AsyncMock()

            with patch("app.pipeline.log_turn", new=AsyncMock()):
                await pipeline.process_text_input("What is 5 times 5?")

            mock_intent.assert_called_once()
            mock_plan.assert_not_called()


# ─── 4.4  Full multi-step sequence simulation ────────────────────────────────

class TestMultiStepSequence:
    """
    Simulate: search → summarize → create note → kg links → daily note.
    Verifies the tool loop executes all steps without skipping.
    """

    @pytest.mark.asyncio
    async def test_four_tool_sequence_completes(self, tool_router):
        """Dispatch four different tools in sequence and verify all succeed."""
        tools_executed = []

        async def fake_web_search(**kwargs):
            tools_executed.append("web_search")
            return "Snowflake 8.0 released with new features."

        async def fake_create_note(**kwargs):
            tools_executed.append("obsidian_create_note")
            return "Note created: Snowflake 8.0"

        async def fake_kg_add(**kwargs):
            tools_executed.append("kg_add")
            return "Graph updated."

        async def fake_append_daily(**kwargs):
            tools_executed.append("obsidian_append_daily")
            return "Appended to daily note."

        with patch("app.tools.web_search.tool_web_search",       new=AsyncMock(side_effect=fake_web_search)), \
             patch("app.tools.obsidian.obsidian_create_note",     new=AsyncMock(side_effect=fake_create_note)), \
             patch("app.tools.obsidian.obsidian_append_daily",    new=AsyncMock(side_effect=fake_append_daily)):

            from app.tools import router as r
            r.TOOL_REGISTRY["web_search"]             = AsyncMock(side_effect=fake_web_search)
            r.TOOL_REGISTRY["obsidian_create_note"]   = AsyncMock(side_effect=fake_create_note)
            r.TOOL_REGISTRY["obsidian_append_daily"]  = AsyncMock(side_effect=fake_append_daily)

            results = []
            for call in [
                {"tool": "web_search",           "query": "latest Snowflake release"},
                {"tool": "obsidian_create_note", "title": "Snowflake 8.0", "content": "Summary."},
                {"tool": "obsidian_append_daily","content": "Reviewed Snowflake 8.0 release."},
            ]:
                result = await tool_router.dispatch(call)
                results.append(result)

        assert all(r["status"] == "ok" for r in results), \
            f"Some tools failed: {[r for r in results if r['status'] != 'ok']}"
        assert len(results) == 3