"""
Suite 7 — Skills System Tests

Verifies that skills can be created, loaded, updated, and deleted,
and that the skill index appears in system prompts.
"""
import pytest
from pathlib import Path


class TestSkillsCRUD:

    @pytest.fixture(autouse=True)
    def patch_skills_dir(self, tmp_path, monkeypatch):
        import app.memory.skills as sk
        monkeypatch.setattr(sk, "SKILLS_DIR", tmp_path / "skills")
        (tmp_path / "skills").mkdir()

    def test_create_skill(self):
        from app.memory.skills import skill_create, skill_load
        result = skill_create(
            name="deploy-snowflake",
            description="Steps to deploy a Snowflake pipeline.",
            content="1. Connect to Snowflake.\n2. Run migration scripts.\n3. Verify row counts.",
        )
        assert "created" in result.lower()
        loaded = skill_load("deploy-snowflake")
        assert "Connect to Snowflake" in loaded

    def test_create_duplicate_rejected(self):
        from app.memory.skills import skill_create
        skill_create("my-skill", "desc", "content")
        result = skill_create("my-skill", "desc2", "content2")
        assert "already exists" in result.lower()

    def test_load_nonexistent_skill(self):
        from app.memory.skills import skill_load
        result = skill_load("this-skill-does-not-exist")
        assert "not found" in result.lower()

    def test_fuzzy_match_suggestion(self):
        from app.memory.skills import skill_create, skill_load
        skill_create("deploy-snowflake", "Deploy pipeline.", "steps...")
        result = skill_load("snowflake")  # partial match
        assert "deploy-snowflake" in result or "snowflake" in result.lower()

    def test_update_skill_patch(self):
        from app.memory.skills import skill_create, skill_update, skill_load
        skill_create("my-workflow", "desc", "Step 1: old approach.\nStep 2: finish.")
        result = skill_update("my-workflow", "old approach", "new improved approach")
        assert "updated" in result.lower()
        loaded = skill_load("my-workflow")
        assert "new improved approach" in loaded
        assert "old approach" not in loaded

    def test_update_nonexistent_skill(self):
        from app.memory.skills import skill_update
        result = skill_update("ghost-skill", "old", "new")
        assert "not found" in result.lower()

    def test_update_ambiguous_text_rejected(self):
        from app.memory.skills import skill_create, skill_update
        skill_create("ambiguous", "desc", "do thing\ndo thing")  # 'do thing' appears twice
        result = skill_update("ambiguous", "do thing", "do better thing")
        assert "specific" in result.lower() or "multiple" in result.lower() or "appears" in result.lower()

    def test_rewrite_skill(self):
        from app.memory.skills import skill_create, skill_rewrite, skill_load
        skill_create("old-workflow", "Old desc.", "Old content.")
        result = skill_rewrite("old-workflow", "New description.", "Completely new content.")
        assert "rewritten" in result.lower()
        loaded = skill_load("old-workflow")
        assert "Completely new content" in loaded

    def test_delete_skill(self):
        from app.memory.skills import skill_create, skill_delete, skill_load
        skill_create("temp-skill", "Temporary.", "content.")
        skill_delete("temp-skill")
        loaded = skill_load("temp-skill")
        assert "not found" in loaded.lower()

    def test_delete_nonexistent_skill(self):
        from app.memory.skills import skill_delete
        result = skill_delete("never-existed")
        assert "not found" in result.lower()

    def test_slug_sanitization(self):
        """Skill names with spaces or caps should be slugified."""
        from app.memory.skills import skill_create, skill_load
        result = skill_create(
            name="Deploy Snowflake Pipeline",  # mixed case + spaces
            description="Deploy steps.",
            content="Steps here.",
        )
        assert "created" in result.lower()
        # Should be loadable with the slugified name
        loaded = skill_load("deploy-snowflake-pipeline")
        assert "Steps here" in loaded


class TestSkillsIndex:

    @pytest.fixture(autouse=True)
    def patch_skills_dir(self, tmp_path, monkeypatch):
        import app.memory.skills as sk
        monkeypatch.setattr(sk, "SKILLS_DIR", tmp_path / "skills")
        (tmp_path / "skills").mkdir()

    def test_index_empty_when_no_skills(self):
        from app.memory.skills import skills_list
        result = skills_list()
        assert result == ""

    def test_index_contains_created_skill(self):
        from app.memory.skills import skill_create, skills_list
        skill_create("rag-pipeline", "How to build a RAG pipeline.", "Steps...")
        index = skills_list()
        assert "rag-pipeline" in index
        assert "RAG pipeline" in index

    def test_index_contains_multiple_skills(self):
        from app.memory.skills import skill_create, skills_list
        skill_create("skill-a", "First skill.", "content a")
        skill_create("skill-b", "Second skill.", "content b")
        index = skills_list()
        assert "skill-a" in index
        assert "skill-b" in index

    def test_index_injected_into_prompt(self, tmp_path, monkeypatch):
        """skills_list() output appears in the system prompt."""
        import app.memory.skills as sk
        import app.memory.hot_memory as hm
        monkeypatch.setattr(sk, "SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr(hm, "MEMORY_PATH", tmp_path / "MEMORY.md")
        monkeypatch.setattr(hm, "USER_PATH",   tmp_path / "USER.md")
        (tmp_path / "skills").mkdir()

        sk.skill_create("obsidian-capture", "How to capture ideas in Obsidian.", "steps...")

        from app.prompts.templates import build_system_prompt
        prompt = build_system_prompt()
        assert "obsidian-capture" in prompt or "AVAILABLE SKILLS" in prompt
