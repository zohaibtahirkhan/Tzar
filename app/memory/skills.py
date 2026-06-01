"""
Self-improving Skills system.

Skills are on-demand SKILL.md documents the LLM can:
  - Load when relevant to the current task
  - Create after completing a complex multi-tool workflow
  - Update when it discovers a better approach
  - Delete when a skill is no longer useful

Skills live in data/skills/<name>/SKILL.md
The LLM sees a skill index (name + description) in every prompt.
Full skill content is loaded on demand to save tokens.
"""
import os
import re
from pathlib import Path
from datetime import datetime

from loguru import logger

from app.config import settings

SKILLS_DIR = settings.data_dir / "skills"
SKILLS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Index ────────────────────────────────────────────────────────────────────

def _parse_frontmatter(text: str) -> dict:
    """Extract simple key: value pairs from a --- frontmatter block."""
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta


def skills_list() -> str:
    """
    Return a compact index of all skills.
    This is injected into every prompt so the LLM knows what's available.
    """
    skill_dirs = sorted(SKILLS_DIR.iterdir()) if SKILLS_DIR.exists() else []
    skills = []

    for d in skill_dirs:
        skill_md = d / "SKILL.md"
        if not d.is_dir() or not skill_md.exists():
            continue
        meta = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        name = meta.get("name", d.name)
        desc = meta.get("description", "No description.")
        skills.append(f"  /{name}: {desc}")

    if not skills:
        return ""

    return "AVAILABLE SKILLS (load with skill_load):\n" + "\n".join(skills)


# ─── Load ─────────────────────────────────────────────────────────────────────

def skill_load(name: str) -> str:
    """Load and return full skill content for injection into the prompt."""
    skill_dir = SKILLS_DIR / name
    skill_md  = skill_dir / "SKILL.md"

    if not skill_md.exists():
        # Fuzzy match
        candidates = [d.name for d in SKILLS_DIR.iterdir() if d.is_dir() and name.lower() in d.name.lower()]
        if candidates:
            return f"Skill '{name}' not found. Did you mean: {', '.join(candidates)}?"
        return f"Skill '{name}' not found. Use skill_create to make one."

    content = skill_md.read_text(encoding="utf-8")
    return f"=== SKILL: {name} ===\n{content}\n=== END SKILL ==="


# ─── Create / Update / Delete ─────────────────────────────────────────────────

def skill_create(name: str, description: str, content: str, category: str = "general") -> str:
    """
    Create a new skill.
    name: slug like "obsidian-idea-capture"
    description: one sentence shown in the skill index
    content: full SKILL.md body (procedure, pitfalls, examples)
    """
    name = re.sub(r"[^a-z0-9\-_]", "-", name.lower()).strip("-")
    skill_dir = SKILLS_DIR / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md  = skill_dir / "SKILL.md"

    if skill_md.exists():
        return f"Skill '{name}' already exists. Use skill_update to modify it."

    now = datetime.utcnow().strftime("%Y-%m-%d")
    full_content = (
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"category: {category}\n"
        f"created: {now}\n"
        f"---\n\n"
        f"{content.strip()}\n"
    )

    skill_md.write_text(full_content, encoding="utf-8")
    logger.info("Skill created: {}", name)
    return f"Skill '{name}' created and saved. It will appear in the skill index on the next turn."


def skill_update(name: str, old_text: str, new_text: str) -> str:
    """
    Patch a skill — surgical replacement, preferred over full rewrite.
    old_text must appear exactly once in the skill.
    """
    skill_md = SKILLS_DIR / name / "SKILL.md"
    if not skill_md.exists():
        return f"Skill '{name}' not found."

    content = skill_md.read_text(encoding="utf-8")
    count   = content.count(old_text)

    if count == 0:
        return f"old_text not found in skill '{name}'."
    if count > 1:
        return f"old_text appears {count} times — be more specific."

    updated = content.replace(old_text, new_text, 1)
    skill_md.write_text(updated, encoding="utf-8")
    logger.info("Skill updated: {}", name)
    return f"Skill '{name}' updated."


def skill_rewrite(name: str, description: str, content: str) -> str:
    """Full rewrite of a skill (when patch isn't enough)."""
    skill_md = SKILLS_DIR / name / "SKILL.md"
    if not skill_md.exists():
        return f"Skill '{name}' not found."

    meta     = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    created  = meta.get("created", datetime.utcnow().strftime("%Y-%m-%d"))
    category = meta.get("category", "general")
    now      = datetime.utcnow().strftime("%Y-%m-%d")

    full_content = (
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"category: {category}\n"
        f"created: {created}\n"
        f"updated: {now}\n"
        f"---\n\n"
        f"{content.strip()}\n"
    )

    skill_md.write_text(full_content, encoding="utf-8")
    logger.info("Skill rewritten: {}", name)
    return f"Skill '{name}' rewritten."


def skill_delete(name: str) -> str:
    """Delete a skill entirely."""
    import shutil
    skill_dir = SKILLS_DIR / name
    if not skill_dir.exists():
        return f"Skill '{name}' not found."
    shutil.rmtree(skill_dir)
    logger.info("Skill deleted: {}", name)
    return f"Skill '{name}' deleted."