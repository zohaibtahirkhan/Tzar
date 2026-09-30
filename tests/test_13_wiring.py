"""
Wiring tests — guard against "built but never connected".

The recurring failure mode in this codebase has been capability that exists
but is unreachable: a tool registered but rejected by the validator, a settings
field defined but never read, a tool name advertised to the LLM that does not
exist. Each of those shipped silently. These tests make them fail loudly.

They are pure static checks: no models, no network, no database writes.
"""
import re
from pathlib import Path

import pytest

# app.* is imported lazily inside each test: importing app.config instantiates
# Settings(), and a stale .env key makes that raise — which would otherwise
# kill collection of this whole module before the .env test could explain why.

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"


def _app_source_excluding(*skip_names: str) -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in APP.rglob("*.py")
        if p.name not in skip_names
    )


# ─── Tools ────────────────────────────────────────────────────────────────────

# Tools the router handles inline (they need the memory manager) rather than
# through TOOL_REGISTRY.
INLINE_TOOLS = {"save_memory", "recall_memory"}


def test_every_registered_tool_passes_validation():
    """A tool in TOOL_REGISTRY but not TOOL_SCHEMA is rejected by _validate()
    as 'Unknown tool' before it can ever run."""
    from app.tools.router import TOOL_REGISTRY, TOOL_SCHEMA
    unreachable = sorted(set(TOOL_REGISTRY) - set(TOOL_SCHEMA))
    assert not unreachable, (
        f"Registered but rejected by the validator (add to TOOL_SCHEMA): {unreachable}"
    )


def test_every_schema_entry_has_a_handler():
    """A tool in TOOL_SCHEMA passes validation, then dispatch() looks it up in
    TOOL_REGISTRY — a missing key there is a KeyError at runtime."""
    from app.tools.router import TOOL_REGISTRY, TOOL_SCHEMA
    orphaned = sorted(k for k in TOOL_SCHEMA if k not in TOOL_REGISTRY and k not in INLINE_TOOLS)
    assert not orphaned, f"In TOOL_SCHEMA with no handler: {orphaned}"


def test_registry_entries_are_callable_or_inline():
    """None in TOOL_REGISTRY means 'handled inline by ToolRouter'. Anything
    else must actually be callable."""
    from app.tools.router import TOOL_REGISTRY, TOOL_SCHEMA
    bad = sorted(
        name for name, fn in TOOL_REGISTRY.items()
        if fn is not None and not callable(fn)
    )
    assert not bad, f"Non-callable registry entries: {bad}"


def _advertised_tool_names() -> dict[str, set[str]]:
    """Tool names each prompt tells the LLM it may call."""
    out: dict[str, set[str]] = {}

    templates = (APP / "prompts" / "templates.py").read_text(encoding="utf-8")
    out["prompts/templates.py"] = set(re.findall(r"^- ([a-z][a-z0-9_]+):", templates, re.M))

    coordinator = (APP / "agents" / "coordinator.py").read_text(encoding="utf-8")
    block = re.search(r"AVAILABLE TOOLS:\n(.*?)\n\n", coordinator, re.S)
    out["agents/coordinator.py"] = set(re.findall(r"\b([a-z][a-z0-9_]+)\b", block.group(1))) if block else set()

    planner = (APP / "planner.py").read_text(encoding="utf-8")
    block = re.search(r"from this set:\n(.*?)\n- complexity", planner, re.S)
    names = set(re.findall(r"\b([a-z][a-z0-9_]+)\b", block.group(1))) if block else set()
    # research_agent is a routing hint for the planner, not a dispatchable tool.
    out["planner.py"] = names - {"research_agent", "special", "triggers", "multi", "source",
                                 "web", "research", "vault", "report"}
    return out


def test_advertised_tools_exist():
    """If a prompt names a tool the validator does not know, the LLM will keep
    trying to call it and every attempt will fail."""
    from app.tools.router import TOOL_REGISTRY, TOOL_SCHEMA
    problems = {
        src: sorted(n for n in names if n not in TOOL_SCHEMA)
        for src, names in _advertised_tool_names().items()
    }
    problems = {k: v for k, v in problems.items() if v}
    assert not problems, f"Advertised to the LLM but not dispatchable: {problems}"


# ─── Settings ─────────────────────────────────────────────────────────────────

# Fields that exist for introspection or are only consumed inside config.py.
# Each needs a reason — this list should shrink, not grow.
INTROSPECTION_ONLY_SETTINGS = {
    "platform_system": "read-only platform report, surfaced via get_platform_optimizations()",
    "is_windows":      "read-only platform report",
    "is_macos":        "read-only platform report",
    "is_linux":        "read-only platform report",
    "path_separator":  "read-only platform report",
    "base_dir":        "root used to build the other path defaults inside config.py",
}


def test_every_setting_is_read_somewhere():
    """A settings field nothing reads is a lie in .env: the user changes it and
    nothing happens. Wire it or delete it."""
    from app.config import Settings
    source = _app_source_excluding("config.py")
    unread = sorted(
        name for name in Settings.model_fields
        if name not in INTROSPECTION_ONLY_SETTINGS
        and not re.search(rf"\bsettings\.{re.escape(name)}\b", source)
    )
    assert not unread, f"Defined in Settings but never read: {unread}"


def test_introspection_allowlist_is_not_stale():
    """Keep the allowlist honest: every entry must still be a real field, and
    must genuinely be unread (otherwise it should be removed from the list)."""
    from app.config import Settings
    source = _app_source_excluding("config.py")
    for name in INTROSPECTION_ONLY_SETTINGS:
        assert name in Settings.model_fields, f"Allowlisted setting no longer exists: {name}"
        assert not re.search(rf"\bsettings\.{re.escape(name)}\b", source), (
            f"'{name}' is now read in app/ — drop it from INTROSPECTION_ONLY_SETTINGS"
        )


def _settings_field_names_from_source() -> set[str]:
    """
    Read field names straight from config.py rather than importing Settings.
    Importing instantiates Settings(), which is exactly what a stale .env key
    breaks — so the import-based check could never report this case cleanly.
    """
    src = (APP / "config.py").read_text(encoding="utf-8")
    body = src.split("class Settings(BaseSettings):", 1)[1].split("\nsettings = Settings()", 1)[0]
    return {m.group(1).lower() for m in re.finditer(r"^\s{4}([A-Za-z_][A-Za-z0-9_]*)\s*:", body, re.M)}


def test_env_files_only_reference_real_settings():
    """pydantic-settings raises on unknown keys in .env, so a stale key there
    crashes startup. Catch it here with a clear message instead."""
    fields = _settings_field_names_from_source()
    assert "llm_model" in fields, "field extraction from config.py broke"
    for env_name in (".env.example", ".env"):
        env_path = ROOT / env_name
        if not env_path.exists():
            continue
        stale = []
        for line in env_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            if m and m.group(1).lower() not in fields:
                stale.append(m.group(1))
        assert not stale, f"{env_name} sets keys that are not Settings fields: {stale}"
