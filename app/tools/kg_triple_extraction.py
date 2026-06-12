"""
app/tools/kg_triple_extraction.py

LLM-powered subject-relation-object triple extraction.
Replaces the co-occurrence heuristic in infer_relations() with real triples.

Usage — drop-in replacement inside kg_extract_and_index():
    from app.tools.kg_triple_extraction import extract_triples_llm, merge_into_kg
    triples = await extract_triples_llm(text)
    await merge_into_kg(note_title, entities, triples)

Relation vocabulary (canonical set — forces consistency across notes):
    USES          — X uses Y as a tool/library/service
    OWNS          — X owns/maintains Y
    WORKS_ON      — X is actively working on Y
    PART_OF       — X is a component/subproject of Y
    DEPENDS_ON    — X requires Y to function
    CREATED_BY    — Y was created/authored by X
    LOCATED_AT    — X is hosted/running at Y
    KNOWS         — X knows/understands Y (person→concept)
    RELATED_TO    — generic fallback (co-occurrence)
    IMPLEMENTS    — X implements Y (interface/protocol/spec)
    EXTENDS       — X extends/inherits Y
    GENERATES     — X produces/generates Y
    CALLS         — X calls/invokes Y (function/API level)
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

from loguru import logger

# ─── Canonical relation set ───────────────────────────────────────────────────

CANONICAL_RELATIONS = {
    "uses", "owns", "works_on", "part_of", "depends_on", "created_by",
    "located_at", "knows", "related_to", "implements", "extends",
    "generates", "calls",
}


def _normalise_relation(rel: str) -> str:
    """
    Normalise a raw relation string to the canonical vocabulary.
    USES / uses / Uses / HAS_USES → 'uses'
    """
    r = rel.strip().lower().replace(" ", "_").replace("-", "_")
    if r in CANONICAL_RELATIONS:
        return r

    # Fuzzy mapping for common synonyms the LLM produces
    _synonyms: dict[str, str] = {
        "use":           "uses",
        "utilized_by":   "uses",
        "utilizes":      "uses",
        "has":           "owns",
        "maintained_by": "owns",
        "maintains":     "owns",
        "built_by":      "created_by",
        "built":         "created_by",
        "authored_by":   "created_by",
        "runs_on":       "located_at",
        "hosted_on":     "located_at",
        "connected_to":  "related_to",
        "associated_with": "related_to",
        "inherits":      "extends",
        "inherits_from": "extends",
        "produces":      "generates",
        "outputs":       "generates",
        "invokes":       "calls",
        "depends":       "depends_on",
        "requires":      "depends_on",
    }
    return _synonyms.get(r, "related_to")   # default to generic relation


# ─── Prompt ───────────────────────────────────────────────────────────────────

_TRIPLE_EXTRACTION_PROMPT = """Extract all meaningful subject-relation-object triples from this text.

RULES:
- subject and object must be specific named entities, concepts, or systems (not pronouns or generic nouns like "it", "the system").
- relation must be one of: USES, OWNS, WORKS_ON, PART_OF, DEPENDS_ON, CREATED_BY, LOCATED_AT, KNOWS, RELATED_TO, IMPLEMENTS, EXTENDS, GENERATES, CALLS
- Only extract relationships that are clearly stated or strongly implied. Do not hallucinate.
- Keep subject/object names concise (1–4 words).

Return ONLY a JSON object — no prose, no markdown.
{{"triples": [["subject", "RELATION", "object"], ...]}}

Text:
{text}"""


# ─── Extraction ───────────────────────────────────────────────────────────────

async def extract_triples_llm(
    text: str,
    max_chars: int = 1500,
) -> list[tuple[str, str, str]]:
    """
    Call the LLM to extract (subject, relation, object) triples from text.
    Returns empty list on any failure — caller should fall back to co-occurrence.
    """
    if not text or not text.strip():
        return []

    truncated = text[:max_chars]

    try:
        from app.llm.engine import llm_engine
        raw = await asyncio.wait_for(
            llm_engine.generate(
                [{"role": "user", "content": _TRIPLE_EXTRACTION_PROMPT.format(text=truncated)}],
                system_prompt="Return only valid JSON. Extract only clearly stated relationships.",
            ),
            timeout=20.0,
        )

        # Strip markdown fences if present
        clean = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        m     = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            logger.debug("KG triple extraction: no JSON in LLM response")
            return []

        data = json.loads(m.group())
        raw_triples = data.get("triples", [])

        triples: list[tuple[str, str, str]] = []
        for t in raw_triples:
            if not isinstance(t, list) or len(t) != 3:
                continue
            subj, rel, obj = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
            if not subj or not obj or not rel:
                continue
            # Skip triples where subject or object is a stop word / too short
            if len(subj) < 2 or len(obj) < 2:
                continue
            norm_rel = _normalise_relation(rel)
            triples.append((subj, norm_rel, obj))

        logger.debug("KG triple extraction: {} triples from {} chars", len(triples), len(truncated))
        return triples

    except asyncio.TimeoutError:
        logger.warning("KG triple extraction timed out — falling back to co-occurrence")
        return []
    except json.JSONDecodeError as exc:
        logger.warning("KG triple extraction JSON parse error: {}", exc)
        return []
    except Exception as exc:
        logger.warning("KG triple extraction failed: {}", exc)
        return []


# ─── Merge triples into graph ─────────────────────────────────────────────────

async def merge_into_kg(
    note_title: str,
    entities: list[str],
    triples: list[tuple[str, str, str]],
) -> str:
    """
    Add entities and LLM-extracted triples to the knowledge graph.
    Entities that don't appear in triples still get added as isolated nodes.
    This replaces kg_add_from_note for the enhanced extraction path.
    """
    import asyncio as _asyncio
    from app.tools.knowledge_graph import _get_kg_conn, _add_node, _upsert_edge

    loop = asyncio.get_running_loop()

    def _write():
        conn = _get_kg_conn()
        added_nodes = 0
        added_edges = 0

        # Add note as a node
        note_node_id = _add_node(conn, note_title, label="note")
        added_nodes += 1

        # Collect all names that appear as subjects or objects in triples
        triple_names: set[str] = set()
        for subj, _, obj in triples:
            triple_names.add(subj)
            triple_names.add(obj)

        # Add entities from regex pass — ensure all nodes exist
        name_to_id: dict[str, int] = {}
        all_names = set(entities) | triple_names | {note_title}
        for name in all_names:
            if not name or name == note_title:
                continue
            # Infer node label from triple context
            # (nodes that appear in triples with CREATED_BY are typically people/tools)
            label = _infer_label(name, triples)
            node_id = _add_node(conn, name, label=label)
            name_to_id[name] = node_id
            added_nodes += 1

        # Add LLM-extracted triples as edges
        for subj, rel, obj in triples:
            src_id = name_to_id.get(subj)
            tgt_id = name_to_id.get(obj)
            if src_id and tgt_id and src_id != tgt_id:
                _upsert_edge(conn, src_id, tgt_id, rel, source_note=note_title)
                added_edges += 1

        # Connect orphan entities (those not in any triple) to the note node
        orphan_names = set(entities) - triple_names - {note_title}
        for name in orphan_names:
            oid = name_to_id.get(name)
            if oid:
                _upsert_edge(conn, note_node_id, oid, "related_to", source_note=note_title)
                added_edges += 1

        conn.commit()
        conn.close()
        return added_nodes, added_edges

    added_nodes, added_edges = await loop.run_in_executor(None, _write)
    return (
        f"Updated Knowledge Graph: +{added_nodes} nodes, +{added_edges} edges "
        f"(from '{note_title}', {len(triples)} triples)"
    )


def _infer_label(name: str, triples: list[tuple[str, str, str]]) -> str:
    """
    Use triple context to guess a node's label.
    e.g. if X CREATED_BY Y, then Y is probably a person.
    """
    for subj, rel, obj in triples:
        if rel == "created_by" and obj == name:
            return "person"
        if rel == "created_by" and subj == name:
            return "tool"
        if rel == "works_on" and subj == name:
            return "person"
        if rel == "part_of" and obj == name:
            return "project"
        if rel == "uses" and subj == name:
            # could be person or tool
            pass
    return "concept"