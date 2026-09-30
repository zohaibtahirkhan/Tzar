"""
Research Agent

Turns a research question into a structured, multi-source report
and optionally saves it to the Obsidian vault.

Flow:
    research question
        ↓
    ResearchPlanner  — decide what angles to search
        ↓
    MultiSearch      — 3-5 targeted web searches
        ↓
    SourceReader     — fetch + extract key content per source
        ↓
    FactMerger       — deduplicate, resolve contradictions
        ↓
    ReportWriter     — structured markdown report
        ↓
    obsidian_create_note  (if save=True)
        ↓
    return summary string to pipeline

Usage from pipeline.py:

    from app.agents.researcher import research_agent
    result = await research_agent.run(user_text)

The agent is self-contained — it imports what it needs internally
to avoid circular import issues.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    full_text: str = ""         # populated if we fetch the page


@dataclass
class ResearchTask:
    question: str
    search_queries: list[str] = field(default_factory=list)
    raw_results: list[SearchResult] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    summary: str = ""
    report: str = ""
    note_title: str = ""
    sources: list[str] = field(default_factory=list)
    saved_to_vault: bool = False


# ─── Prompts ─────────────────────────────────────────────────────────────────

_QUERY_GEN_SYSTEM = """You are a research query generator.
Given a research question, generate 3-5 targeted search queries that together
will cover the topic from different angles.

Respond ONLY with a JSON array of strings. No prose. No markdown.
Example: ["query one", "query two", "query three"]

Rules:
- Each query should target a different aspect (release notes, benchmarks,
  community reactions, comparisons, documentation)
- Queries should be concise (3-8 words)
- Do NOT repeat the same query with minor variations
"""

_SYNTHESIS_SYSTEM = """You are a research analyst.
Given search results about a topic, extract and synthesize the key findings.

Respond ONLY in JSON:
{
  "findings": ["fact 1", "fact 2", ...],
  "contradictions": ["source A says X but source B says Y", ...],
  "key_entities": ["entity1", "entity2", ...],
  "summary": "2-3 sentence executive summary",
  "note_title": "concise note title for this research"
}

Rules:
- findings: concrete, specific facts only — no vague generalities
- contradictions: only genuine conflicts between sources
- summary: written for a human, no markdown
- note_title: 4-8 words, titlecase
"""

_REPORT_SYSTEM = """You are a technical writer.
Given research findings, write a structured Markdown research report.

The report should have these sections (use ## headings):
## Summary
## Key Findings
## Details
## Contradictions or Uncertainties (only if there are any)
## Sources

Rules:
- Be specific and factual — cite sources inline as [Source N]
- Summary: 2-3 sentences max
- Key Findings: bullet points, each one concrete
- Details: prose expanding on the findings
- Sources: numbered list of URLs
- Total length: 300-600 words
"""


# ─── Core agent ──────────────────────────────────────────────────────────────

class ResearchAgent:
    """
    Orchestrates multi-source research. Uses the LLM for planning and
    synthesis, DuckDuckGo for search.
    """

    def __init__(self):
        self._max_queries = 4
        self._max_results_per_query = 3
        self._fetch_top_n = 2        # full-page fetch for top N results
        self._fetch_timeout = 8.0    # seconds per fetch
        self._max_snippet_chars = 800

    # ── Public entry point ────────────────────────────────────────────────

    async def run(
        self,
        question: str,
        save_to_vault: bool = True,
    ) -> str:
        """
        Run a full research cycle.

        Returns the executive summary (short, for TTS/terminal).
        Full report is saved to Obsidian if save_to_vault=True.
        """
        t0 = time.perf_counter()
        logger.info("ResearchAgent: starting — '{}'", question[:80])

        task = ResearchTask(question=question)

        try:
            # 1. Generate search queries
            task.search_queries = await self._plan_queries(question)
            logger.info("ResearchAgent: {} queries planned", len(task.search_queries))

            # 2. Execute searches
            task.raw_results = await self._multi_search(task.search_queries)
            logger.info("ResearchAgent: {} results collected", len(task.raw_results))

            if not task.raw_results:
                return (
                    "I ran the searches but found no usable results. "
                    "Web search may be disabled — say 'enable web search' first."
                )

            # 3. Optionally fetch top page content
            await self._enrich_top_results(task.raw_results)

            # 4. Synthesise findings
            await self._synthesise(task)
            logger.info(
                "ResearchAgent: {} findings, {} contradictions",
                len(task.findings),
                len(task.contradictions),
            )

            # 5. Write report
            await self._write_report(task)

            # 6. Save to Obsidian
            if save_to_vault and task.report:
                await self._save_to_vault(task)

            elapsed = time.perf_counter() - t0
            logger.info("ResearchAgent: done in {:.1f}s", elapsed)

            # Return short summary for TTS / terminal display
            return self._build_response(task)

        except Exception as e:
            logger.error("ResearchAgent failed: {}", e)
            return f"Research failed: {e}"

    # ── Step 1: Query planning ────────────────────────────────────────────

    async def _plan_queries(self, question: str) -> list[str]:
        """Ask the LLM to generate 3-5 targeted search queries."""
        from app.llm.engine import llm_engine

        messages = [{"role": "user", "content": f"Research question: {question}"}]
        try:
            raw = await llm_engine.generate(
                messages,
                system_prompt=_QUERY_GEN_SYSTEM,
                max_tokens=256,
                temperature=0.3,
            )
            queries = self._parse_json_list(raw)
            # Fallback: use the original question as a single query
            if not queries:
                queries = [question]
            return queries[: self._max_queries]
        except Exception as e:
            logger.warning("Query planning failed ({}), using raw question", e)
            return [question]

    # ── Step 2: Multi-search ──────────────────────────────────────────────
    async def _multi_search(self, queries: list[str]) -> list[SearchResult]:
        """
        Run queries SEQUENTIALLY with a delay to prevent 202 Ratelimit errors.
        """
        seen_urls: set[str] = set()
        combined: list[SearchResult] = []

        for i, query in enumerate(queries):
            # Run searches one by one
            try:
                results = await self._search_one(query)
                
                for r in results:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        combined.append(r)
                
                # CRITICAL: Sleep between requests to avoid rate limits
                # Wait 3-4 seconds between requests
                if i < len(queries) - 1:
                    await asyncio.sleep(3.5)
                    
            except Exception as e:
                logger.warning("Search batch failed: {}", e)
                continue

        return combined

    async def _search_one(self, query: str) -> list[SearchResult]:
        """Single DuckDuckGo search using the 'lite' backend to avoid blocking."""
        from app.config import settings

        if not settings.web_search_enabled:
            logger.warning("Web search disabled — ResearchAgent cannot search")
            return []

        try:
            from duckduckgo_search import DDGS

            loop = asyncio.get_running_loop()

            def _do_search():
                results = []
                with DDGS() as ddgs:
                    # Using backend="lite" helps avoid the heavy HTML scraping blocks
                    search_gen = ddgs.text(
                        query, 
                        max_results=self._max_results_per_query,
                        backend="lite"  # <--- Key change
                    )
                    if search_gen:
                        for r in search_gen:
                            results.append(
                                SearchResult(
                                    title=r.get("title", ""),
                                    url=r.get("href", ""),
                                    snippet=r.get("body", "")[:self._max_snippet_chars],
                                )
                            )
                return results

            return await loop.run_in_executor(None, _do_search)

        except ImportError:
            logger.error("duckduckgo-search not installed")
            return []
        except Exception as e:
            logger.warning("Search '{}' failed: {}", query[:40], e)
            return []

    # ── Step 3: Page content fetching ─────────────────────────────────────

    async def _enrich_top_results(self, results: list[SearchResult]) -> None:
        """
        Fetch full page content for top N results and store in .full_text.
        Non-blocking — failures just leave full_text empty.
        """
        top = results[: self._fetch_top_n]
        tasks = [self._fetch_page(r) for r in top]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_page(self, result: SearchResult) -> None:
        """Fetch a URL and extract plain text (best-effort)."""
        if not result.url.startswith("http"):
            return

        try:
            import httpx

            async with httpx.AsyncClient(
                timeout=self._fetch_timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (research bot)"},
            ) as client:
                resp = await client.get(result.url)
                if resp.status_code == 200:
                    result.full_text = self._extract_text(resp.text)[:3000]

        except Exception as e:
            logger.debug("Page fetch failed for {}: {}", result.url[:60], e)

    @staticmethod
    def _extract_text(html: str) -> str:
        """Naïve HTML → plain text: strip tags, collapse whitespace."""
        # Remove script and style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        # Strip remaining tags
        html = re.sub(r"<[^>]+>", " ", html)
        # Decode common entities
        for ent, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                          ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"')]:
            html = html.replace(ent, char)
        # Collapse whitespace
        return re.sub(r"\s+", " ", html).strip()

    # ── Step 4: Synthesis ─────────────────────────────────────────────────

    async def _synthesise(self, task: ResearchTask) -> None:
        """Extract findings, contradictions, and summary using the LLM."""
        from app.llm.engine import llm_engine

        # Build context string from all results
        context_parts = []
        for i, r in enumerate(task.raw_results, 1):
            body = r.full_text if r.full_text else r.snippet
            context_parts.append(
                f"[Source {i}] {r.title}\nURL: {r.url}\n{body}"
            )
            task.sources.append(f"{i}. {r.title} — {r.url}")

        context = "\n\n---\n\n".join(context_parts)

        user_msg = (
            f"Research question: {task.question}\n\n"
            f"Search results:\n\n{context}"
        )

        messages = [{"role": "user", "content": user_msg}]

        try:
            raw = await llm_engine.generate(
                messages,
                system_prompt=_SYNTHESIS_SYSTEM,
                max_tokens=512,
                temperature=0.2,
            )
            data = self._parse_json_dict(raw)
            task.findings = self._str_list(data.get("findings"))
            task.contradictions = self._str_list(data.get("contradictions"))
            task.summary = str(data.get("summary") or "")
            task.note_title = str(data.get("note_title") or task.question[:60])
        except Exception as e:
            logger.error("Synthesis LLM call failed: {}", e)
            # Fallback: use raw snippets
            task.findings = [r.snippet for r in task.raw_results[:5] if r.snippet]
            task.summary = f"Research on: {task.question}"
            task.note_title = task.question[:60]

    # ── Step 5: Report writing ────────────────────────────────────────────

    async def _write_report(self, task: ResearchTask) -> None:
        """Use the LLM to write a full structured markdown report."""
        from app.llm.engine import llm_engine

        findings_text = "\n".join(f"- {f}" for f in task.findings)
        contradictions_text = "\n".join(f"- {c}" for c in task.contradictions)
        sources_text = "\n".join(task.sources)

        user_msg = (
            f"Research question: {task.question}\n\n"
            f"Summary: {task.summary}\n\n"
            f"Key findings:\n{findings_text}\n\n"
            f"Contradictions:\n{contradictions_text or 'None identified.'}\n\n"
            f"Sources:\n{sources_text}"
        )

        messages = [{"role": "user", "content": user_msg}]

        try:
            task.report = await llm_engine.generate(
                messages,
                system_prompt=_REPORT_SYSTEM,
                max_tokens=800,
                temperature=0.3,
            )
        except Exception as e:
            logger.error("Report writing failed: {}", e)
            # Fallback: construct minimal report
            task.report = self._minimal_report(task)

    @staticmethod
    def _minimal_report(task: ResearchTask) -> str:
        lines = [
            f"# {task.note_title}",
            "",
            "## Summary",
            task.summary,
            "",
            "## Key Findings",
        ]
        for f in task.findings:
            lines.append(f"- {f}")
        lines += ["", "## Sources"]
        for s in task.sources:
            lines.append(s)
        return "\n".join(lines)

    # ── Step 6: Vault save ────────────────────────────────────────────────

    async def _save_to_vault(self, task: ResearchTask) -> None:
        """Save the research report as an Obsidian note."""
        try:
            from app.tools.obsidian import obsidian_create_note

            result = await obsidian_create_note(
                title=task.note_title,
                content=task.report,
                folder="Research",
                tags=["research", "auto-generated"],
            )
            task.saved_to_vault = True
            logger.info("ResearchAgent: saved to vault — {}", result)
        except Exception as e:
            logger.warning("Failed to save research to vault: {}", e)

    # ── Response builder ──────────────────────────────────────────────────

    @staticmethod
    def _build_response(task: ResearchTask) -> str:
        """
        Build a concise spoken response with the key takeaway.
        The full report is in the vault; this is for TTS / terminal.
        """
        parts = []

        if task.summary:
            parts.append(task.summary)

        if task.findings:
            top = task.findings[:3]
            bullet_str = "; ".join(top)
            parts.append(f"Key findings: {bullet_str}.")

        if task.contradictions:
            detail = (
                " — see the full report in your vault for details."
                if task.saved_to_vault else "."
            )
            parts.append(
                f"Note: I found {len(task.contradictions)} conflicting claim(s) "
                f"across sources{detail}"
            )

        # Only claim the save when it actually succeeded — _save_to_vault
        # swallows write failures, and it is skipped entirely when
        # save_to_vault=False.
        if task.saved_to_vault:
            parts.append(
                f"I've saved a full report to your vault under '{task.note_title}'."
            )

        if parts:
            return " ".join(parts)
        return (
            "Research complete. See your vault for the report."
            if task.saved_to_vault else "Research complete, but I couldn't extract any findings."
        )

    # ── JSON parsing helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_json_list(raw: str) -> list[str]:
        """Extract a JSON array from LLM output."""
        import json

        clean = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\[.*?\]", clean, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return ResearchAgent._str_list(data)
            except json.JSONDecodeError:
                pass
        # Line-by-line fallback: extract quoted strings
        return re.findall(r'"([^"]{5,})"', clean)

    @staticmethod
    def _str_list(value) -> list[str]:
        """Coerce an LLM-supplied field to a list of non-empty strings."""
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(x).strip() for x in value if str(x).strip()]

    @staticmethod
    def _parse_json_dict(raw: str) -> dict:
        """Extract a JSON object from LLM output."""
        from app.llm.engine import parse_json_object
        return parse_json_object(raw) or {}


# ─── Singleton ────────────────────────────────────────────────────────────────

research_agent = ResearchAgent()
