"""
Web search tool — disabled by default.
Only activated when user explicitly requests it.
"""
from loguru import logger

from app.config import settings


async def tool_web_search(query: str) -> str:
    if not settings.web_search_enabled:
        return (
            "Web search is currently disabled. "
            "To enable it, say 'enable web search' or set WEB_SEARCH_ENABLED=true in your .env file."
        )

    try:
        from duckduckgo_search import DDGS

        logger.info("web_search: {}", query)
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=settings.web_search_max_results):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('href', 'N/A')}\n"
                    f"Snippet: {r.get('body', 'N/A')}\n"
                )

        if not results:
            return "No results found."

        return "\n---\n".join(results)

    except ImportError:
        return "Web search library (duckduckgo-search) is not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        logger.error("web_search error: {}", e)
        return f"Search failed: {e}"


def enable_web_search() -> str:
    settings.web_search_enabled = True
    logger.info("Web search enabled by user request")
    return "Web search has been enabled for this session."


def disable_web_search() -> str:
    settings.web_search_enabled = False
    logger.info("Web search disabled")
    return "Web search has been disabled."
