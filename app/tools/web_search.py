"""
Web search tool — disabled by default.
Includes retry with backoff for DuckDuckGo rate limits.
"""
import asyncio
from loguru import logger
from app.config import settings


async def tool_web_search(query: str) -> str:
    if not settings.web_search_enabled:
        return (
            "Web search is currently disabled. "
            "Say 'enable web search' or set WEB_SEARCH_ENABLED=true in .env."
        )
    return await _search_with_retry(query)


async def _search_with_retry(query: str, max_retries: int = 2) -> str:
    """DuckDuckGo search with exponential backoff on rate limit / timeout."""
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            from duckduckgo_search import DDGS
            logger.info("web_search attempt {}/{}: {}", attempt + 1, max_retries + 1, query)

            # Run blocking DDGS in executor so it doesn't block the event loop
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, _do_ddg_search, query)

            if results:
                return "\n---\n".join(results)
            return "No results found."

        except ImportError:
            return "Web search library not installed. Run: pip install duckduckgo-search"

        except Exception as exc:
            last_error  = str(exc)
            is_rate     = "202" in last_error or "ratelimit" in last_error.lower()
            is_timeout  = "timeout" in last_error.lower() or "TimeoutError" in last_error

            if (is_rate or is_timeout) and attempt < max_retries:
                wait = 4 * (attempt + 1)   # 4s, 8s
                logger.warning(
                    "web_search {}: retrying in {}s",
                    "rate-limited" if is_rate else "timed out",
                    wait,
                )
                await asyncio.sleep(wait)
                continue

            logger.error("web_search failed after {} attempts: {}", attempt + 1, exc)
            return f"Search failed: {exc}"

    return f"Search unavailable after {max_retries + 1} attempts: {last_error}"


def _do_ddg_search(query: str) -> list[str]:
    """Blocking DuckDuckGo search — must be run in executor."""
    from duckduckgo_search import DDGS
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=settings.web_search_max_results):
            results.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('href', 'N/A')}\n"
                f"Snippet: {r.get('body', 'N/A')}\n"
            )
    return results


def enable_web_search() -> str:
    settings.web_search_enabled = True
    logger.info("Web search enabled")
    return "Web search enabled for this session."


def disable_web_search() -> str:
    settings.web_search_enabled = False
    logger.info("Web search disabled")
    return "Web search disabled."