"""
Integration tests for voice assistant critical components.

Tests the interaction between multiple subsystems without requiring
actual LLM/STT/TTS models or hardware.

Coverage:
- Cache integration with pipeline
- Path traversal protection
- CORS configuration
- Tool execution with timeout
- Coordinator flow
"""
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.cache import ResponseCache
from app.tools.filesystem import _safe_path, ALLOWED_ROOT
from app.agents.coordinator import ToolExecutor, ToolStep, StepResult
from app.pipeline import _truncate_tool_result, _clean_llm_response


# ─── Cache Integration Tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_basic_flow():
    """Test cache set/get with TTL."""
    cache = ResponseCache(max_size=10, ttl_seconds=1, enabled=True)
    
    # Set and retrieve
    await cache.set("hello", "world")
    result = await cache.get("hello")
    assert result == "world"
    
    # Cache hit
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0


@pytest.mark.asyncio
async def test_cache_normalization():
    """Test query normalization for cache keys."""
    cache = ResponseCache(max_size=10, ttl_seconds=60, enabled=True)
    
    # Different whitespace, same normalized form
    await cache.set("  hello   world  ", "response1")
    result = await cache.get("hello world")
    assert result == "response1"
    
    # Case insensitive
    result = await cache.get("HELLO WORLD")
    assert result == "response1"


@pytest.mark.asyncio
async def test_cache_lru_eviction():
    """Test LRU eviction when cache is full."""
    cache = ResponseCache(max_size=3, ttl_seconds=60, enabled=True)
    
    await cache.set("key1", "val1")
    await cache.set("key2", "val2")
    await cache.set("key3", "val3")
    
    # Cache is full, adding new entry should evict oldest
    await cache.set("key4", "val4")
    
    # key1 should be evicted
    result = await cache.get("key1")
    assert result is None
    
    # Others should exist
    assert await cache.get("key2") == "val2"
    assert await cache.get("key4") == "val4"
    
    stats = cache.stats()
    assert stats["evictions"] == 1


@pytest.mark.asyncio
async def test_cache_ttl_expiration():
    """Test cache entry expiration after TTL."""
    cache = ResponseCache(max_size=10, ttl_seconds=0.1, enabled=True)
    
    await cache.set("temp", "value")
    
    # Should exist immediately
    result = await cache.get("temp")
    assert result == "value"
    
    # Wait for expiration
    await asyncio.sleep(0.15)
    
    # Should be expired
    result = await cache.get("temp")
    assert result is None


@pytest.mark.asyncio
async def test_cache_prune():
    """Test manual pruning of expired entries."""
    cache = ResponseCache(max_size=10, ttl_seconds=0.1, enabled=True)
    
    await cache.set("key1", "val1")
    await cache.set("key2", "val2")
    
    # Wait for expiration
    await asyncio.sleep(0.15)
    
    # Prune expired entries
    count = await cache.prune_expired()
    assert count == 2
    
    # Cache should be empty
    stats = cache.stats()
    assert stats["size"] == 0


# ─── Path Traversal Protection Tests ──────────────────────────────────────────

def test_safe_path_normal():
    """Test safe_path with valid relative paths."""
    result = _safe_path("test.txt")
    assert result == ALLOWED_ROOT / "test.txt"
    
    result = _safe_path("subdir/file.md")
    assert result == ALLOWED_ROOT / "subdir" / "file.md"


def test_safe_path_blocks_traversal():
    """Test that path traversal attempts are blocked."""
    with pytest.raises(PermissionError, match="Path traversal denied"):
        _safe_path("../etc/passwd")
    
    with pytest.raises(PermissionError, match="Path traversal denied"):
        _safe_path("../../root/.ssh/id_rsa")
    
    with pytest.raises(PermissionError, match="Path traversal denied"):
        _safe_path("subdir/../../etc/hosts")


def test_safe_path_blocks_absolute():
    """Test that absolute paths outside workspace are blocked."""
    with pytest.raises(PermissionError, match="Path traversal denied"):
        _safe_path("/etc/passwd")
    
    with pytest.raises(PermissionError, match="Path traversal denied"):
        _safe_path("/root/.bashrc")


def test_safe_path_null_byte_filtering():
    """Test that null bytes are filtered from paths."""
    # Null bytes should be stripped
    result = _safe_path("test\x00.txt")
    assert "\x00" not in str(result)
    assert result == ALLOWED_ROOT / "test.txt"


def test_safe_path_control_char_filtering():
    """Test that control characters are filtered."""
    result = _safe_path("test\n\r\t.txt")
    # Control chars should be stripped
    assert "\n" not in str(result)
    assert "\r" not in str(result)


# ─── Tool Execution Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_executor_basic():
    """Test basic tool execution."""
    executor = ToolExecutor()
    
    step = ToolStep(
        description="List directory",
        tool="list_directory",
        params={"path": "."},
    )
    
    result = await executor.run(step)
    assert result.tool == "list_directory"
    assert result.status in ("ok", "error")


@pytest.mark.asyncio
async def test_tool_executor_timeout():
    """Test tool execution timeout."""
    executor = ToolExecutor()
    
    # Mock a tool that takes too long
    with patch('app.tools.router.ToolRouter.dispatch', new_callable=AsyncMock) as mock_dispatch:
        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(35)  # Longer than timeout
            return {"status": "ok", "result": "done"}
        
        mock_dispatch.side_effect = slow_tool
        
        step = ToolStep(
            description="Slow tool",
            tool="slow_tool",
            params={},
        )
        
        result = await executor.run(step)
        assert result.status == "error"
        assert "timed out" in result.result.lower()


# ─── Pipeline Helper Tests ────────────────────────────────────────────────────

def test_truncate_tool_result():
    """Test tool result truncation logic."""
    short_text = "Short result"
    result = _truncate_tool_result(short_text, "test_tool")
    assert result == short_text
    
    # Long text should be truncated
    long_text = "x" * 2000
    result = _truncate_tool_result(long_text, "test_tool")
    assert len(result) < len(long_text)
    assert "truncated" in result


def test_clean_llm_response():
    """Test LLM response cleaning."""
    # Remove assistant prefix
    dirty = "Assistant: Hello world"
    clean = _clean_llm_response(dirty)
    assert clean == "Hello world"
    
    # Remove repeated prefix
    dirty = "Assistant: Assistant: Hello"
    clean = _clean_llm_response(dirty)
    assert clean == "Hello"
    
    # Remove fake conversation
    dirty = "Answer is 42\nUser says 'what'"
    clean = _clean_llm_response(dirty)
    assert "User says" not in clean


def test_clean_llm_response_safe():
    """Test that clean_llm_response handles None/empty safely."""
    assert _clean_llm_response(None) == ""
    assert _clean_llm_response("") == ""
    assert _clean_llm_response("   ") == ""


# ─── Coordinator Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_coordinator_tool_step():
    """Test coordinator ToolStep execution."""
    from app.agents.coordinator import tool_executor
    
    step = ToolStep(
        description="List workspace",
        tool="list_directory",
        params={"path": "."},
        required=True,
    )
    
    result = await tool_executor.run(step)
    assert isinstance(result, StepResult)
    assert result.tool == "list_directory"
    assert result.elapsed_ms > 0


# ─── API CORS Tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cors_allowed_origins():
    """Test that CORS is configured with explicit origins."""
    from app.router import ALLOWED_ORIGINS
    
    # Should not contain wildcard
    assert "*" not in ALLOWED_ORIGINS
    
    # Should contain localhost
    assert any("localhost" in origin for origin in ALLOWED_ORIGINS)
    assert any("127.0.0.1" in origin for origin in ALLOWED_ORIGINS)


# ─── Cache Integration with Pipeline ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_integration_disabled():
    """Test pipeline behavior when cache is disabled."""
    cache = ResponseCache(max_size=10, ttl_seconds=60, enabled=False)
    
    # Set should not store
    await cache.set("test", "value")
    
    # Get should return None
    result = await cache.get("test")
    assert result is None


@pytest.mark.asyncio
async def test_cache_stats_tracking():
    """Test cache statistics tracking."""
    cache = ResponseCache(max_size=10, ttl_seconds=60, enabled=True)
    
    # Generate some activity
    await cache.set("q1", "a1")
    await cache.get("q1")  # hit
    await cache.get("q2")  # miss
    await cache.get("q1")  # hit
    
    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 2 / 3


# ─── Constants Tests ──────────────────────────────────────────────────────────

def test_pipeline_constants_defined():
    """Test that pipeline constants are properly defined."""
    from app.pipeline import (
        TOOL_RESULT_MAX_CHARS,
        MAX_TOOL_ITERATIONS,
        TOOL_EXECUTION_TIMEOUT_SECONDS,
        COMPLEX_TASK_TOOL_THRESHOLD,
    )
    
    assert TOOL_RESULT_MAX_CHARS == 1200
    assert MAX_TOOL_ITERATIONS == 4
    assert TOOL_EXECUTION_TIMEOUT_SECONDS == 30.0
    assert COMPLEX_TASK_TOOL_THRESHOLD == 3


def test_coordinator_constants_defined():
    """Test that coordinator constants are properly defined."""
    from app.agents.coordinator import (
        MAX_STEP_RETRIES,
        TOOL_EXECUTION_TIMEOUT_SECONDS,
        PLAN_GENERATION_TIMEOUT_SECONDS,
    )
    
    assert MAX_STEP_RETRIES == 1
    assert TOOL_EXECUTION_TIMEOUT_SECONDS == 30.0
    assert PLAN_GENERATION_TIMEOUT_SECONDS == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
