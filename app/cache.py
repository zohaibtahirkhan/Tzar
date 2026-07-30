"""
Response caching system with TTL support.

Caches LLM responses for repeated queries to reduce latency and computational cost.
Uses in-memory LRU cache with configurable TTL and size limits.

Features:
- TTL-based expiration (default: 1 hour)
- LRU eviction when cache is full
- Query normalization (lowercase, strip whitespace)
- Cache statistics tracking
- Thread-safe operations
"""
import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from app.config import settings


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    response: str
    created_at: float
    hits: int = 0
    last_accessed: float = 0.0


class ResponseCache:
    """
    TTL-based LRU cache for LLM responses.
    
    Cache keys are normalized query hashes.
    Entries expire after TTL seconds or when cache is full (LRU eviction).
    """
    
    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 3600,  # 1 hour default
        enabled: bool = True,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(
            "Response cache initialized: max_size={}, ttl={}s, enabled={}",
            max_size, ttl_seconds, enabled
        )
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for cache key generation.
        
        - Lowercase
        - Strip whitespace
        - Remove multiple spaces
        """
        return " ".join(query.lower().strip().split())
    
    def _hash_query(self, query: str) -> str:
        """Generate cache key from normalized query."""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - entry.created_at) > self.ttl_seconds
    
    async def get(self, query: str) -> Optional[str]:
        """
        Retrieve cached response for query.
        
        Returns None if:
        - Cache disabled
        - Query not in cache
        - Entry has expired
        """
        if not self.enabled:
            return None
        
        async with self._lock:
            key = self._hash_query(query)
            
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if self._is_expired(entry):
                del self._cache[key]
                self._misses += 1
                logger.debug("Cache entry expired: {}", query[:60])
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            
            # Update stats
            entry.hits += 1
            entry.last_accessed = time.time()
            self._hits += 1
            
            logger.debug("Cache HIT: {} (hits: {})", query[:60], entry.hits)
            return entry.response
    
    async def set(self, query: str, response: str) -> None:
        """
        Store response in cache.
        
        If cache is full, evicts least recently used entry.
        """
        if not self.enabled or not response:
            return
        
        async with self._lock:
            key = self._hash_query(query)
            
            # Check if we need to evict
            if key not in self._cache and len(self._cache) >= self.max_size:
                # Remove oldest entry (LRU)
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug("Cache eviction (LRU): size={}", len(self._cache))
            
            # Store new entry
            self._cache[key] = CacheEntry(
                response=response,
                created_at=time.time(),
                last_accessed=time.time(),
            )
            
            # Move to end
            self._cache.move_to_end(key)
            
            logger.debug("Cache SET: {} (size: {})", query[:60], len(self._cache))
    
    async def invalidate(self, query: str) -> bool:
        """Remove specific query from cache. Returns True if entry existed."""
        if not self.enabled:
            return False
        
        async with self._lock:
            key = self._hash_query(query)
            if key in self._cache:
                del self._cache[key]
                logger.debug("Cache invalidated: {}", query[:60])
                return True
            return False
    
    async def clear(self) -> int:
        """Clear entire cache. Returns number of entries removed."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cache cleared: {} entries removed", count)
            return count
    
    async def prune_expired(self) -> int:
        """Remove all expired entries. Returns number of entries removed."""
        if not self.enabled:
            return 0
        
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info("Pruned {} expired cache entries", len(expired_keys))
            
            return len(expired_keys)
    
    def stats(self) -> dict:
        """Return cache statistics."""
        hit_rate = (
            self._hits / (self._hits + self._misses)
            if (self._hits + self._misses) > 0
            else 0.0
        )
        
        return {
            "enabled": self.enabled,
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
        }
    
    def enable(self) -> None:
        """Enable caching."""
        self.enabled = True
        logger.info("Response cache enabled")
    
    def disable(self) -> None:
        """Disable caching (does not clear existing entries)."""
        self.enabled = False
        logger.info("Response cache disabled")


# Singleton instance
response_cache = ResponseCache(
    max_size=100,
    ttl_seconds=3600,  # 1 hour
    enabled=True,
)
