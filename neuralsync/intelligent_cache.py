#!/usr/bin/env python3
"""
Intelligent Caching System for NeuralSync v2
Provides high-performance, TTL-based caching with smart invalidation and preloading
"""

import asyncio
import time
import threading
import hashlib
import json
import logging
import pickle
import lz4.frame  # For compression
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from pathlib import Path
import mmap
import os
from contextlib import asynccontextmanager
from .utils import now_ms

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""
    key: str
    data: Any
    created_at: int
    expires_at: int
    hit_count: int = 0
    last_accessed: int = 0
    compressed: bool = False
    size_bytes: int = 0
    priority_score: float = 1.0
    
    def is_expired(self, current_time: Optional[int] = None) -> bool:
        """Check if entry has expired"""
        if current_time is None:
            current_time = now_ms()
        return current_time >= self.expires_at
    
    def touch(self):
        """Update access metadata"""
        self.hit_count += 1
        self.last_accessed = now_ms()
        # Boost priority based on usage frequency
        self.priority_score = min(10.0, self.priority_score * 1.1)

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_requests: int = 0
    avg_response_time_ms: float = 0.0
    memory_usage_bytes: int = 0
    cache_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        return self.hits / max(1, self.total_requests)
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return self.misses / max(1, self.total_requests)

class IntelligentCache:
    """High-performance cache with intelligent eviction and preloading"""
    
    def __init__(self, 
                 max_size: int = 10000,
                 default_ttl_ms: int = 300000,  # 5 minutes
                 max_memory_mb: int = 256,
                 enable_compression: bool = True,
                 enable_persistence: bool = True,
                 cache_file: Optional[str] = None):
        
        self.max_size = max_size
        self.default_ttl_ms = default_ttl_ms
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_compression = enable_compression
        self.enable_persistence = enable_persistence
        
        # Cache storage using OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_usage = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats()
        
        # Persistence
        self.cache_file = cache_file or "/tmp/neuralsync_cache.db"
        
        # Background tasks
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = False
        
        # Preloader registry
        self._preloaders: Dict[str, Callable] = {}
        
        # Smart invalidation patterns
        self._invalidation_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Load persisted cache
        if self.enable_persistence:
            self._load_cache()
            
        # Start background cleanup
        self._start_cleanup_thread()
        
        logger.info(f"Initialized intelligent cache: max_size={max_size}, "
                   f"max_memory_mb={max_memory_mb}, ttl={default_ttl_ms}ms")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with async support"""
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                self.stats.total_requests += 1
                
                if key in self._cache:
                    entry = self._cache[key]
                    
                    # Check expiration
                    if entry.is_expired():
                        self._remove_entry(key)
                        self.stats.expirations += 1
                        self.stats.misses += 1
                        
                        # Try preloader as fallback
                        if key in self._preloaders:
                            try:
                                data = await self._run_preloader(key)
                                if data is not None:
                                    await self.set(key, data)
                                    return data
                            except Exception as e:
                                logger.warning(f"Preloader failed for key {key}: {e}")
                        
                        return default
                    
                    # Update access metadata
                    entry.touch()
                    
                    # Move to end for LRU
                    self._cache.move_to_end(key)
                    
                    # Decompress if needed
                    data = entry.data
                    if entry.compressed:
                        try:
                            data = pickle.loads(lz4.frame.decompress(data))
                        except Exception as e:
                            logger.error(f"Cache decompression failed for {key}: {e}")
                            self._remove_entry(key)
                            return default
                    
                    self.stats.hits += 1
                    return data
                
                else:
                    self.stats.misses += 1
                    
                    # Try preloader
                    if key in self._preloaders:
                        try:
                            data = await self._run_preloader(key)
                            if data is not None:
                                await self.set(key, data)
                                return data
                        except Exception as e:
                            logger.warning(f"Preloader failed for key {key}: {e}")
                    
                    return default
                    
        finally:
            # Update response time statistics
            response_time_ms = (time.perf_counter() - start_time) * 1000
            self.stats.avg_response_time_ms = (
                0.9 * self.stats.avg_response_time_ms + 0.1 * response_time_ms
            )
    
    async def set(self, key: str, value: Any, ttl_ms: Optional[int] = None) -> bool:
        """Set item in cache with optional TTL"""
        
        if ttl_ms is None:
            ttl_ms = self.default_ttl_ms
            
        expires_at = now_ms() + ttl_ms
        
        # Serialize and optionally compress data
        data = value
        compressed = False
        
        if self.enable_compression:
            try:
                # Only compress if data is significant size
                serialized = pickle.dumps(value)
                if len(serialized) > 1024:  # 1KB threshold
                    compressed_data = lz4.frame.compress(serialized)
                    if len(compressed_data) < len(serialized):
                        data = compressed_data
                        compressed = True
                    else:
                        data = serialized
                else:
                    data = serialized
            except Exception as e:
                logger.warning(f"Cache compression failed for {key}: {e}")
                # Fall back to uncompressed
                data = value
        
        # Calculate size
        size_bytes = self._estimate_size(data)
        
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Check memory limits and evict if necessary
            while (self._memory_usage + size_bytes > self.max_memory_bytes or 
                   len(self._cache) >= self.max_size):
                if not self._evict_lru():
                    break  # Can't evict anything else
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=now_ms(),
                expires_at=expires_at,
                compressed=compressed,
                size_bytes=size_bytes,
                priority_score=1.0
            )
            
            # Store in cache
            self._cache[key] = entry
            self._memory_usage += size_bytes
            
            # Update statistics
            self.stats.cache_size = len(self._cache)
            self.stats.memory_usage_bytes = self._memory_usage
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    async def clear(self):
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0
            self.stats.cache_size = 0
            self.stats.memory_usage_bytes = 0
    
    def _remove_entry(self, key: str):
        """Remove entry and update memory usage"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._memory_usage -= entry.size_bytes
            self.stats.cache_size = len(self._cache)
            self.stats.memory_usage_bytes = self._memory_usage
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item with priority consideration"""
        if not self._cache:
            return False
            
        # Find candidate for eviction (balance LRU with priority)
        candidates = []
        current_time = now_ms()
        
        for key, entry in self._cache.items():
            # Calculate eviction score (lower = more likely to evict)
            age_factor = (current_time - entry.last_accessed) / 3600000  # Hours since last access
            priority_factor = 1.0 / entry.priority_score
            frequency_factor = 1.0 / max(1, entry.hit_count)
            
            eviction_score = age_factor * priority_factor * frequency_factor
            candidates.append((eviction_score, key))
        
        # Sort by eviction score (highest first - most likely to evict)
        candidates.sort(reverse=True)
        
        if candidates:
            _, key_to_evict = candidates[0]
            self._remove_entry(key_to_evict)
            self.stats.evictions += 1
            return True
            
        return False
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data"""
        try:
            if isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in data.items())
            else:
                # Fallback: serialize to estimate
                return len(pickle.dumps(data))
        except:
            return 1024  # Default estimate
    
    def register_preloader(self, key_pattern: str, preloader_func: Callable):
        """Register a preloader function for cache misses"""
        self._preloaders[key_pattern] = preloader_func
        logger.info(f"Registered preloader for pattern: {key_pattern}")
    
    async def _run_preloader(self, key: str) -> Any:
        """Run preloader for given key"""
        for pattern, preloader in self._preloaders.items():
            if pattern in key or key.startswith(pattern):
                try:
                    if asyncio.iscoroutinefunction(preloader):
                        return await preloader(key)
                    else:
                        return preloader(key)
                except Exception as e:
                    logger.error(f"Preloader error for {key}: {e}")
        return None
    
    def add_invalidation_pattern(self, pattern: str, dependent_keys: List[str]):
        """Add smart invalidation pattern"""
        self._invalidation_patterns[pattern].extend(dependent_keys)
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        keys_to_remove = []
        
        with self._lock:
            # Direct pattern matches
            if pattern in self._invalidation_patterns:
                keys_to_remove.extend(self._invalidation_patterns[pattern])
            
            # Key pattern matches
            for key in self._cache.keys():
                if pattern in key:
                    keys_to_remove.append(key)
            
            # Remove matched keys
            for key in keys_to_remove:
                if key in self._cache:
                    self._remove_entry(key)
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while not self._stop_cleanup:
                try:
                    self._cleanup_expired()
                    
                    # Persist cache periodically if enabled
                    if self.enable_persistence:
                        self._persist_cache()
                    
                    time.sleep(30)  # Cleanup every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        with self._lock:
            current_time = now_ms()
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired(current_time):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self.stats.expirations += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _persist_cache(self):
        """Persist cache to disk"""
        try:
            cache_data = {}
            with self._lock:
                for key, entry in self._cache.items():
                    if not entry.is_expired():
                        cache_data[key] = {
                            'data': entry.data,
                            'expires_at': entry.expires_at,
                            'compressed': entry.compressed,
                            'priority_score': entry.priority_score
                        }
            
            # Write to temporary file first, then rename for atomicity
            temp_file = self.cache_file + '.tmp'
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            os.rename(temp_file, self.cache_file)
            logger.debug(f"Persisted {len(cache_data)} cache entries")
            
        except Exception as e:
            logger.error(f"Cache persistence failed: {e}")
    
    def _load_cache(self):
        """Load persisted cache from disk"""
        try:
            if not os.path.exists(self.cache_file):
                return
            
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            current_time = now_ms()
            loaded_count = 0
            
            with self._lock:
                for key, entry_data in cache_data.items():
                    # Skip expired entries
                    if entry_data['expires_at'] <= current_time:
                        continue
                    
                    entry = CacheEntry(
                        key=key,
                        data=entry_data['data'],
                        created_at=current_time,  # Reset creation time
                        expires_at=entry_data['expires_at'],
                        compressed=entry_data.get('compressed', False),
                        size_bytes=self._estimate_size(entry_data['data']),
                        priority_score=entry_data.get('priority_score', 1.0)
                    )
                    
                    self._cache[key] = entry
                    self._memory_usage += entry.size_bytes
                    loaded_count += 1
            
            self.stats.cache_size = len(self._cache)
            self.stats.memory_usage_bytes = self._memory_usage
            
            logger.info(f"Loaded {loaded_count} cache entries from disk")
            
        except Exception as e:
            logger.error(f"Cache loading failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'hit_rate': self.stats.hit_rate,
            'miss_rate': self.stats.miss_rate,
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'evictions': self.stats.evictions,
            'expirations': self.stats.expirations,
            'total_requests': self.stats.total_requests,
            'avg_response_time_ms': self.stats.avg_response_time_ms,
            'cache_size': self.stats.cache_size,
            'memory_usage_mb': self.stats.memory_usage_bytes / (1024 * 1024),
            'memory_utilization': self.stats.memory_usage_bytes / self.max_memory_bytes,
            'preloaders_registered': len(self._preloaders),
            'invalidation_patterns': len(self._invalidation_patterns)
        }
    
    def optimize_performance(self):
        """Perform cache optimization"""
        with self._lock:
            # Remove expired entries
            self._cleanup_expired()
            
            # Rebalance priority scores
            for entry in self._cache.values():
                # Decay priority over time to prevent permanent high priority
                entry.priority_score *= 0.99
                entry.priority_score = max(0.1, entry.priority_score)
            
            logger.info("Cache optimization completed")
    
    def close(self):
        """Clean shutdown"""
        self._stop_cleanup = True
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        if self.enable_persistence:
            self._persist_cache()
        
        logger.info("Cache shutdown completed")


class NeuralSyncCache:
    """Specialized cache for NeuralSync with context-aware optimization"""
    
    def __init__(self, cache_dir: str = "/tmp/neuralsync"):
        os.makedirs(cache_dir, exist_ok=True)
        
        # Separate caches for different data types
        self.persona_cache = IntelligentCache(
            max_size=100,
            default_ttl_ms=600000,  # 10 minutes for persona
            max_memory_mb=16,
            cache_file=os.path.join(cache_dir, "persona.cache")
        )
        
        self.memory_cache = IntelligentCache(
            max_size=5000,
            default_ttl_ms=300000,  # 5 minutes for memories  
            max_memory_mb=128,
            cache_file=os.path.join(cache_dir, "memory.cache")
        )
        
        self.context_cache = IntelligentCache(
            max_size=1000,
            default_ttl_ms=180000,  # 3 minutes for assembled context
            max_memory_mb=64,
            cache_file=os.path.join(cache_dir, "context.cache")
        )
        
        # Register cross-cache invalidation patterns
        self._setup_invalidation_patterns()
        
        logger.info("NeuralSync cache system initialized")
    
    def _setup_invalidation_patterns(self):
        """Setup intelligent invalidation patterns between caches"""
        
        # When persona changes, invalidate context cache
        self.persona_cache.add_invalidation_pattern("persona", ["context:*"])
        
        # When memories change, invalidate related context
        self.memory_cache.add_invalidation_pattern("recall:", ["context:*"])
    
    async def get_persona(self, key: str = "current") -> Optional[str]:
        """Get cached persona"""
        return await self.persona_cache.get(f"persona:{key}")
    
    async def set_persona(self, persona: str, key: str = "current", ttl_ms: Optional[int] = None) -> bool:
        """Cache persona with cross-cache invalidation"""
        success = await self.persona_cache.set(f"persona:{key}", persona, ttl_ms)
        if success:
            # Invalidate context cache since persona changed
            await self.context_cache.invalidate_pattern("context:")
        return success
    
    async def get_memories(self, query_hash: str) -> Optional[List[Dict]]:
        """Get cached memory recall results"""
        return await self.memory_cache.get(f"recall:{query_hash}")
    
    async def set_memories(self, query_hash: str, memories: List[Dict], ttl_ms: Optional[int] = None) -> bool:
        """Cache memory recall results"""
        return await self.memory_cache.set(f"recall:{query_hash}", memories, ttl_ms)
    
    async def get_context(self, context_hash: str) -> Optional[str]:
        """Get cached assembled context"""
        return await self.context_cache.get(f"context:{context_hash}")
    
    async def set_context(self, context_hash: str, context: str, ttl_ms: Optional[int] = None) -> bool:
        """Cache assembled context"""
        return await self.context_cache.set(f"context:{context_hash}", context, ttl_ms)
    
    def hash_query(self, query: str, tool: Optional[str] = None, scope: str = "any", top_k: int = 3) -> str:
        """Generate hash for query parameters"""
        query_params = f"{query}|{tool}|{scope}|{top_k}"
        return hashlib.md5(query_params.encode()).hexdigest()
    
    def hash_context(self, query: str, tool: Optional[str] = None) -> str:
        """Generate hash for context parameters"""
        context_params = f"{query}|{tool}"
        return hashlib.md5(context_params.encode()).hexdigest()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        return {
            'persona_cache': self.persona_cache.get_stats(),
            'memory_cache': self.memory_cache.get_stats(),
            'context_cache': self.context_cache.get_stats()
        }
    
    def optimize_all(self):
        """Optimize all caches"""
        self.persona_cache.optimize_performance()
        self.memory_cache.optimize_performance()
        self.context_cache.optimize_performance()
    
    def close(self):
        """Shutdown all caches"""
        self.persona_cache.close()
        self.memory_cache.close()
        self.context_cache.close()


# Global cache instance
_global_cache: Optional[NeuralSyncCache] = None

def get_neuralsync_cache() -> NeuralSyncCache:
    """Get global NeuralSync cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = NeuralSyncCache()
    return _global_cache

async def cached_persona_fetch(endpoint: str, headers: Dict) -> Optional[str]:
    """Cached persona fetching with fallback"""
    cache = get_neuralsync_cache()
    
    # Try cache first
    cached_persona = await cache.get_persona()
    if cached_persona is not None:
        return cached_persona
    
    # Fallback to network (this would be implemented by caller)
    return None

async def cached_memory_recall(query_hash: str, query_func: Callable) -> Optional[List[Dict]]:
    """Cached memory recall with fallback"""
    cache = get_neuralsync_cache()
    
    # Try cache first
    cached_memories = await cache.get_memories(query_hash)
    if cached_memories is not None:
        return cached_memories
    
    # Fallback to actual recall (implemented by caller)
    return None