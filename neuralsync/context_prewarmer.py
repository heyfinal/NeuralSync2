#!/usr/bin/env python3
"""
Context Pre-warming Service for NeuralSync v2
Background daemon that intelligently pre-loads and caches context data
"""

import asyncio
import threading
import time
import logging
import os
import json
import hashlib
import psutil
from typing import Dict, Any, Optional, List, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path

from .intelligent_cache import get_neuralsync_cache
from .async_network import get_network_client
from .utils import now_ms

logger = logging.getLogger(__name__)

@dataclass
class PrewarmRequest:
    """Request for pre-warming context data"""
    tool: Optional[str]
    query: str
    priority: float
    timestamp: int
    frequency: int = 1
    last_accessed: int = 0
    
    def get_cache_key(self) -> str:
        """Generate cache key for this request"""
        return hashlib.md5(f"{self.tool}:{self.query}".encode()).hexdigest()

@dataclass 
class PrewarmStats:
    """Pre-warming service statistics"""
    total_prewarms: int = 0
    successful_prewarms: int = 0
    failed_prewarms: int = 0
    cache_hits_served: int = 0
    avg_prewarm_time_ms: float = 0.0
    patterns_learned: int = 0
    active_patterns: int = 0

class UsagePatternLearner:
    """Learn usage patterns from CLI invocations"""
    
    def __init__(self, max_patterns: int = 1000):
        self.max_patterns = max_patterns
        self.patterns: Dict[str, PrewarmRequest] = {}
        self.access_history: deque = deque(maxlen=10000)
        self.tool_frequency: defaultdict = defaultdict(int)
        self.query_sequences: deque = deque(maxlen=1000)
        
        # Pattern learning thresholds
        self.min_frequency_threshold = 3
        self.recency_weight = 0.7
        self.frequency_weight = 0.3
        
    def record_access(self, tool: Optional[str], query: str, context_size: int = 0):
        """Record a CLI access pattern"""
        
        timestamp = now_ms()
        access_key = f"{tool}:{query}"
        
        # Record in history
        self.access_history.append({
            'tool': tool,
            'query': query,
            'timestamp': timestamp,
            'context_size': context_size,
            'key': access_key
        })
        
        # Update tool frequency
        self.tool_frequency[tool or 'none'] += 1
        
        # Record query sequences for pattern detection
        if len(self.query_sequences) > 0:
            last_query = self.query_sequences[-1]
            if timestamp - last_query['timestamp'] < 300000:  # Within 5 minutes
                # This could indicate a pattern
                sequence_key = f"{last_query['key']}→{access_key}"
                logger.debug(f"Detected potential sequence: {sequence_key}")
        
        self.query_sequences.append({
            'key': access_key,
            'timestamp': timestamp
        })
        
        # Update or create pattern
        if access_key in self.patterns:
            pattern = self.patterns[access_key]
            pattern.frequency += 1
            pattern.last_accessed = timestamp
            # Boost priority for frequent patterns
            pattern.priority = min(10.0, pattern.priority * 1.1)
        else:
            # Create new pattern if we have room or if it's worth replacing
            if len(self.patterns) < self.max_patterns:
                self.patterns[access_key] = PrewarmRequest(
                    tool=tool,
                    query=query,
                    priority=1.0,
                    timestamp=timestamp,
                    frequency=1,
                    last_accessed=timestamp
                )
            else:
                # Replace least valuable pattern
                self._replace_least_valuable_pattern(tool, query, timestamp)
    
    def _replace_least_valuable_pattern(self, tool: Optional[str], query: str, timestamp: int):
        """Replace the least valuable pattern with a new one"""
        
        # Calculate value scores for all patterns
        current_time = now_ms()
        pattern_values = []
        
        for key, pattern in self.patterns.items():
            # Value based on frequency and recency
            age_hours = (current_time - pattern.last_accessed) / 3600000
            recency_score = 1.0 / (1.0 + age_hours)
            frequency_score = min(1.0, pattern.frequency / 10.0)
            
            value = (self.recency_weight * recency_score + 
                    self.frequency_weight * frequency_score)
            
            pattern_values.append((value, key))
        
        # Sort by value (lowest first)
        pattern_values.sort()
        
        if pattern_values and pattern_values[0][0] < 0.5:  # Low value threshold
            # Replace the least valuable pattern
            key_to_replace = pattern_values[0][1]
            del self.patterns[key_to_replace]
            
            # Add new pattern
            access_key = f"{tool}:{query}"
            self.patterns[access_key] = PrewarmRequest(
                tool=tool,
                query=query,
                priority=1.5,  # Slightly higher since it beat an existing pattern
                timestamp=timestamp,
                frequency=1,
                last_accessed=timestamp
            )
    
    def get_top_patterns(self, count: int = 20) -> List[PrewarmRequest]:
        """Get top patterns for pre-warming"""
        
        current_time = now_ms()
        scored_patterns = []
        
        for pattern in self.patterns.values():
            # Skip patterns that haven't been accessed frequently enough
            if pattern.frequency < self.min_frequency_threshold:
                continue
                
            # Skip very old patterns (more than 24 hours)
            age_hours = (current_time - pattern.last_accessed) / 3600000
            if age_hours > 24:
                continue
            
            # Calculate combined score
            recency_score = 1.0 / (1.0 + age_hours)
            frequency_score = min(1.0, pattern.frequency / 10.0)
            priority_score = min(1.0, pattern.priority / 10.0)
            
            combined_score = (0.4 * recency_score + 
                            0.4 * frequency_score + 
                            0.2 * priority_score)
            
            scored_patterns.append((combined_score, pattern))
        
        # Sort by score (highest first) and return top patterns
        scored_patterns.sort(reverse=True)
        return [pattern for _, pattern in scored_patterns[:count]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'total_patterns': len(self.patterns),
            'access_history_size': len(self.access_history),
            'top_tools': dict(sorted(self.tool_frequency.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]),
            'recent_accesses': len([a for a in self.access_history 
                                  if now_ms() - a['timestamp'] < 3600000])  # Last hour
        }

class ContextPrewarmer:
    """Background service for pre-warming context data"""
    
    def __init__(self, 
                 enable_learning: bool = True,
                 prewarm_interval: float = 60.0,  # 1 minute
                 max_concurrent_prewarms: int = 3,
                 adaptive_scheduling: bool = True):
        
        self.enable_learning = enable_learning
        self.prewarm_interval = prewarm_interval
        self.max_concurrent_prewarms = max_concurrent_prewarms
        self.adaptive_scheduling = adaptive_scheduling
        
        # Components
        self.cache = get_neuralsync_cache()
        self.network_client = get_network_client(fast_mode=False)  # Use full context for pre-warming
        
        # Pattern learning
        if enable_learning:
            self.pattern_learner = UsagePatternLearner()
        else:
            self.pattern_learner = None
        
        # Service state
        self.running = False
        self.prewarm_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = PrewarmStats()
        
        # Semaphore for controlling concurrent pre-warms
        self.prewarm_semaphore = asyncio.Semaphore(max_concurrent_prewarms)
        
        # System resource monitoring for adaptive behavior
        self.system_load_threshold = 80  # Don't pre-warm if system is busy
        
        logger.info(f"ContextPrewarmer initialized: learning={enable_learning}, "
                   f"interval={prewarm_interval}s")
    
    def start(self):
        """Start the pre-warming service"""
        if self.running:
            return
            
        self.running = True
        self.stop_event.clear()
        
        self.prewarm_thread = threading.Thread(target=self._prewarm_loop, daemon=True)
        self.prewarm_thread.start()
        
        logger.info("Context pre-warming service started")
    
    def stop(self):
        """Stop the pre-warming service"""
        self.running = False
        self.stop_event.set()
        
        if self.prewarm_thread:
            self.prewarm_thread.join(timeout=10)
        
        logger.info("Context pre-warming service stopped")
    
    def record_cli_usage(self, tool: Optional[str], query: str = "", context_size: int = 0):
        """Record CLI usage for pattern learning"""
        if self.pattern_learner and self.enable_learning:
            self.pattern_learner.record_access(tool, query, context_size)
    
    def _prewarm_loop(self):
        """Main pre-warming loop running in background thread"""
        
        async def async_prewarm_loop():
            while self.running:
                try:
                    # Check system resources
                    if self.adaptive_scheduling and self._is_system_busy():
                        logger.debug("System busy, skipping pre-warm cycle")
                        await asyncio.sleep(self.prewarm_interval * 2)
                        continue
                    
                    # Get patterns to pre-warm
                    patterns = self._get_prewarm_candidates()
                    
                    if patterns:
                        # Execute pre-warming in parallel
                        tasks = []
                        for pattern in patterns[:self.max_concurrent_prewarms]:
                            task = asyncio.create_task(self._prewarm_context(pattern))
                            tasks.append(task)
                        
                        if tasks:
                            await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Wait for next cycle
                    await asyncio.sleep(self.prewarm_interval)
                    
                except Exception as e:
                    logger.error(f"Pre-warm loop error: {e}")
                    await asyncio.sleep(self.prewarm_interval * 2)
        
        # Run async loop in thread
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(async_prewarm_loop())
        except Exception as e:
            logger.error(f"Pre-warm async loop failed: {e}")
        finally:
            try:
                loop.close()
            except:
                pass
    
    def _is_system_busy(self) -> bool:
        """Check if system is too busy for pre-warming"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.system_load_threshold:
                return True
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                return True
            
            # Check if we're in power save mode (for laptops)
            try:
                battery = psutil.sensors_battery()
                if battery and battery.percent < 20 and not battery.power_plugged:
                    return True
            except:
                pass
            
            return False
            
        except Exception as e:
            logger.debug(f"System load check failed: {e}")
            return False
    
    def _get_prewarm_candidates(self) -> List[PrewarmRequest]:
        """Get candidates for pre-warming"""
        candidates = []
        
        # Get patterns from learner
        if self.pattern_learner:
            learned_patterns = self.pattern_learner.get_top_patterns(10)
            candidates.extend(learned_patterns)
        
        # Add some static high-value patterns
        common_patterns = [
            PrewarmRequest(tool="claude-code", query="", priority=2.0, timestamp=now_ms()),
            PrewarmRequest(tool="codexcli", query="", priority=1.8, timestamp=now_ms()),
            PrewarmRequest(tool=None, query="", priority=1.5, timestamp=now_ms()),
        ]
        candidates.extend(common_patterns)
        
        # Filter out already cached items
        filtered_candidates = []
        for candidate in candidates:
            context_hash = self.cache.hash_context(candidate.query, candidate.tool)
            cache_key = f"context:{context_hash}"
            
            # Check if already cached
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                cached = loop.run_until_complete(self.cache.get_context(context_hash))
                if not cached:
                    filtered_candidates.append(candidate)
            except:
                # If we can't check cache, include it
                filtered_candidates.append(candidate)
        
        # Sort by priority
        filtered_candidates.sort(key=lambda x: x.priority, reverse=True)
        
        return filtered_candidates[:self.max_concurrent_prewarms]
    
    async def _prewarm_context(self, pattern: PrewarmRequest):
        """Pre-warm context for a specific pattern"""
        
        start_time = time.perf_counter()
        
        try:
            async with self.prewarm_semaphore:
                # Fetch context
                context = await self.network_client.get_context_fast(
                    tool=pattern.tool,
                    query=pattern.query
                )
                
                # Update statistics
                self.stats.total_prewarms += 1
                
                if context:
                    self.stats.successful_prewarms += 1
                    logger.debug(f"Pre-warmed context for {pattern.tool}:{pattern.query[:50]}")
                else:
                    self.stats.failed_prewarms += 1
                    
        except Exception as e:
            self.stats.failed_prewarms += 1
            logger.warning(f"Pre-warm failed for {pattern.tool}:{pattern.query}: {e}")
        
        # Update timing statistics
        prewarm_time_ms = (time.perf_counter() - start_time) * 1000
        self.stats.avg_prewarm_time_ms = (
            0.9 * self.stats.avg_prewarm_time_ms + 0.1 * prewarm_time_ms
        )
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive pre-warming statistics"""
        
        base_stats = {
            'service_running': self.running,
            'prewarm_interval': self.prewarm_interval,
            'max_concurrent': self.max_concurrent_prewarms,
            'adaptive_scheduling': self.adaptive_scheduling,
            'learning_enabled': self.enable_learning,
            'stats': {
                'total_prewarms': self.stats.total_prewarms,
                'successful_prewarms': self.stats.successful_prewarms,
                'failed_prewarms': self.stats.failed_prewarms,
                'success_rate': (self.stats.successful_prewarms / 
                               max(1, self.stats.total_prewarms)),
                'avg_prewarm_time_ms': self.stats.avg_prewarm_time_ms
            }
        }
        
        if self.pattern_learner:
            base_stats['pattern_learning'] = self.pattern_learner.get_stats()
        
        return base_stats
    
    def force_prewarm(self, tool: Optional[str], query: str = "") -> bool:
        """Force immediate pre-warming of specific context"""
        try:
            pattern = PrewarmRequest(
                tool=tool,
                query=query,
                priority=10.0,  # Highest priority
                timestamp=now_ms()
            )
            
            # Run in background
            async def run_prewarm():
                await self._prewarm_context(pattern)
            
            # Create task in background
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(run_prewarm())
            except RuntimeError:
                # No event loop, run in thread
                def thread_prewarm():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(run_prewarm())
                    loop.close()
                
                thread = threading.Thread(target=thread_prewarm, daemon=True)
                thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Force pre-warm failed: {e}")
            return False


class PrewarmingLogMonitor:
    """Monitor log files and system usage to trigger pre-warming"""
    
    def __init__(self, prewarmer: ContextPrewarmer):
        self.prewarmer = prewarmer
        self.log_files = [
            "/var/log/neuralsync.log",
            "/tmp/neuralsync.log",
            "~/.neuralsync/usage.log"
        ]
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start log monitoring"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Pre-warming log monitor started")
        
    def stop(self):
        """Stop log monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Pre-warming log monitor stopped")
        
    def _monitor_loop(self):
        """Monitor log files for usage patterns"""
        
        # This is a simplified implementation
        # In a real system, you'd use inotify or similar for file monitoring
        
        while self.running:
            try:
                # Monitor for tool invocations
                # This would parse actual log files for patterns
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Log monitor error: {e}")
                time.sleep(60)


# Global prewarmer instance
_global_prewarmer: Optional[ContextPrewarmer] = None

def get_context_prewarmer() -> ContextPrewarmer:
    """Get global context prewarmer instance"""
    global _global_prewarmer
    if _global_prewarmer is None:
        _global_prewarmer = ContextPrewarmer()
    return _global_prewarmer

def start_prewarming_service(enable_learning: bool = True, 
                           adaptive: bool = True,
                           interval: float = 60.0) -> ContextPrewarmer:
    """Start context pre-warming service with specified settings"""
    
    prewarmer = ContextPrewarmer(
        enable_learning=enable_learning,
        adaptive_scheduling=adaptive,
        prewarm_interval=interval
    )
    
    prewarmer.start()
    
    # Set as global instance
    global _global_prewarmer
    _global_prewarmer = prewarmer
    
    return prewarmer

def record_cli_usage(tool: Optional[str], query: str = "", context_size: int = 0):
    """Record CLI usage for pattern learning"""
    prewarmer = get_context_prewarmer()
    prewarmer.record_cli_usage(tool, query, context_size)