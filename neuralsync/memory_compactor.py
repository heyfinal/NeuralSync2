"""
Automated Memory Compaction and Garbage Collection System
Provides intelligent memory cleanup, compaction, and optimization for NeuralSync2
"""

import os
import time
import threading
import logging
import gc
import psutil
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import pickle
import hashlib

from .utils import now_ms

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memories: int
    active_memories: int
    tombstoned_memories: int
    expired_memories: int
    duplicate_memories: int
    total_size_bytes: int
    fragmentation_ratio: float
    access_frequency: Dict[str, int]

@dataclass
class CompactionTask:
    """Represents a compaction task"""
    task_id: str
    task_type: str
    priority: int
    estimated_duration_ms: int
    target_memories: List[str]
    scheduled_time: int
    status: str = "pending"  # pending, running, completed, failed

class MemoryAnalyzer:
    """Analyzes memory patterns and identifies optimization opportunities"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.cache_ttl = 300000  # 5 minutes
        
    def analyze_memory_usage(self, storage) -> MemoryStats:
        """Analyze current memory usage patterns"""
        
        cache_key = f"memory_analysis_{int(time.time() / 300)}"  # 5-minute buckets
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
            
        try:
            # Query database for memory statistics
            con = storage.con if hasattr(storage, 'con') else storage
            
            # Basic counts
            counts = con.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN tombstone = 0 THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN tombstone = 1 THEN 1 ELSE 0 END) as tombstoned,
                    SUM(CASE WHEN expires_at IS NOT NULL AND expires_at < ? THEN 1 ELSE 0 END) as expired
                FROM items
            """, (now_ms(),)).fetchone()
            
            total, active, tombstoned, expired = counts or (0, 0, 0, 0)
            
            # Estimate memory size
            size_result = con.execute("""
                SELECT SUM(LENGTH(text) + COALESCE(LENGTH(vector), 0) + COALESCE(LENGTH(meta), 0)) 
                FROM items WHERE tombstone = 0
            """).fetchone()
            total_size = size_result[0] or 0
            
            # Find duplicates (same text content)
            duplicates = con.execute("""
                SELECT COUNT(*) 
                FROM (
                    SELECT text, COUNT(*) as cnt 
                    FROM items 
                    WHERE tombstone = 0 AND text IS NOT NULL 
                    GROUP BY text 
                    HAVING cnt > 1
                )
            """).fetchone()[0] or 0
            
            # Calculate fragmentation (simplified)
            fragmentation = 0.0
            if hasattr(storage, 'mmap') and storage.mmap:
                allocated = storage.mmap_offset
                total_size_mmap = len(storage.mmap)
                fragmentation = 1.0 - (allocated / total_size_mmap) if total_size_mmap > 0 else 0.0
                
            # Access frequency analysis
            access_freq = {}
            if hasattr(storage, 'stats'):
                # Use recent access patterns from storage stats
                access_freq = storage.stats.get('access_patterns', {})
                
            stats = MemoryStats(
                total_memories=total,
                active_memories=active,
                tombstoned_memories=tombstoned,
                expired_memories=expired,
                duplicate_memories=duplicates,
                total_size_bytes=total_size,
                fragmentation_ratio=fragmentation,
                access_frequency=access_freq
            )
            
            # Cache results
            self.analysis_cache[cache_key] = stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Memory analysis failed: {e}")
            return MemoryStats(0, 0, 0, 0, 0, 0, 0.0, {})
            
    def identify_cleanup_opportunities(self, storage, stats: MemoryStats) -> List[CompactionTask]:
        """Identify specific cleanup and optimization opportunities"""
        
        tasks = []
        current_time = now_ms()
        
        # Task 1: Remove expired memories
        if stats.expired_memories > 0:
            tasks.append(CompactionTask(
                task_id=f"cleanup_expired_{current_time}",
                task_type="remove_expired",
                priority=1,  # High priority
                estimated_duration_ms=stats.expired_memories * 2,
                target_memories=[],  # Will be populated during execution
                scheduled_time=current_time
            ))
            
        # Task 2: Remove tombstoned memories (if many accumulated)
        if stats.tombstoned_memories > 1000:
            tasks.append(CompactionTask(
                task_id=f"cleanup_tombstones_{current_time}",
                task_type="remove_tombstones",
                priority=2,
                estimated_duration_ms=stats.tombstoned_memories,
                target_memories=[],
                scheduled_time=current_time
            ))
            
        # Task 3: Deduplicate similar memories
        if stats.duplicate_memories > 100:
            tasks.append(CompactionTask(
                task_id=f"deduplicate_{current_time}",
                task_type="deduplicate",
                priority=3,
                estimated_duration_ms=stats.duplicate_memories * 10,
                target_memories=[],
                scheduled_time=current_time
            ))
            
        # Task 4: Memory-mapped file compaction
        if stats.fragmentation_ratio > 0.5:  # More than 50% fragmentation
            tasks.append(CompactionTask(
                task_id=f"compact_mmap_{current_time}",
                task_type="compact_mmap",
                priority=4,
                estimated_duration_ms=5000,  # Fixed time estimate
                target_memories=[],
                scheduled_time=current_time
            ))
            
        # Task 5: Index optimization
        if stats.active_memories > 10000:
            tasks.append(CompactionTask(
                task_id=f"optimize_indexes_{current_time}",
                task_type="optimize_indexes",
                priority=5,
                estimated_duration_ms=10000,
                target_memories=[],
                scheduled_time=current_time
            ))
            
        return sorted(tasks, key=lambda t: t.priority)
        
    def estimate_memory_growth(self, storage, days: int = 7) -> Dict[str, Any]:
        """Estimate memory growth patterns"""
        
        try:
            con = storage.con if hasattr(storage, 'con') else storage
            
            # Get recent memory creation rates
            week_ago = now_ms() - (days * 24 * 60 * 60 * 1000)
            
            growth_data = con.execute("""
                SELECT 
                    DATE(created_at/1000, 'unixepoch') as date,
                    COUNT(*) as daily_count,
                    AVG(LENGTH(text)) as avg_size
                FROM items 
                WHERE created_at > ?
                GROUP BY DATE(created_at/1000, 'unixepoch')
                ORDER BY date
            """, (week_ago,)).fetchall()
            
            if not growth_data:
                return {'trend': 'stable', 'daily_rate': 0, 'size_growth': 0}
                
            # Calculate trends
            daily_counts = [row[1] for row in growth_data]
            avg_sizes = [row[2] for row in growth_data if row[2]]
            
            daily_rate = sum(daily_counts) / len(daily_counts)
            size_trend = sum(avg_sizes) / len(avg_sizes) if avg_sizes else 0
            
            # Determine trend
            if len(daily_counts) > 1:
                trend = "growing" if daily_counts[-1] > daily_counts[0] else "shrinking"
            else:
                trend = "stable"
                
            return {
                'trend': trend,
                'daily_rate': daily_rate,
                'size_growth': size_trend,
                'projection_7d': daily_rate * 7,
                'projection_30d': daily_rate * 30
            }
            
        except Exception as e:
            logger.error(f"Growth estimation failed: {e}")
            return {'trend': 'unknown', 'daily_rate': 0, 'size_growth': 0}


class GarbageCollector:
    """Advanced garbage collection for memory entries"""
    
    def __init__(self):
        self.collection_stats = {
            'total_collections': 0,
            'memories_collected': 0,
            'bytes_freed': 0,
            'avg_collection_time_ms': 0.0
        }
        
    def collect_expired_memories(self, storage) -> Dict[str, Any]:
        """Remove expired memories from storage"""
        
        start_time = time.time()
        collected_count = 0
        bytes_freed = 0
        
        try:
            con = storage.con if hasattr(storage, 'con') else storage
            current_time = now_ms()
            
            # Find expired memories
            expired_items = con.execute("""
                SELECT id, LENGTH(text) + COALESCE(LENGTH(vector), 0) as size
                FROM items 
                WHERE expires_at IS NOT NULL AND expires_at < ? AND tombstone = 0
            """, (current_time,)).fetchall()
            
            if not expired_items:
                return {'collected': 0, 'bytes_freed': 0, 'duration_ms': 0}
                
            # Mark as tombstoned
            expired_ids = [item[0] for item in expired_items]
            placeholders = ','.join(['?' for _ in expired_ids])
            
            con.execute(f"""
                UPDATE items 
                SET tombstone = 1, updated_at = ?
                WHERE id IN ({placeholders})
            """, [current_time] + expired_ids)
            
            # Update FTS index
            con.execute(f"""
                DELETE FROM items_fts WHERE id IN ({placeholders})
            """, expired_ids)
            
            con.commit()
            
            collected_count = len(expired_items)
            bytes_freed = sum(item[1] for item in expired_items)
            
            # Update B+ tree indexes if available
            if hasattr(storage, 'primary_index'):
                for item_id in expired_ids:
                    try:
                        storage.primary_index.delete(item_id)
                    except:
                        pass  # Ignore index deletion errors
                        
            logger.info(f"Collected {collected_count} expired memories, freed {bytes_freed} bytes")
            
        except Exception as e:
            logger.error(f"Expired memory collection failed: {e}")
            return {'error': str(e)}
            
        finally:
            duration = (time.time() - start_time) * 1000
            self._update_collection_stats(collected_count, bytes_freed, duration)
            
        return {
            'collected': collected_count,
            'bytes_freed': bytes_freed,
            'duration_ms': duration
        }
        
    def collect_tombstoned_memories(self, storage, batch_size: int = 1000) -> Dict[str, Any]:
        """Permanently remove tombstoned memories"""
        
        start_time = time.time()
        total_collected = 0
        total_bytes_freed = 0
        
        try:
            con = storage.con if hasattr(storage, 'con') else storage
            
            while True:
                # Get batch of tombstoned items
                tombstoned_items = con.execute("""
                    SELECT id, LENGTH(text) + COALESCE(LENGTH(vector), 0) as size
                    FROM items 
                    WHERE tombstone = 1
                    LIMIT ?
                """, (batch_size,)).fetchall()
                
                if not tombstoned_items:
                    break
                    
                # Delete batch
                tombstoned_ids = [item[0] for item in tombstoned_items]
                placeholders = ','.join(['?' for _ in tombstoned_ids])
                
                con.execute(f"DELETE FROM items WHERE id IN ({placeholders})", tombstoned_ids)
                con.execute(f"DELETE FROM items_fts WHERE id IN ({placeholders})", tombstoned_ids)
                con.commit()
                
                batch_collected = len(tombstoned_items)
                batch_bytes = sum(item[1] for item in tombstoned_items)
                
                total_collected += batch_collected
                total_bytes_freed += batch_bytes
                
                logger.debug(f"Deleted batch of {batch_collected} tombstoned memories")
                
            logger.info(f"Collected {total_collected} tombstoned memories, freed {total_bytes_freed} bytes")
            
        except Exception as e:
            logger.error(f"Tombstone collection failed: {e}")
            return {'error': str(e)}
            
        finally:
            duration = (time.time() - start_time) * 1000
            self._update_collection_stats(total_collected, total_bytes_freed, duration)
            
        return {
            'collected': total_collected,
            'bytes_freed': total_bytes_freed,
            'duration_ms': duration
        }
        
    def deduplicate_memories(self, storage) -> Dict[str, Any]:
        """Remove duplicate memories keeping the most recent/relevant ones"""
        
        start_time = time.time()
        duplicates_removed = 0
        bytes_freed = 0
        
        try:
            con = storage.con if hasattr(storage, 'con') else storage
            
            # Find duplicates by text content
            duplicate_groups = con.execute("""
                SELECT text, COUNT(*) as cnt, GROUP_CONCAT(id) as ids
                FROM items 
                WHERE tombstone = 0 AND text IS NOT NULL
                GROUP BY text
                HAVING cnt > 1
                ORDER BY cnt DESC
            """).fetchall()
            
            for text_content, count, ids_str in duplicate_groups:
                item_ids = ids_str.split(',')
                
                # Get detailed info for each duplicate
                items_info = []
                for item_id in item_ids:
                    item_info = con.execute("""
                        SELECT id, updated_at, consistency, benefit, confidence, 
                               LENGTH(text) + COALESCE(LENGTH(vector), 0) as size
                        FROM items WHERE id = ?
                    """, (item_id,)).fetchone()
                    
                    if item_info:
                        items_info.append(item_info)
                        
                if len(items_info) <= 1:
                    continue
                    
                # Sort by quality score (consistency * benefit * confidence + recency)
                def quality_score(item):
                    id_, updated_at, consistency, benefit, confidence, size = item
                    recency = updated_at or 0
                    quality = ((consistency or 0.5) * (benefit or 0.5) * 
                              (confidence or 0.5)) + (recency / 1e12)  # Normalize timestamp
                    return quality
                    
                items_info.sort(key=quality_score, reverse=True)
                
                # Keep the best one, mark others as tombstoned
                to_remove = items_info[1:]  # Remove all except the first (best)
                
                for item in to_remove:
                    con.execute("""
                        UPDATE items SET tombstone = 1, updated_at = ? WHERE id = ?
                    """, (now_ms(), item[0]))
                    
                    con.execute("DELETE FROM items_fts WHERE id = ?", (item[0],))
                    
                    duplicates_removed += 1
                    bytes_freed += item[5]  # size column
                    
            con.commit()
            
            logger.info(f"Deduplicated {duplicates_removed} memories, freed {bytes_freed} bytes")
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return {'error': str(e)}
            
        finally:
            duration = (time.time() - start_time) * 1000
            self._update_collection_stats(duplicates_removed, bytes_freed, duration)
            
        return {
            'removed': duplicates_removed,
            'bytes_freed': bytes_freed,
            'duration_ms': duration
        }
        
    def _update_collection_stats(self, collected: int, bytes_freed: int, duration_ms: float):
        """Update garbage collection statistics"""
        self.collection_stats['total_collections'] += 1
        self.collection_stats['memories_collected'] += collected
        self.collection_stats['bytes_freed'] += bytes_freed
        
        # Update average collection time
        current_avg = self.collection_stats['avg_collection_time_ms']
        collection_count = self.collection_stats['total_collections']
        
        self.collection_stats['avg_collection_time_ms'] = (
            (current_avg * (collection_count - 1) + duration_ms) / collection_count
        )


class MemoryCompactor:
    """Main compaction coordinator that manages all compaction activities"""
    
    def __init__(self, storage, enable_auto_compact: bool = True):
        self.storage = storage
        self.analyzer = MemoryAnalyzer()
        self.garbage_collector = GarbageCollector()
        
        # Configuration
        self.enable_auto_compact = enable_auto_compact
        self.compact_interval = 300  # 5 minutes
        self.max_concurrent_tasks = 3
        
        # Task management
        self.pending_tasks: List[CompactionTask] = []
        self.running_tasks: Dict[str, CompactionTask] = {}
        self.completed_tasks: List[CompactionTask] = []
        self.task_lock = threading.RLock()
        
        # Performance tracking
        self.compaction_stats = {
            'total_compactions': 0,
            'memories_compacted': 0,
            'bytes_reclaimed': 0,
            'avg_compaction_time_ms': 0.0,
            'last_compaction': 0
        }
        
        # Background compaction thread
        self.compaction_thread = None
        self.stop_event = threading.Event()
        
        if enable_auto_compact:
            self.start_auto_compaction()
            
    def start_auto_compaction(self):
        """Start automatic background compaction"""
        if self.compaction_thread and self.compaction_thread.is_alive():
            return
            
        self.stop_event.clear()
        self.compaction_thread = threading.Thread(target=self._compaction_loop, daemon=True)
        self.compaction_thread.start()
        logger.info("Started automatic memory compaction")
        
    def stop_auto_compaction(self):
        """Stop automatic compaction"""
        self.stop_event.set()
        if self.compaction_thread:
            self.compaction_thread.join(timeout=10)
        logger.info("Stopped automatic memory compaction")
        
    def _compaction_loop(self):
        """Main compaction loop"""
        while not self.stop_event.wait(self.compact_interval):
            try:
                # Check system resources before compacting
                if not self._should_compact():
                    continue
                    
                # Perform automatic compaction
                self.compact_memory(auto_mode=True)
                
            except Exception as e:
                logger.error(f"Auto-compaction error: {e}")
                
    def _should_compact(self) -> bool:
        """Determine if compaction should run based on system state"""
        
        # Check system memory pressure
        memory = psutil.virtual_memory()
        if memory.percent > 85:  # Don't compact if memory is very tight
            logger.debug("Skipping compaction due to high memory usage")
            return False
            
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:  # Don't compact if CPU is very busy
            logger.debug("Skipping compaction due to high CPU usage")
            return False
            
        # Check if we've compacted recently
        time_since_last = now_ms() - self.compaction_stats['last_compaction']
        if time_since_last < 60000:  # Don't compact more than once per minute
            return False
            
        return True
        
    def compact_memory(self, auto_mode: bool = False) -> Dict[str, Any]:
        """Perform comprehensive memory compaction"""
        
        start_time = time.time()
        total_reclaimed = 0
        
        try:
            # Analyze current memory state
            stats = self.analyzer.analyze_memory_usage(self.storage)
            logger.info(f"Memory analysis: {stats.active_memories} active, "
                       f"{stats.tombstoned_memories} tombstoned, "
                       f"{stats.expired_memories} expired")
            
            # Identify compaction tasks
            tasks = self.analyzer.identify_cleanup_opportunities(self.storage, stats)
            
            if not tasks:
                logger.info("No compaction opportunities identified")
                return {'status': 'no_work', 'duration_ms': 0}
                
            # Execute tasks in priority order
            results = {}
            
            for task in tasks:
                if auto_mode and len(self.running_tasks) >= self.max_concurrent_tasks:
                    break  # Don't overload system in auto mode
                    
                task_result = self._execute_compaction_task(task)
                results[task.task_type] = task_result
                
                if 'bytes_freed' in task_result:
                    total_reclaimed += task_result['bytes_freed']
                    
            # Compact memory-mapped file if needed
            if hasattr(self.storage, 'mmap') and stats.fragmentation_ratio > 0.3:
                mmap_result = self._compact_memory_mapped_file()
                results['mmap_compaction'] = mmap_result
                
            # Optimize indexes
            if hasattr(self.storage, 'primary_index'):
                self._optimize_indexes()
                results['index_optimization'] = {'status': 'completed'}
                
            # Force Python garbage collection
            collected = gc.collect()
            results['python_gc'] = {'collected_objects': collected}
            
            # Update statistics
            duration = (time.time() - start_time) * 1000
            self._update_compaction_stats(total_reclaimed, duration)
            
            logger.info(f"Memory compaction completed in {duration:.1f}ms, "
                       f"reclaimed {total_reclaimed} bytes")
            
            return {
                'status': 'completed',
                'duration_ms': duration,
                'bytes_reclaimed': total_reclaimed,
                'tasks_executed': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Memory compaction failed: {e}")
            return {'status': 'failed', 'error': str(e)}
            
    def _execute_compaction_task(self, task: CompactionTask) -> Dict[str, Any]:
        """Execute a specific compaction task"""
        
        task.status = "running"
        
        try:
            if task.task_type == "remove_expired":
                result = self.garbage_collector.collect_expired_memories(self.storage)
                
            elif task.task_type == "remove_tombstones":
                result = self.garbage_collector.collect_tombstoned_memories(self.storage)
                
            elif task.task_type == "deduplicate":
                result = self.garbage_collector.deduplicate_memories(self.storage)
                
            elif task.task_type == "compact_mmap":
                result = self._compact_memory_mapped_file()
                
            elif task.task_type == "optimize_indexes":
                result = self._optimize_indexes()
                
            else:
                result = {'status': 'unknown_task_type'}
                
            task.status = "completed"
            return result
            
        except Exception as e:
            task.status = "failed"
            logger.error(f"Task {task.task_id} failed: {e}")
            return {'status': 'failed', 'error': str(e)}
            
    def _compact_memory_mapped_file(self) -> Dict[str, Any]:
        """Compact memory-mapped file to reduce fragmentation"""
        
        if not hasattr(self.storage, 'mmap') or not self.storage.mmap:
            return {'status': 'not_applicable'}
            
        try:
            start_time = time.time()
            
            # Flush current data
            self.storage.mmap.flush()
            
            # Create new compacted file
            old_path = self.storage.mmap_path
            new_path = old_path + '.compact'
            
            # Calculate actual used space
            used_space = self.storage.mmap_offset
            
            # Create new file with just the used space
            with open(new_path, 'wb') as new_file:
                new_file.write(self.storage.mmap[:used_space])
                
            # Replace old file
            self.storage.mmap.close()
            self.storage.mmap_file.close()
            
            os.replace(new_path, old_path)
            
            # Reopen memory-mapped file
            self.storage._init_memory_map()
            
            duration = (time.time() - start_time) * 1000
            
            logger.info(f"Memory-mapped file compacted in {duration:.1f}ms")
            
            return {
                'status': 'completed',
                'duration_ms': duration,
                'space_reclaimed': len(self.storage.mmap) - used_space if self.storage.mmap else 0
            }
            
        except Exception as e:
            logger.error(f"Memory-mapped file compaction failed: {e}")
            return {'status': 'failed', 'error': str(e)}
            
    def _optimize_indexes(self) -> Dict[str, Any]:
        """Optimize B+ tree indexes for better performance"""
        
        if not hasattr(self.storage, 'primary_index'):
            return {'status': 'not_applicable'}
            
        try:
            start_time = time.time()
            
            # Flush and optimize each index
            indexes = [
                self.storage.primary_index,
                self.storage.text_index,
                self.storage.scope_index,
                self.storage.time_index
            ]
            
            for index in indexes:
                if hasattr(index, 'flush'):
                    index.flush()
                    
            # Force SQLite to optimize as well
            con = self.storage.con
            con.execute('VACUUM')
            con.execute('REINDEX')
            con.commit()
            
            duration = (time.time() - start_time) * 1000
            
            logger.info(f"Indexes optimized in {duration:.1f}ms")
            
            return {
                'status': 'completed',
                'duration_ms': duration,
                'indexes_optimized': len(indexes)
            }
            
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
            
    def _update_compaction_stats(self, bytes_reclaimed: int, duration_ms: float):
        """Update compaction statistics"""
        self.compaction_stats['total_compactions'] += 1
        self.compaction_stats['bytes_reclaimed'] += bytes_reclaimed
        self.compaction_stats['last_compaction'] = now_ms()
        
        # Update average duration
        current_avg = self.compaction_stats['avg_compaction_time_ms']
        compaction_count = self.compaction_stats['total_compactions']
        
        self.compaction_stats['avg_compaction_time_ms'] = (
            (current_avg * (compaction_count - 1) + duration_ms) / compaction_count
        )
        
    def get_compaction_stats(self) -> Dict[str, Any]:
        """Get comprehensive compaction statistics"""
        
        # Current memory analysis
        current_stats = self.analyzer.analyze_memory_usage(self.storage)
        
        # Growth projections
        growth_projection = self.analyzer.estimate_memory_growth(self.storage)
        
        return {
            'compaction_stats': self.compaction_stats,
            'gc_stats': self.garbage_collector.collection_stats,
            'current_memory': current_stats.__dict__,
            'growth_projection': growth_projection,
            'auto_compact_enabled': self.enable_auto_compact,
            'pending_tasks': len(self.pending_tasks),
            'running_tasks': len(self.running_tasks),
            'system_resources': {
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(),
                'disk_usage': psutil.disk_usage('/').percent
            }
        }
        
    def force_full_compaction(self) -> Dict[str, Any]:
        """Force immediate comprehensive compaction"""
        logger.info("Starting forced full memory compaction")
        
        # Temporarily disable auto-compaction to avoid conflicts
        was_auto = self.enable_auto_compact
        self.enable_auto_compact = False
        
        try:
            result = self.compact_memory(auto_mode=False)
            
            # Additional aggressive cleanup
            if hasattr(self.storage, 'storage'):
                # Optimize underlying enhanced storage
                self.storage.optimize_performance()
                
            return result
            
        finally:
            self.enable_auto_compact = was_auto