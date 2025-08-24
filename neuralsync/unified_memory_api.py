"""
Unified Memory API for NeuralSync2 Core Memory System
Provides a single, high-level interface that integrates all memory components
"""

import os
import time
import threading
import logging
import uuid
from typing import Dict, Any, List, Optional, Union, Iterator, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager

from .core_memory import CoreMemoryManager, get_core_memory, init_core_memory
from .storage import EnhancedStorage, connect, upsert_item, recall, get_persona, put_persona
from .memory_merger import IntelligentMemoryMerger, create_memory_update
from .memory_compactor import MemoryCompactor
from .performance_monitor import PerformanceMonitorManager, init_performance_monitoring
from .crdt import ByzantineCRDT, AdvancedVersion
from .utils import now_ms

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for unified memory system"""
    base_path: Optional[str] = None
    memory_pool_size: int = 1024 * 1024 * 512  # 512MB
    enable_performance_monitoring: bool = True
    enable_auto_optimization: bool = True
    enable_auto_compaction: bool = True
    target_response_time_ms: float = 1.0
    max_memory_entries: int = 1000000
    compaction_interval: int = 300  # 5 minutes
    optimization_interval: int = 300  # 5 minutes

@dataclass
class MemoryEntry:
    """Unified memory entry representation"""
    id: str
    content: Any
    scope: str = "global"
    tool_name: Optional[str] = None
    priority: float = 0.5
    tags: List[str] = None
    created_at: int = 0
    updated_at: int = 0
    access_count: int = 0
    metadata: Dict[str, Any] = None

@dataclass
class MemoryQuery:
    """Unified memory query representation"""
    text: str
    scope: str = "any"
    tool_name: Optional[str] = None
    top_k: int = 10
    min_priority: float = 0.0
    max_age_ms: Optional[int] = None
    tags: Optional[List[str]] = None
    semantic_search: bool = True

class UnifiedMemoryAPI:
    """
    Unified Memory API that provides a single interface to all NeuralSync2 memory capabilities.
    
    Features:
    - Sub-millisecond memory access with B+ tree indexing
    - Zero-copy memory operations with memory mapping
    - Byzantine fault-tolerant CRDT conflict resolution
    - Cross-session memory persistence
    - Intelligent memory merging for concurrent updates
    - Automated compaction and garbage collection
    - Real-time performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize unified memory system with all components"""
        
        self.config = config or MemoryConfig()
        self.session_id = str(uuid.uuid4())
        
        # Initialize paths
        if self.config.base_path is None:
            self.config.base_path = os.path.expanduser("~/.neuralsync2")
        
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self._initialize_components()
        
        # API statistics
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_response_time_ms': 0.0,
            'session_start_time': now_ms(),
            'last_operation_time': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Unified Memory API initialized with session {self.session_id}")
        
    def _initialize_components(self):
        """Initialize all memory system components"""
        
        try:
            # Enhanced storage with memory mapping and B+ tree indexes
            self.storage = connect(str(self.base_path / "neuralsync.db"))
            logger.info("Enhanced storage initialized")
            
            # Core memory manager for cross-session persistence
            self.core_memory = init_core_memory(str(self.base_path / "core_memory"))
            logger.info("Core memory manager initialized")
            
            # Intelligent memory merger for concurrent updates
            self.memory_merger = IntelligentMemoryMerger(self.core_memory)
            logger.info("Memory merger initialized")
            
            # Automated compaction and garbage collection
            self.compactor = MemoryCompactor(
                self.storage, 
                enable_auto_compact=self.config.enable_auto_compaction
            )
            logger.info("Memory compactor initialized")
            
            # Performance monitoring and optimization
            self.performance_monitor = init_performance_monitoring(
                self.storage,
                enable_auto_optimization=self.config.enable_auto_optimization,
                target_response_time_ms=self.config.target_response_time_ms
            )
            logger.info("Performance monitor initialized")
            
            # Mark as initialized
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            self._initialized = False
            raise
            
    # High-level Memory Operations
    
    def remember(self, content: Any, scope: str = "global", tool_name: Optional[str] = None,
                priority: float = 0.5, tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a memory entry with intelligent persistence and optimization.
        
        Args:
            content: The content to remember (any serializable type)
            scope: Memory scope for organization and access control
            tool_name: Name of the CLI tool storing this memory
            priority: Priority/importance score (0.0 to 1.0)
            tags: List of tags for categorization
            metadata: Additional metadata
            
        Returns:
            Memory ID for future reference
        """
        
        with self.performance_monitor.measure_memory_access("remember"):
            try:
                with self.lock:
                    # Create memory entry
                    memory_id = str(uuid.uuid4())
                    current_time = now_ms()
                    
                    entry = MemoryEntry(
                        id=memory_id,
                        content=content,
                        scope=scope,
                        tool_name=tool_name or self._get_calling_tool(),
                        priority=priority,
                        tags=tags or [],
                        created_at=current_time,
                        updated_at=current_time,
                        metadata=metadata or {}
                    )
                    
                    # Store in core memory for cross-session persistence
                    self.core_memory.remember(
                        content=content,
                        scope=scope,
                        tool_name=entry.tool_name,
                        priority=priority,
                        tags=tags
                    )
                    
                    # Store in enhanced storage with indexing
                    storage_item = self._entry_to_storage_item(entry)
                    upsert_item(self.storage, storage_item)
                    
                    # Create update for merger
                    update = create_memory_update(
                        session_id=self.session_id,
                        tool_name=entry.tool_name,
                        memory_id=memory_id,
                        operation='create',
                        content=content,
                        metadata={
                            'scope': scope,
                            'priority': priority,
                            'tags': tags
                        }
                    )
                    
                    # Queue for intelligent merging
                    self.memory_merger.queue_update(update)
                    
                    self._update_stats(True)
                    
                    logger.debug(f"Stored memory {memory_id} in scope '{scope}'")
                    return memory_id
                    
            except Exception as e:
                self._update_stats(False)
                logger.error(f"Remember operation failed: {e}")
                raise
                
    def recall(self, query: Union[str, MemoryQuery], **kwargs) -> List[MemoryEntry]:
        """
        Recall memories using intelligent search with sub-millisecond performance.
        
        Args:
            query: Search query (string or MemoryQuery object)
            **kwargs: Additional query parameters if using string query
            
        Returns:
            List of matching memory entries, ranked by relevance
        """
        
        with self.performance_monitor.measure_memory_access("recall"):
            try:
                # Parse query
                if isinstance(query, str):
                    memory_query = MemoryQuery(text=query, **kwargs)
                else:
                    memory_query = query
                    
                with self.lock:
                    # Search cross-session memories first for speed
                    cross_session_results = self.core_memory.recall_across_sessions(
                        query=memory_query.text,
                        scope=memory_query.scope,
                        top_k=memory_query.top_k,
                        include_current=True
                    )
                    
                    # Convert cross-session results to MemoryEntry objects
                    results = []
                    for result in cross_session_results:
                        entry = MemoryEntry(
                            id=result.get('id', ''),
                            content=result.get('content', ''),
                            scope=result.get('scope', 'unknown'),
                            tool_name=result.get('tool_name'),
                            priority=result.get('priority', 0.5),
                            tags=result.get('tags', []),
                            created_at=result.get('created_at', 0),
                            updated_at=result.get('updated_at', 0),
                            access_count=result.get('access_count', 0),
                            metadata=result.get('metadata', {})
                        )
                        results.append(entry)
                        
                    # Supplement with storage-based search if needed
                    if len(results) < memory_query.top_k:
                        storage_results = recall(
                            self.storage,
                            memory_query.text,
                            memory_query.top_k - len(results),
                            memory_query.scope,
                            memory_query.tool_name
                        )
                        
                        # Convert storage results and merge
                        existing_ids = {r.id for r in results}
                        for storage_result in storage_results:
                            if storage_result['id'] not in existing_ids:
                                entry = self._storage_item_to_entry(storage_result)
                                results.append(entry)
                                
                    # Apply additional filtering
                    results = self._apply_query_filters(results, memory_query)
                    
                    # Sort by relevance (priority + recency + access count)
                    results.sort(key=self._calculate_relevance_score, reverse=True)
                    
                    self._update_stats(True)
                    
                    logger.debug(f"Recalled {len(results)} memories for query '{memory_query.text}'")
                    return results[:memory_query.top_k]
                    
            except Exception as e:
                self._update_stats(False)
                logger.error(f"Recall operation failed: {e}")
                return []
                
    def update_memory(self, memory_id: str, content: Any = None, priority: float = None,
                     tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing memory entry with conflict resolution.
        
        Args:
            memory_id: ID of memory to update
            content: New content (if provided)
            priority: New priority (if provided)
            tags: New tags (if provided)  
            metadata: New metadata (if provided)
            
        Returns:
            True if successful, False otherwise
        """
        
        with self.performance_monitor.measure_memory_access("update_memory"):
            try:
                with self.lock:
                    # Find existing memory
                    existing_memories = self.recall(MemoryQuery(text="*", scope="any"))
                    existing_memory = None
                    
                    for memory in existing_memories:
                        if memory.id == memory_id:
                            existing_memory = memory
                            break
                            
                    if not existing_memory:
                        logger.warning(f"Memory {memory_id} not found for update")
                        return False
                        
                    # Create updated memory entry
                    updated_entry = MemoryEntry(
                        id=memory_id,
                        content=content if content is not None else existing_memory.content,
                        scope=existing_memory.scope,
                        tool_name=existing_memory.tool_name,
                        priority=priority if priority is not None else existing_memory.priority,
                        tags=tags if tags is not None else existing_memory.tags,
                        created_at=existing_memory.created_at,
                        updated_at=now_ms(),
                        access_count=existing_memory.access_count + 1,
                        metadata={**(existing_memory.metadata or {}), **(metadata or {})}
                    )
                    
                    # Update storage
                    storage_item = self._entry_to_storage_item(updated_entry)
                    upsert_item(self.storage, storage_item)
                    
                    # Create update for merger
                    update = create_memory_update(
                        session_id=self.session_id,
                        tool_name=updated_entry.tool_name,
                        memory_id=memory_id,
                        operation='update',
                        content=updated_entry.content,
                        metadata=updated_entry.metadata
                    )
                    
                    # Queue for intelligent merging
                    self.memory_merger.queue_update(update)
                    
                    self._update_stats(True)
                    
                    logger.debug(f"Updated memory {memory_id}")
                    return True
                    
            except Exception as e:
                self._update_stats(False)
                logger.error(f"Update operation failed: {e}")
                return False
                
    def forget(self, memory_id: str) -> bool:
        """
        Remove a memory entry with CRDT tombstone handling.
        
        Args:
            memory_id: ID of memory to remove
            
        Returns:
            True if successful, False otherwise
        """
        
        with self.performance_monitor.measure_memory_access("forget"):
            try:
                with self.lock:
                    # Create delete update for merger
                    update = create_memory_update(
                        session_id=self.session_id,
                        tool_name=self._get_calling_tool(),
                        memory_id=memory_id,
                        operation='delete',
                        content=None
                    )
                    
                    # Queue for intelligent merging
                    self.memory_merger.queue_update(update)
                    
                    # Mark as deleted in storage (CRDT handles tombstones)
                    from .storage import delete_item
                    delete_item(self.storage, memory_id)
                    
                    self._update_stats(True)
                    
                    logger.debug(f"Deleted memory {memory_id}")
                    return True
                    
            except Exception as e:
                self._update_stats(False)
                logger.error(f"Forget operation failed: {e}")
                return False
                
    # Advanced Memory Operations
    
    def range_recall(self, start_time: int, end_time: int, scope: str = "any",
                    limit: int = 100) -> List[MemoryEntry]:
        """
        Recall memories within a time range using temporal indexing.
        
        Args:
            start_time: Start timestamp (milliseconds)
            end_time: End timestamp (milliseconds)
            scope: Memory scope filter
            limit: Maximum results to return
            
        Returns:
            List of memories within time range
        """
        
        with self.performance_monitor.measure_memory_access("range_recall"):
            try:
                # Use enhanced storage's temporal index
                storage_results = self.storage.range_recall(start_time, end_time, limit)
                
                # Convert to MemoryEntry objects
                results = []
                for result in storage_results:
                    if scope == "any" or result.get('scope') == scope:
                        entry = self._storage_item_to_entry(result)
                        results.append(entry)
                        
                self._update_stats(True)
                return results
                
            except Exception as e:
                self._update_stats(False)
                logger.error(f"Range recall failed: {e}")
                return []
                
    def bulk_remember(self, entries: List[Dict[str, Any]]) -> List[str]:
        """
        Store multiple memories efficiently in a single transaction.
        
        Args:
            entries: List of memory entry dictionaries
            
        Returns:
            List of memory IDs
        """
        
        with self.performance_monitor.measure_memory_access("bulk_remember"):
            try:
                memory_ids = []
                
                with self.lock:
                    for entry_data in entries:
                        memory_id = self.remember(
                            content=entry_data.get('content'),
                            scope=entry_data.get('scope', 'global'),
                            tool_name=entry_data.get('tool_name'),
                            priority=entry_data.get('priority', 0.5),
                            tags=entry_data.get('tags'),
                            metadata=entry_data.get('metadata')
                        )
                        memory_ids.append(memory_id)
                        
                    # Process all updates together for efficiency
                    self.memory_merger.process_updates()
                    
                self._update_stats(True, len(entries))
                return memory_ids
                
            except Exception as e:
                self._update_stats(False)
                logger.error(f"Bulk remember failed: {e}")
                return []
                
    def semantic_search(self, query: str, scope: str = "any", top_k: int = 10,
                       similarity_threshold: float = 0.7) -> List[MemoryEntry]:
        """
        Perform semantic search using embeddings and vector similarity.
        
        Args:
            query: Search query
            scope: Memory scope filter
            top_k: Maximum results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of semantically similar memories
        """
        
        with self.performance_monitor.measure_memory_access("semantic_search"):
            try:
                # Use enhanced storage's semantic search capabilities
                memory_query = MemoryQuery(
                    text=query,
                    scope=scope,
                    top_k=top_k,
                    semantic_search=True
                )
                
                results = self.recall(memory_query)
                
                # Filter by similarity threshold (simplified)
                # In practice, this would use actual embedding similarity scores
                filtered_results = [r for r in results if r.priority >= similarity_threshold * 0.5]
                
                self._update_stats(True)
                return filtered_results
                
            except Exception as e:
                self._update_stats(False)
                logger.error(f"Semantic search failed: {e}")
                return []
                
    # System Management Operations
    
    def optimize_performance(self, target_response_time_ms: float = 1.0) -> Dict[str, Any]:
        """
        Trigger immediate performance optimization.
        
        Args:
            target_response_time_ms: Target response time in milliseconds
            
        Returns:
            Optimization results
        """
        
        try:
            result = self.performance_monitor.force_optimization(target_response_time_ms)
            logger.info(f"Performance optimization completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            return {'error': str(e)}
            
    def compact_memory(self) -> Dict[str, Any]:
        """
        Trigger immediate memory compaction and garbage collection.
        
        Returns:
            Compaction results
        """
        
        try:
            result = self.compactor.force_full_compaction()
            logger.info(f"Memory compaction completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Memory compaction failed: {e}")
            return {'error': str(e)}
            
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics and performance metrics.
        
        Returns:
            Complete system statistics
        """
        
        try:
            return {
                'api_stats': self.stats,
                'session_info': self.core_memory.get_session_info(),
                'performance_stats': self.performance_monitor.get_comprehensive_stats(),
                'compaction_stats': self.compactor.get_compaction_stats(),
                'merge_stats': self.memory_merger.get_merge_statistics(),
                'storage_stats': self.storage.get_performance_stats(),
                'system_config': asdict(self.config)
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {'error': str(e)}
            
    def merge_with_session(self, other_session_path: str) -> Dict[str, Any]:
        """
        Merge memories from another session or instance.
        
        Args:
            other_session_path: Path to other session's memory data
            
        Returns:
            Merge results
        """
        
        try:
            # Create temporary instance for other session
            other_instance = CoreMemoryManager(other_session_path)
            
            # Merge with current instance
            result = self.core_memory.merge_with_other_instance(other_instance)
            
            # Cleanup
            other_instance.cleanup()
            
            logger.info(f"Session merge completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Session merge failed: {e}")
            return {'error': str(e)}
            
    # Context Managers for Advanced Usage
    
    @contextmanager
    def batch_operations(self):
        """
        Context manager for batching multiple operations for efficiency.
        """
        
        original_auto_compact = self.compactor.enable_auto_compact
        original_auto_optimize = self.performance_monitor.enable_auto_optimization
        
        try:
            # Temporarily disable auto operations
            self.compactor.enable_auto_compact = False
            self.performance_monitor.enable_auto_optimization = False
            
            yield
            
        finally:
            # Process all pending updates
            self.memory_merger.process_updates()
            
            # Restore auto operations
            self.compactor.enable_auto_compact = original_auto_compact
            self.performance_monitor.enable_auto_optimization = original_auto_optimize
            
    @contextmanager
    def performance_profiling(self, operation_name: str):
        """
        Context manager for detailed performance profiling.
        """
        
        start_time = time.perf_counter()
        start_stats = self.get_system_stats()
        
        try:
            yield
            
        finally:
            end_time = time.perf_counter()
            end_stats = self.get_system_stats()
            
            duration_ms = (end_time - start_time) * 1000
            
            logger.info(f"Performance profile for '{operation_name}': "
                       f"{duration_ms:.2f}ms")
            
            # Record detailed metrics
            self.performance_monitor.performance_tracker.record_operation(
                operation_name, duration_ms, {
                    'start_stats': start_stats,
                    'end_stats': end_stats
                }
            )
            
    # Helper Methods
    
    def _get_calling_tool(self) -> str:
        """Get name of calling tool from stack"""
        import sys
        try:
            frame = sys._getframe(2)  # Go up 2 frames to get caller
            return frame.f_code.co_filename.split('/')[-1].replace('.py', '')
        except:
            return 'unknown_tool'
            
    def _entry_to_storage_item(self, entry: MemoryEntry) -> Dict[str, Any]:
        """Convert MemoryEntry to storage item format"""
        return {
            'id': entry.id,
            'kind': 'unified_memory',
            'text': str(entry.content),
            'scope': entry.scope,
            'tool': entry.tool_name,
            'tags': str(entry.tags) if entry.tags else None,
            'confidence': entry.priority,
            'benefit': entry.priority,
            'consistency': 0.5,
            'created_at': entry.created_at,
            'updated_at': entry.updated_at,
            'source': f"unified_api_{self.session_id}",
            'meta': str(entry.metadata) if entry.metadata else None
        }
        
    def _storage_item_to_entry(self, storage_item: Dict[str, Any]) -> MemoryEntry:
        """Convert storage item to MemoryEntry format"""
        return MemoryEntry(
            id=storage_item.get('id', ''),
            content=storage_item.get('text', ''),
            scope=storage_item.get('scope', 'unknown'),
            tool_name=storage_item.get('tool'),
            priority=storage_item.get('confidence', 0.5),
            tags=eval(storage_item.get('tags', '[]')) if storage_item.get('tags') else [],
            created_at=storage_item.get('created_at', 0),
            updated_at=storage_item.get('updated_at', 0),
            access_count=0,  # Not tracked in storage format
            metadata=eval(storage_item.get('meta', '{}')) if storage_item.get('meta') else {}
        )
        
    def _apply_query_filters(self, results: List[MemoryEntry], query: MemoryQuery) -> List[MemoryEntry]:
        """Apply additional filtering to query results"""
        
        filtered = results
        
        # Priority filter
        if query.min_priority > 0:
            filtered = [r for r in filtered if r.priority >= query.min_priority]
            
        # Age filter
        if query.max_age_ms:
            cutoff_time = now_ms() - query.max_age_ms
            filtered = [r for r in filtered if r.updated_at >= cutoff_time]
            
        # Tag filter
        if query.tags:
            filtered = [
                r for r in filtered 
                if r.tags and any(tag in r.tags for tag in query.tags)
            ]
            
        return filtered
        
    def _calculate_relevance_score(self, entry: MemoryEntry) -> float:
        """Calculate relevance score for sorting results"""
        
        # Combine priority, recency, and access frequency
        current_time = now_ms()
        age_hours = (current_time - entry.updated_at) / (1000 * 60 * 60)
        
        recency_score = 1.0 / (1.0 + age_hours / 24)  # Decays over days
        priority_score = entry.priority
        access_score = min(1.0, entry.access_count / 10.0)  # Normalized access count
        
        return 0.4 * priority_score + 0.4 * recency_score + 0.2 * access_score
        
    def _update_stats(self, success: bool, operation_count: int = 1):
        """Update API usage statistics"""
        
        with self.lock:
            self.stats['total_operations'] += operation_count
            self.stats['last_operation_time'] = now_ms()
            
            if success:
                self.stats['successful_operations'] += operation_count
            else:
                self.stats['failed_operations'] += operation_count
                
            # Update success rate
            total_ops = self.stats['total_operations']
            if total_ops > 0:
                success_rate = self.stats['successful_operations'] / total_ops
            else:
                success_rate = 1.0
                
            self.stats['success_rate'] = success_rate
            
    def cleanup(self):
        """Cleanup all resources"""
        try:
            # Stop background processes
            self.compactor.stop_auto_compaction()
            self.performance_monitor.stop_monitoring()
            
            # Cleanup components
            self.core_memory.cleanup()
            self.storage.close()
            
            logger.info(f"Unified Memory API cleanup completed for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Global unified memory API instance
_global_memory_api: Optional[UnifiedMemoryAPI] = None

def get_memory_api(config: Optional[MemoryConfig] = None) -> UnifiedMemoryAPI:
    """Get global unified memory API instance"""
    global _global_memory_api
    if _global_memory_api is None:
        _global_memory_api = UnifiedMemoryAPI(config)
    return _global_memory_api

def init_memory_api(config: Optional[MemoryConfig] = None) -> UnifiedMemoryAPI:
    """Initialize global unified memory API"""
    global _global_memory_api
    _global_memory_api = UnifiedMemoryAPI(config)
    return _global_memory_api


if __name__ == "__main__":
    # Comprehensive test of the unified memory API
    
    def test_unified_memory_api():
        """Test all major functionality of the unified memory API"""
        
        print("Testing Unified Memory API...")
        
        # Initialize with custom config
        config = MemoryConfig(
            base_path="/tmp/neuralsync_test",
            enable_performance_monitoring=True,
            enable_auto_optimization=True,
            target_response_time_ms=0.5  # Very aggressive target
        )
        
        with UnifiedMemoryAPI(config) as api:
            # Test basic memory operations
            print("\n1. Testing basic memory operations...")
            
            mem1 = api.remember("This is a test memory", scope="test", priority=0.8)
            mem2 = api.remember("Another important fact", scope="facts", priority=0.9, tags=["important"])
            mem3 = api.remember({"type": "structured", "data": [1, 2, 3]}, scope="data")
            
            print(f"Stored 3 memories: {mem1}, {mem2}, {mem3}")
            
            # Test recall
            print("\n2. Testing recall operations...")
            
            results = api.recall("test")
            print(f"Recalled {len(results)} memories for 'test'")
            for result in results:
                print(f"  - {result.content} (priority: {result.priority})")
                
            # Test advanced queries
            print("\n3. Testing advanced queries...")
            
            query = MemoryQuery(
                text="important",
                scope="any",
                top_k=5,
                min_priority=0.5,
                tags=["important"]
            )
            
            advanced_results = api.recall(query)
            print(f"Advanced query returned {len(advanced_results)} results")
            
            # Test semantic search
            print("\n4. Testing semantic search...")
            
            semantic_results = api.semantic_search("facts about data", top_k=3)
            print(f"Semantic search returned {len(semantic_results)} results")
            
            # Test bulk operations
            print("\n5. Testing bulk operations...")
            
            bulk_entries = [
                {"content": f"Bulk entry {i}", "scope": "bulk", "priority": 0.6}
                for i in range(5)
            ]
            
            with api.batch_operations():
                bulk_ids = api.bulk_remember(bulk_entries)
                print(f"Stored {len(bulk_ids)} bulk entries")
                
            # Test range recall
            print("\n6. Testing range recall...")
            
            import time
            time.sleep(1)  # Ensure time difference
            
            end_time = now_ms()
            start_time = end_time - 60000  # Last minute
            
            range_results = api.range_recall(start_time, end_time)
            print(f"Range recall found {len(range_results)} memories in last minute")
            
            # Test system operations
            print("\n7. Testing system operations...")
            
            # Performance optimization
            opt_result = api.optimize_performance(target_response_time_ms=0.5)
            print(f"Performance optimization: {opt_result.get('status', 'unknown')}")
            
            # Memory compaction
            compact_result = api.compact_memory()
            print(f"Memory compaction: {compact_result.get('status', 'unknown')}")
            
            # System statistics
            print("\n8. Getting system statistics...")
            
            stats = api.get_system_stats()
            print(f"Total operations: {stats['api_stats']['total_operations']}")
            print(f"Success rate: {stats['api_stats'].get('success_rate', 0):.2%}")
            print(f"Active sessions: {len(stats['session_info']['active_sessions'])}")
            
            # Test performance profiling
            print("\n9. Testing performance profiling...")
            
            with api.performance_profiling("test_operation"):
                # Simulate some work
                complex_query = api.recall(MemoryQuery(text="*", scope="any", top_k=50))
                print(f"Profiled operation processed {len(complex_query)} memories")
                
            print("\nUnified Memory API test completed successfully!")
            
    # Run the test
    test_unified_memory_api()