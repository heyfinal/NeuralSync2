"""
Persistent Core Memory System for NeuralSync2
Provides unified memory substrate that persists across all CLI tool sessions
"""

import os
import json
import time
import threading
import atexit
import logging
import uuid
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
from dataclasses import dataclass, asdict

from .storage import EnhancedStorage, upsert_item, recall, get_persona, put_persona
from .crdt import ByzantineCRDT, AdvancedVersion
from .memory_manager import get_memory_manager
from .utils import now_ms

logger = logging.getLogger(__name__)

@dataclass
class MemorySession:
    """Represents a CLI tool session"""
    session_id: str
    tool_name: str
    pid: int
    start_time: int
    last_active: int
    active: bool = True

@dataclass 
class CrossSessionMemory:
    """Memory entry that persists across sessions"""
    id: str
    content: Any
    source_session: str
    tool_name: str
    scope: str
    priority: float
    created_at: int
    updated_at: int
    access_count: int = 0
    tags: List[str] = None

class CoreMemoryManager:
    """
    Unified core memory system that persists across all CLI tool sessions.
    Provides intelligent memory merging, conflict resolution, and performance optimization.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize core memory manager"""
        
        # Setup paths
        if base_path is None:
            base_path = os.path.expanduser("~/.neuralsync2/core_memory")
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Core storage
        self.storage = EnhancedStorage(str(self.base_path / "core_memory.db"))
        
        # Session management
        self.sessions: Dict[str, MemorySession] = {}
        self.current_session_id = str(uuid.uuid4())
        self.sessions_file = self.base_path / "active_sessions.json"
        
        # Cross-session memory
        self.cross_session_memories: Dict[str, CrossSessionMemory] = {}
        self.memory_file = self.base_path / "cross_session_memory.json"
        
        # Inter-process communication
        self.ipc_dir = self.base_path / "ipc"
        self.ipc_dir.mkdir(exist_ok=True)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'cross_session_merges': 0,
            'conflict_resolutions': 0,
            'total_sessions': 0
        }
        
        # Initialize
        self._load_sessions()
        self._load_cross_session_memories()
        self._register_current_session()
        self._start_background_tasks()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        
    def _register_current_session(self):
        """Register current CLI tool session"""
        try:
            import sys
            tool_name = sys.argv[0] if sys.argv else "unknown"
            
            session = MemorySession(
                session_id=self.current_session_id,
                tool_name=os.path.basename(tool_name),
                pid=os.getpid(),
                start_time=now_ms(),
                last_active=now_ms()
            )
            
            with self.lock:
                self.sessions[self.current_session_id] = session
                self.stats['total_sessions'] += 1
                
            self._save_sessions()
            logger.info(f"Registered session {self.current_session_id} for {session.tool_name}")
            
        except Exception as e:
            logger.error(f"Failed to register session: {e}")
            
    def _load_sessions(self):
        """Load active sessions from disk"""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    data = json.load(f)
                    
                # Reconstruct sessions, marking stale ones as inactive
                current_time = now_ms()
                for session_id, session_data in data.items():
                    session = MemorySession(**session_data)
                    
                    # Check if session is still active (5 minute timeout)
                    if current_time - session.last_active > 300000:
                        session.active = False
                        
                    self.sessions[session_id] = session
                    
        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")
            
    def _save_sessions(self):
        """Save active sessions to disk"""
        try:
            with self.lock:
                # Only save active sessions
                active_sessions = {
                    sid: asdict(session) for sid, session in self.sessions.items()
                    if session.active
                }
                
            with open(self.sessions_file, 'w') as f:
                json.dump(active_sessions, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
            
    def _load_cross_session_memories(self):
        """Load cross-session memories from disk"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    
                for memory_id, memory_data in data.items():
                    memory = CrossSessionMemory(**memory_data)
                    self.cross_session_memories[memory_id] = memory
                    
                logger.info(f"Loaded {len(self.cross_session_memories)} cross-session memories")
                
        except Exception as e:
            logger.warning(f"Failed to load cross-session memories: {e}")
            
    def _save_cross_session_memories(self):
        """Save cross-session memories to disk"""
        try:
            with self.lock:
                memory_data = {
                    mid: asdict(memory) for mid, memory in self.cross_session_memories.items()
                }
                
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save cross-session memories: {e}")
            
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        def maintenance_worker():
            while True:
                try:
                    time.sleep(30)  # Run every 30 seconds
                    
                    # Update session activity
                    self._update_session_activity()
                    
                    # Cleanup stale sessions
                    self._cleanup_stale_sessions()
                    
                    # Merge memories from other sessions
                    self._merge_session_memories()
                    
                    # Optimize storage performance
                    if time.time() % 300 < 30:  # Every 5 minutes
                        self.storage.optimize_performance()
                        
                except Exception as e:
                    logger.error(f"Background maintenance error: {e}")
                    
        maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        maintenance_thread.start()
        
    def _update_session_activity(self):
        """Update current session activity timestamp"""
        try:
            with self.lock:
                if self.current_session_id in self.sessions:
                    self.sessions[self.current_session_id].last_active = now_ms()
                    
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
            
    def _cleanup_stale_sessions(self):
        """Remove stale sessions"""
        try:
            current_time = now_ms()
            stale_sessions = []
            
            with self.lock:
                for session_id, session in self.sessions.items():
                    # Mark sessions inactive after 5 minutes
                    if current_time - session.last_active > 300000:
                        session.active = False
                        
                    # Remove sessions after 1 hour
                    if current_time - session.last_active > 3600000:
                        stale_sessions.append(session_id)
                        
                for session_id in stale_sessions:
                    del self.sessions[session_id]
                    
            if stale_sessions:
                self._save_sessions()
                logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")
                
        except Exception as e:
            logger.error(f"Failed to cleanup stale sessions: {e}")
            
    def _merge_session_memories(self):
        """Merge memories from other active sessions"""
        try:
            # Look for memory files from other sessions
            for ipc_file in self.ipc_dir.glob("session_*.json"):
                if ipc_file.stem.endswith(self.current_session_id):
                    continue  # Skip our own file
                    
                try:
                    with open(ipc_file, 'r') as f:
                        other_memories = json.load(f)
                        
                    # Merge memories
                    merged_count = 0
                    with self.lock:
                        for memory_data in other_memories:
                            memory = CrossSessionMemory(**memory_data)
                            
                            if memory.id not in self.cross_session_memories:
                                self.cross_session_memories[memory.id] = memory
                                merged_count += 1
                                
                            elif (memory.updated_at > 
                                  self.cross_session_memories[memory.id].updated_at):
                                # Update with newer version
                                self.cross_session_memories[memory.id] = memory
                                merged_count += 1
                                self.stats['conflict_resolutions'] += 1
                                
                    if merged_count > 0:
                        self.stats['cross_session_merges'] += 1
                        self._save_cross_session_memories()
                        logger.info(f"Merged {merged_count} memories from {ipc_file.name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to merge from {ipc_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to merge session memories: {e}")
            
    def remember(self, content: Any, scope: str = "global", 
                tool_name: Optional[str] = None, priority: float = 0.5,
                tags: Optional[List[str]] = None) -> str:
        """Store memory that persists across sessions"""
        
        memory_id = str(uuid.uuid4())
        current_time = now_ms()
        
        if tool_name is None:
            tool_name = self.sessions.get(self.current_session_id, {}).get('tool_name', 'unknown')
            
        # Create cross-session memory
        memory = CrossSessionMemory(
            id=memory_id,
            content=content,
            source_session=self.current_session_id,
            tool_name=tool_name,
            scope=scope,
            priority=priority,
            created_at=current_time,
            updated_at=current_time,
            tags=tags or []
        )
        
        with self.lock:
            self.cross_session_memories[memory_id] = memory
            
        # Store in enhanced storage for persistence
        storage_item = {
            'id': memory_id,
            'kind': 'cross_session_memory',
            'text': str(content),
            'scope': scope,
            'tool': tool_name,
            'tags': json.dumps(tags or []),
            'confidence': priority,
            'benefit': priority,
            'consistency': 0.5,
            'source': f"session_{self.current_session_id}",
            'meta': json.dumps({
                'session_id': self.current_session_id,
                'priority': priority
            })
        }
        
        upsert_item(self.storage, storage_item)
        
        # Save to disk and share with other sessions
        self._save_cross_session_memories()
        self._share_memory_with_sessions([memory])
        
        logger.info(f"Stored cross-session memory {memory_id} with scope {scope}")
        return memory_id
        
    def recall_across_sessions(self, query: str, scope: str = "any",
                             top_k: int = 10, include_current: bool = True) -> List[Dict[str, Any]]:
        """Recall memories across all sessions"""
        
        results = []
        
        # Search cross-session memories first
        with self.lock:
            matching_memories = []
            for memory in self.cross_session_memories.values():
                if scope != "any" and memory.scope != scope:
                    continue
                    
                # Simple text matching (could be enhanced with embeddings)
                content_str = str(memory.content).lower()
                if query.lower() in content_str:
                    memory.access_count += 1
                    memory.updated_at = now_ms()
                    matching_memories.append(memory)
                    self.stats['memory_hits'] += 1
                else:
                    self.stats['memory_misses'] += 1
                    
            # Sort by priority and recency
            matching_memories.sort(key=lambda m: (m.priority, m.updated_at), reverse=True)
            
            # Convert to result format
            for memory in matching_memories[:top_k]:
                results.append({
                    'id': memory.id,
                    'content': memory.content,
                    'scope': memory.scope,
                    'tool_name': memory.tool_name,
                    'priority': memory.priority,
                    'created_at': memory.created_at,
                    'updated_at': memory.updated_at,
                    'access_count': memory.access_count,
                    'tags': memory.tags,
                    'source': 'cross_session'
                })
                
        # Supplement with storage-based recall if needed
        if len(results) < top_k:
            storage_results = recall(self.storage, query, top_k - len(results), scope, None)
            for result in storage_results:
                results.append({
                    **result,
                    'source': 'storage'
                })
                
        return results[:top_k]
        
    def _share_memory_with_sessions(self, memories: List[CrossSessionMemory]):
        """Share memories with other active sessions via IPC"""
        try:
            # Create IPC file for our session
            ipc_file = self.ipc_dir / f"session_{self.current_session_id}.json"
            
            memory_data = [asdict(memory) for memory in memories]
            
            with open(ipc_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to share memories via IPC: {e}")
            
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about all active sessions"""
        with self.lock:
            return {
                'current_session': self.current_session_id,
                'active_sessions': [
                    asdict(session) for session in self.sessions.values()
                    if session.active
                ],
                'total_sessions': len(self.sessions),
                'cross_session_memories': len(self.cross_session_memories)
            }
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        storage_stats = self.storage.get_performance_stats()
        
        return {
            'core_memory': self.stats,
            'storage': storage_stats,
            'sessions': {
                'active': sum(1 for s in self.sessions.values() if s.active),
                'total': len(self.sessions),
                'current': self.current_session_id
            },
            'memories': {
                'cross_session': len(self.cross_session_memories),
                'total_access_count': sum(m.access_count for m in self.cross_session_memories.values())
            }
        }
        
    def merge_with_other_instance(self, other_instance: 'CoreMemoryManager') -> Dict[str, Any]:
        """Merge memories with another CoreMemoryManager instance"""
        try:
            # Merge enhanced storage
            merge_stats = self.storage.merge_memories(other_instance.storage)
            
            # Merge cross-session memories
            merged_memories = 0
            with self.lock:
                for memory_id, memory in other_instance.cross_session_memories.items():
                    if memory_id not in self.cross_session_memories:
                        self.cross_session_memories[memory_id] = memory
                        merged_memories += 1
                    elif memory.updated_at > self.cross_session_memories[memory_id].updated_at:
                        self.cross_session_memories[memory_id] = memory
                        merged_memories += 1
                        
            # Save merged state
            self._save_cross_session_memories()
            
            merge_stats['cross_session_memories'] = merged_memories
            logger.info(f"Merged with other instance: {merge_stats}")
            
            return merge_stats
            
        except Exception as e:
            logger.error(f"Failed to merge with other instance: {e}")
            return {'error': str(e)}
            
    def cleanup(self):
        """Cleanup resources and mark session as inactive"""
        try:
            # Mark current session as inactive
            with self.lock:
                if self.current_session_id in self.sessions:
                    self.sessions[self.current_session_id].active = False
                    
            self._save_sessions()
            self._save_cross_session_memories()
            
            # Remove IPC file
            ipc_file = self.ipc_dir / f"session_{self.current_session_id}.json"
            if ipc_file.exists():
                ipc_file.unlink()
                
            # Close storage
            self.storage.close()
            
            logger.info(f"Core memory cleanup completed for session {self.current_session_id}")
            
        except Exception as e:
            logger.error(f"Core memory cleanup error: {e}")


# Global core memory instance
_global_core_memory: Optional[CoreMemoryManager] = None

def get_core_memory() -> CoreMemoryManager:
    """Get global core memory manager instance"""
    global _global_core_memory
    if _global_core_memory is None:
        _global_core_memory = CoreMemoryManager()
    return _global_core_memory

def init_core_memory(base_path: Optional[str] = None) -> CoreMemoryManager:
    """Initialize core memory with custom path"""
    global _global_core_memory
    _global_core_memory = CoreMemoryManager(base_path)
    return _global_core_memory


if __name__ == "__main__":
    # Test the core memory system
    import asyncio
    
    async def test_core_memory():
        """Test core memory functionality"""
        
        core_mem = get_core_memory()
        
        # Store some test memories
        mem1 = core_mem.remember("Test memory 1", scope="test", priority=0.8)
        mem2 = core_mem.remember("Important information about AI", scope="ai", priority=0.9)
        mem3 = core_mem.remember("Configuration details", scope="config", priority=0.6)
        
        print(f"Stored memories: {mem1}, {mem2}, {mem3}")
        
        # Recall memories
        results = core_mem.recall_across_sessions("AI", top_k=5)
        print(f"Recall results for 'AI': {len(results)} items")
        for result in results:
            print(f"  - {result['content']} (priority: {result.get('priority', 0)})")
            
        # Get session info
        session_info = core_mem.get_session_info()
        print(f"Session info: {session_info}")
        
        # Get performance stats
        stats = core_mem.get_performance_stats()
        print(f"Performance stats: {stats}")
        
    asyncio.run(test_core_memory())