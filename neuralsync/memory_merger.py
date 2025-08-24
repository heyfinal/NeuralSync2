"""
Intelligent Memory Merging System for Concurrent CLI Tool Updates
Handles real-time memory synchronization across multiple CLI tools with conflict resolution
"""

import os
import time
import json
import threading
import logging
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path

from .crdt import ByzantineCRDT, AdvancedVersion
from .utils import now_ms

logger = logging.getLogger(__name__)

@dataclass
class MemoryUpdate:
    """Represents a memory update from a CLI tool"""
    update_id: str
    session_id: str
    tool_name: str
    memory_id: str
    operation: str  # 'create', 'update', 'delete'
    content: Any
    version: Dict[str, Any]  # Vector clock version
    timestamp: int
    checksum: str
    metadata: Dict[str, Any] = None

@dataclass
class MergeConflict:
    """Represents a conflict during memory merging"""
    conflict_id: str
    memory_id: str
    conflicting_updates: List[MemoryUpdate]
    resolution_strategy: str
    resolved_content: Any
    confidence: float
    timestamp: int

class ConflictResolver:
    """Advanced conflict resolution strategies"""
    
    def __init__(self):
        self.resolution_stats = {
            'last_writer_wins': 0,
            'content_merge': 0,
            'semantic_merge': 0,
            'manual_required': 0
        }
        
    def resolve_conflict(self, updates: List[MemoryUpdate]) -> MergeConflict:
        """Resolve conflict between multiple memory updates"""
        
        if len(updates) < 2:
            raise ValueError("Cannot resolve conflict with less than 2 updates")
            
        conflict_id = hashlib.md5(
            ''.join(update.update_id for update in updates).encode()
        ).hexdigest()
        
        # Determine resolution strategy based on content type and update patterns
        strategy, resolved_content, confidence = self._determine_resolution(updates)
        
        self.resolution_stats[strategy] += 1
        
        return MergeConflict(
            conflict_id=conflict_id,
            memory_id=updates[0].memory_id,
            conflicting_updates=updates,
            resolution_strategy=strategy,
            resolved_content=resolved_content,
            confidence=confidence,
            timestamp=now_ms()
        )
        
    def _determine_resolution(self, updates: List[MemoryUpdate]) -> Tuple[str, Any, float]:
        """Determine the best resolution strategy and apply it"""
        
        # Sort updates by timestamp
        sorted_updates = sorted(updates, key=lambda u: u.timestamp)
        latest_update = sorted_updates[-1]
        
        # Check if all updates have the same content
        contents = [update.content for update in updates]
        if len(set(str(c) for c in contents)) == 1:
            return 'last_writer_wins', latest_update.content, 1.0
            
        # Analyze content types
        content_types = [type(update.content) for update in updates]
        
        if all(isinstance(content, str) for content in contents):
            return self._resolve_string_conflict(updates)
        elif all(isinstance(content, dict) for content in contents):
            return self._resolve_dict_conflict(updates)
        elif all(isinstance(content, list) for content in contents):
            return self._resolve_list_conflict(updates)
        else:
            # Mixed types or complex objects - use last writer wins
            return 'last_writer_wins', latest_update.content, 0.7
            
    def _resolve_string_conflict(self, updates: List[MemoryUpdate]) -> Tuple[str, Any, float]:
        """Resolve conflicts between string contents"""
        
        contents = [update.content for update in updates]
        
        # Check for append-only pattern
        if self._is_append_pattern(contents):
            # Merge all unique additions
            merged = self._merge_append_strings(contents)
            return 'content_merge', merged, 0.9
            
        # Check for semantic similarity
        similarity = self._calculate_semantic_similarity(contents)
        if similarity > 0.8:
            # Use semantic merge
            merged = self._semantic_merge_strings(contents)
            return 'semantic_merge', merged, 0.85
            
        # Default to last writer wins
        latest = sorted(updates, key=lambda u: u.timestamp)[-1]
        return 'last_writer_wins', latest.content, 0.6
        
    def _resolve_dict_conflict(self, updates: List[MemoryUpdate]) -> Tuple[str, Any, float]:
        """Resolve conflicts between dictionary contents"""
        
        # Deep merge dictionaries
        merged_dict = {}
        all_keys = set()
        
        for update in updates:
            if isinstance(update.content, dict):
                all_keys.update(update.content.keys())
                
        # Merge each key independently
        for key in all_keys:
            key_updates = []
            for update in updates:
                if isinstance(update.content, dict) and key in update.content:
                    key_update = MemoryUpdate(
                        update_id=f"{update.update_id}_{key}",
                        session_id=update.session_id,
                        tool_name=update.tool_name,
                        memory_id=f"{update.memory_id}_{key}",
                        operation=update.operation,
                        content=update.content[key],
                        version=update.version,
                        timestamp=update.timestamp,
                        checksum=update.checksum
                    )
                    key_updates.append(key_update)
                    
            if key_updates:
                if len(key_updates) == 1:
                    merged_dict[key] = key_updates[0].content
                else:
                    # Recursively resolve conflicts for this key
                    _, resolved_value, _ = self._determine_resolution(key_updates)
                    merged_dict[key] = resolved_value
                    
        return 'content_merge', merged_dict, 0.85
        
    def _resolve_list_conflict(self, updates: List[MemoryUpdate]) -> Tuple[str, Any, float]:
        """Resolve conflicts between list contents"""
        
        # Union merge for lists
        merged_list = []
        seen_items = set()
        
        for update in sorted(updates, key=lambda u: u.timestamp):
            if isinstance(update.content, list):
                for item in update.content:
                    item_str = str(item)
                    if item_str not in seen_items:
                        merged_list.append(item)
                        seen_items.add(item_str)
                        
        return 'content_merge', merged_list, 0.8
        
    def _is_append_pattern(self, contents: List[str]) -> bool:
        """Check if strings follow an append-only pattern"""
        if len(contents) < 2:
            return False
            
        sorted_contents = sorted(contents, key=len)
        
        for i in range(1, len(sorted_contents)):
            if not sorted_contents[i].startswith(sorted_contents[i-1]):
                return False
                
        return True
        
    def _merge_append_strings(self, contents: List[str]) -> str:
        """Merge strings that follow append pattern"""
        return max(contents, key=len)
        
    def _calculate_semantic_similarity(self, contents: List[str]) -> float:
        """Calculate semantic similarity between string contents"""
        # Simplified similarity calculation
        # In practice, this could use embeddings or NLP techniques
        
        if len(contents) < 2:
            return 1.0
            
        # Use Jaccard similarity on words
        word_sets = [set(content.lower().split()) for content in contents]
        
        total_similarity = 0
        comparisons = 0
        
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                similarity = intersection / union if union > 0 else 0
                total_similarity += similarity
                comparisons += 1
                
        return total_similarity / comparisons if comparisons > 0 else 0
        
    def _semantic_merge_strings(self, contents: List[str]) -> str:
        """Semantically merge string contents"""
        # Simple implementation - combine unique sentences
        sentences = set()
        
        for content in contents:
            # Split by sentences (simplified)
            content_sentences = content.replace('!', '.').replace('?', '.').split('.')
            for sentence in content_sentences:
                sentence = sentence.strip()
                if sentence:
                    sentences.add(sentence)
                    
        return '. '.join(sorted(sentences)) + '.'


class IntelligentMemoryMerger:
    """
    Intelligent memory merger that handles concurrent updates from multiple CLI tools
    """
    
    def __init__(self, core_memory_manager):
        self.core_memory = core_memory_manager
        self.conflict_resolver = ConflictResolver()
        
        # Pending updates queue
        self.pending_updates: Dict[str, List[MemoryUpdate]] = defaultdict(list)
        self.update_lock = threading.RLock()
        
        # Merge statistics
        self.merge_stats = {
            'total_updates': 0,
            'successful_merges': 0,
            'conflicts_resolved': 0,
            'failed_merges': 0,
            'avg_merge_time_ms': 0.0
        }
        
        # Real-time update monitoring
        self.update_monitor = UpdateMonitor(self)
        self.update_monitor.start()
        
        # Resolved conflicts cache
        self.resolved_conflicts: Dict[str, MergeConflict] = {}
        self.max_conflicts_cache = 1000
        
    def queue_update(self, update: MemoryUpdate) -> bool:
        """Queue a memory update for processing"""
        try:
            # Validate update
            if not self._validate_update(update):
                logger.warning(f"Invalid update rejected: {update.update_id}")
                return False
                
            with self.update_lock:
                self.pending_updates[update.memory_id].append(update)
                self.merge_stats['total_updates'] += 1
                
            logger.debug(f"Queued update {update.update_id} for memory {update.memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue update {update.update_id}: {e}")
            return False
            
    def process_updates(self, memory_id: Optional[str] = None) -> Dict[str, Any]:
        """Process pending updates for specific memory or all memories"""
        start_time = time.time()
        processed_count = 0
        conflicts_resolved = 0
        
        try:
            with self.update_lock:
                memory_ids = [memory_id] if memory_id else list(self.pending_updates.keys())
                
                for mid in memory_ids:
                    if mid not in self.pending_updates:
                        continue
                        
                    updates = self.pending_updates[mid]
                    if not updates:
                        continue
                        
                    # Process updates for this memory
                    result = self._process_memory_updates(mid, updates)
                    
                    if result['success']:
                        # Clear processed updates
                        del self.pending_updates[mid]
                        processed_count += len(updates)
                        
                        if result.get('conflict_resolved'):
                            conflicts_resolved += 1
                    else:
                        logger.error(f"Failed to process updates for memory {mid}: {result.get('error')}")
                        
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.merge_stats['successful_merges'] += processed_count
            self.merge_stats['conflicts_resolved'] += conflicts_resolved
            self.merge_stats['avg_merge_time_ms'] = (
                0.9 * self.merge_stats['avg_merge_time_ms'] + 
                0.1 * processing_time
            )
            
            return {
                'processed_updates': processed_count,
                'conflicts_resolved': conflicts_resolved,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Update processing failed: {e}")
            self.merge_stats['failed_merges'] += 1
            return {'error': str(e)}
            
    def _process_memory_updates(self, memory_id: str, updates: List[MemoryUpdate]) -> Dict[str, Any]:
        """Process all updates for a specific memory"""
        
        if len(updates) == 1:
            # Single update - no conflict
            update = updates[0]
            success = self._apply_single_update(update)
            return {'success': success, 'conflict_resolved': False}
            
        # Multiple updates - potential conflict
        return self._resolve_and_apply_updates(memory_id, updates)
        
    def _apply_single_update(self, update: MemoryUpdate) -> bool:
        """Apply a single update without conflicts"""
        try:
            if update.operation == 'create' or update.operation == 'update':
                # Store/update memory
                self.core_memory.remember(
                    content=update.content,
                    scope=update.metadata.get('scope', 'global'),
                    tool_name=update.tool_name,
                    priority=update.metadata.get('priority', 0.5),
                    tags=update.metadata.get('tags')
                )
                
            elif update.operation == 'delete':
                # Mark memory as deleted (handled by CRDT)
                if hasattr(self.core_memory.storage.crdt, 'delete'):
                    self.core_memory.storage.crdt.delete(update.memory_id)
                    
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply single update {update.update_id}: {e}")
            return False
            
    def _resolve_and_apply_updates(self, memory_id: str, updates: List[MemoryUpdate]) -> Dict[str, Any]:
        """Resolve conflicts and apply merged updates"""
        try:
            # Group updates by operation type
            creates = [u for u in updates if u.operation == 'create']
            updates_ops = [u for u in updates if u.operation == 'update'] 
            deletes = [u for u in updates if u.operation == 'delete']
            
            # Handle deletes first
            if deletes:
                # If any delete operation exists, the memory should be deleted
                latest_delete = max(deletes, key=lambda u: u.timestamp)
                if hasattr(self.core_memory.storage.crdt, 'delete'):
                    self.core_memory.storage.crdt.delete(memory_id)
                return {'success': True, 'conflict_resolved': True}
                
            # Handle creates and updates
            content_updates = creates + updates_ops
            if not content_updates:
                return {'success': True, 'conflict_resolved': False}
                
            # Resolve conflicts
            conflict = self.conflict_resolver.resolve_conflict(content_updates)
            
            # Cache resolved conflict
            self.resolved_conflicts[conflict.conflict_id] = conflict
            
            # Trim conflicts cache if needed
            if len(self.resolved_conflicts) > self.max_conflicts_cache:
                # Remove oldest conflicts
                sorted_conflicts = sorted(
                    self.resolved_conflicts.items(),
                    key=lambda x: x[1].timestamp
                )
                for conflict_id, _ in sorted_conflicts[:100]:  # Remove oldest 100
                    del self.resolved_conflicts[conflict_id]
                    
            # Apply resolved content
            latest_update = max(content_updates, key=lambda u: u.timestamp)
            success = self.core_memory.remember(
                content=conflict.resolved_content,
                scope=latest_update.metadata.get('scope', 'global'),
                tool_name=f"merged_{latest_update.tool_name}",
                priority=latest_update.metadata.get('priority', 0.5),
                tags=latest_update.metadata.get('tags')
            )
            
            logger.info(f"Resolved conflict {conflict.conflict_id} using {conflict.resolution_strategy}")
            
            return {
                'success': bool(success),
                'conflict_resolved': True,
                'conflict': asdict(conflict)
            }
            
        except Exception as e:
            logger.error(f"Failed to resolve conflicts for memory {memory_id}: {e}")
            return {'success': False, 'error': str(e)}
            
    def _validate_update(self, update: MemoryUpdate) -> bool:
        """Validate update integrity and structure"""
        
        # Check required fields
        required_fields = ['update_id', 'session_id', 'memory_id', 'operation', 'timestamp']
        for field in required_fields:
            if not hasattr(update, field) or getattr(update, field) is None:
                return False
                
        # Validate operation type
        if update.operation not in ['create', 'update', 'delete']:
            return False
            
        # Validate timestamp (not too far in future or past)
        current_time = now_ms()
        if abs(update.timestamp - current_time) > 86400000:  # 24 hours
            return False
            
        # Validate checksum if provided
        if update.checksum:
            expected_checksum = self._calculate_update_checksum(update)
            if update.checksum != expected_checksum:
                logger.warning(f"Checksum mismatch for update {update.update_id}")
                return False
                
        return True
        
    def _calculate_update_checksum(self, update: MemoryUpdate) -> str:
        """Calculate checksum for update integrity"""
        content_str = f"{update.memory_id}{update.operation}{update.content}{update.timestamp}"
        return hashlib.md5(content_str.encode()).hexdigest()
        
    def get_merge_statistics(self) -> Dict[str, Any]:
        """Get comprehensive merging statistics"""
        return {
            'merge_stats': self.merge_stats,
            'resolution_stats': self.conflict_resolver.resolution_stats,
            'pending_updates': {
                memory_id: len(updates) 
                for memory_id, updates in self.pending_updates.items()
            },
            'resolved_conflicts': len(self.resolved_conflicts),
            'monitor_status': self.update_monitor.get_status()
        }
        
    def force_merge_all(self) -> Dict[str, Any]:
        """Force processing of all pending updates"""
        return self.process_updates()
        
    def get_conflict_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get history of resolved conflicts"""
        conflicts = sorted(
            self.resolved_conflicts.values(),
            key=lambda c: c.timestamp,
            reverse=True
        )
        
        return [asdict(conflict) for conflict in conflicts[:limit]]


class UpdateMonitor:
    """Monitors for real-time memory updates from other CLI tools"""
    
    def __init__(self, merger: IntelligentMemoryMerger):
        self.merger = merger
        self.monitor_thread = None
        self.running = False
        self.monitor_interval = 1.0  # Check every second
        
    def start(self):
        """Start monitoring for updates"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check for new updates from other sessions
                self._check_for_updates()
                
                # Process any pending updates
                if self.merger.pending_updates:
                    self.merger.process_updates()
                    
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Update monitoring error: {e}")
                time.sleep(self.monitor_interval * 2)  # Back off on error
                
    def _check_for_updates(self):
        """Check for new updates from other CLI tool sessions"""
        try:
            # This would implement actual monitoring logic
            # For now, it's a placeholder that would:
            # 1. Watch IPC files for new updates
            # 2. Parse update messages
            # 3. Queue them for processing
            pass
            
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status"""
        return {
            'running': self.running,
            'monitor_interval': self.monitor_interval,
            'thread_alive': self.monitor_thread.is_alive() if self.monitor_thread else False
        }


def create_memory_update(session_id: str, tool_name: str, memory_id: str,
                        operation: str, content: Any, metadata: Optional[Dict] = None) -> MemoryUpdate:
    """Helper function to create a properly formatted memory update"""
    
    import uuid
    
    update = MemoryUpdate(
        update_id=str(uuid.uuid4()),
        session_id=session_id,
        tool_name=tool_name,
        memory_id=memory_id,
        operation=operation,
        content=content,
        version={session_id: now_ms()},  # Simple version vector
        timestamp=now_ms(),
        checksum="",  # Will be calculated
        metadata=metadata or {}
    )
    
    # Calculate checksum
    content_str = f"{memory_id}{operation}{content}{update.timestamp}"
    update.checksum = hashlib.md5(content_str.encode()).hexdigest()
    
    return update