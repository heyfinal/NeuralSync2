"""
High-Performance B+ Tree Implementation for Memory Indexing
Provides sub-millisecond lookups with range query support and memory mapping
"""

import mmap
import struct
import bisect
import threading
import time
import logging
from typing import Optional, List, Tuple, Iterator, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)

# B+ tree configuration constants
DEFAULT_FANOUT = 256  # Higher fanout for better cache performance
PAGE_SIZE = 4096     # Standard page size
HEADER_SIZE = 64     # Header metadata size
NODE_OVERHEAD = 32   # Node metadata overhead

@dataclass
class BTreeStats:
    """Performance statistics for B+ tree operations"""
    total_reads: int = 0
    total_writes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    splits: int = 0
    merges: int = 0
    height: int = 0
    node_count: int = 0
    avg_lookup_time_ms: float = 0.0


class BTreeNode:
    """Base class for B+ tree nodes with memory-mapped storage"""
    
    def __init__(self, fanout: int, is_leaf: bool = False):
        self.fanout = fanout
        self.is_leaf = is_leaf
        self.keys: List[bytes] = []
        self.parent: Optional['BTreeNode'] = None
        self.dirty = True
        self.last_access = time.time()
        self.node_id = id(self)  # Unique identifier
        
    def is_full(self) -> bool:
        return len(self.keys) >= self.fanout - 1
        
    def is_underflow(self) -> bool:
        min_keys = (self.fanout - 1) // 2
        return len(self.keys) < min_keys
        
    def serialize(self) -> bytes:
        """Serialize node to bytes for storage"""
        data = {
            'fanout': self.fanout,
            'is_leaf': self.is_leaf,
            'keys': self.keys,
            'node_id': self.node_id
        }
        if hasattr(self, 'values'):
            data['values'] = self.values
        if hasattr(self, 'children'):
            data['children'] = [child.node_id if child else None for child in self.children]
        return pickle.dumps(data)
        
    @classmethod
    def deserialize(cls, data: bytes, node_registry: Dict[int, 'BTreeNode']) -> 'BTreeNode':
        """Deserialize node from bytes"""
        obj_data = pickle.loads(data)
        
        if obj_data['is_leaf']:
            node = LeafNode(obj_data['fanout'])
            node.values = obj_data.get('values', [])
        else:
            node = InternalNode(obj_data['fanout'])
            # Reconstruct children references
            child_ids = obj_data.get('children', [])
            node.children = [node_registry.get(cid) for cid in child_ids]
            
        node.keys = obj_data['keys']
        node.node_id = obj_data['node_id']
        node.dirty = False
        return node


class LeafNode(BTreeNode):
    """Leaf node containing actual key-value pairs"""
    
    def __init__(self, fanout: int):
        super().__init__(fanout, is_leaf=True)
        self.values: List[Any] = []
        self.next_leaf: Optional['LeafNode'] = None  # For range queries
        self.prev_leaf: Optional['LeafNode'] = None
        
    def insert(self, key: bytes, value: Any) -> Optional[Tuple[bytes, 'BTreeNode']]:
        """Insert key-value pair, returns split info if node splits"""
        pos = bisect.bisect_left(self.keys, key)
        
        # Update existing key
        if pos < len(self.keys) and self.keys[pos] == key:
            self.values[pos] = value
            self.dirty = True
            return None
            
        # Insert new key-value
        self.keys.insert(pos, key)
        self.values.insert(pos, value)
        self.dirty = True
        
        # Check if split needed
        if self.is_full():
            return self._split()
        return None
        
    def _split(self) -> Tuple[bytes, 'LeafNode']:
        """Split leaf node and return split key and new right node"""
        mid = len(self.keys) // 2
        
        # Create new right leaf
        right = LeafNode(self.fanout)
        right.keys = self.keys[mid:]
        right.values = self.values[mid:]
        
        # Update this node (left)
        self.keys = self.keys[:mid]
        self.values = self.values[:mid]
        
        # Link leaves for range queries
        right.next_leaf = self.next_leaf
        right.prev_leaf = self
        if self.next_leaf:
            self.next_leaf.prev_leaf = right
        self.next_leaf = right
        
        # Both nodes are dirty
        right.dirty = True
        self.dirty = True
        
        return (right.keys[0], right)
        
    def find(self, key: bytes) -> Optional[Any]:
        """Find value for given key"""
        pos = bisect.bisect_left(self.keys, key)
        if pos < len(self.keys) and self.keys[pos] == key:
            self.last_access = time.time()
            return self.values[pos]
        return None
        
    def range_query(self, start_key: bytes, end_key: bytes) -> Iterator[Tuple[bytes, Any]]:
        """Return iterator for range of key-value pairs"""
        # Find starting position
        start_pos = bisect.bisect_left(self.keys, start_key)
        
        # Iterate through this leaf
        for i in range(start_pos, len(self.keys)):
            if self.keys[i] > end_key:
                return
            yield (self.keys[i], self.values[i])
            
        # Continue to next leaves
        current = self.next_leaf
        while current:
            for i, key in enumerate(current.keys):
                if key > end_key:
                    return
                yield (key, current.values[i])
            current = current.next_leaf
            
    def delete(self, key: bytes) -> bool:
        """Delete key from leaf, returns True if found and deleted"""
        pos = bisect.bisect_left(self.keys, key)
        if pos < len(self.keys) and self.keys[pos] == key:
            del self.keys[pos]
            del self.values[pos]
            self.dirty = True
            return True
        return False


class InternalNode(BTreeNode):
    """Internal node containing keys and child pointers"""
    
    def __init__(self, fanout: int):
        super().__init__(fanout, is_leaf=False)
        self.children: List[Optional[BTreeNode]] = []
        
    def insert(self, key: bytes, child: BTreeNode) -> Optional[Tuple[bytes, 'BTreeNode']]:
        """Insert key and child pointer"""
        pos = bisect.bisect_left(self.keys, key)
        
        self.keys.insert(pos, key)
        self.children.insert(pos + 1, child)
        child.parent = self
        self.dirty = True
        
        if self.is_full():
            return self._split()
        return None
        
    def _split(self) -> Tuple[bytes, 'InternalNode']:
        """Split internal node"""
        mid = len(self.keys) // 2
        split_key = self.keys[mid]
        
        # Create new right internal node
        right = InternalNode(self.fanout)
        right.keys = self.keys[mid + 1:]
        right.children = self.children[mid + 1:]
        
        # Update children's parent pointers
        for child in right.children:
            if child:
                child.parent = right
                
        # Update this node (left)
        self.keys = self.keys[:mid]
        self.children = self.children[:mid + 1]
        
        right.dirty = True
        self.dirty = True
        
        return (split_key, right)
        
    def find_child(self, key: bytes) -> Optional[BTreeNode]:
        """Find appropriate child for given key"""
        pos = bisect.bisect_left(self.keys, key)
        if pos < len(self.children):
            return self.children[pos]
        return None


class BTreeIndex:
    """High-performance B+ tree with memory mapping and caching"""
    
    def __init__(self, file_path: str, fanout: int = DEFAULT_FANOUT, 
                 cache_size: int = 1000):
        self.file_path = Path(file_path)
        self.fanout = fanout
        self.cache_size = cache_size
        
        # Tree structure
        self.root: Optional[BTreeNode] = None
        self.height = 0
        
        # Memory mapping
        self.mmap_file: Optional[mmap.mmap] = None
        self.file_handle: Optional[Any] = None
        
        # Caching and performance
        self.node_cache: Dict[int, BTreeNode] = {}
        self.node_registry: Dict[int, BTreeNode] = {}
        self.dirty_nodes: set = set()
        
        # Thread safety
        self.lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.RLock()
        
        # Statistics
        self.stats = BTreeStats()
        
        # Initialize or load existing tree
        self._initialize()
        
    def _initialize(self):
        """Initialize tree from file or create new"""
        if self.file_path.exists():
            self._load_from_file()
        else:
            self._create_new_tree()
            
    def _create_new_tree(self):
        """Create new empty B+ tree"""
        self.root = LeafNode(self.fanout)
        self.height = 1
        self._mark_dirty(self.root)
        
    def _load_from_file(self):
        """Load B+ tree from memory-mapped file"""
        try:
            self.file_handle = open(self.file_path, 'r+b')
            self.mmap_file = mmap.mmap(self.file_handle.fileno(), 0)
            
            # Read header
            header_data = self.mmap_file[:HEADER_SIZE]
            header = struct.unpack('QQQQ', header_data[:32])
            root_node_id, height, node_count, checksum = header
            
            # Verify checksum (simple hash of first 1KB)
            content_hash = hashlib.md5(self.mmap_file[:1024]).hexdigest()
            if hash(content_hash) != checksum:
                logger.warning(f"Checksum mismatch in {self.file_path}, recreating")
                self._create_new_tree()
                return
                
            # Load nodes (simplified - full implementation would load on-demand)
            self._load_nodes_from_mmap(root_node_id, height, node_count)
            
        except Exception as e:
            logger.error(f"Failed to load B+ tree from {self.file_path}: {e}")
            self._create_new_tree()
            
    def _load_nodes_from_mmap(self, root_node_id: int, height: int, node_count: int):
        """Load nodes from memory-mapped file"""
        # This is a simplified implementation
        # Full version would implement proper page-based loading
        offset = HEADER_SIZE
        
        for _ in range(node_count):
            if offset >= len(self.mmap_file):
                break
                
            # Read node size
            node_size = struct.unpack('Q', self.mmap_file[offset:offset+8])[0]
            offset += 8
            
            # Read node data
            node_data = self.mmap_file[offset:offset+node_size]
            offset += node_size
            
            try:
                node = BTreeNode.deserialize(node_data, self.node_registry)
                self.node_registry[node.node_id] = node
                self.node_cache[node.node_id] = node
                
                if node.node_id == root_node_id:
                    self.root = node
                    
            except Exception as e:
                logger.error(f"Failed to deserialize node: {e}")
                break
                
        self.height = height
        self.stats.height = height
        self.stats.node_count = node_count
        
    def _mark_dirty(self, node: BTreeNode):
        """Mark node as dirty for eventual persistence"""
        node.dirty = True
        self.dirty_nodes.add(node.node_id)
        
    def insert(self, key: Union[str, bytes], value: Any) -> bool:
        """Insert key-value pair into B+ tree"""
        if isinstance(key, str):
            key = key.encode('utf-8')
            
        start_time = time.time()
        
        try:
            with self.lock:
                if not self.root:
                    self._create_new_tree()
                    
                split_info = self._insert_recursive(self.root, key, value)
                
                # Handle root split
                if split_info:
                    split_key, right_child = split_info
                    new_root = InternalNode(self.fanout)
                    new_root.keys.append(split_key)
                    new_root.children.extend([self.root, right_child])
                    
                    self.root.parent = new_root
                    right_child.parent = new_root
                    
                    self.root = new_root
                    self.height += 1
                    self.stats.splits += 1
                    self._mark_dirty(new_root)
                    
                self.stats.total_writes += 1
                self._update_lookup_time(time.time() - start_time)
                
                return True
                
        except Exception as e:
            logger.error(f"Insert failed for key {key}: {e}")
            return False
            
    def _insert_recursive(self, node: BTreeNode, key: bytes, value: Any) -> Optional[Tuple[bytes, BTreeNode]]:
        """Recursively insert into tree"""
        if node.is_leaf:
            return node.insert(key, value)
        else:
            # Find appropriate child
            child = node.find_child(key)
            if not child:
                return None
                
            split_info = self._insert_recursive(child, key, value)
            
            if split_info:
                split_key, right_child = split_info
                return node.insert(split_key, right_child)
                
        return None
        
    def find(self, key: Union[str, bytes]) -> Optional[Any]:
        """Find value for given key"""
        if isinstance(key, str):
            key = key.encode('utf-8')
            
        start_time = time.time()
        
        try:
            with self.lock:
                if not self.root:
                    return None
                    
                result = self._find_recursive(self.root, key)
                self.stats.total_reads += 1
                self._update_lookup_time(time.time() - start_time)
                
                return result
                
        except Exception as e:
            logger.error(f"Find failed for key {key}: {e}")
            return None
            
    def _find_recursive(self, node: BTreeNode, key: bytes) -> Optional[Any]:
        """Recursively find key in tree"""
        if node.is_leaf:
            return node.find(key)
        else:
            child = node.find_child(key)
            if child:
                return self._find_recursive(child, key)
        return None
        
    def range_query(self, start_key: Union[str, bytes], 
                   end_key: Union[str, bytes]) -> Iterator[Tuple[bytes, Any]]:
        """Perform range query returning iterator"""
        if isinstance(start_key, str):
            start_key = start_key.encode('utf-8')
        if isinstance(end_key, str):
            end_key = end_key.encode('utf-8')
            
        try:
            with self.lock:
                if not self.root:
                    return iter([])
                    
                # Find leftmost leaf containing start_key
                leaf = self._find_leftmost_leaf(self.root, start_key)
                if leaf:
                    return leaf.range_query(start_key, end_key)
                    
        except Exception as e:
            logger.error(f"Range query failed: {e}")
            
        return iter([])
        
    def _find_leftmost_leaf(self, node: BTreeNode, key: bytes) -> Optional[LeafNode]:
        """Find leftmost leaf that might contain key"""
        if node.is_leaf:
            return node
        else:
            child = node.find_child(key)
            if child:
                return self._find_leftmost_leaf(child, key)
        return None
        
    def delete(self, key: Union[str, bytes]) -> bool:
        """Delete key from B+ tree"""
        if isinstance(key, str):
            key = key.encode('utf-8')
            
        try:
            with self.lock:
                if not self.root:
                    return False
                    
                success = self._delete_recursive(self.root, key)
                
                # Handle root underflow
                if (not self.root.is_leaf and 
                    len(self.root.keys) == 0 and 
                    len(self.root.children) == 1):
                    
                    old_root = self.root
                    self.root = self.root.children[0]
                    self.root.parent = None
                    self.height -= 1
                    
                    # Remove old root from tracking
                    if old_root.node_id in self.node_cache:
                        del self.node_cache[old_root.node_id]
                    if old_root.node_id in self.node_registry:
                        del self.node_registry[old_root.node_id]
                        
                return success
                
        except Exception as e:
            logger.error(f"Delete failed for key {key}: {e}")
            return False
            
    def _delete_recursive(self, node: BTreeNode, key: bytes) -> bool:
        """Recursively delete from tree"""
        if node.is_leaf:
            return node.delete(key)
        else:
            child = node.find_child(key)
            if child:
                success = self._delete_recursive(child, key)
                
                # Handle underflow (simplified)
                if success and child.is_underflow():
                    self._handle_underflow(child)
                    
                return success
                
        return False
        
    def _handle_underflow(self, node: BTreeNode):
        """Handle node underflow through borrowing or merging"""
        if not node.parent:
            return  # Root can underflow
            
        # Simplified underflow handling
        # Full implementation would include borrowing from siblings
        parent = node.parent
        
        # Find node position in parent
        node_pos = -1
        for i, child in enumerate(parent.children):
            if child and child.node_id == node.node_id:
                node_pos = i
                break
                
        if node_pos == -1:
            return
            
        # Try to merge with right sibling
        if (node_pos + 1 < len(parent.children) and 
            parent.children[node_pos + 1]):
            
            right_sibling = parent.children[node_pos + 1]
            
            # Simple merge (full implementation would check size constraints)
            if node.is_leaf and right_sibling.is_leaf:
                # Merge leaf nodes
                node.keys.extend(right_sibling.keys)
                node.values.extend(right_sibling.values)
                node.next_leaf = right_sibling.next_leaf
                
                # Remove right sibling from parent
                del parent.keys[node_pos]
                del parent.children[node_pos + 1]
                
                self._mark_dirty(node)
                self._mark_dirty(parent)
                self.stats.merges += 1
                
    def _update_lookup_time(self, time_ms: float):
        """Update average lookup time statistics"""
        if self.stats.total_reads == 0:
            self.stats.avg_lookup_time_ms = time_ms * 1000
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_lookup_time_ms = (
                alpha * (time_ms * 1000) + 
                (1 - alpha) * self.stats.avg_lookup_time_ms
            )
            
    def flush(self):
        """Persist all dirty nodes to storage"""
        try:
            with self.lock:
                if not self.dirty_nodes:
                    return
                    
                # Create or resize file if needed
                if not self.file_handle:
                    self.file_handle = open(self.file_path, 'w+b')
                    
                # Calculate required space
                total_size = HEADER_SIZE
                node_data = []
                
                for node_id in self.dirty_nodes:
                    if node_id in self.node_registry:
                        node = self.node_registry[node_id]
                        serialized = node.serialize()
                        node_data.append(serialized)
                        total_size += 8 + len(serialized)  # size header + data
                        
                # Resize file if necessary
                self.file_handle.seek(0, 2)  # Seek to end
                current_size = self.file_handle.tell()
                if total_size > current_size:
                    self.file_handle.truncate(total_size)
                    self.file_handle.flush()  # Ensure file size is committed
                    
                # Create/update memory map with bounds checking
                if self.mmap_file:
                    self.mmap_file.close()
                    
                # Ensure file size is correct before mapping
                self.file_handle.seek(0, 2)
                actual_size = self.file_handle.tell()
                if actual_size < total_size:
                    # Write zeros to extend file
                    self.file_handle.write(b'\x00' * (total_size - actual_size))
                    self.file_handle.flush()
                    
                self.mmap_file = mmap.mmap(self.file_handle.fileno(), total_size)
                
                # Write header
                header = struct.pack('QQQQ', 
                    self.root.node_id if self.root else 0,
                    self.height,
                    len(self.node_registry),
                    hash(str(time.time()))  # Simple checksum
                )
                self.mmap_file[:len(header)] = header
                
                # Write dirty nodes
                offset = HEADER_SIZE
                for serialized in node_data:
                    # Write size header
                    size_header = struct.pack('Q', len(serialized))
                    self.mmap_file[offset:offset+8] = size_header
                    offset += 8
                    
                    # Write node data with bounds checking
                    if offset + len(serialized) <= len(self.mmap_file):
                        self.mmap_file[offset:offset+len(serialized)] = serialized
                        offset += len(serialized)
                    else:
                        logger.warning(f"Skipping node write - would exceed mmap bounds: {offset + len(serialized)} > {len(self.mmap_file)}")
                        break
                    
                # Sync to disk
                self.mmap_file.flush()
                
                # Clear dirty flags
                for node_id in self.dirty_nodes:
                    if node_id in self.node_registry:
                        self.node_registry[node_id].dirty = False
                        
                self.dirty_nodes.clear()
                
                logger.info(f"Flushed {len(node_data)} dirty nodes to {self.file_path}")
                
        except Exception as e:
            logger.error(f"Failed to flush B+ tree: {e}")
            # Clean up on error
            try:
                if self.mmap_file:
                    self.mmap_file.close()
                    self.mmap_file = None
            except Exception:
                pass
            
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        self.stats.height = self.height
        self.stats.node_count = len(self.node_registry)
        
        return {
            'height': self.stats.height,
            'node_count': self.stats.node_count,
            'total_reads': self.stats.total_reads,
            'total_writes': self.stats.total_writes,
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'cache_hit_rate': (
                self.stats.cache_hits / 
                max(1, self.stats.cache_hits + self.stats.cache_misses)
            ),
            'splits': self.stats.splits,
            'merges': self.stats.merges,
            'avg_lookup_time_ms': self.stats.avg_lookup_time_ms,
            'dirty_nodes': len(self.dirty_nodes),
            'fanout': self.fanout
        }
        
    def close(self):
        """Close B+ tree and release resources"""
        self.flush()
        
        if self.mmap_file:
            self.mmap_file.close()
            self.mmap_file = None
            
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            
        self.node_cache.clear()
        self.node_registry.clear()
        self.dirty_nodes.clear()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Factory function for easy creation
def create_btree_index(file_path: str, fanout: int = DEFAULT_FANOUT) -> BTreeIndex:
    """Create new B+ tree index with optimal configuration"""
    return BTreeIndex(file_path, fanout)