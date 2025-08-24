"""
Zero-Copy Memory Manager with Advanced Allocation Strategies
Provides sub-millisecond memory operations with reference counting
"""

import mmap
import os
import ctypes
import struct
import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class MemoryRegion:
    """Represents an allocated memory region with metadata"""
    offset: int
    size: int
    ref_count: int
    allocated_at: float
    last_accessed: float
    flags: int = 0

class SlabAllocator:
    """High-performance slab allocator for fixed-size allocations"""
    
    def __init__(self, memory_pool: mmap.mmap, slab_sizes: List[int] = None):
        self.memory_pool = memory_pool
        self.slab_sizes = slab_sizes or [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        self.free_lists: Dict[int, List[int]] = defaultdict(list)
        self.allocated_regions: Dict[int, MemoryRegion] = {}
        self.lock = threading.RLock()
        self.pool_size = len(memory_pool)
        self.current_offset = 0
        
        # Initialize slab free lists
        self._initialize_slabs()
        
    def _initialize_slabs(self):
        """Pre-allocate slab regions for common sizes"""
        for slab_size in self.slab_sizes:
            # Allocate 10% of pool for each slab size initially
            slab_count = max(10, (self.pool_size // slab_size) // 20)
            
            for _ in range(slab_count):
                if self.current_offset + slab_size >= self.pool_size:
                    break
                    
                self.free_lists[slab_size].append(self.current_offset)
                self.current_offset += slab_size
                
    def alloc(self, size: int, alignment: int = 8) -> Optional[int]:
        """Allocate memory with specified size and alignment"""
        # Round up to nearest slab size
        slab_size = self._find_slab_size(size)
        
        with self.lock:
            # Try to get from free list first
            if self.free_lists[slab_size]:
                offset = self.free_lists[slab_size].pop()
                region = MemoryRegion(
                    offset=offset,
                    size=slab_size,
                    ref_count=1,
                    allocated_at=time.time(),
                    last_accessed=time.time()
                )
                self.allocated_regions[offset] = region
                return offset
                
            # Allocate new region
            if self.current_offset + slab_size >= self.pool_size:
                # Try garbage collection
                self._garbage_collect()
                
                if self.current_offset + slab_size >= self.pool_size:
                    return None  # Out of memory
                    
            offset = self.current_offset
            self.current_offset += slab_size
            
            region = MemoryRegion(
                offset=offset,
                size=slab_size,
                ref_count=1,
                allocated_at=time.time(),
                last_accessed=time.time()
            )
            self.allocated_regions[offset] = region
            
            return offset
            
    def free(self, offset: int) -> bool:
        """Free allocated memory region"""
        with self.lock:
            if offset not in self.allocated_regions:
                return False
                
            region = self.allocated_regions[offset]
            region.ref_count -= 1
            
            if region.ref_count <= 0:
                # Return to free list
                self.free_lists[region.size].append(offset)
                del self.allocated_regions[offset]
                
            return True
            
    def _find_slab_size(self, size: int) -> int:
        """Find appropriate slab size for allocation"""
        for slab_size in self.slab_sizes:
            if size <= slab_size:
                return slab_size
        # For very large allocations, round up to next power of 2
        return 1 << (size - 1).bit_length()
        
    def _garbage_collect(self):
        """Compact memory and reclaim unused regions"""
        current_time = time.time()
        to_free = []
        
        # Find regions that haven't been accessed in 5 minutes
        for offset, region in self.allocated_regions.items():
            if current_time - region.last_accessed > 300 and region.ref_count <= 1:
                to_free.append(offset)
                
        for offset in to_free:
            self.free(offset)


class ZeroCopyMemoryManager:
    """Advanced memory manager with zero-copy operations and reference counting"""
    
    def __init__(self, pool_size: int = 1024*1024*1024):  # 1GB default
        self.pool_size = pool_size
        self.pool = mmap.mmap(-1, pool_size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self.allocator = SlabAllocator(self.pool)
        self.message_cache: Dict[str, Tuple[int, int]] = {}  # message_id -> (offset, size)
        self.ref_counts: Dict[int, ctypes.c_uint32] = {}
        self.lock = threading.RLock()
        
        # Performance metrics
        self.allocation_count = 0
        self.deallocation_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
    def allocate_message(self, size: int, message_id: Optional[str] = None) -> Optional[memoryview]:
        """Zero-copy message allocation with optional caching"""
        
        # Check cache first if message_id provided
        if message_id and message_id in self.message_cache:
            offset, cached_size = self.message_cache[message_id]
            if cached_size >= size:
                self.cache_hits += 1
                self._increment_ref_count(offset)
                return self._create_memoryview(offset, size)
                
        self.cache_misses += 1
        
        # Allocate new region (include 8 bytes for ref count header)
        offset = self.allocator.alloc(size + 8)
        if offset is None:
            logger.warning(f"Failed to allocate {size} bytes")
            return None
            
        # Initialize reference count
        self._set_ref_count(offset, 1)
        
        # Cache if message_id provided
        if message_id:
            self.message_cache[message_id] = (offset, size)
            
        self.allocation_count += 1
        
        # Return memoryview skipping ref count header
        return self._create_memoryview(offset + 8, size)
        
    def _create_memoryview(self, offset: int, size: int) -> memoryview:
        """Create memoryview for given offset and size"""
        return memoryview(self.pool)[offset:offset + size]
        
    def _set_ref_count(self, offset: int, count: int):
        """Set reference count for memory region"""
        ref_count_ptr = ctypes.cast(
            ctypes.c_void_p.from_buffer(self.pool, offset),
            ctypes.POINTER(ctypes.c_uint32)
        )
        ref_count_ptr.contents.value = count
        
    def _increment_ref_count(self, offset: int) -> int:
        """Increment and return reference count"""
        with self.lock:
            ref_count_ptr = ctypes.cast(
                ctypes.c_void_p.from_buffer(self.pool, offset),
                ctypes.POINTER(ctypes.c_uint32)
            )
            ref_count_ptr.contents.value += 1
            return ref_count_ptr.contents.value
            
    def _decrement_ref_count(self, offset: int) -> int:
        """Decrement and return reference count"""
        with self.lock:
            ref_count_ptr = ctypes.cast(
                ctypes.c_void_p.from_buffer(self.pool, offset),
                ctypes.POINTER(ctypes.c_uint32)
            )
            ref_count_ptr.contents.value -= 1
            count = ref_count_ptr.contents.value
            
            if count == 0:
                # Free the memory region
                self.allocator.free(offset)
                self.deallocation_count += 1
                
            return count
            
    def release_message(self, mv: memoryview):
        """Release reference to message, freeing if no more references"""
        # Calculate original offset (subtract header size)
        buffer_info = mv.obj
        if hasattr(buffer_info, '_offset'):
            offset = buffer_info._offset - 8
            self._decrement_ref_count(offset)
            
    def share_fd(self, data: bytes, target_pid: int) -> Optional[int]:
        """Share file descriptor with target process via SCM_RIGHTS"""
        try:
            # Create memory file descriptor
            fd = os.memfd_create("neuralsync_msg", 0)
            os.write(fd, data)
            os.lseek(fd, 0, os.SEEK_SET)  # Reset to beginning
            
            # Send fd via Unix domain socket (implementation depends on target)
            self._send_fd_via_socket(fd, target_pid)
            
            return fd
            
        except Exception as e:
            logger.error(f"Failed to share fd: {e}")
            return None
            
    def _send_fd_via_socket(self, fd: int, target_pid: int):
        """Send file descriptor via Unix domain socket with SCM_RIGHTS"""
        import socket
        import struct
        
        # This is a simplified version - full implementation would 
        # require proper socket setup and error handling
        sock_path = f"/tmp/neuralsync2_{target_pid}.sock"
        
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(sock_path)
            
            # Send file descriptor using SCM_RIGHTS
            sock.sendmsg([b"fd_transfer"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))])
            sock.close()
            
        except Exception as e:
            logger.error(f"Failed to send fd via socket: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager performance statistics"""
        return {
            'pool_size': self.pool_size,
            'allocations': self.allocation_count,
            'deallocations': self.deallocation_count,
            'active_allocations': self.allocation_count - self.deallocation_count,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'cached_messages': len(self.message_cache),
            'allocator_stats': {
                'free_lists': {size: len(offsets) for size, offsets in self.allocator.free_lists.items()},
                'allocated_regions': len(self.allocator.allocated_regions)
            }
        }
        
    def cleanup(self):
        """Cleanup resources"""
        self.message_cache.clear()
        self.pool.close()


# Global memory manager instance
_global_memory_manager: Optional[ZeroCopyMemoryManager] = None

def get_memory_manager() -> ZeroCopyMemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = ZeroCopyMemoryManager()
    return _global_memory_manager


async def benchmark_memory_manager():
    """Benchmark memory manager performance"""
    manager = get_memory_manager()
    
    # Test allocation/deallocation speed
    start_time = time.time()
    allocations = []
    
    for i in range(10000):
        mv = manager.allocate_message(1024, f"msg_{i}")
        if mv:
            allocations.append(mv)
            
    alloc_time = time.time() - start_time
    
    # Test deallocation speed
    start_time = time.time()
    for mv in allocations:
        manager.release_message(mv)
    dealloc_time = time.time() - start_time
    
    stats = manager.get_stats()
    
    print(f"Allocation time: {alloc_time:.3f}s ({10000/alloc_time:.0f} ops/sec)")
    print(f"Deallocation time: {dealloc_time:.3f}s ({10000/dealloc_time:.0f} ops/sec)")
    print(f"Memory stats: {stats}")


if __name__ == "__main__":
    asyncio.run(benchmark_memory_manager())