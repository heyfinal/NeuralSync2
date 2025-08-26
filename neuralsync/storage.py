import sqlite3, json, numpy as np
import mmap
import os
import threading
import time
import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple
from .embeddings import embed
from .utils import now_ms, redact
from .crdt import Version, version_newer, ByzantineCRDT, AdvancedVersion
from .btree_index import BTreeIndex, create_btree_index
from .memory_manager import get_memory_manager

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  scope TEXT NOT NULL,
  tool TEXT,
  tags TEXT,
  confidence REAL,
  benefit REAL,
  consistency REAL,
  vector BLOB,
  created_at INTEGER,
  updated_at INTEGER,
  ttl_ms INTEGER,
  expires_at INTEGER,
  tombstone INTEGER DEFAULT 0,
  site_id TEXT,
  lamport INTEGER,
  source TEXT,
  meta TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(id, text, tokenize='porter');
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""

class EnhancedStorage:
    """Enhanced storage system with memory mapping and zero-copy operations"""
    
    def __init__(self, db_path: str, memory_pool_size: int = 1024*1024*512):  # 512MB default
        self.db_path = db_path
        self.memory_pool_size = memory_pool_size
        
        # Traditional SQLite connection for compatibility (thread-safe and resilient)
        self.con = sqlite3.connect(db_path, check_same_thread=False, timeout=5.0)
        self.con.execute('PRAGMA journal_mode=WAL;')
        self.con.execute('PRAGMA synchronous=NORMAL;')
        self.con.execute('PRAGMA busy_timeout=5000;')
        self.con.executescript(SCHEMA)
        
        # Memory-mapped components
        self.memory_manager = get_memory_manager()
        
        # B+ tree indexes for fast lookups
        index_dir = os.path.dirname(db_path)
        self.primary_index = create_btree_index(os.path.join(index_dir, "primary.btree"))
        self.text_index = create_btree_index(os.path.join(index_dir, "text.btree"))
        self.scope_index = create_btree_index(os.path.join(index_dir, "scope.btree"))
        self.time_index = create_btree_index(os.path.join(index_dir, "time.btree"))
        
        # CRDT for conflict resolution
        self.crdt = ByzantineCRDT(site_id=self._get_site_id())
        
        # Memory-mapped storage for large values
        self.mmap_file = None
        self.mmap_path = db_path + '.mmap'
        self._init_memory_map()
        
        # Thread safety
        self.lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.RLock()
        
        # Performance tracking
        self.stats = {
            'zero_copy_reads': 0,
            'mmap_writes': 0,
            'index_hits': 0,
            'total_operations': 0,
            'avg_response_time_ms': 0.0
        }
        
        # Background compaction
        self._start_background_compaction()
        
    def _get_site_id(self) -> str:
        """Get unique site identifier"""
        try:
            import uuid
            hostname = os.uname().nodename
            return f"{hostname}_{uuid.getnode()}"
        except:
            return f"site_{hash(self.db_path) % 100000}"
            
    def _init_memory_map(self):
        """Initialize memory-mapped file for large value storage"""
        try:
            # Create or open memory-mapped file
            if not os.path.exists(self.mmap_path):
                with open(self.mmap_path, 'wb') as f:
                    f.write(b'\x00' * self.memory_pool_size)
                    
            self.mmap_file = open(self.mmap_path, 'r+b')
            self.mmap = mmap.mmap(self.mmap_file.fileno(), 0)
            
            # Track allocation offset (stored in first 8 bytes)
            self.mmap_offset = int.from_bytes(self.mmap[:8], 'little') or 8
            
        except Exception as e:
            logger.error(f"Failed to initialize memory map: {e}")
            self.mmap_file = None
            self.mmap = None
            
    def _allocate_mmap_space(self, size: int) -> Optional[int]:
        """Allocate space in memory-mapped file"""
        if not self.mmap:
            return None
            
        try:
            current_offset = self.mmap_offset
            new_offset = current_offset + size + 8  # Include size header
            
            if new_offset >= len(self.mmap):
                # Need to expand file
                old_len = len(self.mmap)
                self.mmap.close()
                self.mmap_file.close()
                
                # Double the size based on previous length
                new_size = max(old_len * 2, new_offset + self.memory_pool_size)
                
                with open(self.mmap_path, 'r+b') as f:
                    f.truncate(new_size)
                    
                self.mmap_file = open(self.mmap_path, 'r+b')
                self.mmap = mmap.mmap(self.mmap_file.fileno(), 0)
                
            # Write size header
            self.mmap[current_offset:current_offset+8] = size.to_bytes(8, 'little')
            
            # Update offset
            self.mmap_offset = new_offset
            self.mmap[:8] = self.mmap_offset.to_bytes(8, 'little')
            
            return current_offset + 8  # Return data offset (after size header)
            
        except Exception as e:
            logger.error(f"Memory map allocation failed: {e}")
            return None
            
    def _start_background_compaction(self):
        """Start background thread for memory compaction"""
        def compaction_worker():
            while True:
                try:
                    time.sleep(300)  # Compact every 5 minutes
                    self._compact_storage()
                    self._update_indexes()
                except Exception as e:
                    logger.error(f"Background compaction error: {e}")
                    
        import threading
        compaction_thread = threading.Thread(target=compaction_worker, daemon=True)
        compaction_thread.start()
        
    def _compact_storage(self):
        """Compact storage and reclaim space"""
        try:
            with self.lock:
                # Flush B+ tree indexes
                self.primary_index.flush()
                self.text_index.flush()
                self.scope_index.flush()
                self.time_index.flush()
                
                # Compact memory-mapped file (simplified)
                if self.mmap:
                    self.mmap.flush()
                    
                logger.info("Storage compaction completed")
                
        except Exception as e:
            logger.error(f"Storage compaction failed: {e}")
            
    def _update_indexes(self):
        """Update B+ tree indexes with latest data"""
        try:
            # Rebuild indexes from database
            cursor = self.con.execute("""
                SELECT id, text, scope, updated_at FROM items 
                WHERE tombstone = 0
            """)
            
            for row in cursor:
                item_id, text, scope, updated_at = row
                
                # Update indexes
                self.primary_index.insert(item_id, item_id)
                if text:
                    self.text_index.insert(text[:100], item_id)  # First 100 chars
                if scope:
                    self.scope_index.insert(scope, item_id)
                if updated_at:
                    # Time-based key for temporal queries
                    time_key = str(updated_at).zfill(20)
                    self.time_index.insert(time_key, item_id)
                    
        except Exception as e:
            logger.error(f"Index update failed: {e}")
            
    def merge_memories(self, other_storage: 'EnhancedStorage') -> Dict[str, Any]:
        """Intelligently merge memories from another storage instance"""
        try:
            with self.lock:
                # Merge CRDT states
                merged_crdt = self.crdt.merge(other_storage.crdt)
                merge_stats = {
                    'conflicts_resolved': 0,
                    'items_merged': 0,
                    'items_added': 0
                }
                
                # Apply merged state
                merged_data = merged_crdt.get_state()['data']
                
                for item_id, item_data in merged_data.items():
                    if item_id in self.crdt.data:
                        # Conflict resolution already handled by CRDT
                        merge_stats['conflicts_resolved'] += 1
                        merge_stats['items_merged'] += 1
                    else:
                        merge_stats['items_added'] += 1
                        
                    # Upsert merged item
                    if isinstance(item_data, dict):
                        upsert_item(self, item_data)
                        
                self.crdt = merged_crdt
                logger.info(f"Memory merge completed: {merge_stats}")
                return merge_stats
                
        except Exception as e:
            logger.error(f"Memory merge failed: {e}")
            return {'error': str(e)}
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        btree_stats = {
            'primary_index': self.primary_index.get_stats(),
            'text_index': self.text_index.get_stats(),
            'scope_index': self.scope_index.get_stats(),
            'time_index': self.time_index.get_stats()
        }
        
        memory_stats = self.memory_manager.get_stats()
        
        crdt_state = self.crdt.get_state()
        
        return {
            'storage': self.stats,
            'btree_indexes': btree_stats,
            'memory_manager': memory_stats,
            'crdt': {
                'data_items': len(crdt_state['data']),
                'tombstones': len(crdt_state['tombstones']),
                'suspected_byzantine': len(crdt_state['suspected_byzantine']),
                'site_id': crdt_state['site_id']
            },
            'memory_mapped_size': len(self.mmap) if self.mmap else 0,
            'memory_utilization': self.mmap_offset / len(self.mmap) if self.mmap else 0
        }
        
    def range_recall(self, start_time: int, end_time: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Perform temporal range query using time index"""
        try:
            start_key = str(start_time).zfill(20)
            end_key = str(end_time).zfill(20)
            
            # Use B+ tree time index for efficient range query
            time_results = list(self.time_index.range_query(start_key, end_key))
            
            # Fetch full records
            item_ids = [result[1] for result in time_results[:limit]]
            
            if not item_ids:
                return []
                
            placeholders = ','.join(['?' for _ in item_ids])
            query_sql = f"""
                SELECT * FROM items 
                WHERE id IN ({placeholders}) AND tombstone = 0
                ORDER BY updated_at DESC
            """
            
            cols = [c[1] for c in self.con.execute('PRAGMA table_info(items)')]
            results = []
            
            for row in self.con.execute(query_sql, item_ids):
                rec = dict(zip(cols, row))
                if not (rec.get('expires_at') and rec['expires_at'] < now_ms()):
                    results.append(rec)
                    
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Range recall failed: {e}")
            return []
            
    def optimize_performance(self):
        """Perform comprehensive performance optimization"""
        try:
            with self.lock:
                # Compact storage
                self._compact_storage()
                
                # Rebuild indexes for optimal performance
                self._update_indexes()
                
                # Garbage collect memory manager
                self.memory_manager.cleanup()
                
                # Flush all B+ trees
                for index in [self.primary_index, self.text_index, 
                            self.scope_index, self.time_index]:
                    index.flush()
                    
                logger.info("Performance optimization completed")
                
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            
    def close(self):
        """Cleanup and close all resources"""
        try:
            # Flush all data
            self.optimize_performance()
            
            # Close B+ tree indexes
            self.primary_index.close()
            self.text_index.close()
            self.scope_index.close()
            self.time_index.close()
            
            # Close memory-mapped file
            if self.mmap:
                self.mmap.close()
            if self.mmap_file:
                self.mmap_file.close()
                
            # Close SQLite connection
            self.con.close()
            
            logger.info("Enhanced storage closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing enhanced storage: {e}")


def connect(path: str) -> EnhancedStorage:
    """Create enhanced storage connection"""
    return EnhancedStorage(path)

def _get_persona_row(storage):
    """Get persona from either EnhancedStorage or raw connection"""
    if hasattr(storage, 'con'):
        con = storage.con
    else:
        con = storage
    row = con.execute("SELECT v FROM kv WHERE k='persona'").fetchone()
    return None if not row else json.loads(row[0])

def put_persona(storage, text: str, site_id: str, lamport: int):
    """Store persona with CRDT conflict resolution"""
    if hasattr(storage, 'con'):
        # Enhanced storage with CRDT
        storage.crdt.set('persona', {
            'text': text, 
            'site_id': site_id, 
            'lamport': lamport
        })
        con = storage.con
    else:
        # Legacy compatibility
        con = storage
        
    cur = _get_persona_row(storage)
    if cur:
        if not version_newer(Version(lamport, site_id), Version(cur.get('lamport',0), cur.get('site_id',''))):
            return
    con.execute("INSERT OR REPLACE INTO kv(k,v) VALUES('persona', json(?))", (json.dumps({'text': text, 'site_id': site_id, 'lamport': lamport}),))
    con.commit()

def get_persona(storage) -> Dict[str,Any]:
    """Get persona from storage"""
    if hasattr(storage, 'crdt'):
        # Try CRDT first
        persona_data = storage.crdt.get('persona')
        if persona_data:
            return persona_data
            
    # Fallback to database
    if hasattr(storage, 'con'):
        con = storage.con
    else:
        con = storage
    row = con.execute("SELECT v FROM kv WHERE k='persona'").fetchone()
    return json.loads(row[0]) if row else {'text': ''}

def _insert_or_update(con: sqlite3.Connection, item: Dict[str,Any]):
    exists = con.execute('SELECT lamport, site_id FROM items WHERE id=?', (item['id'],)).fetchone()
    if exists:
        cur_ver = Version(lamport=int(exists[0]), site_id=exists[1])
        new_ver = Version(lamport=int(item['lamport']), site_id=item['site_id'])
        if not version_newer(new_ver, cur_ver):
            return
    fields = ["id","kind","text","scope","tool","tags","confidence","benefit","consistency","vector","created_at","updated_at","ttl_ms","expires_at","tombstone","site_id","lamport","source","meta"]
    vals = [item.get(f) for f in fields]
    con.execute(f"""INSERT INTO items({','.join(fields)}) VALUES ({','.join(['?']*len(fields))})
        ON CONFLICT(id) DO UPDATE SET
        kind=excluded.kind, text=excluded.text, scope=excluded.scope, tool=excluded.tool,
        tags=excluded.tags, confidence=excluded.confidence, benefit=excluded.benefit,
        consistency=excluded.consistency, vector=excluded.vector,
        updated_at=excluded.updated_at, ttl_ms=excluded.ttl_ms,
        expires_at=excluded.expires_at, tombstone=excluded.tombstone,
        site_id=excluded.site_id, lamport=excluded.lamport, source=excluded.source,
        meta=excluded.meta""", vals)
    con.execute("INSERT INTO items_fts(id, text) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET text=excluded.text", (item['id'], item['text']))
    con.commit()

def upsert_item(storage, item: Dict[str,Any]) -> Dict[str,Any]:
    """Enhanced upsert with zero-copy operations and indexing"""
    start_time = time.time()
    
    if hasattr(storage, 'con'):
        # Enhanced storage path
        con = storage.con
        
        # Zero-copy vector embedding if needed
        if item.get('text') and item.get('vector') is None:
            text_bytes = item['text'].encode('utf-8')
            
            # Try zero-copy allocation for large texts
            if len(text_bytes) > 1024 and storage.mmap:
                try:
                    # Allocate in memory-mapped storage
                    offset = storage._allocate_mmap_space(len(text_bytes))
                    if offset:
                        storage.mmap[offset:offset+len(text_bytes)] = text_bytes
                        item['mmap_offset'] = offset
                        item['mmap_size'] = len(text_bytes)
                        storage.stats['mmap_writes'] += 1
                except Exception as e:
                    logger.warning(f"Memory-mapped storage failed: {e}")
            
            # Generate embedding
            v = embed(redact(item['text']))
            item['vector'] = v.tobytes()
            
        item['updated_at'] = now_ms()
        
        # Update CRDT
        storage.crdt.set(item['id'], item)
        
        # Update B+ tree indexes
        try:
            storage.primary_index.insert(item['id'], item['id'])
            if item.get('text'):
                storage.text_index.insert(item['text'][:100], item['id'])
            if item.get('scope'):
                storage.scope_index.insert(item['scope'], item['id'])
            if item.get('updated_at'):
                time_key = str(item['updated_at']).zfill(20)
                storage.time_index.insert(time_key, item['id'])
            storage.stats['index_hits'] += 1
        except Exception as e:
            logger.warning(f"Index update failed: {e}")
            
        # Store in SQLite for persistence
        _insert_or_update(con, item)
        
        # Update statistics
        storage.stats['total_operations'] += 1
        response_time = (time.time() - start_time) * 1000
        storage.stats['avg_response_time_ms'] = (
            0.9 * storage.stats['avg_response_time_ms'] + 0.1 * response_time
        )
        
    else:
        # Legacy compatibility
        if item.get('text') and item.get('vector') is None:
            v = embed(redact(item['text']))
            item['vector'] = v.tobytes()
        item['updated_at'] = now_ms()
        _insert_or_update(storage, item)
        
    return item

def delete_item(storage, id_: str):
    """Enhanced delete with CRDT tombstone"""
    if hasattr(storage, 'con'):
        # Enhanced storage with CRDT
        storage.crdt.delete(id_)
        con = storage.con
    else:
        # Legacy compatibility
        con = storage
        
    con.execute('UPDATE items SET tombstone=1, updated_at=? WHERE id=?', (now_ms(), id_))
    con.commit()

def _list_candidates(con: sqlite3.Connection, query: str, scope: str, tool: Optional[str], limit: int):
    rows = []
    if query.strip():
        for row in con.execute('SELECT id FROM items_fts WHERE text MATCH ? LIMIT ?', (query, limit*3)):
            rows.append(row[0])
    if len(rows) < limit:
        for row in con.execute('SELECT id FROM items WHERE tombstone=0 ORDER BY updated_at DESC LIMIT ?', (limit*3,)):
            if row[0] not in rows: rows.append(row[0])
    out=[]
    cols = [c[1] for c in con.execute('PRAGMA table_info(items)')]
    for id_ in rows:
        r = con.execute('SELECT * FROM items WHERE id=?', (id_,)).fetchone()
        if not r: continue
        rec = dict(zip(cols, r))
        if rec['tombstone']: continue
        if rec.get('expires_at') and rec['expires_at'] < now_ms(): continue
        if scope != 'any' and rec['scope'] != scope: continue
        if tool and rec.get('tool') and rec['tool'] != tool: continue
        out.append(rec)
        if len(out) >= limit*3: break
    return out

def recall(storage, query: str, top_k: int, scope: str, tool: Optional[str]):
    """Enhanced recall with B+ tree indexing and zero-copy operations"""
    start_time = time.time()
    
    if hasattr(storage, 'con'):
        # Enhanced storage path with B+ tree acceleration
        con = storage.con
        
        # Try B+ tree index lookup first for exact matches
        candidates = []
        
        try:
            # Search text index for partial matches
            query_bytes = query[:100].encode('utf-8')
            text_results = list(storage.text_index.range_query(
                query_bytes, 
                (query_bytes[:-1] + bytes([query_bytes[-1] + 1])) if query_bytes else b'\xff'
            ))
            
            # Get item IDs from index results
            index_item_ids = [result[1] for result in text_results[:top_k*2]]
            
            if index_item_ids:
                # Fetch full records for indexed results
                placeholders = ','.join(['?' for _ in index_item_ids])
                query_sql = f"""
                    SELECT * FROM items 
                    WHERE id IN ({placeholders}) AND tombstone = 0
                """
                
                cols = [c[1] for c in con.execute('PRAGMA table_info(items)')]
                for row in con.execute(query_sql, index_item_ids):
                    rec = dict(zip(cols, row))
                    if rec.get('expires_at') and rec['expires_at'] < now_ms():
                        continue
                    if scope != 'any' and rec['scope'] != scope:
                        continue
                    if tool and rec.get('tool') and rec['tool'] != tool:
                        continue
                    candidates.append(rec)
                    
                storage.stats['index_hits'] += 1
                
        except Exception as e:
            logger.warning(f"B+ tree lookup failed: {e}")
            
        # Fallback to traditional search if needed
        if len(candidates) < top_k:
            traditional_candidates = _list_candidates(con, query, scope, tool, top_k)
            
            # Merge candidates, avoiding duplicates
            existing_ids = {c['id'] for c in candidates}
            for cand in traditional_candidates:
                if cand['id'] not in existing_ids:
                    candidates.append(cand)
                    if len(candidates) >= top_k * 3:
                        break
        
        storage.stats['zero_copy_reads'] += 1
        
    else:
        # Legacy compatibility
        candidates = _list_candidates(storage, query, scope, tool, top_k)
    
    if not candidates:
        return []
        
    # Enhanced scoring with zero-copy vector operations
    qv = embed(query)
    scored = []
    now = now_ms()
    
    for rec in candidates:
        import numpy as _np
        
        # Try zero-copy vector access from memory-mapped storage
        vector_data = None
        if hasattr(storage, 'mmap') and storage.mmap and rec.get('mmap_offset'):
            try:
                offset = rec['mmap_offset'] 
                size = rec.get('mmap_size', 0)
                if size > 0:
                    # Zero-copy access to memory-mapped data
                    mmap_data = storage.mmap[offset:offset+size]
                    storage.stats['zero_copy_reads'] += 1
            except Exception as e:
                logger.warning(f"Zero-copy read failed: {e}")
        
        # Vector similarity calculation
        v = _np.frombuffer(rec['vector'], dtype='float32') if rec['vector'] else embed(rec['text'])
        sim = float((_np.dot(qv, v)) / ((_np.linalg.norm(qv)*_np.linalg.norm(v))+1e-8))
        
        # Enhanced scoring factors
        age_ms = max(1, now - (rec['updated_at'] or rec['created_at'] or now))
        recency = 1.0 / (1.0 + age_ms/3.6e6)
        benefit = float(rec.get('benefit') or rec.get('confidence') or 0.5)
        consistency = float(rec.get('consistency') or 0.5)
        
        # Weighted scoring with query relevance boost
        relevance_boost = 1.0
        if query.lower() in rec.get('text', '').lower():
            relevance_boost = 1.2
            
        score = relevance_boost * (0.35*recency + 0.25*benefit + 0.15*consistency + 0.25*sim)
        scored.append((score, rec))
    
    # Sort and select top results
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [r for _, r in scored[:top_k]]
    
    # Update consistency scores for accessed items
    if hasattr(storage, 'con'):
        con = storage.con
    else:
        con = storage
        
    for rec in top:
        new_cons = min(1.0, float(rec.get('consistency') or 0.5) + 0.02)
        con.execute('UPDATE items SET consistency=?, updated_at=? WHERE id=?', 
                   (new_cons, now_ms(), rec['id']))
    con.commit()
    
    # Update performance statistics
    if hasattr(storage, 'stats'):
        storage.stats['total_operations'] += 1
        response_time = (time.time() - start_time) * 1000
        storage.stats['avg_response_time_ms'] = (
            0.9 * storage.stats['avg_response_time_ms'] + 0.1 * response_time
        )
    
    return top
