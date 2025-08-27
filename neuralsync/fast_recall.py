#!/usr/bin/env python3
"""
Fast Memory Recall System for NeuralSync v2
Optimized memory recall with advanced indexing, filtering, and caching
"""

import time
import threading
import logging
import numpy as np
import sqlite3
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed

from .storage import connect
from .embeddings import embed
from .utils import now_ms
from .intelligent_cache import get_neuralsync_cache

logger = logging.getLogger(__name__)

@dataclass
class RecallCandidate:
    """Memory recall candidate with scoring metadata"""
    item_id: str
    text: str
    kind: str
    scope: str
    tool: Optional[str]
    confidence: float
    created_at: int
    updated_at: int
    vector: Optional[np.ndarray] = None
    relevance_score: float = 0.0
    semantic_score: float = 0.0
    temporal_score: float = 0.0
    frequency_score: float = 0.0
    final_score: float = 0.0

@dataclass
class RecallQuery:
    """Memory recall query with optimization metadata"""
    query_text: str
    query_vector: Optional[np.ndarray]
    tool_filter: Optional[str]
    scope_filter: str
    top_k: int
    query_hash: str
    timestamp: int
    
    def __post_init__(self):
        if self.query_vector is None and self.query_text:
            self.query_vector = embed(self.query_text)

class FastIndexManager:
    """High-performance indexing for memory recall"""
    
    def __init__(self, storage):
        self.storage = storage
        
        # In-memory indexes for fast lookups
        self.text_tokens: Dict[str, Set[str]] = {}  # token -> item_ids
        self.tool_index: Dict[str, Set[str]] = defaultdict(set)  # tool -> item_ids
        self.scope_index: Dict[str, Set[str]] = defaultdict(set)  # scope -> item_ids
        self.time_buckets: Dict[int, Set[str]] = defaultdict(set)  # time_bucket -> item_ids
        
        # Vector similarity index (simplified FAISS-like structure)
        self.vector_index: Dict[str, np.ndarray] = {}
        self.vector_clusters: List[Tuple[np.ndarray, Set[str]]] = []
        
        # Index metadata
        self.index_timestamp = 0
        self.total_items_indexed = 0
        self.lock = threading.RLock()
        
        # Background indexing
        self.index_thread: Optional[threading.Thread] = None
        self.stop_indexing = False
        
        # Start background indexing
        self._start_background_indexing()
        
        logger.info("FastIndexManager initialized")
    
    def _start_background_indexing(self):
        """Start background thread for index maintenance"""
        def indexing_worker():
            while not self.stop_indexing:
                try:
                    self._rebuild_indexes()
                    time.sleep(300)  # Rebuild every 5 minutes
                except Exception as e:
                    logger.error(f"Background indexing error: {e}")
                    time.sleep(600)  # Wait 10 minutes on error
        
        self.index_thread = threading.Thread(target=indexing_worker, daemon=True)
        self.index_thread.start()
    
    def _rebuild_indexes(self):
        """Rebuild all indexes from storage"""
        start_time = time.perf_counter()
        
        try:
            with self.lock:
                # Clear existing indexes
                self.text_tokens.clear()
                self.tool_index.clear()
                self.scope_index.clear()
                self.time_buckets.clear()
                self.vector_index.clear()
                
                # Get all active items
                if hasattr(self.storage, 'con'):
                    con = self.storage.con
                else:
                    con = self.storage
                
                query = """
                    SELECT id, text, kind, scope, tool, vector, updated_at, confidence
                    FROM items 
                    WHERE tombstone = 0 
                    AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY updated_at DESC
                """
                
                current_time = now_ms()
                cursor = con.execute(query, (current_time,))
                items_processed = 0
                
                for row in cursor:
                    item_id, text, kind, scope, tool, vector_blob, updated_at, confidence = row
                    
                    # Index text tokens
                    if text:
                        tokens = self._tokenize_text(text)
                        for token in tokens:
                            if token not in self.text_tokens:
                                self.text_tokens[token] = set()
                            self.text_tokens[token].add(item_id)
                    
                    # Index by tool
                    if tool:
                        self.tool_index[tool].add(item_id)
                    
                    # Index by scope
                    if scope:
                        self.scope_index[scope].add(item_id)
                    
                    # Time-based indexing (by day)
                    if updated_at:
                        time_bucket = updated_at // (24 * 3600 * 1000)  # Day bucket
                        self.time_buckets[time_bucket].add(item_id)
                    
                    # Vector indexing
                    if vector_blob:
                        try:
                            vector = np.frombuffer(vector_blob, dtype='float32')
                            self.vector_index[item_id] = vector
                        except Exception as e:
                            logger.debug(f"Vector indexing failed for {item_id}: {e}")
                    
                    items_processed += 1
                
                # Build vector clusters for fast similarity search
                self._build_vector_clusters()
                
                self.total_items_indexed = items_processed
                self.index_timestamp = now_ms()
                
                build_time_ms = (time.perf_counter() - start_time) * 1000
                logger.info(f"Rebuilt indexes: {items_processed} items in {build_time_ms:.2f}ms")
                
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
    
    def _tokenize_text(self, text: str, max_tokens: int = 20) -> Set[str]:
        """Simple but effective text tokenization"""
        import re
        
        # Clean and split text
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = set()
        
        # Word tokens
        words = text_clean.split()
        tokens.update(word for word in words[:max_tokens] if len(word) > 2)
        
        # Bigrams for better matching
        for i in range(len(words) - 1):
            if i >= max_tokens // 2:
                break
            bigram = f"{words[i]}_{words[i+1]}"
            if len(bigram) > 4:
                tokens.add(bigram)
        
        return tokens
    
    def _build_vector_clusters(self, num_clusters: int = 50):
        """Build vector clusters for fast similarity search"""
        if len(self.vector_index) < 10:
            return
        
        try:
            # Simple k-means clustering
            vectors = list(self.vector_index.values())
            item_ids = list(self.vector_index.keys())
            
            if len(vectors) < num_clusters:
                num_clusters = max(1, len(vectors) // 3)
            
            # Initialize centroids randomly
            vector_array = np.array(vectors)
            centroids = vector_array[np.random.choice(len(vectors), num_clusters, replace=False)]
            
            # Simple clustering (3 iterations)
            for iteration in range(3):
                clusters = [set() for _ in range(num_clusters)]
                
                # Assign vectors to nearest centroid
                for i, vector in enumerate(vectors):
                    distances = np.linalg.norm(centroids - vector, axis=1)
                    closest_cluster = np.argmin(distances)
                    clusters[closest_cluster].add(item_ids[i])
                
                # Update centroids
                new_centroids = []
                for cluster_items in clusters:
                    if cluster_items:
                        cluster_vectors = [self.vector_index[item_id] for item_id in cluster_items]
                        centroid = np.mean(cluster_vectors, axis=0)
                        new_centroids.append(centroid)
                    else:
                        new_centroids.append(centroids[len(new_centroids)])
                
                centroids = np.array(new_centroids)
            
            # Store clusters
            self.vector_clusters = [(centroid, cluster_items) 
                                  for centroid, cluster_items in zip(centroids, clusters)
                                  if cluster_items]
            
            logger.debug(f"Built {len(self.vector_clusters)} vector clusters")
            
        except Exception as e:
            logger.error(f"Vector clustering failed: {e}")
    
    def get_text_candidates(self, query_tokens: Set[str], max_candidates: int = 100) -> Set[str]:
        """Get candidates based on text token matching"""
        
        candidate_scores = defaultdict(float)
        
        with self.lock:
            for token in query_tokens:
                if token in self.text_tokens:
                    item_ids = self.text_tokens[token]
                    # Weight by token rarity (inverse frequency)
                    weight = 1.0 / max(1, len(item_ids))
                    
                    for item_id in item_ids:
                        candidate_scores[item_id] += weight
        
        # Return top candidates by text score
        top_candidates = heapq.nlargest(max_candidates, candidate_scores.items(), key=lambda x: x[1])
        return set(item_id for item_id, _ in top_candidates)
    
    def get_vector_candidates(self, query_vector: np.ndarray, max_candidates: int = 100) -> List[Tuple[str, float]]:
        """Get candidates based on vector similarity"""
        
        candidates = []
        
        with self.lock:
            if self.vector_clusters:
                # Find relevant clusters first
                cluster_distances = []
                for i, (centroid, _) in enumerate(self.vector_clusters):
                    distance = np.linalg.norm(query_vector - centroid)
                    cluster_distances.append((distance, i))
                
                # Search in closest clusters
                cluster_distances.sort()
                items_found = 0
                
                for _, cluster_idx in cluster_distances:
                    if items_found >= max_candidates:
                        break
                    
                    _, cluster_items = self.vector_clusters[cluster_idx]
                    
                    for item_id in cluster_items:
                        if items_found >= max_candidates:
                            break
                        
                        if item_id in self.vector_index:
                            item_vector = self.vector_index[item_id]
                            similarity = np.dot(query_vector, item_vector) / (
                                np.linalg.norm(query_vector) * np.linalg.norm(item_vector) + 1e-8
                            )
                            candidates.append((item_id, similarity))
                            items_found += 1
            
            else:
                # Fallback: brute force similarity
                for item_id, item_vector in list(self.vector_index.items())[:max_candidates]:
                    similarity = np.dot(query_vector, item_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(item_vector) + 1e-8
                    )
                    candidates.append((item_id, similarity))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
    
    def filter_by_tool(self, item_ids: Set[str], tool: Optional[str]) -> Set[str]:
        """Filter items by tool"""
        if not tool:
            return item_ids
        
        with self.lock:
            if tool in self.tool_index:
                return item_ids.intersection(self.tool_index[tool])
            else:
                return set()
    
    def filter_by_scope(self, item_ids: Set[str], scope: str) -> Set[str]:
        """Filter items by scope"""
        if scope == 'any':
            return item_ids
        
        with self.lock:
            if scope in self.scope_index:
                return item_ids.intersection(self.scope_index[scope])
            else:
                return set()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        with self.lock:
            return {
                'total_items_indexed': self.total_items_indexed,
                'text_tokens': len(self.text_tokens),
                'tools_indexed': len(self.tool_index),
                'scopes_indexed': len(self.scope_index),
                'time_buckets': len(self.time_buckets),
                'vectors_indexed': len(self.vector_index),
                'vector_clusters': len(self.vector_clusters),
                'index_age_ms': now_ms() - self.index_timestamp
            }
    
    def force_rebuild(self):
        """Force immediate index rebuild"""
        self._rebuild_indexes()
    
    def close(self):
        """Clean shutdown"""
        self.stop_indexing = True
        if self.index_thread:
            self.index_thread.join(timeout=5)

class OptimizedRecallEngine:
    """High-performance memory recall engine"""
    
    def __init__(self, storage):
        self.storage = storage
        self.cache = get_neuralsync_cache()
        self.index_manager = FastIndexManager(storage)
        
        # Performance tracking
        self.stats = {
            'total_recalls': 0,
            'cache_hits': 0,
            'avg_recall_time_ms': 0.0,
            'last_recall_time_ms': 0.0
        }
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="recall_")
        
        logger.info("OptimizedRecallEngine initialized")
    
    async def fast_recall(self, 
                         query: str,
                         top_k: int = 8,
                         scope: str = 'any',
                         tool: Optional[str] = None,
                         use_cache: bool = True) -> List[Dict[str, Any]]:
        """High-speed memory recall with all optimizations"""
        
        start_time = time.perf_counter()
        self.stats['total_recalls'] += 1
        
        # Create query object
        query_hash = hashlib.md5(f"{query}|{tool}|{scope}|{top_k}".encode()).hexdigest()
        recall_query = RecallQuery(
            query_text=query,
            query_vector=None,  # Will be computed lazily
            tool_filter=tool,
            scope_filter=scope,
            top_k=top_k,
            query_hash=query_hash,
            timestamp=now_ms()
        )
        
        # Try cache first
        if use_cache:
            cached_result = await self.cache.get_memories(query_hash)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                recall_time_ms = (time.perf_counter() - start_time) * 1000
                self.stats['last_recall_time_ms'] = recall_time_ms
                return cached_result
        
        try:
            # Get candidates using multiple strategies
            candidates = await self._get_recall_candidates(recall_query)
            
            # Score and rank candidates
            ranked_candidates = self._score_candidates(candidates, recall_query)
            
            # Convert to output format
            results = self._format_results(ranked_candidates[:top_k])
            
            # Cache results
            if use_cache and results:
                await self.cache.set_memories(query_hash, results, 300000)  # 5 minutes
            
            # Update performance statistics
            recall_time_ms = (time.perf_counter() - start_time) * 1000
            self.stats['last_recall_time_ms'] = recall_time_ms
            self.stats['avg_recall_time_ms'] = (
                0.9 * self.stats['avg_recall_time_ms'] + 0.1 * recall_time_ms
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Fast recall failed: {e}")
            return []
    
    async def _get_recall_candidates(self, query: RecallQuery) -> List[RecallCandidate]:
        """Get candidates using multiple strategies in parallel"""
        
        candidates_map = {}
        
        # Strategy 1: Text-based matching
        if query.query_text:
            query_tokens = self.index_manager._tokenize_text(query.query_text)
            text_candidates = self.index_manager.get_text_candidates(query_tokens, 200)
            
            # Add to candidates map
            for item_id in text_candidates:
                if item_id not in candidates_map:
                    candidates_map[item_id] = {'sources': set(), 'text_score': 0.0}
                candidates_map[item_id]['sources'].add('text')
                candidates_map[item_id]['text_score'] = 1.0
        
        # Strategy 2: Vector similarity (if we have a meaningful query)
        if query.query_text and len(query.query_text.split()) > 1:
            if query.query_vector is None:
                query.query_vector = embed(query.query_text)
            
            vector_candidates = self.index_manager.get_vector_candidates(query.query_vector, 100)
            
            for item_id, similarity in vector_candidates:
                if item_id not in candidates_map:
                    candidates_map[item_id] = {'sources': set(), 'text_score': 0.0}
                candidates_map[item_id]['sources'].add('vector')
                candidates_map[item_id]['vector_score'] = similarity
        
        # Strategy 3: Recent items (for empty queries)
        if not query.query_text.strip():
            recent_candidates = await self._get_recent_candidates(50)
            for item_id in recent_candidates:
                if item_id not in candidates_map:
                    candidates_map[item_id] = {'sources': set(), 'text_score': 0.0}
                candidates_map[item_id]['sources'].add('recent')
                candidates_map[item_id]['recency_score'] = 1.0
        
        # Apply filters
        filtered_item_ids = set(candidates_map.keys())
        
        if query.tool_filter:
            filtered_item_ids = self.index_manager.filter_by_tool(filtered_item_ids, query.tool_filter)
        
        if query.scope_filter != 'any':
            filtered_item_ids = self.index_manager.filter_by_scope(filtered_item_ids, query.scope_filter)
        
        # Fetch full candidate data
        candidates = await self._fetch_candidate_data(list(filtered_item_ids), candidates_map)
        
        return candidates
    
    async def _get_recent_candidates(self, limit: int) -> List[str]:
        """Get recent items for empty queries"""
        
        try:
            if hasattr(self.storage, 'con'):
                con = self.storage.con
            else:
                con = self.storage
            
            query = """
                SELECT id FROM items 
                WHERE tombstone = 0 
                AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY updated_at DESC 
                LIMIT ?
            """
            
            cursor = con.execute(query, (now_ms(), limit))
            return [row[0] for row in cursor]
            
        except Exception as e:
            logger.error(f"Recent candidates fetch failed: {e}")
            return []
    
    async def _fetch_candidate_data(self, item_ids: List[str], score_data: Dict) -> List[RecallCandidate]:
        """Fetch full data for candidate items"""
        
        if not item_ids:
            return []
        
        try:
            if hasattr(self.storage, 'con'):
                con = self.storage.con
            else:
                con = self.storage
            
            # Build query with placeholders
            placeholders = ','.join(['?' for _ in item_ids])
            query = f"""
                SELECT id, text, kind, scope, tool, confidence, created_at, updated_at, vector
                FROM items 
                WHERE id IN ({placeholders}) 
                AND tombstone = 0
                AND (expires_at IS NULL OR expires_at > ?)
            """
            
            cursor = con.execute(query, item_ids + [now_ms()])
            candidates = []
            
            for row in cursor:
                item_id, text, kind, scope, tool, confidence, created_at, updated_at, vector_blob = row
                
                # Parse vector
                vector = None
                if vector_blob:
                    try:
                        vector = np.frombuffer(vector_blob, dtype='float32')
                    except Exception as e:
                        logger.debug(f"Vector parsing failed for {item_id}: {e}")
                
                # Get scoring metadata
                scores = score_data.get(item_id, {})
                
                candidate = RecallCandidate(
                    item_id=item_id,
                    text=text,
                    kind=kind,
                    scope=scope,
                    tool=tool,
                    confidence=confidence or 0.5,
                    created_at=created_at or 0,
                    updated_at=updated_at or 0,
                    vector=vector,
                    semantic_score=scores.get('vector_score', 0.0),
                    temporal_score=scores.get('recency_score', 0.0)
                )
                
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Candidate data fetch failed: {e}")
            return []
    
    def _score_candidates(self, candidates: List[RecallCandidate], query: RecallQuery) -> List[RecallCandidate]:
        """Score and rank candidates using multiple factors"""
        
        current_time = now_ms()
        
        for candidate in candidates:
            # Temporal scoring
            age_hours = max(1, (current_time - candidate.updated_at) / 3600000)
            candidate.temporal_score = max(0.1, 1.0 / (1.0 + age_hours / 24))  # Decay over days
            
            # Frequency/confidence scoring
            candidate.frequency_score = min(1.0, candidate.confidence)
            
            # Query relevance (text matching)
            if query.query_text:
                query_lower = query.query_text.lower()
                text_lower = candidate.text.lower()
                
                # Exact substring match bonus
                if query_lower in text_lower:
                    candidate.relevance_score = 1.0
                elif any(word in text_lower for word in query_lower.split()):
                    candidate.relevance_score = 0.7
                else:
                    candidate.relevance_score = 0.3
            else:
                candidate.relevance_score = 0.5
            
            # Combined scoring
            weights = {
                'semantic': 0.35,
                'relevance': 0.25,
                'temporal': 0.25,
                'frequency': 0.15
            }
            
            candidate.final_score = (
                weights['semantic'] * candidate.semantic_score +
                weights['relevance'] * candidate.relevance_score +
                weights['temporal'] * candidate.temporal_score +
                weights['frequency'] * candidate.frequency_score
            )
        
        # Sort by final score
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        
        return candidates
    
    def _format_results(self, candidates: List[RecallCandidate]) -> List[Dict[str, Any]]:
        """Format candidates as API response"""
        
        results = []
        
        for candidate in candidates:
            result = {
                'id': candidate.item_id,
                'text': candidate.text,
                'kind': candidate.kind,
                'scope': candidate.scope,
                'tool': candidate.tool,
                'confidence': candidate.confidence,
                'created_at': candidate.created_at,
                'updated_at': candidate.updated_at,
                'score': candidate.final_score
            }
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recall engine statistics"""
        index_stats = self.index_manager.get_stats()
        
        return {
            'recall_stats': self.stats.copy(),
            'index_stats': index_stats,
            'cache_stats': self.cache.get_comprehensive_stats()
        }
    
    def close(self):
        """Clean shutdown"""
        self.index_manager.close()
        self.thread_pool.shutdown(wait=True)


# Global recall engine instance
_global_recall_engine: Optional[OptimizedRecallEngine] = None

def get_fast_recall_engine(storage=None) -> OptimizedRecallEngine:
    """Get global recall engine instance"""
    global _global_recall_engine
    if _global_recall_engine is None and storage:
        _global_recall_engine = OptimizedRecallEngine(storage)
    return _global_recall_engine

async def fast_recall(storage, query: str, top_k: int = 8, scope: str = 'any', tool: Optional[str] = None) -> List[Dict[str, Any]]:
    """High-performance memory recall function"""
    engine = get_fast_recall_engine(storage)
    if engine is None:
        engine = OptimizedRecallEngine(storage)
    
    return await engine.fast_recall(query, top_k, scope, tool)