#!/usr/bin/env python3
"""
Research Deduplication System for NeuralSync2
Prevents redundant API calls by storing and matching past research
"""

import asyncio
import json
import hashlib
import sqlite3
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pickle
import zlib
from collections import defaultdict
import re

# Advanced similarity detection
from difflib import SequenceMatcher
# Optional heavy deps: sklearn, spacy
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
    class TfidfVectorizer:  # minimal stub
        def __init__(self, **kwargs):
            self._vocab = {}
        def fit_transform(self, texts):
            return np.zeros((len(texts), 1), dtype=np.float32)
        def transform(self, texts):
            return np.zeros((len(texts), 1), dtype=np.float32)
    def cosine_similarity(a, b):
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
try:
    import spacy  # type: ignore
except Exception:
    spacy = None

@dataclass
class ResearchEntry:
    """Research entry with metadata"""
    id: str
    query: str
    query_hash: str
    response: str
    response_hash: str
    cli_tool: str
    api_provider: str
    timestamp: float
    token_count: int
    cost_estimate: float
    similarity_vector: Optional[bytes] = None
    tags: List[str] = None
    context: Dict[str, Any] = None

class ResearchDatabase:
    """High-performance research storage with advanced deduplication"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path.home() / ".neuralsync" / "research.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLP model for semantic analysis
        try:
            if spacy is not None:
                self.nlp = spacy.load("en_core_web_sm")
            else:
                raise OSError("spacy not installed")
        except OSError:
            print("⚠️ spaCy model not available, using basic similarity")
            self.nlp = None
            
        # Initialize TF-IDF vectorizer
        # Initialize TF-IDF vectorizer (fallback to stub if sklearn missing)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000 if _HAS_SKLEARN else None,
            stop_words='english' if _HAS_SKLEARN else None,
            ngram_range=(1, 2) if _HAS_SKLEARN else None,
            lowercase=True if _HAS_SKLEARN else None,
            strip_accents='unicode' if _HAS_SKLEARN else None
        )
        
        self.bloom_filter: Set[str] = set()
        self.query_cache: Dict[str, List[ResearchEntry]] = {}
        self.last_cache_update = 0
        
        self._init_database()
        self._load_bloom_filter()
        
    def _init_database(self):
        """Initialize SQLite database with optimized schema"""
        con = sqlite3.connect(self.db_path)
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('PRAGMA synchronous=NORMAL')
        con.execute('PRAGMA cache_size=10000')
        con.execute('PRAGMA temp_store=MEMORY')
        
        # Create tables
        con.executescript('''
        CREATE TABLE IF NOT EXISTS research_entries (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            response TEXT NOT NULL,
            response_hash TEXT NOT NULL,
            cli_tool TEXT NOT NULL,
            api_provider TEXT NOT NULL,
            timestamp REAL NOT NULL,
            token_count INTEGER DEFAULT 0,
            cost_estimate REAL DEFAULT 0.0,
            similarity_vector BLOB,
            tags TEXT,
            context TEXT,
            UNIQUE(query_hash)
        );
        
        CREATE INDEX IF NOT EXISTS idx_query_hash ON research_entries(query_hash);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON research_entries(timestamp);
        CREATE INDEX IF NOT EXISTS idx_cli_tool ON research_entries(cli_tool);
        CREATE INDEX IF NOT EXISTS idx_api_provider ON research_entries(api_provider);
        
        CREATE VIRTUAL TABLE IF NOT EXISTS research_fts 
        USING fts5(id, query, response, tokenize='porter');
        
        CREATE TABLE IF NOT EXISTS similarity_index (
            id1 TEXT,
            id2 TEXT,
            similarity_score REAL,
            PRIMARY KEY(id1, id2)
        );
        
        CREATE TABLE IF NOT EXISTS query_patterns (
            pattern TEXT PRIMARY KEY,
            count INTEGER DEFAULT 1,
            last_seen REAL
        );
        ''')
        
        con.commit()
        con.close()
        
    def _load_bloom_filter(self):
        """Load bloom filter for fast duplicate detection"""
        con = sqlite3.connect(self.db_path)
        try:
            for row in con.execute('SELECT query_hash FROM research_entries'):
                self.bloom_filter.add(row[0])
        finally:
            con.close()
            
    def _compute_query_hash(self, query: str) -> str:
        """Compute normalized query hash"""
        # Normalize query for better deduplication
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        return hashlib.sha256(normalized.encode()).hexdigest()
        
    def _compute_response_hash(self, response: str) -> str:
        """Compute response content hash"""
        return hashlib.sha256(response.encode()).hexdigest()
        
    def _compute_similarity_vector(self, text: str) -> bytes:
        """Compute similarity vector for semantic matching"""
        try:
            # Use TF-IDF vectorization
            if hasattr(self, '_fitted_vectorizer'):
                vector = self._fitted_vectorizer.transform([text]).toarray()[0]
            else:
                # Initial fitting with current text
                vector = self.tfidf_vectorizer.fit_transform([text]).toarray()[0]
                self._fitted_vectorizer = self.tfidf_vectorizer
                
            return zlib.compress(vector.astype(np.float32).tobytes())
            
        except Exception as e:
            print(f"❌ Vector computation error: {e}")
            # Fallback to simple hash-based vector
            hash_val = hashlib.md5(text.encode()).hexdigest()
            return zlib.compress(np.array([int(hash_val[i:i+2], 16) for i in range(0, 32, 2)]).astype(np.float32).tobytes())
            
    async def store_research(
        self, 
        query: str, 
        response: str, 
        cli_tool: str,
        api_provider: str = "unknown",
        token_count: int = 0,
        cost_estimate: float = 0.0,
        tags: List[str] = None,
        context: Dict[str, Any] = None
    ) -> str:
        """Store research entry with deduplication"""
        
        query_hash = self._compute_query_hash(query)
        response_hash = self._compute_response_hash(response)
        
        # Check if already exists
        if query_hash in self.bloom_filter:
            existing = await self.find_similar_research(query, threshold=0.95)
            if existing:
                print(f"🔄 Duplicate research detected, skipping storage")
                return existing[0].id
                
        # Generate unique ID
        entry_id = hashlib.sha256(f"{query_hash}_{response_hash}_{time.time()}".encode()).hexdigest()[:16]
        
        # Compute similarity vector
        similarity_vector = self._compute_similarity_vector(query + " " + response)
        
        # Create research entry
        entry = ResearchEntry(
            id=entry_id,
            query=query,
            query_hash=query_hash,
            response=response,
            response_hash=response_hash,
            cli_tool=cli_tool,
            api_provider=api_provider,
            timestamp=time.time(),
            token_count=token_count,
            cost_estimate=cost_estimate,
            similarity_vector=similarity_vector,
            tags=tags or [],
            context=context or {}
        )
        
        # Store in database
        await self._store_entry(entry)
        
        # Update bloom filter
        self.bloom_filter.add(query_hash)
        
        # Update query patterns
        await self._update_query_patterns(query)
        
        # Clear cache
        self.query_cache.clear()
        
        print(f"💾 Stored research: {entry_id} ({len(response)} chars, ${cost_estimate:.4f})")
        return entry_id
        
    async def _store_entry(self, entry: ResearchEntry):
        """Store research entry in database"""
        con = sqlite3.connect(self.db_path)
        try:
            # Insert main entry
            con.execute('''
                INSERT OR REPLACE INTO research_entries 
                (id, query, query_hash, response, response_hash, cli_tool, api_provider, 
                 timestamp, token_count, cost_estimate, similarity_vector, tags, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.id, entry.query, entry.query_hash, entry.response, entry.response_hash,
                entry.cli_tool, entry.api_provider, entry.timestamp, entry.token_count,
                entry.cost_estimate, entry.similarity_vector, 
                json.dumps(entry.tags), json.dumps(entry.context)
            ))
            
            # Insert into FTS index
            con.execute('''
                INSERT OR REPLACE INTO research_fts (id, query, response)
                VALUES (?, ?, ?)
            ''', (entry.id, entry.query, entry.response))
            
            con.commit()
            
        finally:
            con.close()
            
    async def _update_query_patterns(self, query: str):
        """Update query pattern statistics"""
        # Extract query patterns (simplified)
        pattern = re.sub(r'\b\d+\b', '#NUM', query.lower())
        pattern = re.sub(r'\b[A-Za-z]+\d+[A-Za-z]*\b', '#ALPHANUMERIC', pattern)
        
        con = sqlite3.connect(self.db_path)
        try:
            con.execute('''
                INSERT INTO query_patterns (pattern, count, last_seen) 
                VALUES (?, 1, ?) 
                ON CONFLICT(pattern) DO UPDATE SET 
                count = count + 1, last_seen = ?
            ''', (pattern, time.time(), time.time()))
            con.commit()
        finally:
            con.close()
            
    async def find_similar_research(
        self, 
        query: str, 
        threshold: float = 0.8,
        limit: int = 10,
        cli_tool: str = None
    ) -> List[ResearchEntry]:
        """Find similar research entries"""
        
        query_hash = self._compute_query_hash(query)
        
        # Fast bloom filter check
        if query_hash in self.bloom_filter:
            # Exact match check
            exact_match = await self._get_by_query_hash(query_hash)
            if exact_match:
                return [exact_match]
                
        # Semantic similarity search
        candidates = await self._get_similarity_candidates(query, limit * 3, cli_tool)
        if not candidates:
            return []
            
        # Compute query vector
        query_vector = self._compute_similarity_vector(query)
        similar_entries = []
        
        for candidate in candidates:
            if candidate.similarity_vector:
                similarity = await self._compute_vector_similarity(
                    query_vector, 
                    candidate.similarity_vector
                )
                
                if similarity >= threshold:
                    similar_entries.append((similarity, candidate))
                    
        # Sort by similarity and return top results
        similar_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similar_entries[:limit]]
        
    async def _get_by_query_hash(self, query_hash: str) -> Optional[ResearchEntry]:
        """Get research entry by exact query hash"""
        con = sqlite3.connect(self.db_path)
        try:
            row = con.execute('''
                SELECT * FROM research_entries WHERE query_hash = ?
            ''', (query_hash,)).fetchone()
            
            if row:
                return self._row_to_entry(row)
            return None
        finally:
            con.close()
            
    async def _get_similarity_candidates(
        self, 
        query: str, 
        limit: int, 
        cli_tool: str = None
    ) -> List[ResearchEntry]:
        """Get similarity search candidates"""
        
        con = sqlite3.connect(self.db_path)
        candidates = []
        
        try:
            # FTS search first
            fts_query = ' '.join(query.split()[:10])  # Limit query length
            fts_sql = '''
                SELECT r.* FROM research_entries r
                JOIN research_fts fts ON r.id = fts.id
                WHERE fts MATCH ?
            '''
            fts_params = [fts_query]
            
            if cli_tool:
                fts_sql += ' AND r.cli_tool = ?'
                fts_params.append(cli_tool)
                
            fts_sql += ' ORDER BY rank LIMIT ?'
            fts_params.append(limit)
            
            for row in con.execute(fts_sql, fts_params):
                candidates.append(self._row_to_entry(row))
                
            # If not enough candidates, get recent entries
            if len(candidates) < limit:
                recent_sql = '''
                    SELECT * FROM research_entries 
                    WHERE id NOT IN ({})
                '''.format(','.join('?' * len(candidates)))
                recent_params = [c.id for c in candidates]
                
                if cli_tool:
                    recent_sql += ' AND cli_tool = ?'
                    recent_params.append(cli_tool)
                    
                recent_sql += ' ORDER BY timestamp DESC LIMIT ?'
                recent_params.append(limit - len(candidates))
                
                for row in con.execute(recent_sql, recent_params):
                    candidates.append(self._row_to_entry(row))
                    
        finally:
            con.close()
            
        return candidates
        
    async def _compute_vector_similarity(self, vec1: bytes, vec2: bytes) -> float:
        """Compute cosine similarity between vectors"""
        try:
            # Decompress vectors
            arr1 = np.frombuffer(zlib.decompress(vec1), dtype=np.float32)
            arr2 = np.frombuffer(zlib.decompress(vec2), dtype=np.float32)
            
            # Ensure same dimensions
            min_len = min(len(arr1), len(arr2))
            arr1 = arr1[:min_len]
            arr2 = arr2[:min_len]
            
            # Compute cosine similarity
            dot_product = np.dot(arr1, arr2)
            norms = np.linalg.norm(arr1) * np.linalg.norm(arr2)
            
            if norms == 0:
                return 0.0
                
            return float(dot_product / norms)
            
        except Exception as e:
            print(f"❌ Similarity computation error: {e}")
            return 0.0
            
    def _row_to_entry(self, row) -> ResearchEntry:
        """Convert database row to ResearchEntry"""
        return ResearchEntry(
            id=row[0],
            query=row[1],
            query_hash=row[2],
            response=row[3],
            response_hash=row[4],
            cli_tool=row[5],
            api_provider=row[6],
            timestamp=row[7],
            token_count=row[8] or 0,
            cost_estimate=row[9] or 0.0,
            similarity_vector=row[10],
            tags=json.loads(row[11] or '[]'),
            context=json.loads(row[12] or '{}')
        )
        
    async def get_research_stats(self) -> Dict[str, Any]:
        """Get research database statistics"""
        con = sqlite3.connect(self.db_path)
        try:
            stats = {}
            
            # Total entries
            stats['total_entries'] = con.execute('SELECT COUNT(*) FROM research_entries').fetchone()[0]
            
            # Total cost saved
            stats['total_cost_saved'] = con.execute('SELECT SUM(cost_estimate) FROM research_entries').fetchone()[0] or 0.0
            
            # Entries by CLI tool
            stats['by_cli_tool'] = {}
            for row in con.execute('SELECT cli_tool, COUNT(*) FROM research_entries GROUP BY cli_tool'):
                stats['by_cli_tool'][row[0]] = row[1]
                
            # Entries by API provider
            stats['by_api_provider'] = {}
            for row in con.execute('SELECT api_provider, COUNT(*) FROM research_entries GROUP BY api_provider'):
                stats['by_api_provider'][row[0]] = row[1]
                
            # Recent activity (last 7 days)
            week_ago = time.time() - (7 * 24 * 3600)
            stats['recent_entries'] = con.execute(
                'SELECT COUNT(*) FROM research_entries WHERE timestamp > ?', 
                (week_ago,)
            ).fetchone()[0]
            
            # Top query patterns
            stats['top_patterns'] = []
            for row in con.execute('SELECT pattern, count FROM query_patterns ORDER BY count DESC LIMIT 10'):
                stats['top_patterns'].append({'pattern': row[0], 'count': row[1]})
                
            # Database size
            stats['db_size_mb'] = Path(self.db_path).stat().st_size / (1024 * 1024)
            
            return stats
            
        finally:
            con.close()
            
    async def cleanup_old_research(self, days_old: int = 30) -> int:
        """Clean up research entries older than specified days"""
        cutoff_time = time.time() - (days_old * 24 * 3600)
        
        con = sqlite3.connect(self.db_path)
        try:
            # Get IDs to delete
            old_ids = [row[0] for row in con.execute(
                'SELECT id FROM research_entries WHERE timestamp < ?', 
                (cutoff_time,)
            )]
            
            if not old_ids:
                return 0
                
            # Delete from all tables
            placeholders = ','.join('?' * len(old_ids))
            
            con.execute(f'DELETE FROM research_entries WHERE id IN ({placeholders})', old_ids)
            con.execute(f'DELETE FROM research_fts WHERE id IN ({placeholders})', old_ids)
            con.execute(f'DELETE FROM similarity_index WHERE id1 IN ({placeholders}) OR id2 IN ({placeholders})', old_ids * 2)
            
            # Clean up old query patterns
            con.execute('DELETE FROM query_patterns WHERE last_seen < ?', (cutoff_time,))
            
            con.commit()
            
            # Rebuild bloom filter
            self.bloom_filter.clear()
            self._load_bloom_filter()
            
            print(f"🧹 Cleaned up {len(old_ids)} old research entries")
            return len(old_ids)
            
        finally:
            con.close()
            
    async def export_research(self, output_path: str, format: str = 'json') -> bool:
        """Export research database"""
        try:
            con = sqlite3.connect(self.db_path)
            entries = []
            
            for row in con.execute('SELECT * FROM research_entries ORDER BY timestamp'):
                entry = self._row_to_entry(row)
                entries.append(asdict(entry))
                
            con.close()
            
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(entries, f, indent=2, default=str)
            elif format == 'pickle':
                with open(output_path, 'wb') as f:
                    pickle.dump(entries, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            print(f"📤 Exported {len(entries)} research entries to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            return False
            
    async def import_research(self, input_path: str, format: str = 'json') -> int:
        """Import research database"""
        try:
            if format == 'json':
                with open(input_path, 'r') as f:
                    entries_data = json.load(f)
            elif format == 'pickle':
                with open(input_path, 'rb') as f:
                    entries_data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            imported_count = 0
            
            for entry_data in entries_data:
                # Convert dict back to ResearchEntry
                entry = ResearchEntry(**entry_data)
                
                # Check if already exists
                if not await self._get_by_query_hash(entry.query_hash):
                    await self._store_entry(entry)
                    self.bloom_filter.add(entry.query_hash)
                    imported_count += 1
                    
            print(f"📥 Imported {imported_count} new research entries")
            return imported_count
            
        except Exception as e:
            print(f"❌ Import error: {e}")
            return 0

# High-level API for CLI tools
class ResearchAPI:
    """High-level research API for CLI tools"""
    
    def __init__(self):
        self.db = ResearchDatabase()
        
    async def before_api_call(self, query: str, cli_tool: str) -> Optional[Dict[str, Any]]:
        """Check if research already exists before making API call"""
        similar = await self.db.find_similar_research(query, threshold=0.85, cli_tool=cli_tool)
        
        if similar:
            best_match = similar[0]
            print(f"🎯 Found similar research (saved ${best_match.cost_estimate:.4f})")
            
            return {
                'found_similar': True,
                'response': best_match.response,
                'original_query': best_match.query,
                'timestamp': best_match.timestamp,
                'similarity_score': 0.95,  # Simplified
                'cost_saved': best_match.cost_estimate
            }
            
        return {'found_similar': False}
        
    async def after_api_call(
        self,
        query: str,
        response: str,
        cli_tool: str,
        api_provider: str = "unknown",
        token_count: int = 0,
        cost_estimate: float = 0.0
    ) -> str:
        """Store research after successful API call"""
        return await self.db.store_research(
            query=query,
            response=response,
            cli_tool=cli_tool,
            api_provider=api_provider,
            token_count=token_count,
            cost_estimate=cost_estimate
        )
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get research statistics"""
        return await self.db.get_research_stats()
        
    async def cleanup(self, days_old: int = 30) -> int:
        """Clean up old research"""
        return await self.db.cleanup_old_research(days_old)


# Example usage for CLI integration
async def integrate_with_cli_tool(cli_tool_name: str):
    """Example integration with CLI tool"""
    research_api = ResearchAPI()
    
    async def enhanced_api_call(query: str, original_api_func):
        """Wrapper around API calls with research deduplication"""
        
        # Check for existing research
        result = await research_api.before_api_call(query, cli_tool_name)
        
        if result['found_similar']:
            print(f"💡 Using cached research result")
            return result['response']
            
        # Make actual API call
        print(f"🌐 Making new API call...")
        response = await original_api_func(query)
        
        # Store research
        await research_api.after_api_call(
            query=query,
            response=response,
            cli_tool=cli_tool_name,
            api_provider="openai",  # or detected provider
            token_count=len(response.split()),  # rough estimate
            cost_estimate=0.002 * len(response.split())  # rough estimate
        )
        
        return response
        
    return enhanced_api_call

if __name__ == "__main__":
    async def test_research_system():
        db = ResearchDatabase()
        
        # Store some test research
        await db.store_research(
            query="How to implement async/await in Python?",
            response="Async/await in Python allows you to write asynchronous code...",
            cli_tool="claude-code",
            api_provider="openai",
            cost_estimate=0.05
        )
        
        # Search for similar research
        similar = await db.find_similar_research("How do I use async await in Python?")
        print(f"Found {len(similar)} similar entries")
        
        # Get stats
        stats = await db.get_research_stats()
        print(f"Research stats: {stats}")
        
    asyncio.run(test_research_system())
