#!/usr/bin/env python3
"""
NeuralSync Agent Synchronization System
Shared memory, persona consistency, and agent coordination
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import threading
from collections import defaultdict
import sqlite3
from contextlib import contextmanager

from .config import load_config, DEFAULT_HOME
from .ultra_comm import get_comm_manager, MessageTypes, CommunicationManager
from .crdt import ByzantineCRDT, AdvancedVersion

logger = logging.getLogger(__name__)


@dataclass
class AgentSession:
    """Agent session information"""
    agent_id: str
    cli_tool: str
    capabilities: Set[str]
    last_seen: float
    session_start: float
    memory_version: str
    persona_version: str
    active_tasks: List[str]
    status: str  # active, idle, busy, disconnected


@dataclass
class SharedMemoryItem:
    """Shared memory item between agents"""
    id: str
    content: str
    kind: str
    source_agent: str
    timestamp: float
    relevance_score: float
    access_count: int
    last_accessed: float
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class PersonaState:
    """Persona state information"""
    base_persona: str
    session_adaptations: Dict[str, str]
    global_context: str
    version_hash: str
    last_updated: float
    consistency_level: float


class AgentSynchronizer:
    """Advanced agent synchronization and coordination system"""
    
    def __init__(self, config_dir: Path = DEFAULT_HOME):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for persistent state
        self.db_path = self.config_dir / "agent_sync.db"
        self.init_database()
        
        # In-memory state
        self.active_sessions: Dict[str, AgentSession] = {}
        self.shared_memory: Dict[str, SharedMemoryItem] = {}
        self.persona_state: Optional[PersonaState] = None
        self.memory_locks = defaultdict(threading.RLock)
        
        # CRDT for conflict resolution
        self.ns_config = load_config()
        self.crdt = ByzantineCRDT(self.ns_config.site_id)
        
        # Communication manager
        self.comm_manager: Optional[CommunicationManager] = None
        
        # Background tasks
        self.cleanup_task = None
        self.sync_task = None
        self.running = False
        
        # Load initial state
        self._load_persistent_state()
        
    def init_database(self):
        """Initialize SQLite database for persistent state"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS agent_sessions (
                agent_id TEXT PRIMARY KEY,
                cli_tool TEXT NOT NULL,
                capabilities TEXT NOT NULL,
                last_seen REAL NOT NULL,
                session_start REAL NOT NULL,
                memory_version TEXT NOT NULL,
                persona_version TEXT NOT NULL,
                active_tasks TEXT NOT NULL,
                status TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS shared_memory (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_agent TEXT NOT NULL,
                timestamp REAL NOT NULL,
                relevance_score REAL NOT NULL,
                access_count INTEGER NOT NULL,
                last_accessed REAL NOT NULL,
                tags TEXT NOT NULL,
                metadata TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS persona_state (
                id INTEGER PRIMARY KEY,
                base_persona TEXT NOT NULL,
                session_adaptations TEXT NOT NULL,
                global_context TEXT NOT NULL,
                version_hash TEXT NOT NULL,
                last_updated REAL NOT NULL,
                consistency_level REAL NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_sessions_last_seen ON agent_sessions(last_seen);
            CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON shared_memory(timestamp);
            CREATE INDEX IF NOT EXISTS idx_memory_relevance ON shared_memory(relevance_score);
            """)
            
    def _load_persistent_state(self):
        """Load persistent state from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Load sessions
                for row in conn.execute("SELECT * FROM agent_sessions"):
                    session = AgentSession(
                        agent_id=row['agent_id'],
                        cli_tool=row['cli_tool'],
                        capabilities=set(json.loads(row['capabilities'])),
                        last_seen=row['last_seen'],
                        session_start=row['session_start'],
                        memory_version=row['memory_version'],
                        persona_version=row['persona_version'],
                        active_tasks=json.loads(row['active_tasks']),
                        status=row['status']
                    )
                    self.active_sessions[session.agent_id] = session
                    
                # Load shared memory
                for row in conn.execute("SELECT * FROM shared_memory"):
                    memory_item = SharedMemoryItem(
                        id=row['id'],
                        content=row['content'],
                        kind=row['kind'],
                        source_agent=row['source_agent'],
                        timestamp=row['timestamp'],
                        relevance_score=row['relevance_score'],
                        access_count=row['access_count'],
                        last_accessed=row['last_accessed'],
                        tags=json.loads(row['tags']),
                        metadata=json.loads(row['metadata'])
                    )
                    self.shared_memory[memory_item.id] = memory_item
                    
                # Load persona state
                persona_row = conn.execute("SELECT * FROM persona_state ORDER BY last_updated DESC LIMIT 1").fetchone()
                if persona_row:
                    self.persona_state = PersonaState(
                        base_persona=persona_row['base_persona'],
                        session_adaptations=json.loads(persona_row['session_adaptations']),
                        global_context=persona_row['global_context'],
                        version_hash=persona_row['version_hash'],
                        last_updated=persona_row['last_updated'],
                        consistency_level=persona_row['consistency_level']
                    )
                    
        except Exception as e:
            logger.error(f"Failed to load persistent state: {e}")
            
    def _save_persistent_state(self):
        """Save current state to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing data
                conn.execute("DELETE FROM agent_sessions")
                conn.execute("DELETE FROM shared_memory") 
                conn.execute("DELETE FROM persona_state")
                
                # Save sessions
                for session in self.active_sessions.values():
                    conn.execute("""
                        INSERT INTO agent_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session.agent_id,
                        session.cli_tool,
                        json.dumps(list(session.capabilities)),
                        session.last_seen,
                        session.session_start,
                        session.memory_version,
                        session.persona_version,
                        json.dumps(session.active_tasks),
                        session.status
                    ))
                    
                # Save shared memory
                for memory_item in self.shared_memory.values():
                    conn.execute("""
                        INSERT INTO shared_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        memory_item.id,
                        memory_item.content,
                        memory_item.kind,
                        memory_item.source_agent,
                        memory_item.timestamp,
                        memory_item.relevance_score,
                        memory_item.access_count,
                        memory_item.last_accessed,
                        json.dumps(memory_item.tags),
                        json.dumps(memory_item.metadata)
                    ))
                    
                # Save persona state
                if self.persona_state:
                    conn.execute("""
                        INSERT INTO persona_state VALUES (NULL, ?, ?, ?, ?, ?, ?)
                    """, (
                        self.persona_state.base_persona,
                        json.dumps(self.persona_state.session_adaptations),
                        self.persona_state.global_context,
                        self.persona_state.version_hash,
                        self.persona_state.last_updated,
                        self.persona_state.consistency_level
                    ))
                    
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to save persistent state: {e}")
            
    async def start_synchronization(self):
        """Start the synchronization system"""
        if self.running:
            return
            
        try:
            # Get communication manager
            self.comm_manager = get_comm_manager()
            if not self.comm_manager.running:
                await self.comm_manager.start_system()
                
            # Register sync message handlers
            await self._register_sync_handlers()
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.sync_task = asyncio.create_task(self._synchronization_loop())
            
            self.running = True
            logger.info("🔄 Agent synchronization system started")
            
        except Exception as e:
            logger.error(f"Failed to start synchronization: {e}")
            raise
            
    async def stop_synchronization(self):
        """Stop the synchronization system"""
        self.running = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.sync_task:
            self.sync_task.cancel()
            
        # Wait for tasks to complete
        if self.cleanup_task or self.sync_task:
            tasks = [t for t in [self.cleanup_task, self.sync_task] if t]
            await asyncio.gather(*tasks, return_exceptions=True)
            
        # Save state
        self._save_persistent_state()
        
        logger.info("🛑 Agent synchronization system stopped")
        
    async def _register_sync_handlers(self):
        """Register synchronization message handlers"""
        
        async def handle_session_register(data):
            """Handle agent session registration"""
            agent_id = data.get('agent_id')
            cli_tool = data.get('cli_tool')
            capabilities = set(data.get('capabilities', []))
            
            if agent_id:
                await self.register_agent_session(agent_id, cli_tool, capabilities)
                
        async def handle_memory_share(data):
            """Handle shared memory update"""
            memory_data = data.get('memory', {})
            if memory_data:
                await self.add_shared_memory(
                    content=memory_data.get('content', ''),
                    kind=memory_data.get('kind', 'note'),
                    source_agent=memory_data.get('source_agent', ''),
                    tags=memory_data.get('tags', []),
                    metadata=memory_data.get('metadata', {})
                )
                
        async def handle_persona_sync(data):
            """Handle persona synchronization"""
            persona_updates = data.get('updates', {})
            if persona_updates:
                await self.update_persona_state(persona_updates)
                
        # Register handlers with communication manager
        if self.comm_manager:
            # We would register these with the actual message broker
            pass
            
    async def register_agent_session(self, agent_id: str, cli_tool: str, capabilities: Set[str]) -> bool:
        """Register a new agent session"""
        try:
            current_time = time.time()
            
            session = AgentSession(
                agent_id=agent_id,
                cli_tool=cli_tool,
                capabilities=capabilities,
                last_seen=current_time,
                session_start=current_time,
                memory_version=self._generate_version_hash(),
                persona_version=self.persona_state.version_hash if self.persona_state else "",
                active_tasks=[],
                status="active"
            )
            
            self.active_sessions[agent_id] = session
            
            # Broadcast session registration to other agents
            if self.comm_manager:
                await self.comm_manager.broadcast_to_agents(
                    MessageTypes.AGENT_STATUS.value,
                    {
                        'event': 'agent_registered',
                        'agent_id': agent_id,
                        'cli_tool': cli_tool,
                        'capabilities': list(capabilities),
                        'timestamp': current_time
                    },
                    exclude={agent_id}
                )
                
            logger.info(f"✅ Registered agent session: {agent_id} ({cli_tool})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent session {agent_id}: {e}")
            return False
            
    async def unregister_agent_session(self, agent_id: str):
        """Unregister an agent session"""
        if agent_id in self.active_sessions:
            session = self.active_sessions[agent_id]
            session.status = "disconnected"
            session.last_seen = time.time()
            
            # Broadcast disconnection
            if self.comm_manager:
                await self.comm_manager.broadcast_to_agents(
                    MessageTypes.AGENT_STATUS.value,
                    {
                        'event': 'agent_disconnected',
                        'agent_id': agent_id,
                        'timestamp': time.time()
                    }
                )
                
            logger.info(f"❌ Unregistered agent session: {agent_id}")
            
    async def add_shared_memory(self, content: str, kind: str, source_agent: str, 
                               tags: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Add item to shared memory"""
        try:
            memory_id = self._generate_memory_id(content, source_agent)
            current_time = time.time()
            
            with self.memory_locks[memory_id]:
                memory_item = SharedMemoryItem(
                    id=memory_id,
                    content=content,
                    kind=kind,
                    source_agent=source_agent,
                    timestamp=current_time,
                    relevance_score=self._calculate_relevance_score(content, tags or []),
                    access_count=0,
                    last_accessed=current_time,
                    tags=tags or [],
                    metadata=metadata or {}
                )
                
                self.shared_memory[memory_id] = memory_item
                
                # Store in CRDT for conflict resolution
                self.crdt.set(memory_id, asdict(memory_item))
                
                # Broadcast to other agents
                if self.comm_manager:
                    await self.comm_manager.broadcast_to_agents(
                        MessageTypes.MEMORY_STORE.value,
                        {
                            'memory': asdict(memory_item),
                            'source': source_agent
                        },
                        exclude={source_agent}
                    )
                    
            logger.debug(f"Added shared memory item: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add shared memory: {e}")
            return ""
            
    async def get_shared_memory(self, query: str = "", agent_id: str = "", 
                               limit: int = 10) -> List[SharedMemoryItem]:
        """Retrieve shared memory items"""
        try:
            # Simple filtering - in production would use vector similarity
            items = list(self.shared_memory.values())
            
            # Filter by query if provided
            if query:
                filtered_items = []
                query_lower = query.lower()
                for item in items:
                    if (query_lower in item.content.lower() or 
                        any(query_lower in tag.lower() for tag in item.tags) or
                        query_lower in item.kind.lower()):
                        # Update access count
                        item.access_count += 1
                        item.last_accessed = time.time()
                        filtered_items.append(item)
                items = filtered_items
                
            # Sort by relevance and recency
            items.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
            
            return items[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get shared memory: {e}")
            return []
            
    async def update_persona_state(self, updates: Dict[str, Any]) -> bool:
        """Update persona state with consistency checking"""
        try:
            current_time = time.time()
            
            if not self.persona_state:
                # Initialize persona state
                self.persona_state = PersonaState(
                    base_persona=updates.get('base_persona', ''),
                    session_adaptations={},
                    global_context=updates.get('global_context', ''),
                    version_hash=self._generate_version_hash(),
                    last_updated=current_time,
                    consistency_level=1.0
                )
            else:
                # Update existing state
                if 'base_persona' in updates:
                    self.persona_state.base_persona = updates['base_persona']
                if 'global_context' in updates:
                    self.persona_state.global_context = updates['global_context']
                if 'session_adaptations' in updates:
                    self.persona_state.session_adaptations.update(updates['session_adaptations'])
                    
                self.persona_state.version_hash = self._generate_version_hash()
                self.persona_state.last_updated = current_time
                
            # Store in CRDT
            self.crdt.set("persona_state", asdict(self.persona_state))
            
            # Broadcast persona update to all agents
            if self.comm_manager:
                await self.comm_manager.broadcast_to_agents(
                    MessageTypes.PERSONALITY_UPDATE.value,
                    {
                        'persona_state': asdict(self.persona_state),
                        'timestamp': current_time
                    }
                )
                
            # Update all agent sessions with new persona version
            for session in self.active_sessions.values():
                session.persona_version = self.persona_state.version_hash
                
            logger.info("🔄 Updated persona state")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update persona state: {e}")
            return False
            
    def get_unified_context(self, agent_id: str = "") -> Dict[str, Any]:
        """Get unified context for an agent"""
        try:
            context = {
                'persona': self.persona_state.base_persona if self.persona_state else "",
                'session_adaptations': {},
                'global_context': self.persona_state.global_context if self.persona_state else "",
                'recent_memories': [],
                'active_agents': [],
                'timestamp': time.time()
            }
            
            # Add session-specific adaptations
            if agent_id and self.persona_state:
                context['session_adaptations'] = self.persona_state.session_adaptations.get(agent_id, {})
                
            # Add recent shared memories
            recent_memories = sorted(
                self.shared_memory.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )[:5]
            context['recent_memories'] = [
                {
                    'content': mem.content[:200] + '...' if len(mem.content) > 200 else mem.content,
                    'kind': mem.kind,
                    'source': mem.source_agent,
                    'timestamp': mem.timestamp
                }
                for mem in recent_memories
            ]
            
            # Add active agents info
            context['active_agents'] = [
                {
                    'agent_id': session.agent_id,
                    'cli_tool': session.cli_tool,
                    'capabilities': list(session.capabilities),
                    'status': session.status
                }
                for session in self.active_sessions.values()
                if session.status == 'active'
            ]
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get unified context: {e}")
            return {}
            
    def _generate_memory_id(self, content: str, source: str) -> str:
        """Generate unique memory ID"""
        hash_input = f"{content}{source}{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
    def _generate_version_hash(self) -> str:
        """Generate version hash for state tracking"""
        hash_input = f"{time.time()}{self.ns_config.site_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        
    def _calculate_relevance_score(self, content: str, tags: List[str]) -> float:
        """Calculate relevance score for memory item"""
        # Simple scoring - in production would use ML models
        base_score = 0.5
        
        # Boost for length (more content = potentially more valuable)
        length_score = min(len(content) / 1000, 0.3)
        
        # Boost for tags
        tag_score = min(len(tags) * 0.1, 0.2)
        
        return base_score + length_score + tag_score
        
    async def _cleanup_loop(self):
        """Background cleanup of stale sessions and memories"""
        while self.running:
            try:
                current_time = time.time()
                
                # Remove stale sessions (inactive for >30 minutes)
                stale_sessions = []
                for agent_id, session in self.active_sessions.items():
                    if current_time - session.last_seen > 1800:  # 30 minutes
                        stale_sessions.append(agent_id)
                        
                for agent_id in stale_sessions:
                    await self.unregister_agent_session(agent_id)
                    del self.active_sessions[agent_id]
                    
                # Clean up old shared memory items (>24 hours, low relevance, low access)
                stale_memories = []
                for memory_id, memory_item in self.shared_memory.items():
                    age_hours = (current_time - memory_item.timestamp) / 3600
                    if (age_hours > 24 and 
                        memory_item.relevance_score < 0.3 and 
                        memory_item.access_count < 2):
                        stale_memories.append(memory_id)
                        
                for memory_id in stale_memories:
                    del self.shared_memory[memory_id]
                    
                # Persist state periodically
                self._save_persistent_state()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
                
    async def _synchronization_loop(self):
        """Background synchronization with other instances"""
        while self.running:
            try:
                # Sync with CRDT - resolve conflicts
                # This would sync with other NeuralSync instances
                
                # For now, just update session timestamps
                current_time = time.time()
                for session in self.active_sessions.values():
                    if session.status == 'active':
                        session.last_seen = current_time
                        
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Synchronization loop error: {e}")
                await asyncio.sleep(60)
                
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization system statistics"""
        return {
            'running': self.running,
            'active_sessions': len([s for s in self.active_sessions.values() if s.status == 'active']),
            'total_sessions': len(self.active_sessions),
            'shared_memory_items': len(self.shared_memory),
            'persona_version': self.persona_state.version_hash if self.persona_state else "",
            'persona_consistency': self.persona_state.consistency_level if self.persona_state else 0.0,
            'timestamp': time.time()
        }


# Global synchronizer instance
_agent_synchronizer: Optional[AgentSynchronizer] = None


def get_agent_synchronizer() -> AgentSynchronizer:
    """Get singleton agent synchronizer instance"""
    global _agent_synchronizer
    if _agent_synchronizer is None:
        _agent_synchronizer = AgentSynchronizer()
    return _agent_synchronizer


async def ensure_synchronization_system() -> bool:
    """Ensure the synchronization system is running"""
    try:
        synchronizer = get_agent_synchronizer()
        if not synchronizer.running:
            await synchronizer.start_synchronization()
        return True
    except Exception as e:
        logger.error(f"Failed to ensure synchronization system: {e}")
        return False