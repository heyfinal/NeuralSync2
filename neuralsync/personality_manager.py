"""
Unified Personality Manager
Maintains consistent personality and context across all CLI tools
"""

import json
import time
import hashlib
import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import threading
from pathlib import Path
import pickle
import zlib

logger = logging.getLogger(__name__)

@dataclass
class PersonalityTraits:
    """Core personality traits and characteristics"""
    communication_style: str = "professional_friendly"  # casual, professional, technical, friendly, etc.
    verbosity_level: int = 5  # 1-10 scale
    technical_depth: int = 7  # How deep to go into technical details
    humor_level: int = 3  # Amount of humor/personality
    formality_level: int = 4  # Level of formality
    creativity_level: int = 6  # Creative vs conservative responses
    patience_level: int = 8  # How patient with user questions
    teaching_style: str = "adaptive"  # step_by_step, conceptual, practical, adaptive
    
    def to_prompt_context(self) -> str:
        """Convert traits to prompt context"""
        styles = {
            "casual": "Use a casual, conversational tone",
            "professional": "Maintain a professional, business-appropriate tone",
            "technical": "Focus on technical accuracy and precision", 
            "friendly": "Be warm and approachable",
            "professional_friendly": "Be professional yet approachable"
        }
        
        style_desc = styles.get(self.communication_style, "Be natural and helpful")
        
        context = f"""
Communication Style: {style_desc}
Verbosity: {"Concise" if self.verbosity_level < 4 else "Detailed" if self.verbosity_level > 6 else "Moderate"}
Technical Depth: {"High technical detail" if self.technical_depth > 7 else "Moderate technical detail" if self.technical_depth > 4 else "Simple explanations"}
Teaching Approach: {self.teaching_style.replace('_', ' ').title()}
        """.strip()
        
        return context


@dataclass 
class InteractionMemory:
    """Memory of past interactions"""
    timestamp: float
    cli_tool: str
    session_id: str
    user_input: str
    ai_response: str
    context_tags: Set[str] = field(default_factory=set)
    user_satisfaction: Optional[int] = None  # 1-5 rating if available
    interaction_type: str = "general"  # coding, research, chat, etc.
    success_indicators: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> str:
        """Get concise summary of interaction"""
        return f"{self.interaction_type}: {self.user_input[:100]}..."


@dataclass
class SessionContext:
    """Context for specific CLI tool session"""
    session_id: str
    cli_tool: str
    started_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    interaction_count: int = 0
    current_task: str = ""
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_variables: Dict[str, Any] = field(default_factory=dict)
    working_directory: str = ""
    project_context: str = ""
    recent_files: List[str] = field(default_factory=list)
    
    def update_activity(self):
        """Update last activity time"""
        self.last_active = time.time()
        self.interaction_count += 1


class PersonalityStore:
    """Storage backend for personality data"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".neuralsync2" / "personality"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self.base_personality: Optional[PersonalityTraits] = None
        self.cli_personalities: Dict[str, PersonalityTraits] = {}
        self.session_contexts: Dict[str, SessionContext] = {}
        self.interaction_history: deque = deque(maxlen=10000)
        
        self.lock = threading.RLock()
        
        # Load existing data
        self._load_from_disk()
        
    def _load_from_disk(self):
        """Load personality data from disk"""
        try:
            # Load base personality
            base_file = self.storage_path / "base_personality.json"
            if base_file.exists():
                with open(base_file, 'r') as f:
                    data = json.load(f)
                    self.base_personality = PersonalityTraits(**data)
            else:
                self.base_personality = PersonalityTraits()
                
            # Load CLI-specific personalities
            cli_dir = self.storage_path / "cli_specific"
            if cli_dir.exists():
                for cli_file in cli_dir.glob("*.json"):
                    cli_tool = cli_file.stem
                    with open(cli_file, 'r') as f:
                        data = json.load(f)
                        self.cli_personalities[cli_tool] = PersonalityTraits(**data)
                        
            # Load interaction history (compressed)
            history_file = self.storage_path / "interaction_history.pkl.gz"
            if history_file.exists():
                with open(history_file, 'rb') as f:
                    compressed_data = f.read()
                    data = pickle.loads(zlib.decompress(compressed_data))
                    self.interaction_history = deque(data, maxlen=10000)
                    
            logger.info("Personality data loaded from disk")
            
        except Exception as e:
            logger.error(f"Error loading personality data: {e}")
            # Initialize defaults
            self.base_personality = PersonalityTraits()
            
    def _save_to_disk(self):
        """Save personality data to disk"""
        try:
            # Save base personality
            base_file = self.storage_path / "base_personality.json"
            with open(base_file, 'w') as f:
                json.dump(asdict(self.base_personality), f, indent=2)
                
            # Save CLI-specific personalities
            cli_dir = self.storage_path / "cli_specific"
            cli_dir.mkdir(exist_ok=True)
            
            for cli_tool, personality in self.cli_personalities.items():
                cli_file = cli_dir / f"{cli_tool}.json"
                with open(cli_file, 'w') as f:
                    json.dump(asdict(personality), f, indent=2)
                    
            # Save interaction history (compressed)
            history_file = self.storage_path / "interaction_history.pkl.gz"
            with open(history_file, 'wb') as f:
                data = list(self.interaction_history)
                compressed = zlib.compress(pickle.dumps(data))
                f.write(compressed)
                
            logger.debug("Personality data saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving personality data: {e}")
            
    async def get_base_personality(self) -> PersonalityTraits:
        """Get base personality traits"""
        with self.lock:
            if self.base_personality is None:
                self.base_personality = PersonalityTraits()
            return self.base_personality
            
    async def get_cli_personality(self, cli_tool: str) -> PersonalityTraits:
        """Get CLI-specific personality traits"""
        with self.lock:
            if cli_tool not in self.cli_personalities:
                # Create CLI-specific personality based on tool type
                base = await self.get_base_personality()
                cli_personality = PersonalityTraits(**asdict(base))
                
                # Customize based on CLI tool
                if cli_tool == 'claude-code':
                    cli_personality.technical_depth = 8
                    cli_personality.communication_style = "professional_friendly"
                    cli_personality.teaching_style = "step_by_step"
                elif cli_tool == 'gemini':
                    cli_personality.creativity_level = 8
                    cli_personality.verbosity_level = 6
                elif cli_tool == 'codex-cli':
                    cli_personality.technical_depth = 9
                    cli_personality.communication_style = "technical"
                    cli_personality.verbosity_level = 4
                    
                self.cli_personalities[cli_tool] = cli_personality
                
            return self.cli_personalities[cli_tool]
            
    async def get_session_context(self, session_id: str) -> SessionContext:
        """Get session context"""
        with self.lock:
            if session_id not in self.session_contexts:
                # Extract CLI tool from session ID if possible
                cli_tool = session_id.split('_')[0] if '_' in session_id else 'unknown'
                self.session_contexts[session_id] = SessionContext(
                    session_id=session_id,
                    cli_tool=cli_tool
                )
                
            context = self.session_contexts[session_id]
            context.update_activity()
            return context
            
    async def update_personality(self, cli_tool: str, updates: Dict[str, Any]) -> bool:
        """Update personality traits"""
        with self.lock:
            try:
                personality = await self.get_cli_personality(cli_tool)
                
                # Apply updates
                for key, value in updates.items():
                    if hasattr(personality, key):
                        setattr(personality, key, value)
                        
                # Save to disk
                self._save_to_disk()
                
                logger.info(f"Personality updated for {cli_tool}: {updates}")
                return True
                
            except Exception as e:
                logger.error(f"Error updating personality: {e}")
                return False
                
    async def add_interaction(self, interaction: InteractionMemory):
        """Add interaction to history"""
        with self.lock:
            self.interaction_history.append(interaction)
            
            # Save periodically
            if len(self.interaction_history) % 100 == 0:
                self._save_to_disk()


class ContextMerger:
    """Intelligent context merging system"""
    
    def __init__(self):
        self.merge_strategies = {
            'personality_traits': self._merge_personality_traits,
            'interaction_history': self._merge_interaction_history,
            'session_context': self._merge_session_context,
            'user_preferences': self._merge_user_preferences
        }
        
    async def merge(self,
                   base_personality: PersonalityTraits,
                   cli_personality: PersonalityTraits,
                   session_context: SessionContext,
                   recent_interactions: List[InteractionMemory]) -> Dict[str, Any]:
        """Merge all context sources into unified context"""
        
        # Start with base personality
        merged_traits = self._merge_personality_traits(base_personality, cli_personality)
        
        # Extract insights from recent interactions
        interaction_insights = self._analyze_interaction_patterns(recent_interactions)
        
        # Build unified context
        unified_context = {
            'personality_traits': merged_traits,
            'session_context': session_context,
            'interaction_insights': interaction_insights,
            'context_summary': self._generate_context_summary(
                merged_traits, session_context, interaction_insights
            )
        }
        
        return unified_context
        
    def _merge_personality_traits(self, base: PersonalityTraits, cli: PersonalityTraits) -> PersonalityTraits:
        """Merge base and CLI-specific personality traits"""
        
        # CLI-specific traits take precedence
        merged = PersonalityTraits(**asdict(cli))
        
        # But fall back to base for any missing values
        for field_name, field_def in PersonalityTraits.__dataclass_fields__.items():
            cli_value = getattr(cli, field_name)
            base_value = getattr(base, field_name)
            
            # Use CLI value if significantly different from default
            if cli_value == field_def.default and base_value != field_def.default:
                setattr(merged, field_name, base_value)
                
        return merged
        
    def _merge_interaction_history(self, histories: List[List[InteractionMemory]]) -> List[InteractionMemory]:
        """Merge multiple interaction histories"""
        
        all_interactions = []
        for history in histories:
            all_interactions.extend(history)
            
        # Sort by timestamp and deduplicate
        all_interactions.sort(key=lambda x: x.timestamp, reverse=True)
        
        seen_hashes = set()
        unique_interactions = []
        
        for interaction in all_interactions:
            # Create hash based on input/response content
            content_hash = hashlib.md5(
                f"{interaction.user_input}{interaction.ai_response}".encode()
            ).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_interactions.append(interaction)
                
        return unique_interactions[:100]  # Keep most recent 100
        
    def _analyze_interaction_patterns(self, interactions: List[InteractionMemory]) -> Dict[str, Any]:
        """Analyze patterns in user interactions"""
        
        if not interactions:
            return {}
            
        # Analyze interaction types
        type_counts = defaultdict(int)
        satisfaction_scores = []
        common_topics = defaultdict(int)
        
        for interaction in interactions:
            type_counts[interaction.interaction_type] += 1
            
            if interaction.user_satisfaction:
                satisfaction_scores.append(interaction.user_satisfaction)
                
            # Extract topics from context tags
            for tag in interaction.context_tags:
                common_topics[tag] += 1
                
        # Calculate insights
        insights = {
            'primary_interaction_type': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'general',
            'avg_satisfaction': sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else None,
            'interaction_count': len(interactions),
            'common_topics': dict(sorted(common_topics.items(), key=lambda x: x[1], reverse=True)[:10]),
            'recent_focus': self._extract_recent_focus(interactions[:10])  # Last 10 interactions
        }
        
        return insights
        
    def _extract_recent_focus(self, recent_interactions: List[InteractionMemory]) -> str:
        """Extract what the user has been focusing on recently"""
        
        if not recent_interactions:
            return "general assistance"
            
        # Look for patterns in recent interactions
        keywords = defaultdict(int)
        
        for interaction in recent_interactions:
            # Extract keywords from user input
            words = interaction.user_input.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    keywords[word] += 1
                    
        if keywords:
            top_keyword = max(keywords.items(), key=lambda x: x[1])[0]
            return f"working with {top_keyword}"
        else:
            return "general assistance"
            
    def _generate_context_summary(self,
                                 traits: PersonalityTraits,
                                 session: SessionContext,
                                 insights: Dict[str, Any]) -> str:
        """Generate human-readable context summary"""
        
        summary_parts = [
            f"Communication style: {traits.communication_style}",
            f"Technical depth: {traits.technical_depth}/10",
            f"Session: {session.cli_tool}"
        ]
        
        if session.current_task:
            summary_parts.append(f"Current task: {session.current_task}")
            
        if insights.get('recent_focus'):
            summary_parts.append(f"Recent focus: {insights['recent_focus']}")
            
        if insights.get('primary_interaction_type'):
            summary_parts.append(f"Primary interaction type: {insights['primary_interaction_type']}")
            
        return " | ".join(summary_parts)


class TTLCache:
    """Simple TTL cache for personality contexts"""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.default_ttl = default_ttl
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        with self.lock:
            if key not in self.cache:
                return None
                
            value, expiry = self.cache[key]
            
            if time.time() > expiry:
                del self.cache[key]
                return None
                
            return value
            
    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        """Cache value with TTL"""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        with self.lock:
            self.cache[key] = (value, expiry)
            
    def invalidate(self, key: str):
        """Invalidate cached value"""
        with self.lock:
            self.cache.pop(key, None)
            
    def clear(self):
        """Clear all cached values"""
        with self.lock:
            self.cache.clear()


class UnifiedPersonalityManager:
    """Main personality management system"""
    
    def __init__(self):
        self.personality_store = PersonalityStore()
        self.context_merger = ContextMerger()
        self.personality_cache = TTLCache(default_ttl=300)  # 5min TTL
        
        # CLI tool tracking
        self.connected_tools: Set[str] = set()
        self.tool_connections: Dict[str, float] = {}  # tool -> last_seen
        
        # Personality propagation
        self.propagation_tasks: Dict[str, asyncio.Task] = {}
        
        self.lock = threading.RLock()
        
    async def get_unified_personality(self, cli_tool: str, session_id: str) -> Dict[str, Any]:
        """Get unified personality context for CLI tool session"""
        
        cache_key = f"{cli_tool}:{session_id}"
        
        # Check cache first
        cached = self.personality_cache.get(cache_key)
        if cached:
            return cached
            
        # Merge personality from all sources
        base_personality = await self.personality_store.get_base_personality()
        cli_personality = await self.personality_store.get_cli_personality(cli_tool)
        session_context = await self.personality_store.get_session_context(session_id)
        
        # Get recent interactions for this CLI tool
        recent_interactions = await self._get_recent_interactions(cli_tool, limit=50)
        
        # Merge all contexts
        unified = await self.context_merger.merge(
            base_personality,
            cli_personality, 
            session_context,
            recent_interactions
        )
        
        # Add prompt-ready context
        unified['prompt_context'] = self._generate_prompt_context(unified)
        
        # Cache result
        self.personality_cache.put(cache_key, unified)
        
        # Track CLI tool connection
        self._track_cli_connection(cli_tool)
        
        return unified
        
    async def _get_recent_interactions(self, cli_tool: str, limit: int = 50) -> List[InteractionMemory]:
        """Get recent interactions for CLI tool"""
        
        with self.lock:
            # Filter interactions by CLI tool
            cli_interactions = [
                interaction for interaction in self.personality_store.interaction_history
                if interaction.cli_tool == cli_tool
            ]
            
            # Sort by timestamp (most recent first)
            cli_interactions.sort(key=lambda x: x.timestamp, reverse=True)
            
            return cli_interactions[:limit]
            
    def _generate_prompt_context(self, unified_context: Dict[str, Any]) -> str:
        """Generate prompt context from unified personality"""
        
        traits = unified_context['personality_traits']
        session = unified_context['session_context']
        insights = unified_context['interaction_insights']
        
        # Base personality prompt
        prompt_parts = [
            "# Personality Context",
            traits.to_prompt_context()
        ]
        
        # Session context
        if session.current_task:
            prompt_parts.extend([
                "",
                "# Current Session Context",
                f"Current task: {session.current_task}",
                f"CLI tool: {session.cli_tool}"
            ])
            
        if session.project_context:
            prompt_parts.append(f"Project context: {session.project_context}")
            
        # Recent insights
        if insights:
            prompt_parts.extend([
                "",
                "# Recent Interaction Insights"
            ])
            
            if insights.get('recent_focus'):
                prompt_parts.append(f"Recent focus: {insights['recent_focus']}")
                
            if insights.get('primary_interaction_type'):
                prompt_parts.append(f"Primary interaction type: {insights['primary_interaction_type']}")
                
            if insights.get('avg_satisfaction'):
                satisfaction = insights['avg_satisfaction']
                if satisfaction >= 4:
                    prompt_parts.append("User satisfaction: High - continue current approach")
                elif satisfaction <= 2:
                    prompt_parts.append("User satisfaction: Low - adjust communication style")
                    
        return "\n".join(prompt_parts)
        
    async def update_personality_state(self, 
                                     cli_tool: str, 
                                     session_id: str,
                                     updates: Dict[str, Any]) -> None:
        """Update personality state and propagate to connected tools"""
        
        # Apply updates locally
        success = await self.personality_store.update_personality(cli_tool, updates)
        
        if not success:
            return
            
        # Invalidate cache for affected sessions
        self._invalidate_personality_cache(cli_tool)
        
        # Propagate to connected CLI tools
        await self._propagate_personality_updates(cli_tool, updates)
        
    async def _propagate_personality_updates(self, source_cli: str, updates: Dict[str, Any]):
        """Propagate personality updates to other connected CLI tools"""
        
        connected_tools = self._get_connected_tools()
        
        propagation_tasks = []
        for tool in connected_tools:
            if tool != source_cli:
                task = asyncio.create_task(
                    self._send_personality_update(tool, updates)
                )
                propagation_tasks.append(task)
                
        # Wait for propagation with timeout
        if propagation_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*propagation_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Personality propagation timeout")
                
    async def _send_personality_update(self, target_cli: str, updates: Dict[str, Any]):
        """Send personality update to specific CLI tool"""
        
        # This would integrate with the communication system
        from .ultra_comm import get_comm_manager, MessageTypes
        
        try:
            comm = get_comm_manager()
            
            payload = json.dumps({
                'type': 'personality_update',
                'updates': updates,
                'timestamp': time.time()
            }).encode()
            
            await comm.send_message(target_cli, MessageTypes.PERSONALITY_UPDATE, payload)
            
        except Exception as e:
            logger.error(f"Failed to send personality update to {target_cli}: {e}")
            
    def _track_cli_connection(self, cli_tool: str):
        """Track CLI tool connection"""
        with self.lock:
            self.connected_tools.add(cli_tool)
            self.tool_connections[cli_tool] = time.time()
            
    def _get_connected_tools(self) -> List[str]:
        """Get currently connected CLI tools"""
        current_time = time.time()
        connection_timeout = 3600  # 1 hour
        
        with self.lock:
            active_tools = []
            
            for tool, last_seen in list(self.tool_connections.items()):
                if current_time - last_seen < connection_timeout:
                    active_tools.append(tool)
                else:
                    # Remove stale connections
                    self.connected_tools.discard(tool)
                    del self.tool_connections[tool]
                    
            return active_tools
            
    def _invalidate_personality_cache(self, cli_tool: str):
        """Invalidate personality cache entries for CLI tool"""
        
        # Find and invalidate all cache keys for this CLI tool
        keys_to_invalidate = []
        for key in self.personality_cache.cache.keys():
            if key.startswith(f"{cli_tool}:"):
                keys_to_invalidate.append(key)
                
        for key in keys_to_invalidate:
            self.personality_cache.invalidate(key)
            
    async def record_interaction(self,
                               cli_tool: str,
                               session_id: str,
                               user_input: str,
                               ai_response: str,
                               interaction_type: str = "general",
                               context_tags: Optional[Set[str]] = None):
        """Record interaction in personality system"""
        
        interaction = InteractionMemory(
            timestamp=time.time(),
            cli_tool=cli_tool,
            session_id=session_id,
            user_input=user_input,
            ai_response=ai_response,
            interaction_type=interaction_type,
            context_tags=context_tags or set()
        )
        
        await self.personality_store.add_interaction(interaction)
        
        # Invalidate cache to force fresh personality context
        cache_key = f"{cli_tool}:{session_id}"
        self.personality_cache.invalidate(cache_key)
        
    def get_personality_stats(self) -> Dict[str, Any]:
        """Get personality system statistics"""
        
        with self.lock:
            return {
                'connected_tools': len(self.connected_tools),
                'active_sessions': len(self.personality_store.session_contexts),
                'cached_personalities': len(self.personality_cache.cache),
                'interaction_history_size': len(self.personality_store.interaction_history),
                'cli_personalities': len(self.personality_store.cli_personalities),
                'last_disk_save': getattr(self.personality_store, '_last_save_time', 0)
            }


# Global personality manager
_global_personality_manager: Optional[UnifiedPersonalityManager] = None

def get_personality_manager() -> UnifiedPersonalityManager:
    """Get global personality manager"""
    global _global_personality_manager
    if _global_personality_manager is None:
        _global_personality_manager = UnifiedPersonalityManager()
    return _global_personality_manager


async def demo_personality_system():
    """Demonstrate personality system functionality"""
    
    manager = get_personality_manager()
    
    # Get unified personality for claude-code
    personality = await manager.get_unified_personality('claude-code', 'demo_session')
    
    print("Unified Personality Context:")
    print("=" * 40)
    print(personality['prompt_context'])
    print()
    
    # Record an interaction
    await manager.record_interaction(
        'claude-code',
        'demo_session', 
        "How do I implement a binary search tree?",
        "I'll help you implement a binary search tree...",
        'coding',
        {'data_structures', 'algorithms'}
    )
    
    # Update personality
    await manager.update_personality_state(
        'claude-code',
        'demo_session',
        {'technical_depth': 9, 'teaching_style': 'step_by_step'}
    )
    
    # Get updated personality
    updated_personality = await manager.get_unified_personality('claude-code', 'demo_session')
    
    print("Updated Personality Context:")
    print("=" * 40)
    print(updated_personality['prompt_context'])
    
    # Get stats
    stats = manager.get_personality_stats()
    print(f"\nPersonality Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(demo_personality_system())