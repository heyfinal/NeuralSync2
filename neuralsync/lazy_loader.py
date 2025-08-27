#!/usr/bin/env python3
"""
Lazy Loading System for NeuralSync v2
Smart context loading with bypass modes and performance trade-offs
"""

import asyncio
import time
import os
import logging
import threading
from typing import Optional, Dict, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from .intelligent_cache import get_neuralsync_cache
from .async_network import get_network_client
from .context_prewarmer import get_context_prewarmer

logger = logging.getLogger(__name__)

class LoadingMode(Enum):
    """Context loading modes with different speed/context trade-offs"""
    BYPASS = "bypass"           # No context loading - maximum speed
    MINIMAL = "minimal"         # Only persona, no memories - fast
    BALANCED = "balanced"       # Persona + reduced memories - moderate  
    COMPLETE = "complete"       # Full context - slower but comprehensive
    ADAPTIVE = "adaptive"       # Dynamic mode selection based on conditions

@dataclass
class LoadingStrategy:
    """Configuration for lazy loading strategy"""
    mode: LoadingMode
    max_wait_ms: int = 800
    enable_caching: bool = True
    enable_prewarming: bool = True
    fallback_mode: LoadingMode = LoadingMode.MINIMAL
    adaptive_threshold_ms: int = 500
    
    # Context size limits for different modes
    minimal_memory_count: int = 1
    balanced_memory_count: int = 2
    complete_memory_count: int = 5

@dataclass
class LoadingResult:
    """Result of lazy loading operation"""
    context: str
    mode_used: LoadingMode
    loading_time_ms: float
    from_cache: bool
    prewarmed: bool
    fallback_used: bool
    persona_loaded: bool
    memories_loaded: int
    total_size_bytes: int

class ContextAnalyzer:
    """Analyze context requirements and determine optimal loading strategy"""
    
    def __init__(self):
        self.tool_patterns = {
            # Fast tools that rarely need context
            'ls': LoadingMode.BYPASS,
            'pwd': LoadingMode.BYPASS,
            'cd': LoadingMode.BYPASS,
            'clear': LoadingMode.BYPASS,
            'echo': LoadingMode.BYPASS,
            
            # Tools that benefit from minimal context
            'git': LoadingMode.MINIMAL,
            'npm': LoadingMode.MINIMAL,
            'pip': LoadingMode.MINIMAL,
            
            # Tools that need balanced context
            'python': LoadingMode.BALANCED,
            'node': LoadingMode.BALANCED,
            'ruby': LoadingMode.BALANCED,
            
            # Tools that benefit from full context
            'claude-code': LoadingMode.COMPLETE,
            'codexcli': LoadingMode.COMPLETE,
            'gemini': LoadingMode.BALANCED,
        }
        
        self.command_patterns = {
            '--help': LoadingMode.BYPASS,
            '--version': LoadingMode.BYPASS,
            '-h': LoadingMode.BYPASS,
            '-v': LoadingMode.BYPASS,
            'help': LoadingMode.BYPASS,
            'version': LoadingMode.BYPASS,
        }
        
        self.query_complexity_thresholds = {
            LoadingMode.BYPASS: 0,
            LoadingMode.MINIMAL: 5,      # Very short queries
            LoadingMode.BALANCED: 20,    # Medium queries
            LoadingMode.COMPLETE: 50     # Long, complex queries
        }
    
    def analyze_context_needs(self, tool: Optional[str], query: str = "", 
                             command_args: Optional[list] = None) -> LoadingMode:
        """Analyze and recommend optimal loading mode"""
        
        # Check for bypass conditions first
        if command_args:
            first_arg = command_args[0] if command_args else ""
            if first_arg in self.command_patterns:
                return self.command_patterns[first_arg]
        
        # Tool-specific patterns
        if tool and tool in self.tool_patterns:
            tool_mode = self.tool_patterns[tool]
            
            # Override for complex queries even on simple tools
            if len(query) > self.query_complexity_thresholds[LoadingMode.COMPLETE]:
                return max(tool_mode, LoadingMode.BALANCED)
            
            return tool_mode
        
        # Query complexity analysis
        query_len = len(query.strip())
        
        if query_len == 0:
            return LoadingMode.MINIMAL
        elif query_len <= self.query_complexity_thresholds[LoadingMode.MINIMAL]:
            return LoadingMode.MINIMAL
        elif query_len <= self.query_complexity_thresholds[LoadingMode.BALANCED]:
            return LoadingMode.BALANCED
        else:
            return LoadingMode.COMPLETE
    
    def should_use_prewarming(self, tool: Optional[str], mode: LoadingMode) -> bool:
        """Determine if prewarming should be used"""
        
        # Don't prewarm for bypass operations
        if mode == LoadingMode.BYPASS:
            return False
        
        # Always prewarm for AI tools
        if tool in ['claude-code', 'codexcli', 'gemini']:
            return True
        
        # Prewarm for balanced and complete modes
        return mode in [LoadingMode.BALANCED, LoadingMode.COMPLETE]

class SystemConditionMonitor:
    """Monitor system conditions to inform adaptive loading decisions"""
    
    def __init__(self):
        self.last_check = 0
        self.check_interval_ms = 5000  # Check every 5 seconds
        self.cached_conditions = {
            'cpu_busy': False,
            'memory_pressure': False,
            'network_slow': False,
            'battery_low': False
        }
    
    def get_system_conditions(self) -> Dict[str, bool]:
        """Get current system conditions (cached for performance)"""
        
        current_time = time.time() * 1000
        
        if current_time - self.last_check > self.check_interval_ms:
            self._update_conditions()
            self.last_check = current_time
        
        return self.cached_conditions.copy()
    
    def _update_conditions(self):
        """Update system condition cache"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cached_conditions['cpu_busy'] = cpu_percent > 80
            
            # Memory pressure
            memory = psutil.virtual_memory()
            self.cached_conditions['memory_pressure'] = memory.percent > 85
            
            # Battery status (for laptops)
            try:
                battery = psutil.sensors_battery()
                if battery:
                    self.cached_conditions['battery_low'] = (
                        battery.percent < 20 and not battery.power_plugged
                    )
                else:
                    self.cached_conditions['battery_low'] = False
            except:
                self.cached_conditions['battery_low'] = False
            
            # Network speed (simplified - could be enhanced with actual testing)
            self.cached_conditions['network_slow'] = False  # Placeholder
            
        except ImportError:
            # psutil not available, assume good conditions
            self.cached_conditions = {k: False for k in self.cached_conditions}
        except Exception as e:
            logger.debug(f"System condition check failed: {e}")

class AdaptiveLoadingEngine:
    """Engine for adaptive loading mode selection"""
    
    def __init__(self):
        self.condition_monitor = SystemConditionMonitor()
        self.performance_history = []
        self.max_history = 100
        
        # Adaptive thresholds
        self.performance_thresholds = {
            'fast_response_ms': 300,
            'acceptable_response_ms': 800,
            'slow_response_ms': 1500
        }
    
    def select_adaptive_mode(self, base_mode: LoadingMode, 
                           tool: Optional[str] = None) -> LoadingMode:
        """Select optimal loading mode based on current conditions"""
        
        conditions = self.condition_monitor.get_system_conditions()
        recent_performance = self._get_recent_performance()
        
        # Start with base mode
        selected_mode = base_mode
        
        # Downgrade if system is under stress
        if conditions['cpu_busy'] or conditions['memory_pressure']:
            if selected_mode == LoadingMode.COMPLETE:
                selected_mode = LoadingMode.BALANCED
            elif selected_mode == LoadingMode.BALANCED:
                selected_mode = LoadingMode.MINIMAL
        
        # Downgrade if on battery power
        if conditions['battery_low']:
            if selected_mode in [LoadingMode.COMPLETE, LoadingMode.BALANCED]:
                selected_mode = LoadingMode.MINIMAL
        
        # Consider recent performance
        if recent_performance and recent_performance['avg_time_ms'] > self.performance_thresholds['slow_response_ms']:
            # Recent responses are slow, be more conservative
            if selected_mode == LoadingMode.COMPLETE:
                selected_mode = LoadingMode.BALANCED
        
        logger.debug(f"Adaptive mode: {base_mode} -> {selected_mode} (conditions: {conditions})")
        
        return selected_mode
    
    def record_performance(self, mode: LoadingMode, time_ms: float, success: bool):
        """Record performance data for adaptive learning"""
        
        record = {
            'mode': mode,
            'time_ms': time_ms,
            'success': success,
            'timestamp': time.time() * 1000
        }
        
        self.performance_history.append(record)
        
        # Trim history
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history:]
    
    def _get_recent_performance(self, window_ms: int = 60000) -> Optional[Dict[str, float]]:
        """Get recent performance statistics"""
        
        if not self.performance_history:
            return None
        
        current_time = time.time() * 1000
        cutoff_time = current_time - window_ms
        
        recent_records = [
            r for r in self.performance_history 
            if r['timestamp'] > cutoff_time
        ]
        
        if not recent_records:
            return None
        
        times = [r['time_ms'] for r in recent_records]
        
        return {
            'avg_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'success_rate': sum(1 for r in recent_records if r['success']) / len(recent_records)
        }

class LazyContextLoader:
    """Main lazy loading implementation with multiple strategies"""
    
    def __init__(self):
        self.cache = get_neuralsync_cache()
        self.network_client = get_network_client()
        self.prewarmer = get_context_prewarmer()
        self.analyzer = ContextAnalyzer()
        self.adaptive_engine = AdaptiveLoadingEngine()
        
        # Performance tracking
        self.stats = {
            'total_loads': 0,
            'cache_hits': 0,
            'prewarmed_hits': 0,
            'fallbacks_used': 0,
            'mode_usage': {mode: 0 for mode in LoadingMode},
            'avg_load_time_ms': 0.0
        }
        
        # Configuration
        self.default_strategy = LoadingStrategy(
            mode=LoadingMode.ADAPTIVE,
            max_wait_ms=int(os.environ.get('NS_MAX_WAIT_MS', '800')),
            enable_caching=True,
            enable_prewarming=True
        )
        
        logger.info("LazyContextLoader initialized")
    
    async def load_context(self, 
                          tool: Optional[str], 
                          query: str = "",
                          command_args: Optional[list] = None,
                          strategy: Optional[LoadingStrategy] = None) -> LoadingResult:
        """Load context using lazy loading strategy"""
        
        start_time = time.perf_counter()
        self.stats['total_loads'] += 1
        
        # Use provided strategy or default
        if strategy is None:
            strategy = self.default_strategy
        
        # Determine loading mode
        if strategy.mode == LoadingMode.ADAPTIVE:
            base_mode = self.analyzer.analyze_context_needs(tool, query, command_args)
            selected_mode = self.adaptive_engine.select_adaptive_mode(base_mode, tool)
        else:
            selected_mode = strategy.mode
        
        self.stats['mode_usage'][selected_mode] += 1
        
        # Execute loading based on selected mode
        try:
            result = await self._execute_loading_mode(
                selected_mode, tool, query, strategy
            )
            
            # Record performance for adaptive learning
            loading_time_ms = (time.perf_counter() - start_time) * 1000
            self.adaptive_engine.record_performance(
                selected_mode, loading_time_ms, result.context != ""
            )
            
            # Update statistics
            self.stats['avg_load_time_ms'] = (
                0.9 * self.stats['avg_load_time_ms'] + 0.1 * loading_time_ms
            )
            
            if result.from_cache:
                self.stats['cache_hits'] += 1
            if result.prewarmed:
                self.stats['prewarmed_hits'] += 1
            if result.fallback_used:
                self.stats['fallbacks_used'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Context loading failed: {e}")
            
            # Return fallback result
            fallback_result = LoadingResult(
                context="",
                mode_used=selected_mode,
                loading_time_ms=(time.perf_counter() - start_time) * 1000,
                from_cache=False,
                prewarmed=False,
                fallback_used=True,
                persona_loaded=False,
                memories_loaded=0,
                total_size_bytes=0
            )
            
            self.stats['fallbacks_used'] += 1
            return fallback_result
    
    async def _execute_loading_mode(self, 
                                   mode: LoadingMode,
                                   tool: Optional[str],
                                   query: str,
                                   strategy: LoadingStrategy) -> LoadingResult:
        """Execute specific loading mode"""
        
        start_time = time.perf_counter()
        
        if mode == LoadingMode.BYPASS:
            return LoadingResult(
                context="",
                mode_used=mode,
                loading_time_ms=0.0,
                from_cache=False,
                prewarmed=False,
                fallback_used=False,
                persona_loaded=False,
                memories_loaded=0,
                total_size_bytes=0
            )
        
        # Check cache first
        if strategy.enable_caching:
            context_hash = self.cache.hash_context(query, tool)
            cached_context = await self.cache.get_context(context_hash)
            
            if cached_context:
                return LoadingResult(
                    context=cached_context,
                    mode_used=mode,
                    loading_time_ms=(time.perf_counter() - start_time) * 1000,
                    from_cache=True,
                    prewarmed=False,
                    fallback_used=False,
                    persona_loaded=True,  # Assume cached context includes persona
                    memories_loaded=1,    # Estimate
                    total_size_bytes=len(cached_context.encode('utf-8'))
                )
        
        # Load based on mode
        try:
            persona = ""
            memories = []
            
            if mode in [LoadingMode.MINIMAL, LoadingMode.BALANCED, LoadingMode.COMPLETE]:
                # Load persona for non-bypass modes
                try:
                    persona_data = await asyncio.wait_for(
                        self._load_persona_with_cache(),
                        timeout=strategy.max_wait_ms / 2000.0  # Half the total timeout
                    )
                    if persona_data:
                        persona = persona_data
                except asyncio.TimeoutError:
                    logger.debug("Persona loading timed out")
                    persona = ""
            
            if mode in [LoadingMode.BALANCED, LoadingMode.COMPLETE]:
                # Load memories for balanced and complete modes
                memory_count = {
                    LoadingMode.BALANCED: strategy.balanced_memory_count,
                    LoadingMode.COMPLETE: strategy.complete_memory_count
                }.get(mode, strategy.minimal_memory_count)
                
                try:
                    memories_data = await asyncio.wait_for(
                        self._load_memories_with_cache(query, tool, memory_count),
                        timeout=strategy.max_wait_ms / 1000.0
                    )
                    if memories_data:
                        memories = memories_data
                except asyncio.TimeoutError:
                    logger.debug("Memory loading timed out")
                    memories = []
            
            # Assemble context
            context_parts = []
            
            if persona and mode != LoadingMode.MINIMAL:
                context_parts.append(f"Persona: {persona}")
                context_parts.append("")
            
            for i, memory in enumerate(memories, 1):
                context_line = f"[M{i}] ({memory.get('kind', 'unknown')},{memory.get('scope', 'global')},conf={memory.get('confidence', '')})"
                context_line += f" {memory.get('text', '')}"
                context_parts.append(context_line)
            
            assembled_context = "\n".join(context_parts)
            if assembled_context:
                assembled_context += "\n\n"
            
            # Cache the result
            if strategy.enable_caching and assembled_context:
                context_hash = self.cache.hash_context(query, tool)
                await self.cache.set_context(context_hash, assembled_context, 180000)  # 3 minutes
            
            return LoadingResult(
                context=assembled_context,
                mode_used=mode,
                loading_time_ms=(time.perf_counter() - start_time) * 1000,
                from_cache=False,
                prewarmed=False,  # Could enhance this detection
                fallback_used=False,
                persona_loaded=bool(persona),
                memories_loaded=len(memories),
                total_size_bytes=len(assembled_context.encode('utf-8'))
            )
            
        except Exception as e:
            logger.warning(f"Loading mode {mode} failed: {e}")
            
            # Try fallback mode
            if strategy.fallback_mode != mode:
                logger.info(f"Falling back from {mode} to {strategy.fallback_mode}")
                return await self._execute_loading_mode(
                    strategy.fallback_mode, tool, query, strategy
                )
            else:
                raise
    
    async def _load_persona_with_cache(self) -> Optional[str]:
        """Load persona with caching"""
        
        # Try cache first
        cached_persona = await self.cache.get_persona()
        if cached_persona:
            return cached_persona
        
        # Fetch from network
        try:
            client = get_network_client(fast_mode=True)
            persona_data, _ = await client.get_persona_and_memories("", "", 0)
            
            if persona_data:
                await self.cache.set_persona(persona_data)
                return persona_data
                
        except Exception as e:
            logger.debug(f"Network persona fetch failed: {e}")
        
        return None
    
    async def _load_memories_with_cache(self, query: str, tool: Optional[str], count: int) -> Optional[list]:
        """Load memories with caching"""
        
        # Try cache first
        query_hash = self.cache.hash_query(query, tool, "any", count)
        cached_memories = await self.cache.get_memories(query_hash)
        if cached_memories:
            return cached_memories
        
        # Fetch from network
        try:
            client = get_network_client(fast_mode=True)
            _, memories_data = await client.get_persona_and_memories(tool, query, count)
            
            if memories_data:
                await self.cache.set_memories(query_hash, memories_data)
                return memories_data
                
        except Exception as e:
            logger.debug(f"Network memory fetch failed: {e}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive loading statistics"""
        return {
            'loading_stats': self.stats.copy(),
            'adaptive_performance': self.adaptive_engine._get_recent_performance(),
            'system_conditions': self.condition_monitor.get_system_conditions()
        }
    
    def configure_strategy(self, **kwargs) -> LoadingStrategy:
        """Create custom loading strategy"""
        return LoadingStrategy(**kwargs)

# Global lazy loader instance
_global_lazy_loader: Optional[LazyContextLoader] = None

def get_lazy_loader() -> LazyContextLoader:
    """Get global lazy loader instance"""
    global _global_lazy_loader
    if _global_lazy_loader is None:
        _global_lazy_loader = LazyContextLoader()
    return _global_lazy_loader

async def load_context_lazy(tool: Optional[str], 
                           query: str = "",
                           command_args: Optional[list] = None,
                           mode: Union[str, LoadingMode] = LoadingMode.ADAPTIVE,
                           max_wait_ms: int = 800) -> str:
    """Convenient function for lazy context loading"""
    
    loader = get_lazy_loader()
    
    # Convert string mode to enum if needed
    if isinstance(mode, str):
        mode = LoadingMode(mode)
    
    strategy = LoadingStrategy(
        mode=mode,
        max_wait_ms=max_wait_ms
    )
    
    result = await loader.load_context(tool, query, command_args, strategy)
    return result.context

def get_bypass_mode_from_env() -> LoadingMode:
    """Get loading mode from environment variables"""
    
    env_mode = os.environ.get('NS_LOADING_MODE', 'adaptive').lower()
    
    mode_mapping = {
        'bypass': LoadingMode.BYPASS,
        'minimal': LoadingMode.MINIMAL,
        'balanced': LoadingMode.BALANCED,
        'complete': LoadingMode.COMPLETE,
        'adaptive': LoadingMode.ADAPTIVE
    }
    
    return mode_mapping.get(env_mode, LoadingMode.ADAPTIVE)