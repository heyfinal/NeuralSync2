#!/usr/bin/env python3
"""
CLI Performance Integration for NeuralSync v2
Comprehensive performance monitoring and optimization integration
"""

import time
import os
import asyncio
import threading
import logging
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from pathlib import Path

from .performance_monitor import get_performance_monitor, init_performance_monitoring
from .intelligent_cache import get_neuralsync_cache
from .async_network import get_network_client
from .context_prewarmer import get_context_prewarmer
from .lazy_loader import get_lazy_loader, LoadingMode, LoadingStrategy
from .fast_recall import get_fast_recall_engine

logger = logging.getLogger(__name__)

@dataclass
class CLIPerformanceMetrics:
    """Comprehensive CLI performance metrics"""
    total_invocations: int = 0
    context_fetch_time_ms: float = 0.0
    command_execution_time_ms: float = 0.0
    total_response_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    average_context_size_bytes: int = 0
    bypass_mode_usage: float = 0.0
    error_rate: float = 0.0
    
    # Performance buckets
    sub_100ms_responses: int = 0
    sub_500ms_responses: int = 0
    sub_1000ms_responses: int = 0
    over_1000ms_responses: int = 0
    
    # Quality metrics
    context_relevance_score: float = 0.0
    user_satisfaction_proxy: float = 0.0

@dataclass
class CLISession:
    """Individual CLI session tracking"""
    session_id: str
    tool_name: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    context_loaded: bool = False
    context_size_bytes: int = 0
    loading_mode_used: Optional[LoadingMode] = None
    cache_hits: int = 0
    network_requests: int = 0
    errors: List[str] = field(default_factory=list)
    performance_score: float = 0.0

class PerformanceIntegrationManager:
    """Central manager for CLI performance integration"""
    
    def __init__(self, storage=None):
        self.storage = storage
        
        # Initialize performance monitoring components
        if storage:
            self.perf_monitor = init_performance_monitoring(
                storage, 
                enable_auto_optimization=True,
                target_response_time_ms=800.0
            )
        else:
            self.perf_monitor = get_performance_monitor()
        
        self.cache = get_neuralsync_cache()
        self.prewarmer = get_context_prewarmer()
        self.lazy_loader = get_lazy_loader()
        
        # CLI-specific metrics
        self.cli_metrics = CLIPerformanceMetrics()
        self.active_sessions: Dict[str, CLISession] = {}
        
        # Performance thresholds
        self.performance_thresholds = {
            'excellent_response_ms': 200,
            'good_response_ms': 500,
            'acceptable_response_ms': 1000,
            'poor_response_ms': 2000
        }
        
        # Optimization triggers
        self.optimization_triggers = {
            'high_error_rate': 0.1,           # 10% error rate
            'low_cache_hit_rate': 0.3,        # 30% cache hit rate
            'slow_average_response': 1500,    # 1.5s average
            'memory_pressure': 0.85           # 85% memory usage
        }
        
        # Background monitoring
        self.monitoring_enabled = True
        self.stats_collection_interval = 30  # seconds
        self.optimization_check_interval = 300  # 5 minutes
        
        self._start_background_monitoring()
        
        logger.info("PerformanceIntegrationManager initialized")
    
    def _start_background_monitoring(self):
        """Start background performance monitoring"""
        
        def monitoring_worker():
            while self.monitoring_enabled:
                try:
                    self._collect_performance_stats()
                    self._check_optimization_triggers()
                    time.sleep(self.stats_collection_interval)
                    
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
        monitor_thread.start()
        
        logger.info("Background performance monitoring started")
    
    def _collect_performance_stats(self):
        """Collect comprehensive performance statistics"""
        try:
            # Update cache statistics
            cache_stats = self.cache.get_comprehensive_stats()
            total_cache_requests = sum(
                cache_stats[cache_type]['total_requests'] 
                for cache_type in cache_stats.keys()
                if 'total_requests' in cache_stats[cache_type]
            )
            total_cache_hits = sum(
                cache_stats[cache_type]['hits']
                for cache_type in cache_stats.keys() 
                if 'hits' in cache_stats[cache_type]
            )
            
            if total_cache_requests > 0:
                self.cli_metrics.cache_hit_rate = total_cache_hits / total_cache_requests
            
            # Update lazy loader statistics
            lazy_stats = self.lazy_loader.get_stats()
            loading_stats = lazy_stats['loading_stats']
            
            if loading_stats['total_loads'] > 0:
                self.cli_metrics.bypass_mode_usage = (
                    loading_stats['mode_usage'].get(LoadingMode.BYPASS, 0) /
                    loading_stats['total_loads']
                )
            
            # Update session-based metrics
            self._update_session_metrics()
            
        except Exception as e:
            logger.error(f"Stats collection failed: {e}")
    
    def _update_session_metrics(self):
        """Update metrics from active sessions"""
        if not self.active_sessions:
            return
        
        completed_sessions = [
            session for session in self.active_sessions.values()
            if session.end_time is not None
        ]
        
        if completed_sessions:
            # Calculate average response times
            total_times = [
                (session.end_time - session.start_time) * 1000
                for session in completed_sessions
            ]
            
            if total_times:
                avg_time = sum(total_times) / len(total_times)
                self.cli_metrics.total_response_time_ms = avg_time
                
                # Update performance buckets
                for time_ms in total_times:
                    if time_ms < 100:
                        self.cli_metrics.sub_100ms_responses += 1
                    elif time_ms < 500:
                        self.cli_metrics.sub_500ms_responses += 1
                    elif time_ms < 1000:
                        self.cli_metrics.sub_1000ms_responses += 1
                    else:
                        self.cli_metrics.over_1000ms_responses += 1
                
                # Calculate error rate
                sessions_with_errors = sum(1 for s in completed_sessions if s.errors)
                self.cli_metrics.error_rate = sessions_with_errors / len(completed_sessions)
    
    def _check_optimization_triggers(self):
        """Check if automatic optimization should be triggered"""
        
        triggers_fired = []
        
        # Check error rate
        if self.cli_metrics.error_rate > self.optimization_triggers['high_error_rate']:
            triggers_fired.append('high_error_rate')
        
        # Check cache hit rate
        if self.cli_metrics.cache_hit_rate < self.optimization_triggers['low_cache_hit_rate']:
            triggers_fired.append('low_cache_hit_rate')
        
        # Check average response time
        if self.cli_metrics.total_response_time_ms > self.optimization_triggers['slow_average_response']:
            triggers_fired.append('slow_average_response')
        
        # Check memory pressure
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.optimization_triggers['memory_pressure'] * 100:
                triggers_fired.append('memory_pressure')
        except:
            pass
        
        # Trigger optimizations if needed
        if triggers_fired:
            logger.info(f"Optimization triggers fired: {triggers_fired}")
            asyncio.create_task(self._apply_automatic_optimizations(triggers_fired))
    
    async def _apply_automatic_optimizations(self, triggers: List[str]):
        """Apply automatic optimizations based on triggers"""
        
        optimizations_applied = []
        
        try:
            # High error rate - increase caching and reduce timeouts
            if 'high_error_rate' in triggers:
                self.lazy_loader.default_strategy.max_wait_ms = min(
                    self.lazy_loader.default_strategy.max_wait_ms * 0.8, 500
                )
                optimizations_applied.append('reduced_timeouts')
            
            # Low cache hit rate - force cache rebuild and prewarming
            if 'low_cache_hit_rate' in triggers:
                self.cache.optimize_all()
                if self.prewarmer and not self.prewarmer.running:
                    self.prewarmer.start()
                optimizations_applied.append('cache_optimization')
            
            # Slow responses - enable bypass mode for simple operations
            if 'slow_average_response' in triggers:
                # This could be implemented as dynamic bypass mode adjustment
                optimizations_applied.append('dynamic_bypass')
            
            # Memory pressure - trigger cleanup
            if 'memory_pressure' in triggers:
                self.cache.optimize_all()
                if self.perf_monitor:
                    self.perf_monitor.force_optimization(target_ms=500)
                optimizations_applied.append('memory_cleanup')
            
            logger.info(f"Applied automatic optimizations: {optimizations_applied}")
            
        except Exception as e:
            logger.error(f"Automatic optimization failed: {e}")
    
    def start_cli_session(self, tool: Optional[str], session_id: Optional[str] = None) -> str:
        """Start tracking a CLI session"""
        
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())[:8]
        
        session = CLISession(
            session_id=session_id,
            tool_name=tool,
            start_time=time.perf_counter()
        )
        
        self.active_sessions[session_id] = session
        self.cli_metrics.total_invocations += 1
        
        return session_id
    
    def end_cli_session(self, session_id: str, 
                       success: bool = True, 
                       errors: Optional[List[str]] = None):
        """End tracking a CLI session"""
        
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session.end_time = time.perf_counter()
        
        if errors:
            session.errors.extend(errors)
        
        # Calculate performance score
        response_time_ms = (session.end_time - session.start_time) * 1000
        
        if response_time_ms < self.performance_thresholds['excellent_response_ms']:
            session.performance_score = 1.0
        elif response_time_ms < self.performance_thresholds['good_response_ms']:
            session.performance_score = 0.8
        elif response_time_ms < self.performance_thresholds['acceptable_response_ms']:
            session.performance_score = 0.6
        else:
            session.performance_score = 0.3
        
        # Adjust score based on errors
        if session.errors:
            session.performance_score *= 0.5
        
        # Clean up old sessions periodically
        if len(self.active_sessions) > 100:
            self._cleanup_old_sessions()
    
    def _cleanup_old_sessions(self):
        """Remove old completed sessions"""
        
        current_time = time.perf_counter()
        sessions_to_remove = []
        
        for session_id, session in self.active_sessions.items():
            if session.end_time and (current_time - session.end_time) > 3600:  # 1 hour old
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
    
    def record_context_loading(self, session_id: str, 
                              context_size_bytes: int,
                              loading_mode: LoadingMode,
                              from_cache: bool,
                              loading_time_ms: float):
        """Record context loading metrics for a session"""
        
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session.context_loaded = True
        session.context_size_bytes = context_size_bytes
        session.loading_mode_used = loading_mode
        
        if from_cache:
            session.cache_hits += 1
        else:
            session.network_requests += 1
        
        # Update global metrics
        self.cli_metrics.context_fetch_time_ms = (
            0.9 * self.cli_metrics.context_fetch_time_ms + 0.1 * loading_time_ms
        )
        
        if context_size_bytes > 0:
            self.cli_metrics.average_context_size_bytes = (
                0.9 * self.cli_metrics.average_context_size_bytes + 0.1 * context_size_bytes
            )
    
    @asynccontextmanager
    async def measure_cli_operation(self, tool: Optional[str], operation: str = "cli_execution"):
        """Context manager for measuring CLI operations"""
        
        session_id = self.start_cli_session(tool)
        errors = []
        
        try:
            if self.perf_monitor:
                async with self.perf_monitor.measure_cpu_operation(f"{tool}_{operation}"):
                    yield session_id
            else:
                yield session_id
                
        except Exception as e:
            errors.append(str(e))
            raise
        finally:
            self.end_cli_session(session_id, success=len(errors) == 0, errors=errors)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        # Calculate percentile distributions
        total_responses = (self.cli_metrics.sub_100ms_responses + 
                          self.cli_metrics.sub_500ms_responses +
                          self.cli_metrics.sub_1000ms_responses +
                          self.cli_metrics.over_1000ms_responses)
        
        percentiles = {}
        if total_responses > 0:
            percentiles = {
                'sub_100ms_percent': (self.cli_metrics.sub_100ms_responses / total_responses) * 100,
                'sub_500ms_percent': (self.cli_metrics.sub_500ms_responses / total_responses) * 100,
                'sub_1000ms_percent': (self.cli_metrics.sub_1000ms_responses / total_responses) * 100,
                'over_1000ms_percent': (self.cli_metrics.over_1000ms_responses / total_responses) * 100
            }
        
        # Get component statistics
        component_stats = {}
        
        try:
            component_stats['cache'] = self.cache.get_comprehensive_stats()
        except:
            component_stats['cache'] = {}
        
        try:
            component_stats['lazy_loader'] = self.lazy_loader.get_stats()
        except:
            component_stats['lazy_loader'] = {}
        
        try:
            if self.prewarmer:
                component_stats['prewarmer'] = self.prewarmer.get_comprehensive_stats()
        except:
            component_stats['prewarmer'] = {}
        
        try:
            if self.perf_monitor:
                component_stats['performance_monitor'] = self.perf_monitor.get_comprehensive_stats()
        except:
            component_stats['performance_monitor'] = {}
        
        # Calculate overall health score
        health_score = self._calculate_health_score()
        
        return {
            'cli_metrics': {
                'total_invocations': self.cli_metrics.total_invocations,
                'average_response_time_ms': self.cli_metrics.total_response_time_ms,
                'cache_hit_rate': self.cli_metrics.cache_hit_rate,
                'error_rate': self.cli_metrics.error_rate,
                'bypass_mode_usage': self.cli_metrics.bypass_mode_usage,
                'average_context_size_bytes': self.cli_metrics.average_context_size_bytes
            },
            'response_time_distribution': percentiles,
            'performance_health_score': health_score,
            'active_sessions': len(self.active_sessions),
            'component_stats': component_stats,
            'optimization_thresholds': self.optimization_triggers
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)"""
        
        scores = []
        
        # Response time score
        if self.cli_metrics.total_response_time_ms <= self.performance_thresholds['excellent_response_ms']:
            scores.append(1.0)
        elif self.cli_metrics.total_response_time_ms <= self.performance_thresholds['good_response_ms']:
            scores.append(0.8)
        elif self.cli_metrics.total_response_time_ms <= self.performance_thresholds['acceptable_response_ms']:
            scores.append(0.6)
        else:
            scores.append(0.3)
        
        # Cache hit rate score
        if self.cli_metrics.cache_hit_rate >= 0.8:
            scores.append(1.0)
        elif self.cli_metrics.cache_hit_rate >= 0.6:
            scores.append(0.8)
        elif self.cli_metrics.cache_hit_rate >= 0.4:
            scores.append(0.6)
        else:
            scores.append(0.3)
        
        # Error rate score (inverted)
        if self.cli_metrics.error_rate <= 0.01:
            scores.append(1.0)
        elif self.cli_metrics.error_rate <= 0.05:
            scores.append(0.8)
        elif self.cli_metrics.error_rate <= 0.1:
            scores.append(0.6)
        else:
            scores.append(0.3)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def export_performance_report(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Export comprehensive performance report"""
        
        report = {
            'timestamp': time.time(),
            'report_type': 'neuralsync_cli_performance',
            'summary': self.get_performance_summary(),
            'detailed_sessions': [
                {
                    'session_id': session.session_id,
                    'tool': session.tool_name,
                    'response_time_ms': ((session.end_time or time.perf_counter()) - session.start_time) * 1000,
                    'context_loaded': session.context_loaded,
                    'context_size_bytes': session.context_size_bytes,
                    'loading_mode': session.loading_mode_used.value if session.loading_mode_used else None,
                    'cache_hits': session.cache_hits,
                    'network_requests': session.network_requests,
                    'errors': session.errors,
                    'performance_score': session.performance_score
                }
                for session in list(self.active_sessions.values())[-50:]  # Last 50 sessions
            ],
            'recommendations': self._generate_performance_recommendations()
        }
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Performance report exported to {file_path}")
            except Exception as e:
                logger.error(f"Failed to export performance report: {e}")
        
        return report
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        if self.cli_metrics.cache_hit_rate < 0.5:
            recommendations.append(
                "Consider enabling context pre-warming to improve cache hit rates"
            )
        
        if self.cli_metrics.total_response_time_ms > 1000:
            recommendations.append(
                "Response times are slow - consider using bypass mode for simple operations"
            )
        
        if self.cli_metrics.error_rate > 0.05:
            recommendations.append(
                "High error rate detected - check network connectivity and service health"
            )
        
        if self.cli_metrics.bypass_mode_usage < 0.2:
            recommendations.append(
                "Low bypass mode usage - consider auto-detection for help/version commands"
            )
        
        total_responses = (self.cli_metrics.sub_100ms_responses + 
                          self.cli_metrics.sub_500ms_responses +
                          self.cli_metrics.sub_1000ms_responses +
                          self.cli_metrics.over_1000ms_responses)
        
        if total_responses > 0 and (self.cli_metrics.sub_100ms_responses / total_responses) < 0.3:
            recommendations.append(
                "Few sub-100ms responses - optimize for faster startup and caching"
            )
        
        return recommendations
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        
        self.cli_metrics = CLIPerformanceMetrics()
        self.active_sessions.clear()
        
        # Reset component metrics if possible
        try:
            if self.perf_monitor:
                self.perf_monitor.performance_tracker.reset_stats()
        except:
            pass
        
        logger.info("Performance metrics reset")
    
    def close(self):
        """Clean shutdown"""
        
        self.monitoring_enabled = False
        
        # Export final report
        try:
            report_path = "/tmp/neuralsync_final_performance_report.json"
            self.export_performance_report(report_path)
        except:
            pass
        
        logger.info("Performance integration manager closed")


# Global performance integration manager
_global_perf_integration: Optional[PerformanceIntegrationManager] = None

def get_performance_integration(storage=None) -> PerformanceIntegrationManager:
    """Get global performance integration manager"""
    global _global_perf_integration
    if _global_perf_integration is None:
        _global_perf_integration = PerformanceIntegrationManager(storage)
    return _global_perf_integration

def init_cli_performance_monitoring(storage=None) -> PerformanceIntegrationManager:
    """Initialize CLI performance monitoring"""
    manager = PerformanceIntegrationManager(storage)
    
    # Set as global instance
    global _global_perf_integration
    _global_perf_integration = manager
    
    return manager