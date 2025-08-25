#!/usr/bin/env python3
"""
Performance Optimizer Module for NeuralSync2
Advanced startup optimization, error handling, and system performance tuning
"""

import asyncio
import os
import time
import psutil
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import signal
import weakref
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Performance optimization levels"""
    MINIMAL = "minimal"      # Basic optimizations
    BALANCED = "balanced"    # Good balance of performance and resource usage
    AGGRESSIVE = "aggressive" # Maximum performance optimizations
    ADAPTIVE = "adaptive"    # Dynamically adjust based on system conditions

class PerformanceMetric(Enum):
    """Performance metrics to track"""
    STARTUP_TIME = "startup_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"

@dataclass
class PerformanceProfile:
    """Performance profile with optimization settings"""
    name: str
    level: OptimizationLevel
    max_workers: int
    cache_size: int
    batch_size: int
    timeout_multiplier: float
    preload_modules: List[str]
    lazy_load_modules: List[str]
    concurrent_startups: int
    health_check_interval: float
    metrics_collection_interval: float

@dataclass
class PerformanceMetrics:
    """Performance metrics collection"""
    timestamp: float
    startup_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    response_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_count: int = 0
    cache_hit_rate: float = 0.0
    active_connections: int = 0
    queue_length: int = 0

@dataclass
class OptimizationResult:
    """Result of performance optimization"""
    success: bool
    optimization_applied: str
    performance_gain: float
    resource_impact: Dict[str, float]
    warnings: List[str]
    recommendations: List[str]

class CircuitBreaker:
    """Circuit breaker for handling cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        with self._lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                
                raise e

class PerformanceOptimizer:
    """Advanced performance optimization and monitoring system"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self.metrics_lock = threading.RLock()
        self.monitoring_active = False
        
        # Optimization profiles
        self.profiles = self._create_optimization_profiles()
        self.active_profile: Optional[PerformanceProfile] = None
        
        # Resource management
        self.thread_pools: Dict[str, ThreadPoolExecutor] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Caching and performance
        self._startup_cache: Dict[str, Any] = {}
        self._module_cache: Dict[str, Any] = {}
        self._performance_cache: Dict[str, Tuple[Any, float]] = {}
        
        # System monitoring
        self.system_stats = {
            'cpu_count': psutil.cpu_count(),
            'total_memory': psutil.virtual_memory().total,
            'boot_time': psutil.boot_time()
        }
        
        # Adaptive optimization
        self.adaptation_history: List[Dict[str, Any]] = []
        self.optimization_results: List[OptimizationResult] = []
        
    def _create_optimization_profiles(self) -> Dict[str, PerformanceProfile]:
        """Create predefined optimization profiles"""
        
        cpu_count = psutil.cpu_count()
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        return {
            'minimal': PerformanceProfile(
                name='minimal',
                level=OptimizationLevel.MINIMAL,
                max_workers=max(2, cpu_count // 2),
                cache_size=64 * 1024 * 1024,  # 64MB
                batch_size=10,
                timeout_multiplier=1.0,
                preload_modules=['logging', 'json'],
                lazy_load_modules=['psutil', 'requests'],
                concurrent_startups=1,
                health_check_interval=30.0,
                metrics_collection_interval=60.0
            ),
            'balanced': PerformanceProfile(
                name='balanced',
                level=OptimizationLevel.BALANCED,
                max_workers=cpu_count,
                cache_size=256 * 1024 * 1024,  # 256MB
                batch_size=25,
                timeout_multiplier=0.8,
                preload_modules=['logging', 'json', 'asyncio', 'threading'],
                lazy_load_modules=['psutil'],
                concurrent_startups=2,
                health_check_interval=15.0,
                metrics_collection_interval=30.0
            ),
            'aggressive': PerformanceProfile(
                name='aggressive',
                level=OptimizationLevel.AGGRESSIVE,
                max_workers=cpu_count * 2,
                cache_size=min(512 * 1024 * 1024, int(total_memory_gb * 0.1 * 1024**3)),  # 512MB or 10% of RAM
                batch_size=50,
                timeout_multiplier=0.5,
                preload_modules=['logging', 'json', 'asyncio', 'threading', 'psutil', 'time', 'pathlib'],
                lazy_load_modules=[],
                concurrent_startups=max(2, cpu_count // 2),
                health_check_interval=5.0,
                metrics_collection_interval=10.0
            ),
            'adaptive': PerformanceProfile(
                name='adaptive',
                level=OptimizationLevel.ADAPTIVE,
                max_workers=cpu_count,
                cache_size=128 * 1024 * 1024,  # Will be adjusted dynamically
                batch_size=20,
                timeout_multiplier=0.8,
                preload_modules=['logging', 'json', 'asyncio'],
                lazy_load_modules=['psutil'],
                concurrent_startups=2,
                health_check_interval=20.0,
                metrics_collection_interval=30.0
            )
        }
    
    def select_optimal_profile(self, system_load: Optional[float] = None) -> PerformanceProfile:
        """Select optimal performance profile based on system conditions"""
        
        # Get current system metrics
        if system_load is None:
            system_load = self._calculate_system_load()
        
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Profile selection logic
        if memory_usage > 85 or cpu_usage > 90:
            return self.profiles['minimal']
        elif memory_usage > 70 or cpu_usage > 75:
            return self.profiles['balanced']
        elif memory_usage < 50 and cpu_usage < 50:
            return self.profiles['aggressive']
        else:
            return self.profiles['adaptive']
    
    def _calculate_system_load(self) -> float:
        """Calculate normalized system load factor"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # Get load average if available (Unix-like systems)
            load_avg = 0.0
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()[0] / psutil.cpu_count()
            
            # Combine metrics (weighted)
            load_factor = (
                cpu_percent * 0.4 +
                memory_percent * 0.3 +
                load_avg * 100 * 0.3
            ) / 100
            
            return min(load_factor, 1.0)
            
        except Exception:
            return 0.5  # Default moderate load
    
    async def optimize_service_startup(
        self, 
        service_configs: Dict[str, Any],
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    ) -> Dict[str, OptimizationResult]:
        """Optimize service startup sequence and performance"""
        
        profile = self.profiles.get(optimization_level.value, self.profiles['balanced'])
        self.active_profile = profile
        
        # Initialize thread pools
        self._initialize_thread_pools(profile)
        
        # Start performance monitoring
        await self._start_performance_monitoring()
        
        optimization_results = {}
        start_time = time.time()
        
        try:
            # Pre-optimization preparations
            await self._preoptimize_system(profile)
            
            # Parallel module preloading
            await self._preload_modules(profile.preload_modules)
            
            # Optimize startup sequence
            startup_results = await self._optimize_startup_sequence(service_configs, profile)
            optimization_results.update(startup_results)
            
            # Apply runtime optimizations
            runtime_results = await self._apply_runtime_optimizations(profile)
            optimization_results.update(runtime_results)
            
            # Measure overall performance gain
            total_time = time.time() - start_time
            
            # Record successful optimization
            overall_result = OptimizationResult(
                success=True,
                optimization_applied=f"startup_optimization_{profile.name}",
                performance_gain=max(0, 30 - total_time),  # Target 30s baseline
                resource_impact={
                    'memory_mb': self._get_memory_usage(),
                    'cpu_percent': psutil.cpu_percent()
                },
                warnings=[],
                recommendations=self._generate_optimization_recommendations()
            )
            
            self.optimization_results.append(overall_result)
            optimization_results['overall'] = overall_result
            
        except Exception as e:
            logger.error(f"Startup optimization failed: {e}")
            optimization_results['overall'] = OptimizationResult(
                success=False,
                optimization_applied="startup_optimization_failed",
                performance_gain=0.0,
                resource_impact={},
                warnings=[str(e)],
                recommendations=["Review system resources and configuration"]
            )
        
        return optimization_results
    
    def _initialize_thread_pools(self, profile: PerformanceProfile):
        """Initialize optimized thread pools"""
        
        # Main processing pool
        self.thread_pools['main'] = ThreadPoolExecutor(
            max_workers=profile.max_workers,
            thread_name_prefix="neuralsync_main"
        )
        
        # I/O operations pool
        self.thread_pools['io'] = ThreadPoolExecutor(
            max_workers=max(2, profile.max_workers // 2),
            thread_name_prefix="neuralsync_io"
        )
        
        # Health check pool
        self.thread_pools['health'] = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="neuralsync_health"
        )
        
        # Initialize circuit breakers
        self.circuit_breakers['service_start'] = CircuitBreaker(
            failure_threshold=3,
            timeout=30.0
        )
        
        self.circuit_breakers['health_check'] = CircuitBreaker(
            failure_threshold=5,
            timeout=60.0
        )
    
    async def _preoptimize_system(self, profile: PerformanceProfile):
        """Pre-optimization system preparations"""
        
        # Clear any stale caches
        self._startup_cache.clear()
        
        # Optimize Python garbage collection
        import gc
        gc.set_threshold(700, 10, 10)  # More aggressive GC
        gc.collect()
        
        # Set process priority if possible
        try:
            current_process = psutil.Process()
            if profile.level == OptimizationLevel.AGGRESSIVE:
                current_process.nice(-5)  # Higher priority
        except (PermissionError, psutil.AccessDenied):
            pass
        
        # Warm up system caches
        await self._warmup_system_caches()
    
    async def _preload_modules(self, modules: List[str]):
        """Preload modules for faster startup"""
        
        async def load_module(module_name: str):
            try:
                if module_name not in self._module_cache:
                    # Use import caching
                    module = __import__(module_name)
                    self._module_cache[module_name] = module
                    logger.debug(f"Preloaded module: {module_name}")
            except ImportError as e:
                logger.debug(f"Failed to preload module {module_name}: {e}")
        
        # Load modules concurrently
        tasks = [load_module(module) for module in modules]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _warmup_system_caches(self):
        """Warm up system-level caches"""
        loop = asyncio.get_event_loop()
        
        def warmup_filesystem():
            """Warm up filesystem caches"""
            try:
                # Read common system files to warm up filesystem cache
                for path in ['/proc/cpuinfo', '/proc/meminfo', '/etc/passwd']:
                    if Path(path).exists():
                        with open(path, 'r') as f:
                            f.read(1024)  # Read first 1KB
            except Exception:
                pass
        
        def warmup_network():
            """Warm up network stack"""
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                try:
                    sock.connect(('127.0.0.1', 1))  # This will fail but warm up the stack
                except:
                    pass
                finally:
                    sock.close()
            except Exception:
                pass
        
        # Run warmup operations
        await loop.run_in_executor(self.thread_pools['io'], warmup_filesystem)
        await loop.run_in_executor(self.thread_pools['io'], warmup_network)
    
    async def _optimize_startup_sequence(
        self, 
        service_configs: Dict[str, Any], 
        profile: PerformanceProfile
    ) -> Dict[str, OptimizationResult]:
        """Optimize service startup sequence"""
        
        results = {}
        
        # Analyze service dependencies
        dependency_graph = self._build_dependency_graph(service_configs)
        
        # Optimize startup order
        startup_order = self._calculate_optimal_startup_order(dependency_graph)
        
        # Concurrent startup batches
        startup_batches = self._create_startup_batches(startup_order, profile.concurrent_startups)
        
        for batch_idx, batch in enumerate(startup_batches):
            batch_start_time = time.time()
            
            # Start services in current batch concurrently
            batch_tasks = []
            for service_name in batch:
                task = self._start_service_optimized(service_name, service_configs.get(service_name, {}))
                batch_tasks.append((service_name, task))
            
            # Wait for batch completion
            for service_name, task in batch_tasks:
                try:
                    success = await task
                    batch_time = time.time() - batch_start_time
                    
                    results[service_name] = OptimizationResult(
                        success=success,
                        optimization_applied=f"concurrent_startup_batch_{batch_idx}",
                        performance_gain=max(0, 10 - batch_time),  # Target 10s per service
                        resource_impact={'startup_time': batch_time},
                        warnings=[],
                        recommendations=[]
                    )
                    
                except Exception as e:
                    results[service_name] = OptimizationResult(
                        success=False,
                        optimization_applied="startup_failed",
                        performance_gain=0.0,
                        resource_impact={},
                        warnings=[str(e)],
                        recommendations=[f"Review {service_name} configuration"]
                    )
        
        return results
    
    def _build_dependency_graph(self, service_configs: Dict[str, Any]) -> Dict[str, List[str]]:
        """Build service dependency graph"""
        dependency_graph = {}
        
        # Default dependencies for known services
        known_dependencies = {
            'neuralsync-server': ['neuralsync-broker'],
            'neuralsync-broker': [],
            'memory-manager': ['neuralsync-server'],
            'sync-manager': ['neuralsync-server']
        }
        
        for service_name in service_configs.keys():
            # Use known dependencies or empty list
            dependency_graph[service_name] = known_dependencies.get(service_name, [])
        
        return dependency_graph
    
    def _calculate_optimal_startup_order(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Calculate optimal service startup order using topological sort"""
        
        # Topological sort implementation
        in_degree = {service: 0 for service in dependency_graph}
        
        # Calculate in-degrees
        for service in dependency_graph:
            for dependency in dependency_graph[service]:
                if dependency in in_degree:
                    in_degree[dependency] += 1
        
        # Start with services that have no dependencies
        queue = [service for service, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            service = queue.pop(0)
            result.append(service)
            
            # Update in-degrees for dependent services
            for dependent in dependency_graph:
                if service in dependency_graph[dependent]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        return result
    
    def _create_startup_batches(self, startup_order: List[str], concurrent_limit: int) -> List[List[str]]:
        """Create batches of services that can start concurrently"""
        batches = []
        
        for i in range(0, len(startup_order), concurrent_limit):
            batch = startup_order[i:i + concurrent_limit]
            batches.append(batch)
        
        return batches
    
    async def _start_service_optimized(self, service_name: str, service_config: Dict[str, Any]) -> bool:
        """Start service with optimizations applied"""
        
        def start_with_circuit_breaker():
            return self.circuit_breakers['service_start'].call(
                self._actual_service_start, service_name, service_config
            )
        
        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.thread_pools['main'],
                start_with_circuit_breaker
            )
            return success
            
        except Exception as e:
            logger.error(f"Optimized service start failed for {service_name}: {e}")
            return False
    
    def _actual_service_start(self, service_name: str, service_config: Dict[str, Any]) -> bool:
        """Actual service startup implementation (placeholder)"""
        # This would integrate with the daemon manager
        # For now, simulate startup
        time.sleep(0.1)  # Simulate startup time
        return True
    
    async def _apply_runtime_optimizations(self, profile: PerformanceProfile) -> Dict[str, OptimizationResult]:
        """Apply runtime performance optimizations"""
        results = {}
        
        # Memory optimization
        memory_result = await self._optimize_memory_usage(profile)
        results['memory_optimization'] = memory_result
        
        # Cache optimization
        cache_result = await self._optimize_caching(profile)
        results['cache_optimization'] = cache_result
        
        # Network optimization
        network_result = await self._optimize_network_settings(profile)
        results['network_optimization'] = network_result
        
        return results
    
    async def _optimize_memory_usage(self, profile: PerformanceProfile) -> OptimizationResult:
        """Optimize memory usage"""
        start_memory = self._get_memory_usage()
        
        try:
            # Garbage collection optimization
            import gc
            collected = gc.collect()
            
            # Adjust cache sizes based on available memory
            available_memory = psutil.virtual_memory().available
            if available_memory < 512 * 1024 * 1024:  # Less than 512MB
                profile.cache_size = min(profile.cache_size, 32 * 1024 * 1024)  # Reduce to 32MB
            
            end_memory = self._get_memory_usage()
            memory_saved = max(0, start_memory - end_memory)
            
            return OptimizationResult(
                success=True,
                optimization_applied="memory_optimization",
                performance_gain=memory_saved,
                resource_impact={'memory_saved_mb': memory_saved},
                warnings=[],
                recommendations=[]
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                optimization_applied="memory_optimization_failed",
                performance_gain=0.0,
                resource_impact={},
                warnings=[str(e)],
                recommendations=["Review memory configuration"]
            )
    
    async def _optimize_caching(self, profile: PerformanceProfile) -> OptimizationResult:
        """Optimize caching strategies"""
        try:
            # Implement intelligent cache warming
            cache_hits_before = len(self._performance_cache)
            
            # Pre-populate frequently accessed items
            await self._warmup_performance_caches()
            
            cache_hits_after = len(self._performance_cache)
            cache_improvement = cache_hits_after - cache_hits_before
            
            return OptimizationResult(
                success=True,
                optimization_applied="cache_optimization",
                performance_gain=cache_improvement * 0.1,  # Estimate performance gain
                resource_impact={'cache_entries_added': cache_improvement},
                warnings=[],
                recommendations=[]
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                optimization_applied="cache_optimization_failed",
                performance_gain=0.0,
                resource_impact={},
                warnings=[str(e)],
                recommendations=["Review cache configuration"]
            )
    
    async def _warmup_performance_caches(self):
        """Warm up performance-related caches"""
        # This would cache frequently accessed configuration and data
        pass
    
    async def _optimize_network_settings(self, profile: PerformanceProfile) -> OptimizationResult:
        """Optimize network-related settings"""
        try:
            # Optimize socket settings, connection pooling, etc.
            # This is a placeholder for actual network optimizations
            
            return OptimizationResult(
                success=True,
                optimization_applied="network_optimization",
                performance_gain=1.0,  # Estimated improvement
                resource_impact={'network_optimized': True},
                warnings=[],
                recommendations=[]
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                optimization_applied="network_optimization_failed",
                performance_gain=0.0,
                resource_impact={},
                warnings=[str(e)],
                recommendations=["Review network configuration"]
            )
    
    async def _start_performance_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self._collect_current_metrics()
                
                with self.metrics_lock:
                    self.metrics_history.append(metrics)
                    
                    # Keep only recent metrics (last 1000 entries)
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                
                # Adaptive optimization
                if self.active_profile and self.active_profile.level == OptimizationLevel.ADAPTIVE:
                    await self._apply_adaptive_optimizations(metrics)
                
                # Sleep until next collection
                interval = self.active_profile.metrics_collection_interval if self.active_profile else 30.0
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        return PerformanceMetrics(
            timestamp=time.time(),
            memory_usage_mb=self._get_memory_usage(),
            cpu_usage_percent=psutil.cpu_percent(),
            # Other metrics would be collected from running services
        )
    
    async def _apply_adaptive_optimizations(self, current_metrics: PerformanceMetrics):
        """Apply adaptive optimizations based on current metrics"""
        
        # Analyze trends
        if len(self.metrics_history) >= 10:
            recent_metrics = self.metrics_history[-10:]
            
            # Check if memory usage is trending upward
            memory_trend = self._calculate_trend([m.memory_usage_mb for m in recent_metrics])
            
            if memory_trend > 0.1:  # Increasing memory usage
                # Trigger more aggressive garbage collection
                import gc
                gc.collect()
                
                # Reduce cache sizes
                if self.active_profile:
                    self.active_profile.cache_size = max(
                        64 * 1024 * 1024,  # Minimum 64MB
                        int(self.active_profile.cache_size * 0.9)
                    )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * v for i, v in enumerate(values))
        sum_x2 = sum(i * i for i in range(n))
        
        # Linear regression slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on system state"""
        recommendations = []
        
        # System resource analysis
        memory_percent = psutil.virtual_memory().percent
        cpu_count = psutil.cpu_count()
        
        if memory_percent > 85:
            recommendations.append("Consider increasing system memory or reducing cache sizes")
        
        if cpu_count < 4:
            recommendations.append("Consider reducing concurrent operations for low-CPU systems")
        
        # Performance history analysis
        if len(self.metrics_history) > 10:
            recent_cpu = [m.cpu_usage_percent for m in self.metrics_history[-10:]]
            avg_cpu = sum(recent_cpu) / len(recent_cpu)
            
            if avg_cpu > 80:
                recommendations.append("High CPU usage detected - consider optimizing algorithms or reducing load")
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.metrics_lock:
            if not self.metrics_history:
                return {'status': 'no_data'}
            
            recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
            
            summary = {
                'current_metrics': asdict(self.metrics_history[-1]) if self.metrics_history else {},
                'average_metrics': {
                    'memory_usage_mb': sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
                    'cpu_usage_percent': sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
                },
                'active_profile': self.active_profile.name if self.active_profile else None,
                'optimization_count': len(self.optimization_results),
                'recommendations': self._generate_optimization_recommendations(),
                'system_info': self.system_stats
            }
            
            return summary
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
    
    def shutdown(self):
        """Shutdown performance optimizer"""
        self.stop_monitoring()
        
        # Shutdown thread pools
        for pool in self.thread_pools.values():
            pool.shutdown(wait=True)
        
        self.thread_pools.clear()

# Testing function
async def test_performance_optimizer():
    """Test performance optimizer functionality"""
    optimizer = PerformanceOptimizer(Path.home() / ".neuralsync")
    
    try:
        # Test profile selection
        print("Testing profile selection...")
        profile = optimizer.select_optimal_profile()
        print(f"Selected profile: {profile.name} (level: {profile.level.value})")
        
        # Test optimization
        print("\nTesting service startup optimization...")
        service_configs = {
            'neuralsync-server': {'port': 8373},
            'neuralsync-broker': {'port': 8374}
        }
        
        results = await optimizer.optimize_service_startup(
            service_configs,
            OptimizationLevel.BALANCED
        )
        
        print(f"Optimization results: {len(results)} services")
        for service, result in results.items():
            print(f"  {service}: {'success' if result.success else 'failed'} "
                  f"(gain: {result.performance_gain:.2f})")
        
        # Test monitoring
        print("\nTesting performance monitoring...")
        await asyncio.sleep(2)  # Let monitoring run briefly
        
        summary = optimizer.get_performance_summary()
        print(f"Performance summary: {summary.get('active_profile', 'none')} profile active")
        
        if 'current_metrics' in summary:
            current = summary['current_metrics']
            print(f"Current metrics: {current.get('memory_usage_mb', 0):.1f}MB memory, "
                  f"{current.get('cpu_usage_percent', 0):.1f}% CPU")
        
    finally:
        optimizer.shutdown()
        print("\nPerformance optimizer test completed")

if __name__ == "__main__":
    from dataclasses import asdict
    asyncio.run(test_performance_optimizer())