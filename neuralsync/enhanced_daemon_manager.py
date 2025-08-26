#!/usr/bin/env python3
"""
Enhanced Daemon Manager for NeuralSync2
Integrates all optimization modules to provide robust, fast, and reliable service management
"""

import asyncio
import os
import signal
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json
import threading
from concurrent.futures import ThreadPoolExecutor

# Import our specialized modules
from .robust_service_detector import RobustServiceDetector, ServiceDetectionResult, ServiceState
from .smart_process_discovery import SmartProcessDiscovery, DiscoveryStrategy, ProcessMatch, PortConflict
from .configuration_validator import ConfigurationValidator, ValidationIssue, ValidationSeverity
from .performance_optimizer import PerformanceOptimizer, OptimizationLevel, OptimizationResult
from .config import load_config, DEFAULT_HOME

logger = logging.getLogger(__name__)

@dataclass
class EnhancedServiceConfig:
    """Enhanced service configuration with optimization settings"""
    name: str
    command: List[str]
    cwd: str
    env: Dict[str, str]
    health_check_url: str
    health_check_interval: int
    restart_on_failure: bool
    max_restart_attempts: int
    startup_timeout: int
    expected_port: Optional[int] = None
    dependencies: List[str] = None
    priority: int = 0  # Higher number = higher priority
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED

@dataclass
class ServiceStatus:
    """Comprehensive service status"""
    name: str
    state: ServiceState
    detection_result: Optional[ServiceDetectionResult]
    startup_time: Optional[float]
    restart_count: int
    last_health_check: float
    optimization_applied: List[str]
    performance_metrics: Dict[str, float]
    warnings: List[str]
    recommendations: List[str]

class EnhancedDaemonManager:
    """Enhanced daemon manager with integrated optimization modules"""
    
    def __init__(self, config_dir: Path = DEFAULT_HOME):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize specialized modules
        self.service_detector = RobustServiceDetector(self.config_dir)
        self.process_discovery = SmartProcessDiscovery(self.config_dir)
        self.config_validator = ConfigurationValidator(self.config_dir)
        self.performance_optimizer = PerformanceOptimizer(self.config_dir)
        
        # Service management
        self.services: Dict[str, EnhancedServiceConfig] = {}
        self.service_status: Dict[str, ServiceStatus] = {}
        self.startup_lock = threading.RLock()
        
        # Runtime state
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.optimization_active = False
        
        # Load NeuralSync configuration
        self.ns_config = load_config()
        
        # Performance tracking
        self.startup_performance: Dict[str, float] = {}
        self.global_startup_time = 0.0
        
        # Register core services with enhanced configuration
        self._register_enhanced_core_services()
        
    def _register_enhanced_core_services(self):
        """Register core NeuralSync services with enhanced configurations"""
        
        neuralsync_env = os.environ.copy()
        neuralsync_env.update({
            'NS_HOST': self.ns_config.bind_host,
            'NS_PORT': str(self.ns_config.bind_port),
            'NS_TOKEN': self.ns_config.token
        })
        
        # Enhanced NeuralSync server configuration - use regular server if enhanced fails
        server_command = [sys.executable, '-m', 'neuralsync.server']
        if Path(Path(__file__).parent / 'enhanced_server.py').exists():
            try:
                # Test if enhanced server imports work
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "enhanced_server", 
                    Path(__file__).parent / 'enhanced_server.py'
                )
                if spec and spec.loader:
                    server_command = [sys.executable, '-m', 'neuralsync.enhanced_server']
            except Exception:
                pass  # Fall back to regular server
        
        self.services['neuralsync-server'] = EnhancedServiceConfig(
            name='neuralsync-server',
            command=server_command,
            cwd=str(Path(__file__).parent.parent),
            env=neuralsync_env,
            health_check_url=f'http://{self.ns_config.bind_host}:{self.ns_config.bind_port}/health',
            health_check_interval=10,
            restart_on_failure=True,
            max_restart_attempts=3,
            startup_timeout=20,  # Reduced from 30s
            expected_port=self.ns_config.bind_port,
            dependencies=[],
            priority=100,
            optimization_level=OptimizationLevel.BALANCED
        )
        
        # Enhanced message broker configuration
        self.services['neuralsync-broker'] = EnhancedServiceConfig(
            name='neuralsync-broker',
            command=[
                sys.executable, '-c',
                'from neuralsync.ultra_comm import start_message_broker; import asyncio; asyncio.run(start_message_broker())'
            ],
            cwd=str(Path(__file__).parent.parent),
            env=neuralsync_env,
            health_check_url='unix:///tmp/neuralsync_broker.sock',
            health_check_interval=15,
            restart_on_failure=True,
            max_restart_attempts=3,
            startup_timeout=10,  # Reduced from previous timeout
            expected_port=None,
            dependencies=[],
            priority=50,
            optimization_level=OptimizationLevel.BALANCED
        )
    
    async def start_enhanced_monitoring(self):
        """Start enhanced monitoring with all optimizations"""
        if self.running:
            logger.info("Enhanced monitoring already running")
            return True
        
        global_start_time = time.time()
        logger.info("Starting enhanced NeuralSync daemon manager...")
        
        try:
            # Phase 1: Pre-startup validation and optimization
            await self._phase1_validation_and_optimization()
            
            # Phase 2: Intelligent service detection and conflict resolution
            await self._phase2_detection_and_conflict_resolution()
            
            # Phase 3: Optimized service startup
            success = await self._phase3_optimized_startup()
            
            # Phase 4: Runtime monitoring and optimization
            if success:
                await self._phase4_runtime_monitoring()
            
            self.global_startup_time = time.time() - global_start_time
            logger.info(f"Enhanced daemon manager started in {self.global_startup_time:.2f}s")
            
            return success
            
        except Exception as e:
            logger.error(f"Enhanced monitoring startup failed: {e}")
            return False
    
    async def _phase1_validation_and_optimization(self):
        """Phase 1: Configuration validation and system optimization"""
        logger.info("Phase 1: Configuration validation and system optimization")
        
        # Validate configuration
        config_dict = asdict(self.ns_config)
        validation_issues = self.config_validator.validate_configuration(
            config_dict, 'production'  # Use production profile by default
        )
        
        # Auto-fix critical issues
        if validation_issues:
            critical_issues = [i for i in validation_issues if i.severity == ValidationSeverity.CRITICAL]
            if critical_issues:
                logger.warning(f"Found {len(critical_issues)} critical configuration issues")
                fixed_config, fix_log = self.config_validator.auto_fix_issues(config_dict, critical_issues)
                
                if fix_log:
                    logger.info(f"Auto-fixed issues: {', '.join(fix_log)}")
        
        # Initialize performance optimization
        self.optimization_active = True
        
        # Select optimal performance profile
        optimal_profile = self.performance_optimizer.select_optimal_profile()
        logger.info(f"Selected performance profile: {optimal_profile.name}")
    
    async def _phase2_detection_and_conflict_resolution(self):
        """Phase 2: Intelligent service detection and conflict resolution"""
        logger.info("Phase 2: Service detection and conflict resolution")
        
        # Discover existing NeuralSync processes
        existing_processes = await self.process_discovery.discover_neuralsync_processes(
            DiscoveryStrategy.THOROUGH
        )
        
        if existing_processes:
            logger.info(f"Found {len(existing_processes)} existing NeuralSync processes")
            
            # Analyze existing processes to avoid conflicts
            await self._analyze_existing_processes(existing_processes)
        
        # Detect and resolve port conflicts
        service_ports = {
            name: config.expected_port 
            for name, config in self.services.items() 
            if config.expected_port
        }
        
        if service_ports:
            conflicts = await self.process_discovery.detect_port_conflicts(service_ports)
            
            if conflicts:
                logger.warning(f"Found {len(conflicts)} port conflicts")
                resolved = await self.process_discovery.auto_resolve_conflicts(conflicts)
                
                for service_name, resolution_success in resolved.items():
                    if resolution_success:
                        logger.info(f"Resolved port conflict for {service_name}")
                    else:
                        logger.warning(f"Could not auto-resolve conflict for {service_name}")
        
        # Clean up stale resources
        cleanup_results = self.service_detector.cleanup_stale_resources()
        if cleanup_results['pid_files_cleaned']:
            logger.info(f"Cleaned {len(cleanup_results['pid_files_cleaned'])} stale PID files")
    
    async def _analyze_existing_processes(self, existing_processes: List[ProcessMatch]):
        """Analyze existing processes and update service status"""
        
        for process_match in existing_processes:
            # Try to identify which service this process belongs to
            process_info = process_match.process_info
            cmdline_str = ' '.join(process_info.cmdline).lower()
            
            service_name = None
            if 'enhanced_server' in cmdline_str:
                service_name = 'neuralsync-server'
            elif 'message_broker' in cmdline_str or 'ultra_comm' in cmdline_str:
                service_name = 'neuralsync-broker'
            
            if service_name and service_name in self.services:
                # Update service status to reflect existing process
                self.service_status[service_name] = ServiceStatus(
                    name=service_name,
                    state=ServiceState.RUNNING,
                    detection_result=None,
                    startup_time=0.0,
                    restart_count=0,
                    last_health_check=time.time(),
                    optimization_applied=['existing_process_detected'],
                    performance_metrics={
                        'memory_mb': process_info.memory_mb,
                        'cpu_percent': process_info.cpu_percent
                    },
                    warnings=[],
                    recommendations=[]
                )
                
                logger.info(f"Detected existing {service_name} process (PID: {process_info.pid})")
    
    async def _phase3_optimized_startup(self) -> bool:
        """Phase 3: Optimized service startup sequence"""
        logger.info("Phase 3: Optimized service startup")
        
        with self.startup_lock:
            # Check which services actually need to be started
            services_to_start = []
            
            for service_name, service_config in self.services.items():
                current_status = self.service_status.get(service_name)
                
                if not current_status or current_status.state != ServiceState.RUNNING:
                    # Use enhanced detection to verify service state
                    detection_result = self.service_detector.detect_service_comprehensive(
                        service_name, 
                        service_config.expected_port,
                        service_config.health_check_url
                    )
                    
                    if detection_result.state != ServiceState.RUNNING:
                        services_to_start.append(service_name)
                        logger.info(f"Service {service_name} needs to be started (state: {detection_result.state.value})")
                    else:
                        logger.info(f"Service {service_name} already running (confidence: {detection_result.confidence_score:.2f})")
                        
                        # Update status for already running service
                        self.service_status[service_name] = ServiceStatus(
                            name=service_name,
                            state=ServiceState.RUNNING,
                            detection_result=detection_result,
                            startup_time=0.0,
                            restart_count=0,
                            last_health_check=time.time(),
                            optimization_applied=['pre_existing'],
                            performance_metrics={},
                            warnings=[],
                            recommendations=[]
                        )
            
            if not services_to_start:
                logger.info("All services already running - no startup required")
                self.running = True
                return True
            
            # Use performance optimizer for coordinated startup
            service_configs_dict = {
                name: {
                    'command': self.services[name].command,
                    'env': self.services[name].env,
                    'expected_port': self.services[name].expected_port,
                    'optimization_level': self.services[name].optimization_level.value
                }
                for name in services_to_start
            }
            
            optimization_results = await self.performance_optimizer.optimize_service_startup(
                service_configs_dict,
                OptimizationLevel.BALANCED
            )
            
            # Start services with optimizations
            startup_success = await self._start_services_with_optimization(
                services_to_start, optimization_results
            )
            
            if startup_success:
                self.running = True
                logger.info("All services started successfully")
            else:
                logger.error("Some services failed to start")
            
            return startup_success
    
    async def _start_services_with_optimization(
        self, 
        services_to_start: List[str], 
        optimization_results: Dict[str, OptimizationResult]
    ) -> bool:
        """Start services with applied optimizations"""
        
        all_success = True
        
        # Sort services by priority and dependencies
        sorted_services = self._sort_services_by_priority_and_deps(services_to_start)
        
        for service_name in sorted_services:
            service_config = self.services[service_name]
            start_time = time.time()
            
            try:
                success = await self._start_single_service_enhanced(service_name, service_config)
                startup_time = time.time() - start_time
                
                # Update service status
                optimization_applied = []
                if service_name in optimization_results:
                    opt_result = optimization_results[service_name]
                    optimization_applied.append(opt_result.optimization_applied)
                
                self.service_status[service_name] = ServiceStatus(
                    name=service_name,
                    state=ServiceState.RUNNING if success else ServiceState.CRASHED,
                    detection_result=None,
                    startup_time=startup_time,
                    restart_count=0,
                    last_health_check=time.time(),
                    optimization_applied=optimization_applied,
                    performance_metrics={'startup_time': startup_time},
                    warnings=[],
                    recommendations=[]
                )
                
                if success:
                    logger.info(f"Started {service_name} successfully in {startup_time:.2f}s")
                    self.startup_performance[service_name] = startup_time
                else:
                    logger.error(f"Failed to start {service_name}")
                    all_success = False
                
            except Exception as e:
                logger.error(f"Exception starting {service_name}: {e}")
                all_success = False
        
        return all_success
    
    def _sort_services_by_priority_and_deps(self, services: List[str]) -> List[str]:
        """Sort services by priority and dependencies"""
        
        # Create dependency graph
        service_deps = {}
        for service_name in services:
            deps = self.services[service_name].dependencies or []
            service_deps[service_name] = [d for d in deps if d in services]
        
        # Topological sort with priority
        result = []
        remaining = set(services)
        
        while remaining:
            # Find services with no unmet dependencies
            ready = []
            for service in remaining:
                if not any(dep in remaining for dep in service_deps[service]):
                    ready.append(service)
            
            if not ready:
                # Break circular dependencies by picking highest priority
                ready = [max(remaining, key=lambda s: self.services[s].priority)]
            
            # Sort ready services by priority
            ready.sort(key=lambda s: self.services[s].priority, reverse=True)
            
            # Add highest priority service
            service = ready[0]
            result.append(service)
            remaining.remove(service)
        
        return result
    
    async def _start_single_service_enhanced(
        self, 
        service_name: str, 
        service_config: EnhancedServiceConfig
    ) -> bool:
        """Start a single service with enhanced capabilities"""
        
        logger.info(f"Starting enhanced service: {service_name}")
        
        # First check if already running
        try:
            detection_result = self.service_detector.detect_service_comprehensive(
                service_name,
                service_config.expected_port,
                service_config.health_check_url
            )
            
            if detection_result.state == ServiceState.RUNNING:
                logger.info(f"Service {service_name} already running (confidence: {detection_result.confidence_score:.2f})")
                return True
        except Exception as e:
            logger.debug(f"Pre-startup detection failed for {service_name}: {e}")
        
        try:
            # Import the original daemon manager for actual process starting
            from .daemon_manager import DaemonManager
            original_manager = DaemonManager(self.config_dir)
            
            # Convert enhanced config to original format
            from .daemon_manager import ServiceConfig
            original_config = ServiceConfig(
                name=service_config.name,
                command=service_config.command,
                cwd=service_config.cwd,
                env=service_config.env,
                health_check_url=service_config.health_check_url,
                health_check_interval=service_config.health_check_interval,
                restart_on_failure=service_config.restart_on_failure,
                max_restart_attempts=service_config.max_restart_attempts,
                startup_timeout=service_config.startup_timeout
            )
            
            # Register and start with original manager
            original_manager.register_service(original_config)
            success = await original_manager.start_service(service_name)
            
            if success:
                # Give service time to initialize
                await asyncio.sleep(1)
                
                # Verify with our enhanced detection
                try:
                    detection_result = self.service_detector.detect_service_comprehensive(
                        service_name,
                        service_config.expected_port,
                        service_config.health_check_url
                    )
                    
                    final_success = detection_result.state == ServiceState.RUNNING
                    if not final_success:
                        logger.warning(f"Service {service_name} started but not detected as running (state: {detection_result.state.value})")
                    
                    return final_success
                except Exception as e:
                    logger.error(f"Post-startup detection failed for {service_name}: {e}")
                    # If detection fails but startup reported success, assume it's running
                    return True
            
            logger.error(f"Failed to start service {service_name}")
            return False
            
        except Exception as e:
            logger.error(f"Exception starting {service_name}: {e}")
            return False
    
    async def _phase4_runtime_monitoring(self):
        """Phase 4: Runtime monitoring and optimization"""
        logger.info("Phase 4: Runtime monitoring started")
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._enhanced_monitor_loop())
    
    async def _enhanced_monitor_loop(self):
        """Enhanced monitoring loop with all optimization modules"""
        
        monitor_interval = 10.0  # 10 second monitoring interval
        
        while self.running:
            try:
                # Update service statuses with enhanced detection
                await self._update_enhanced_service_statuses()
                
                # Check for failed services and restart with optimization
                await self._check_and_restart_failed_services_enhanced()
                
                # Adaptive optimization based on performance metrics
                if self.optimization_active:
                    await self._apply_adaptive_optimizations()
                
                await asyncio.sleep(monitor_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Enhanced monitor loop error: {e}")
                await asyncio.sleep(5)
    
    async def _update_enhanced_service_statuses(self):
        """Update service statuses using enhanced detection"""
        
        for service_name in self.services.keys():
            try:
                service_config = self.services[service_name]
                
                # Use comprehensive detection
                detection_result = self.service_detector.detect_service_comprehensive(
                    service_name,
                    service_config.expected_port,
                    service_config.health_check_url
                )
                
                # Update or create service status
                current_status = self.service_status.get(service_name)
                if current_status:
                    current_status.detection_result = detection_result
                    current_status.state = detection_result.state
                    current_status.last_health_check = time.time()
                    
                    if detection_result.process_info:
                        current_status.performance_metrics.update({
                            'memory_mb': detection_result.process_info.memory_mb,
                            'cpu_percent': detection_result.process_info.cpu_percent
                        })
                else:
                    self.service_status[service_name] = ServiceStatus(
                        name=service_name,
                        state=detection_result.state,
                        detection_result=detection_result,
                        startup_time=0.0,
                        restart_count=0,
                        last_health_check=time.time(),
                        optimization_applied=[],
                        performance_metrics={},
                        warnings=[],
                        recommendations=[]
                    )
                
            except Exception as e:
                logger.error(f"Failed to update status for {service_name}: {e}")
    
    async def _check_and_restart_failed_services_enhanced(self):
        """Check and restart failed services with enhanced capabilities"""
        
        for service_name, status in self.service_status.items():
            if status.state in [ServiceState.CRASHED, ServiceState.STOPPED, ServiceState.UNHEALTHY]:
                
                service_config = self.services[service_name]
                if not service_config.restart_on_failure:
                    continue
                
                if status.restart_count >= service_config.max_restart_attempts:
                    logger.error(f"Service {service_name} exceeded max restart attempts")
                    continue
                
                logger.info(f"Restarting failed service: {service_name} (attempt {status.restart_count + 1})")
                
                # Use enhanced restart with optimization
                restart_start = time.time()
                success = await self._start_single_service_enhanced(service_name, service_config)
                restart_time = time.time() - restart_start
                
                if success:
                    status.restart_count += 1
                    status.optimization_applied.append(f"restart_optimization_{status.restart_count}")
                    status.performance_metrics['last_restart_time'] = restart_time
                    logger.info(f"Successfully restarted {service_name} in {restart_time:.2f}s")
                else:
                    logger.error(f"Failed to restart {service_name}")
    
    async def _apply_adaptive_optimizations(self):
        """Apply adaptive optimizations based on current performance"""
        
        # Get performance summary
        perf_summary = self.performance_optimizer.get_performance_summary()
        
        if 'current_metrics' in perf_summary:
            current_metrics = perf_summary['current_metrics']
            
            # Check if we need to adjust optimization levels
            memory_usage = current_metrics.get('memory_usage_mb', 0)
            cpu_usage = current_metrics.get('cpu_usage_percent', 0)
            
            # Adaptive optimization based on resource usage
            if memory_usage > 500 or cpu_usage > 80:  # High resource usage
                # Switch to more conservative optimization
                for service_config in self.services.values():
                    if service_config.optimization_level == OptimizationLevel.AGGRESSIVE:
                        service_config.optimization_level = OptimizationLevel.BALANCED
                        logger.info(f"Reduced optimization level for {service_config.name} due to high resource usage")
            
            elif memory_usage < 200 and cpu_usage < 30:  # Low resource usage
                # Switch to more aggressive optimization
                for service_config in self.services.values():
                    if service_config.optimization_level == OptimizationLevel.MINIMAL:
                        service_config.optimization_level = OptimizationLevel.BALANCED
                        logger.debug(f"Increased optimization level for {service_config.name} due to low resource usage")
    
    async def ensure_core_services_running_enhanced(self) -> bool:
        """Enhanced version of ensuring core services are running"""
        
        if not self.running:
            return await self.start_enhanced_monitoring()
        
        # Quick check using fast detection
        all_running = True
        for service_name in self.services.keys():
            if not self.service_detector.is_service_running_fast(service_name):
                all_running = False
                break
        
        if all_running:
            return True
        
        # If not all running, do a comprehensive check and restart
        logger.info("Some services not running, performing comprehensive check...")
        return await self._phase3_optimized_startup()
    
    def get_enhanced_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        
        summary = {
            'daemon_manager': {
                'running': self.running,
                'optimization_active': self.optimization_active,
                'global_startup_time': self.global_startup_time,
                'services_count': len(self.services)
            },
            'services': {},
            'performance': self.performance_optimizer.get_performance_summary(),
            'detection_stats': self.service_detector.get_detection_stats(),
            'discovery_stats': self.process_discovery.get_discovery_stats()
        }
        
        # Add service status details
        for service_name, status in self.service_status.items():
            summary['services'][service_name] = {
                'state': status.state.value,
                'startup_time': status.startup_time,
                'restart_count': status.restart_count,
                'optimization_applied': status.optimization_applied,
                'performance_metrics': status.performance_metrics,
                'warnings': status.warnings,
                'recommendations': status.recommendations
            }
        
        return summary
    
    async def stop_enhanced_monitoring(self):
        """Stop enhanced monitoring and cleanup"""
        logger.info("Stopping enhanced daemon manager...")
        
        self.running = False
        self.optimization_active = False
        
        # Cancel monitoring task
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop services using original daemon manager
        from .daemon_manager import DaemonManager
        original_manager = DaemonManager(self.config_dir)
        
        for service_name in self.services.keys():
            await original_manager.stop_service(service_name)
        
        # Shutdown specialized modules
        self.process_discovery.shutdown()
        self.config_validator.shutdown()
        self.performance_optimizer.shutdown()
        
        logger.info("Enhanced daemon manager stopped")


# Global enhanced daemon manager instance
_enhanced_daemon_manager: Optional[EnhancedDaemonManager] = None


def get_enhanced_daemon_manager() -> EnhancedDaemonManager:
    """Get singleton enhanced daemon manager instance"""
    global _enhanced_daemon_manager
    if _enhanced_daemon_manager is None:
        _enhanced_daemon_manager = EnhancedDaemonManager()
    return _enhanced_daemon_manager


async def ensure_neuralsync_running_enhanced() -> bool:
    """Enhanced version of ensuring NeuralSync services are running"""
    manager = get_enhanced_daemon_manager()
    return await manager.ensure_core_services_running_enhanced()


async def graceful_shutdown_enhanced():
    """Enhanced graceful shutdown"""
    global _enhanced_daemon_manager
    if _enhanced_daemon_manager:
        await _enhanced_daemon_manager.stop_enhanced_monitoring()


def main():
    """CLI entry point for enhanced daemon management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced NeuralSync Daemon Manager')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status', 'monitor'])
    parser.add_argument('--service', help='Specific service name')
    parser.add_argument('--optimization', choices=['minimal', 'balanced', 'aggressive', 'adaptive'], 
                       default='balanced', help='Optimization level')
    
    args = parser.parse_args()
    
    async def run_enhanced_action():
        manager = get_enhanced_daemon_manager()
        
        if args.action == 'start':
            success = await manager.start_enhanced_monitoring()
            print(f"Enhanced daemon manager: {'started' if success else 'failed'}")
            
        elif args.action == 'stop':
            await manager.stop_enhanced_monitoring()
            print("Enhanced daemon manager stopped")
            
        elif args.action == 'restart':
            await manager.stop_enhanced_monitoring()
            success = await manager.start_enhanced_monitoring()
            print(f"Enhanced daemon manager: {'restarted' if success else 'failed'}")
            
        elif args.action == 'status':
            summary = manager.get_enhanced_status_summary()
            print(json.dumps(summary, indent=2, default=str))
            
        elif args.action == 'monitor':
            await manager.start_enhanced_monitoring()
            print("Enhanced monitoring started. Press Ctrl+C to stop.")
            try:
                while manager.running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                await manager.stop_enhanced_monitoring()
    
    # Setup signal handlers for graceful shutdown
    async def signal_handler():
        await graceful_shutdown_enhanced()
    
    if sys.platform != 'win32':
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda s, f: asyncio.create_task(signal_handler()))
    
    # Run the enhanced action
    try:
        asyncio.run(run_enhanced_action())
    except KeyboardInterrupt:
        asyncio.run(graceful_shutdown_enhanced())


if __name__ == "__main__":
    main()