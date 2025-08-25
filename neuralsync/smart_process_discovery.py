#!/usr/bin/env python3
"""
Smart Process Discovery Module for NeuralSync2
Advanced process and port discovery with intelligent conflict resolution
"""

import asyncio
import os
import socket
import time
import psutil
import subprocess
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class DiscoveryStrategy(Enum):
    """Process discovery strategy"""
    FAST = "fast"           # Quick checks only
    THOROUGH = "thorough"   # Comprehensive scanning
    ADAPTIVE = "adaptive"   # Adjusts based on system load

@dataclass 
class PortInfo:
    """Port binding information"""
    port: int
    protocol: str  # TCP/UDP
    state: str     # LISTEN, ESTABLISHED, etc.
    process_pid: Optional[int]
    process_name: Optional[str] 
    process_cmdline: Optional[List[str]]
    bind_address: str
    local_only: bool

@dataclass
class ProcessMatch:
    """Process matching result"""
    pid: int
    confidence: float
    match_criteria: List[str]
    process_info: 'ProcessInfo'

@dataclass
class PortConflict:
    """Port conflict information"""
    port: int
    desired_service: str
    conflicting_processes: List[ProcessMatch]
    resolution_suggestions: List[str]
    can_auto_resolve: bool

class SmartProcessDiscovery:
    """Advanced process discovery with intelligent analysis"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.cache_dir = self.config_dir / "discovery_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Discovery configuration
        self.discovery_config = {
            'cache_ttl': 10.0,  # 10 second cache
            'port_scan_timeout': 2.0,
            'process_scan_batch_size': 50,
            'max_concurrent_checks': 10,
            'adaptive_threshold_cpu': 80.0,  # Switch to fast mode if CPU > 80%
            'adaptive_threshold_load': 2.0   # Switch to fast mode if load > 2.0
        }
        
        # Process pattern matching
        self.neuralsync_patterns = [
            r'python.*neuralsync',
            r'neuralsync.*server',
            r'neuralsync.*broker', 
            r'enhanced_server\.py',
            r'ultra_comm\.py',
            r'message_broker'
        ]
        
        # Port range definitions
        self.neuralsync_port_ranges = [
            (8370, 8380),  # Main service ports
            (9370, 9380),  # Alternative ports  
            (8000, 8010),  # Development ports
        ]
        
        # Caching and performance
        self._process_cache: Dict[str, Tuple[Any, float]] = {}
        self._port_cache: Dict[int, Tuple[PortInfo, float]] = {}
        self._discovery_stats = {
            'total_discoveries': 0,
            'cache_hits': 0,
            'avg_discovery_time': 0.0,
            'port_conflicts_resolved': 0
        }
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(
            max_workers=self.discovery_config['max_concurrent_checks'],
            thread_name_prefix="discovery"
        )
        
        # Locks
        self._cache_lock = threading.RLock()
    
    def _is_cache_valid(self, key: str, cache_dict: Dict, ttl: float = None) -> bool:
        """Check if cached entry is still valid"""
        ttl = ttl or self.discovery_config['cache_ttl']
        
        if key not in cache_dict:
            return False
        
        _, timestamp = cache_dict[key]
        return time.time() - timestamp < ttl
    
    def _update_cache(self, key: str, value: Any, cache_dict: Dict):
        """Update cache entry"""
        with self._cache_lock:
            cache_dict[key] = (value, time.time())
    
    def _get_system_load_factor(self) -> float:
        """Get current system load factor (0.0 = no load, 1.0+ = high load)"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            
            cpu_factor = cpu_percent / 100.0
            load_factor = load_avg / psutil.cpu_count()
            
            return max(cpu_factor, load_factor)
        except:
            return 0.5  # Default moderate load
    
    def _choose_discovery_strategy(self, requested_strategy: DiscoveryStrategy = DiscoveryStrategy.ADAPTIVE) -> DiscoveryStrategy:
        """Choose optimal discovery strategy based on system conditions"""
        if requested_strategy != DiscoveryStrategy.ADAPTIVE:
            return requested_strategy
        
        load_factor = self._get_system_load_factor()
        
        if load_factor > 0.8:  # High load
            return DiscoveryStrategy.FAST
        elif load_factor < 0.3:  # Low load
            return DiscoveryStrategy.THOROUGH
        else:
            return DiscoveryStrategy.FAST  # Default to fast for medium load
    
    async def discover_neuralsync_processes(
        self, 
        strategy: DiscoveryStrategy = DiscoveryStrategy.ADAPTIVE
    ) -> List[ProcessMatch]:
        """Discover all NeuralSync-related processes"""
        
        strategy = self._choose_discovery_strategy(strategy)
        cache_key = f"neuralsync_processes_{strategy.value}"
        
        # Check cache
        with self._cache_lock:
            if self._is_cache_valid(cache_key, self._process_cache):
                result, _ = self._process_cache[cache_key]
                self._discovery_stats['cache_hits'] += 1
                return result
        
        start_time = time.time()
        matches = []
        
        try:
            if strategy == DiscoveryStrategy.FAST:
                matches = await self._discover_processes_fast()
            else:  # THOROUGH
                matches = await self._discover_processes_thorough()
            
            # Update cache
            self._update_cache(cache_key, matches, self._process_cache)
            
            # Update stats
            discovery_time = time.time() - start_time
            self._update_discovery_stats(discovery_time)
            
            return matches
            
        except Exception as e:
            logger.error(f"Process discovery failed: {e}")
            return []
    
    async def _discover_processes_fast(self) -> List[ProcessMatch]:
        """Fast process discovery - basic pattern matching"""
        matches = []
        
        def check_process(proc_info):
            try:
                pid, name, cmdline = proc_info['pid'], proc_info['name'], proc_info['cmdline']
                
                if not cmdline:
                    return None
                
                cmdline_str = ' '.join(cmdline)
                match_criteria = []
                confidence = 0.0
                
                # Check against patterns
                for pattern in self.neuralsync_patterns:
                    if re.search(pattern, cmdline_str, re.IGNORECASE):
                        match_criteria.append(f"cmdline_pattern: {pattern}")
                        confidence += 0.3
                
                # Check process name
                if 'neuralsync' in name.lower():
                    match_criteria.append("process_name")
                    confidence += 0.2
                
                if confidence > 0.2:  # Minimum confidence threshold
                    from .robust_service_detector import RobustServiceDetector, ProcessInfo
                    detector = RobustServiceDetector(self.config_dir)
                    process_info = detector._get_process_info(pid)
                    
                    if process_info:
                        return ProcessMatch(
                            pid=pid,
                            confidence=min(confidence, 1.0),
                            match_criteria=match_criteria,
                            process_info=process_info
                        )
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            return None
        
        # Get all processes efficiently
        process_list = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                process_list.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Check processes concurrently
        loop = asyncio.get_event_loop()
        tasks = []
        
        for i in range(0, len(process_list), self.discovery_config['process_scan_batch_size']):
            batch = process_list[i:i + self.discovery_config['process_scan_batch_size']]
            task = loop.run_in_executor(
                self.executor,
                lambda batch=batch: [check_process(proc) for proc in batch]
            )
            tasks.append(task)
        
        # Collect results
        for task in asyncio.as_completed(tasks):
            batch_results = await task
            for result in batch_results:
                if result:
                    matches.append(result)
        
        return matches
    
    async def _discover_processes_thorough(self) -> List[ProcessMatch]:
        """Thorough process discovery - comprehensive analysis"""
        matches = await self._discover_processes_fast()
        
        # Additional checks for thorough mode
        additional_matches = []
        
        def deep_process_check(proc_info):
            try:
                pid = proc_info['pid']
                
                # Check environment variables
                env_match = False
                try:
                    process = psutil.Process(pid)
                    environ = process.environ()
                    
                    neuralsync_env_vars = [
                        'NS_HOST', 'NS_PORT', 'NS_TOKEN', 'NEURALSYNC_',
                        'TOOL_NAME'
                    ]
                    
                    for env_var in neuralsync_env_vars:
                        if any(env_var in key for key in environ.keys()):
                            env_match = True
                            break
                            
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
                
                # Check working directory
                cwd_match = False
                try:
                    process = psutil.Process(pid)
                    cwd = process.cwd()
                    if 'neuralsync' in cwd.lower():
                        cwd_match = True
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
                
                # Check network connections for NeuralSync ports
                port_match = False
                try:
                    process = psutil.Process(pid)
                    connections = process.connections()
                    
                    for conn in connections:
                        if hasattr(conn, 'laddr') and conn.laddr:
                            port = conn.laddr.port
                            for start_port, end_port in self.neuralsync_port_ranges:
                                if start_port <= port <= end_port:
                                    port_match = True
                                    break
                            if port_match:
                                break
                                
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
                
                # Create match if any deep criteria met
                if env_match or cwd_match or port_match:
                    match_criteria = []
                    confidence = 0.5  # Base confidence for deep matches
                    
                    if env_match:
                        match_criteria.append("environment_variables")
                        confidence += 0.2
                    if cwd_match:
                        match_criteria.append("working_directory")
                        confidence += 0.1
                    if port_match:
                        match_criteria.append("port_binding")
                        confidence += 0.3
                    
                    from .robust_service_detector import RobustServiceDetector
                    detector = RobustServiceDetector(self.config_dir)
                    process_info = detector._get_process_info(pid)
                    
                    if process_info:
                        return ProcessMatch(
                            pid=pid,
                            confidence=min(confidence, 1.0),
                            match_criteria=match_criteria,
                            process_info=process_info
                        )
                        
            except Exception:
                pass
            
            return None
        
        # Check all processes not already found
        existing_pids = {match.pid for match in matches}
        all_processes = []
        
        for proc in psutil.process_iter(['pid']):
            try:
                if proc.info['pid'] not in existing_pids:
                    all_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Run deep checks concurrently
        loop = asyncio.get_event_loop()
        tasks = []
        
        for i in range(0, len(all_processes), self.discovery_config['process_scan_batch_size']):
            batch = all_processes[i:i + self.discovery_config['process_scan_batch_size']]
            task = loop.run_in_executor(
                self.executor,
                lambda batch=batch: [deep_process_check(proc) for proc in batch]
            )
            tasks.append(task)
        
        # Collect additional results
        for task in asyncio.as_completed(tasks):
            batch_results = await task
            for result in batch_results:
                if result:
                    additional_matches.append(result)
        
        return matches + additional_matches
    
    async def discover_port_usage(self, ports: List[int]) -> Dict[int, PortInfo]:
        """Discover detailed port usage information"""
        cache_key = f"port_usage_{'-'.join(map(str, sorted(ports)))}"
        
        # Check cache
        with self._cache_lock:
            if self._is_cache_valid(cache_key, self._process_cache, ttl=5.0):  # Shorter TTL for ports
                result, _ = self._process_cache[cache_key]
                return result
        
        port_info = {}
        
        def check_port(port: int) -> Optional[PortInfo]:
            try:
                # Check all connections for this port
                for conn in psutil.net_connections():
                    if hasattr(conn, 'laddr') and conn.laddr and conn.laddr.port == port:
                        
                        # Get process info
                        process_pid = conn.pid
                        process_name = None
                        process_cmdline = None
                        
                        if process_pid:
                            try:
                                process = psutil.Process(process_pid)
                                process_name = process.name()
                                process_cmdline = process.cmdline()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        
                        # Determine if local-only binding
                        local_only = (conn.laddr.ip == '127.0.0.1' or 
                                     conn.laddr.ip == '::1' or
                                     conn.laddr.ip.startswith('127.'))
                        
                        return PortInfo(
                            port=port,
                            protocol='TCP',  # Most common for our use case
                            state=conn.status,
                            process_pid=process_pid,
                            process_name=process_name,
                            process_cmdline=process_cmdline,
                            bind_address=conn.laddr.ip,
                            local_only=local_only
                        )
                
                # Port not found in connections - check if it's available
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(self.discovery_config['port_scan_timeout'])
                    result = sock.connect_ex(('127.0.0.1', port))
                    sock.close()
                    
                    if result != 0:  # Port is available
                        return PortInfo(
                            port=port,
                            protocol='TCP',
                            state='AVAILABLE',
                            process_pid=None,
                            process_name=None,
                            process_cmdline=None,
                            bind_address='',
                            local_only=True
                        )
                        
                except socket.error:
                    pass
                
                return None
                
            except Exception as e:
                logger.debug(f"Port check failed for {port}: {e}")
                return None
        
        # Check ports concurrently
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, check_port, port)
            for port in ports
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for port, result in zip(ports, results):
            if isinstance(result, PortInfo):
                port_info[port] = result
            elif not isinstance(result, Exception) and result:
                port_info[port] = result
        
        # Update cache
        self._update_cache(cache_key, port_info, self._process_cache)
        
        return port_info
    
    async def detect_port_conflicts(
        self, 
        service_port_map: Dict[str, int]
    ) -> Dict[str, PortConflict]:
        """Detect port conflicts for services"""
        
        all_ports = list(service_port_map.values())
        port_usage = await self.discover_port_usage(all_ports)
        
        conflicts = {}
        
        for service_name, desired_port in service_port_map.items():
            port_info = port_usage.get(desired_port)
            
            if port_info and port_info.state not in ['AVAILABLE']:
                # Find conflicting processes
                conflicting_processes = []
                
                if port_info.process_pid:
                    from .robust_service_detector import RobustServiceDetector
                    detector = RobustServiceDetector(self.config_dir)
                    process_info = detector._get_process_info(port_info.process_pid)
                    
                    if process_info:
                        # Check if this is actually our service
                        is_neuralsync = any(
                            re.search(pattern, ' '.join(process_info.cmdline), re.IGNORECASE)
                            for pattern in self.neuralsync_patterns
                        )
                        
                        if not is_neuralsync:
                            conflicting_processes.append(ProcessMatch(
                                pid=port_info.process_pid,
                                confidence=1.0,
                                match_criteria=["port_conflict"],
                                process_info=process_info
                            ))
                
                if conflicting_processes:
                    # Generate resolution suggestions
                    suggestions = self._generate_conflict_resolutions(
                        service_name, desired_port, conflicting_processes
                    )
                    
                    conflicts[service_name] = PortConflict(
                        port=desired_port,
                        desired_service=service_name,
                        conflicting_processes=conflicting_processes,
                        resolution_suggestions=suggestions,
                        can_auto_resolve=self._can_auto_resolve_conflict(conflicting_processes)
                    )
        
        return conflicts
    
    def _generate_conflict_resolutions(
        self, 
        service_name: str, 
        port: int, 
        conflicting_processes: List[ProcessMatch]
    ) -> List[str]:
        """Generate conflict resolution suggestions"""
        suggestions = []
        
        for process_match in conflicting_processes:
            proc_info = process_match.process_info
            
            # Suggest alternative ports
            suggestions.append(f"Use alternative port for {service_name} (try {port + 10})")
            
            # Suggest stopping conflicting process if safe
            if self._is_safe_to_terminate(proc_info):
                suggestions.append(f"Stop process {proc_info.name} (PID: {proc_info.pid})")
            else:
                suggestions.append(f"Manually resolve conflict with {proc_info.name} (PID: {proc_info.pid})")
            
            # Suggest configuration change
            suggestions.append(f"Configure {service_name} to use different port in config file")
        
        return suggestions
    
    def _is_safe_to_terminate(self, process_info) -> bool:
        """Check if it's safe to terminate a process automatically"""
        # Conservative approach - only terminate processes we recognize as safe
        safe_patterns = [
            r'python.*test',
            r'node.*development',
            r'npm.*start',
        ]
        
        cmdline_str = ' '.join(process_info.cmdline)
        
        for pattern in safe_patterns:
            if re.search(pattern, cmdline_str, re.IGNORECASE):
                return True
        
        return False
    
    def _can_auto_resolve_conflict(self, conflicting_processes: List[ProcessMatch]) -> bool:
        """Check if conflict can be automatically resolved"""
        return all(
            self._is_safe_to_terminate(proc.process_info) 
            for proc in conflicting_processes
        )
    
    async def auto_resolve_conflicts(
        self, 
        conflicts: Dict[str, PortConflict]
    ) -> Dict[str, bool]:
        """Automatically resolve port conflicts where safe"""
        results = {}
        
        for service_name, conflict in conflicts.items():
            if not conflict.can_auto_resolve:
                results[service_name] = False
                continue
            
            try:
                # Terminate conflicting processes
                for process_match in conflict.conflicting_processes:
                    pid = process_match.process_info.pid
                    
                    try:
                        process = psutil.Process(pid)
                        process.terminate()
                        
                        # Wait for termination
                        try:
                            process.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            process.kill()
                            
                        logger.info(f"Terminated conflicting process {pid} for {service_name}")
                        
                    except psutil.NoSuchProcess:
                        pass  # Already terminated
                    
                results[service_name] = True
                self._discovery_stats['port_conflicts_resolved'] += 1
                
            except Exception as e:
                logger.error(f"Failed to resolve conflict for {service_name}: {e}")
                results[service_name] = False
        
        return results
    
    def _update_discovery_stats(self, discovery_time: float):
        """Update discovery performance statistics"""
        self._discovery_stats['total_discoveries'] += 1
        
        current_avg = self._discovery_stats['avg_discovery_time']
        total = self._discovery_stats['total_discoveries']
        
        # Incremental average
        new_avg = current_avg + (discovery_time - current_avg) / total
        self._discovery_stats['avg_discovery_time'] = new_avg
    
    async def get_service_recommendations(
        self, 
        service_name: str
    ) -> Dict[str, Any]:
        """Get recommendations for service configuration"""
        
        # Discover current NeuralSync processes
        processes = await self.discover_neuralsync_processes()
        
        # Check port availability
        recommended_ports = []
        for start_port, end_port in self.neuralsync_port_ranges:
            for port in range(start_port, end_port + 1):
                port_usage = await self.discover_port_usage([port])
                if port not in port_usage or port_usage[port].state == 'AVAILABLE':
                    recommended_ports.append(port)
                    if len(recommended_ports) >= 3:  # Top 3 recommendations
                        break
            if len(recommended_ports) >= 3:
                break
        
        # System resource analysis
        memory_available = psutil.virtual_memory().available / 1024 / 1024  # MB
        cpu_count = psutil.cpu_count()
        load_avg = self._get_system_load_factor()
        
        recommendations = {
            'service_name': service_name,
            'recommended_ports': recommended_ports,
            'existing_neuralsync_processes': len(processes),
            'system_resources': {
                'memory_available_mb': memory_available,
                'cpu_count': cpu_count,
                'load_factor': load_avg,
                'recommended_max_workers': max(1, cpu_count - 1)
            },
            'configuration_suggestions': []
        }
        
        # Generate configuration suggestions
        if memory_available < 512:
            recommendations['configuration_suggestions'].append(
                "Low memory detected - consider reducing cache sizes"
            )
        
        if load_avg > 0.8:
            recommendations['configuration_suggestions'].append(
                "High system load - consider reducing concurrent operations"
            )
        
        if len(processes) > 3:
            recommendations['configuration_suggestions'].append(
                "Multiple NeuralSync processes detected - verify configuration"
            )
        
        return recommendations
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery performance statistics"""
        return self._discovery_stats.copy()
    
    def clear_caches(self):
        """Clear all discovery caches"""
        with self._cache_lock:
            self._process_cache.clear()
            self._port_cache.clear()
    
    def shutdown(self):
        """Shutdown discovery system"""
        self.executor.shutdown(wait=True)

# Testing function
async def test_smart_discovery():
    """Test smart process discovery functionality"""
    discovery = SmartProcessDiscovery(Path.home() / ".neuralsync")
    
    try:
        # Test process discovery
        print("Testing process discovery...")
        processes = await discovery.discover_neuralsync_processes(DiscoveryStrategy.THOROUGH)
        print(f"Found {len(processes)} NeuralSync processes")
        
        for proc in processes[:3]:  # Show first 3
            print(f"  PID: {proc.pid}, Confidence: {proc.confidence:.2f}, Criteria: {proc.match_criteria}")
        
        # Test port discovery
        print("\nTesting port discovery...")
        test_ports = [8373, 8374, 8375, 22, 80]
        port_usage = await discovery.discover_port_usage(test_ports)
        
        for port, info in port_usage.items():
            print(f"  Port {port}: {info.state} ({'local' if info.local_only else 'public'})")
        
        # Test conflict detection
        print("\nTesting conflict detection...")
        service_ports = {
            'neuralsync-server': 8373,
            'neuralsync-broker': 8374
        }
        
        conflicts = await discovery.detect_port_conflicts(service_ports)
        print(f"Found {len(conflicts)} conflicts")
        
        for service, conflict in conflicts.items():
            print(f"  {service}: Port {conflict.port} conflicts with {len(conflict.conflicting_processes)} processes")
        
        # Test recommendations
        print("\nTesting recommendations...")
        recommendations = await discovery.get_service_recommendations("test-service")
        print(f"Recommended ports: {recommendations['recommended_ports']}")
        print(f"System resources: {recommendations['system_resources']}")
        
        # Show stats
        print(f"\nDiscovery stats: {discovery.get_discovery_stats()}")
        
    finally:
        discovery.shutdown()

if __name__ == "__main__":
    asyncio.run(test_smart_discovery())