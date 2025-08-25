#!/usr/bin/env python3
"""
Robust Service Detection Module for NeuralSync2
Provides reliable process and port detection with race condition prevention
"""

import asyncio
import os
import socket
import time
import psutil
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

logger = logging.getLogger(__name__)

class ServiceState(Enum):
    """Service state enumeration"""
    UNKNOWN = "unknown"
    STARTING = "starting" 
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CRASHED = "crashed"
    UNHEALTHY = "unhealthy"

@dataclass
class ProcessInfo:
    """Detailed process information"""
    pid: int
    name: str
    cmdline: List[str]
    create_time: float
    memory_mb: float
    cpu_percent: float
    status: str
    cwd: str
    ports: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)

@dataclass
class ServiceDetectionResult:
    """Service detection result"""
    service_name: str
    state: ServiceState
    process_info: Optional[ProcessInfo]
    pid_file_exists: bool
    pid_file_valid: bool
    port_bound: bool
    health_check_passed: bool
    confidence_score: float
    detection_time: float
    errors: List[str] = field(default_factory=list)

class RobustServiceDetector:
    """Advanced service detection with race condition prevention"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.pid_dir = self.config_dir / "pids"
        self.lock_dir = self.config_dir / "locks"
        
        # Create directories
        self.pid_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        
        # Detection cache with TTL
        self._cache: Dict[str, Tuple[ServiceDetectionResult, float]] = {}
        self._cache_ttl = 5.0  # 5 second cache
        
        # Process monitoring
        self._process_snapshots: Dict[int, ProcessInfo] = {}
        self._port_allocations: Dict[int, str] = {}
        
        # Locking for thread safety
        self._detection_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.RLock()
        
        # Performance metrics
        self._detection_stats = {
            'total_detections': 0,
            'cache_hits': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'avg_detection_time': 0.0
        }
        
    def _get_service_lock(self, service_name: str) -> threading.Lock:
        """Get or create a lock for service detection"""
        with self._global_lock:
            if service_name not in self._detection_locks:
                self._detection_locks[service_name] = threading.Lock()
            return self._detection_locks[service_name]
    
    def _is_cache_valid(self, service_name: str) -> bool:
        """Check if cached detection result is still valid"""
        if service_name not in self._cache:
            return False
        
        result, timestamp = self._cache[service_name]
        return time.time() - timestamp < self._cache_ttl
    
    def _update_cache(self, service_name: str, result: ServiceDetectionResult):
        """Update detection cache"""
        self._cache[service_name] = (result, time.time())
    
    def _get_process_info(self, pid: int) -> Optional[ProcessInfo]:
        """Get detailed process information with error handling"""
        try:
            process = psutil.Process(pid)
            
            # Collect basic info
            create_time = process.create_time()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            # Get process ports
            ports = []
            try:
                connections = process.connections()
                ports = [conn.laddr.port for conn in connections 
                        if conn.status == psutil.CONN_LISTEN]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
            
            # Get child processes
            children = []
            try:
                children = [child.pid for child in process.children()]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
            
            return ProcessInfo(
                pid=pid,
                name=process.name(),
                cmdline=process.cmdline(),
                create_time=create_time,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                status=process.status(),
                cwd=process.cwd(),
                ports=ports,
                children=children
            )
            
        except psutil.NoSuchProcess:
            return None
        except Exception as e:
            logger.debug(f"Failed to get process info for PID {pid}: {e}")
            return None
    
    def _check_pid_file(self, service_name: str) -> Tuple[bool, bool, Optional[int]]:
        """Check PID file existence and validity"""
        pid_file = self.pid_dir / f"{service_name}.pid"
        
        if not pid_file.exists():
            return False, False, None
        
        try:
            with open(pid_file, 'r') as f:
                pid_str = f.read().strip()
                
            if not pid_str.isdigit():
                return True, False, None
            
            pid = int(pid_str)
            
            # Check if PID exists and is valid
            try:
                process = psutil.Process(pid)
                return True, True, pid
            except psutil.NoSuchProcess:
                # Stale PID file
                return True, False, None
                
        except (ValueError, FileNotFoundError) as e:
            logger.debug(f"Invalid PID file for {service_name}: {e}")
            return True, False, None
    
    def _check_port_binding(self, port: int, expected_pid: Optional[int] = None) -> bool:
        """Check if port is bound by expected process"""
        try:
            # First, try to bind to check availability
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            
            try:
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                
                if result == 0:
                    # Port is bound, check if by expected process
                    if expected_pid is not None:
                        return self._is_port_bound_by_pid(port, expected_pid)
                    return True
                else:
                    return False
                    
            except Exception:
                sock.close()
                return False
                
        except Exception as e:
            logger.debug(f"Port check failed for {port}: {e}")
            return False
    
    def _is_port_bound_by_pid(self, port: int, expected_pid: int) -> bool:
        """Check if specific port is bound by specific PID"""
        try:
            process = psutil.Process(expected_pid)
            connections = process.connections()
            
            for conn in connections:
                if (conn.laddr.port == port and 
                    conn.status == psutil.CONN_LISTEN):
                    return True
                    
            return False
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def _clean_stale_pid_file(self, service_name: str):
        """Remove stale PID file"""
        pid_file = self.pid_dir / f"{service_name}.pid"
        try:
            pid_file.unlink(missing_ok=True)
            logger.debug(f"Cleaned stale PID file for {service_name}")
        except Exception as e:
            logger.warning(f"Failed to clean stale PID file for {service_name}: {e}")
    
    def _calculate_confidence_score(
        self, 
        process_info: Optional[ProcessInfo],
        pid_file_valid: bool,
        port_bound: bool,
        health_check_passed: bool
    ) -> float:
        """Calculate confidence score for service detection"""
        score = 0.0
        
        # Process existence and details (40%)
        if process_info:
            score += 0.3
            # Recent start time indicates legitimate process
            if time.time() - process_info.create_time < 3600:  # Within 1 hour
                score += 0.1
        
        # Valid PID file (30%)
        if pid_file_valid:
            score += 0.3
        
        # Port binding (20%)
        if port_bound:
            score += 0.2
        
        # Health check (10%)
        if health_check_passed:
            score += 0.1
        
        return min(score, 1.0)
    
    def detect_service_comprehensive(
        self, 
        service_name: str, 
        expected_port: Optional[int] = None,
        health_check_url: Optional[str] = None
    ) -> ServiceDetectionResult:
        """Comprehensive service detection with race condition prevention"""
        
        start_time = time.time()
        service_lock = self._get_service_lock(service_name)
        
        with service_lock:
            # Check cache first
            if self._is_cache_valid(service_name):
                result, _ = self._cache[service_name]
                self._detection_stats['cache_hits'] += 1
                return result
            
            self._detection_stats['total_detections'] += 1
            errors = []
            
            # Check PID file
            pid_file_exists, pid_file_valid, pid = self._check_pid_file(service_name)
            
            # Get process information
            process_info = None
            if pid:
                process_info = self._get_process_info(pid)
                if not process_info:
                    # Process doesn't exist but PID file does - stale file
                    pid_file_valid = False
                    self._clean_stale_pid_file(service_name)
            
            # Check port binding
            port_bound = False
            if expected_port and pid:
                port_bound = self._check_port_binding(expected_port, pid)
            elif expected_port:
                port_bound = self._check_port_binding(expected_port)
            
            # Health check (placeholder - would be implemented by caller)
            health_check_passed = False  # Will be set by external health check
            
            # Determine service state
            state = self._determine_service_state(
                process_info, pid_file_valid, port_bound, health_check_passed
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                process_info, pid_file_valid, port_bound, health_check_passed
            )
            
            # Create result
            detection_time = time.time() - start_time
            result = ServiceDetectionResult(
                service_name=service_name,
                state=state,
                process_info=process_info,
                pid_file_exists=pid_file_exists,
                pid_file_valid=pid_file_valid,
                port_bound=port_bound,
                health_check_passed=health_check_passed,
                confidence_score=confidence_score,
                detection_time=detection_time,
                errors=errors
            )
            
            # Update cache
            self._update_cache(service_name, result)
            
            # Update performance stats
            self._update_performance_stats(detection_time)
            
            return result
    
    def _determine_service_state(
        self, 
        process_info: Optional[ProcessInfo],
        pid_file_valid: bool,
        port_bound: bool,
        health_check_passed: bool
    ) -> ServiceState:
        """Determine service state based on detection results"""
        
        if not process_info and not pid_file_valid:
            return ServiceState.STOPPED
        
        if process_info and pid_file_valid:
            if port_bound:
                if health_check_passed:
                    return ServiceState.RUNNING
                else:
                    return ServiceState.UNHEALTHY
            else:
                return ServiceState.STARTING
        
        if process_info and not pid_file_valid:
            # Process exists but no valid PID file - might be externally started
            if port_bound:
                return ServiceState.RUNNING
            else:
                return ServiceState.STARTING
        
        if not process_info and pid_file_valid:
            # PID file exists but no process - crashed
            return ServiceState.CRASHED
        
        return ServiceState.UNKNOWN
    
    def _update_performance_stats(self, detection_time: float):
        """Update performance statistics"""
        current_avg = self._detection_stats['avg_detection_time']
        total = self._detection_stats['total_detections']
        
        # Incremental average calculation
        new_avg = current_avg + (detection_time - current_avg) / total
        self._detection_stats['avg_detection_time'] = new_avg
    
    def is_service_running_fast(self, service_name: str) -> bool:
        """Fast service running check - uses cache aggressively"""
        if self._is_cache_valid(service_name):
            result, _ = self._cache[service_name]
            return result.state == ServiceState.RUNNING
        
        # Quick check without full detection
        pid_file_exists, pid_file_valid, pid = self._check_pid_file(service_name)
        
        if not pid_file_valid:
            return False
        
        if pid:
            try:
                process = psutil.Process(pid)
                return process.is_running()
            except psutil.NoSuchProcess:
                return False
        
        return False
    
    def find_services_by_pattern(self, pattern: str) -> List[ServiceDetectionResult]:
        """Find services matching a pattern"""
        results = []
        
        # Search by command line pattern
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline_str = ' '.join(process.info['cmdline'] or [])
                if pattern.lower() in cmdline_str.lower():
                    process_info = self._get_process_info(process.info['pid'])
                    if process_info:
                        result = ServiceDetectionResult(
                            service_name=f"pattern_{process.info['pid']}",
                            state=ServiceState.RUNNING,
                            process_info=process_info,
                            pid_file_exists=False,
                            pid_file_valid=False,
                            port_bound=len(process_info.ports) > 0,
                            health_check_passed=False,
                            confidence_score=0.6,
                            detection_time=0.0
                        )
                        results.append(result)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return results
    
    def get_port_conflicts(self, desired_ports: List[int]) -> Dict[int, List[ProcessInfo]]:
        """Get processes that are using desired ports"""
        conflicts = {}
        
        for port in desired_ports:
            processes = []
            
            for process in psutil.process_iter(['pid']):
                try:
                    process_info = self._get_process_info(process.info['pid'])
                    if process_info and port in process_info.ports:
                        processes.append(process_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if processes:
                conflicts[port] = processes
        
        return conflicts
    
    def cleanup_stale_resources(self) -> Dict[str, List[str]]:
        """Clean up stale PID files and resources"""
        cleanup_results = {
            'pid_files_cleaned': [],
            'locks_cleaned': [],
            'errors': []
        }
        
        try:
            # Clean stale PID files
            for pid_file in self.pid_dir.glob("*.pid"):
                service_name = pid_file.stem
                
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    
                    # Check if process exists
                    if not psutil.pid_exists(pid):
                        pid_file.unlink()
                        cleanup_results['pid_files_cleaned'].append(str(pid_file))
                        
                except (ValueError, FileNotFoundError) as e:
                    pid_file.unlink(missing_ok=True)
                    cleanup_results['pid_files_cleaned'].append(str(pid_file))
                except Exception as e:
                    cleanup_results['errors'].append(f"Failed to clean {pid_file}: {e}")
            
            # Clean stale lock files (if any)
            for lock_file in self.lock_dir.glob("*.lock"):
                try:
                    # Simple time-based cleanup for now
                    if time.time() - lock_file.stat().st_mtime > 3600:  # 1 hour old
                        lock_file.unlink()
                        cleanup_results['locks_cleaned'].append(str(lock_file))
                except Exception as e:
                    cleanup_results['errors'].append(f"Failed to clean {lock_file}: {e}")
                    
        except Exception as e:
            cleanup_results['errors'].append(f"General cleanup error: {e}")
        
        return cleanup_results
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection performance statistics"""
        return self._detection_stats.copy()
    
    def clear_cache(self):
        """Clear detection cache"""
        with self._global_lock:
            self._cache.clear()

# Testing and validation functions
def test_service_detector():
    """Test the service detector functionality"""
    detector = RobustServiceDetector(Path.home() / ".neuralsync")
    
    # Test basic detection
    result = detector.detect_service_comprehensive("test-service", 8373)
    print(f"Detection result: {result}")
    
    # Test cleanup
    cleanup_results = detector.cleanup_stale_resources()
    print(f"Cleanup results: {cleanup_results}")
    
    # Test performance stats
    stats = detector.get_detection_stats()
    print(f"Performance stats: {stats}")

if __name__ == "__main__":
    test_service_detector()