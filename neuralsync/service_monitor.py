#!/usr/bin/env python3
"""
NeuralSync Service Monitor & Health Check System
Provides comprehensive monitoring and auto-recovery for all NeuralSync services
"""

import asyncio
import time
import logging
import subprocess
import json
import requests
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import threading
import signal

from .config import load_config

logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    """Service health status"""
    name: str
    status: str  # running, stopped, unhealthy, unknown
    pid: Optional[int]
    port: Optional[int]
    uptime: float
    memory_mb: float
    cpu_percent: float
    response_time_ms: Optional[float]
    last_check: float
    error_count: int
    restart_count: int

class ServiceMonitor:
    """Comprehensive service monitoring and health management"""
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path.home() / '.neuralsync'
        self.ns_config = load_config()
        
        # Service definitions
        self.services = {
            'neuralsync-server': {
                'command': ['python3', '-m', 'neuralsync.server'],
                'health_url': f'http://{self.ns_config.bind_host}:{self.ns_config.bind_port}/health',
                'port': self.ns_config.bind_port,
                'critical': True,
                'restart_delay': 5
            },
            'neuralsync-broker': {
                'command': ['python3', '-c', 'from neuralsync.ultra_comm import start_message_broker; import asyncio; asyncio.run(start_message_broker())'],
                'health_url': None,  # Unix socket based
                'port': None,
                'critical': False,
                'restart_delay': 3
            }
        }
        
        self.service_status: Dict[str, ServiceHealth] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.lock = threading.RLock()
        
        # Performance tracking
        self.check_interval = 10.0  # seconds
        self.restart_cooldown = 30.0  # seconds between restarts
        self.max_restart_attempts = 3
        
    async def start_monitoring(self) -> bool:
        """Start comprehensive service monitoring"""
        if self.monitoring:
            return True
            
        logger.info("Starting NeuralSync service monitoring...")
        
        # Initialize service status
        for service_name in self.services.keys():
            self.service_status[service_name] = ServiceHealth(
                name=service_name,
                status='unknown',
                pid=None,
                port=self.services[service_name]['port'],
                uptime=0.0,
                memory_mb=0.0,
                cpu_percent=0.0,
                response_time_ms=None,
                last_check=0.0,
                error_count=0,
                restart_count=0
            )
        
        # Start monitoring loop
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        # Ensure critical services are running
        await self._ensure_critical_services()
        
        logger.info("Service monitoring started successfully")
        return True
    
    async def stop_monitoring(self):
        """Stop service monitoring and cleanup"""
        logger.info("Stopping service monitoring...")
        
        self.monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop all managed processes
        await self._stop_all_services()
        
        logger.info("Service monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                await self._check_all_services()
                await self._handle_unhealthy_services()
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)
    
    async def _check_all_services(self):
        """Check health of all services"""
        with self.lock:
            for service_name, config in self.services.items():
                await self._check_service_health(service_name, config)
    
    async def _check_service_health(self, service_name: str, config: Dict[str, Any]):
        """Check health of a specific service"""
        status = self.service_status[service_name]
        status.last_check = time.time()
        
        # Check if process is running
        process = self.processes.get(service_name)
        if process:
            try:
                # Check if process is still alive
                if process.poll() is None:  # Still running
                    # Get process info using psutil
                    try:
                        ps_process = psutil.Process(process.pid)
                        status.pid = process.pid
                        status.memory_mb = ps_process.memory_info().rss / 1024 / 1024
                        status.cpu_percent = ps_process.cpu_percent()
                        status.uptime = time.time() - ps_process.create_time()
                        
                        # Check HTTP health if available
                        if config.get('health_url'):
                            health_ok, response_time = await self._check_http_health(config['health_url'])
                            if health_ok:
                                status.status = 'running'
                                status.response_time_ms = response_time
                                status.error_count = max(0, status.error_count - 1)  # Recover from errors
                            else:
                                status.status = 'unhealthy'
                                status.error_count += 1
                        else:
                            # No HTTP health check - assume running if process exists
                            status.status = 'running'
                            status.error_count = max(0, status.error_count - 1)
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        status.status = 'stopped'
                        status.pid = None
                        status.error_count += 1
                        
                else:
                    # Process has terminated
                    status.status = 'stopped'
                    status.pid = None
                    status.error_count += 1
                    # Remove from process dict
                    del self.processes[service_name]
                    
            except Exception as e:
                logger.error(f"Error checking {service_name}: {e}")
                status.status = 'unknown'
                status.error_count += 1
        else:
            # No process tracked - check if service might be running externally
            if config.get('health_url'):
                health_ok, response_time = await self._check_http_health(config['health_url'])
                if health_ok:
                    status.status = 'running'
                    status.response_time_ms = response_time
                    status.error_count = 0
                    # Try to find the external process
                    await self._find_external_process(service_name, config)
                else:
                    status.status = 'stopped'
                    status.error_count += 1
            else:
                status.status = 'stopped'
                status.error_count += 1
    
    async def _check_http_health(self, url: str) -> tuple[bool, Optional[float]]:
        """Check HTTP health endpoint"""
        try:
            start_time = time.time()
            response = requests.get(url, timeout=5)
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                return True, response_time
            else:
                return False, response_time
                
        except Exception:
            return False, None
    
    async def _find_external_process(self, service_name: str, config: Dict[str, Any]):
        """Try to find externally running process"""
        try:
            port = config.get('port')
            if port:
                # Look for process listening on the expected port
                for proc in psutil.process_iter(['pid', 'cmdline', 'connections']):
                    try:
                        connections = proc.info['connections'] or []
                        for conn in connections:
                            if conn.laddr.port == port and conn.status == 'LISTEN':
                                # Found process listening on our port
                                status = self.service_status[service_name]
                                status.pid = proc.info['pid']
                                logger.info(f"Found external {service_name} process: PID {proc.info['pid']}")
                                return
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
        except Exception as e:
            logger.debug(f"Error finding external process for {service_name}: {e}")
    
    async def _handle_unhealthy_services(self):
        """Handle unhealthy or stopped services"""
        with self.lock:
            for service_name, status in self.service_status.items():
                config = self.services[service_name]
                
                # Check if service needs restart
                should_restart = False
                
                if status.status in ['stopped', 'unhealthy']:
                    if config.get('critical', False):
                        should_restart = True
                    elif status.error_count >= 3:  # Non-critical but consistently failing
                        should_restart = True
                
                if should_restart and status.restart_count < self.max_restart_attempts:
                    # Check restart cooldown
                    time_since_last_restart = time.time() - status.last_check
                    if time_since_last_restart > self.restart_cooldown:
                        await self._restart_service(service_name)
    
    async def _restart_service(self, service_name: str):
        """Restart a specific service"""
        logger.info(f"Restarting service: {service_name}")
        
        status = self.service_status[service_name]
        config = self.services[service_name]
        
        # Stop existing process if any
        await self._stop_service(service_name)
        
        # Wait for restart delay
        await asyncio.sleep(config.get('restart_delay', 5))
        
        # Start service
        success = await self._start_service(service_name, config)
        
        if success:
            status.restart_count += 1
            status.error_count = 0
            logger.info(f"Successfully restarted {service_name}")
        else:
            status.error_count += 1
            logger.error(f"Failed to restart {service_name}")
    
    async def _start_service(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Start a specific service"""
        try:
            # Prepare environment
            env = dict(os.environ)
            env.update({
                'NS_HOST': self.ns_config.bind_host,
                'NS_PORT': str(self.ns_config.bind_port),
                'NS_TOKEN': self.ns_config.token or '',
                'PYTHONPATH': str(Path(__file__).parent.parent)
            })
            
            # Start process
            process = subprocess.Popen(
                config['command'],
                env=env,
                cwd=Path(__file__).parent.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            self.processes[service_name] = process
            
            # Give service time to start
            await asyncio.sleep(2)
            
            # Check if it's actually running
            if process.poll() is None:
                logger.info(f"Started {service_name} successfully (PID: {process.pid})")
                return True
            else:
                logger.error(f"Service {service_name} failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting {service_name}: {e}")
            return False
    
    async def _stop_service(self, service_name: str):
        """Stop a specific service"""
        process = self.processes.get(service_name)
        if process:
            try:
                # Try graceful shutdown first
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    process.kill()
                    process.wait()
                
                logger.info(f"Stopped service: {service_name}")
                
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")
            finally:
                if service_name in self.processes:
                    del self.processes[service_name]
    
    async def _stop_all_services(self):
        """Stop all managed services"""
        for service_name in list(self.processes.keys()):
            await self._stop_service(service_name)
    
    async def _ensure_critical_services(self):
        """Ensure critical services are running"""
        for service_name, config in self.services.items():
            if config.get('critical', False):
                status = self.service_status[service_name]
                if status.status != 'running':
                    await self._start_service(service_name, config)
    
    def get_service_status(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get service status information"""
        if service_name:
            if service_name in self.service_status:
                return asdict(self.service_status[service_name])
            else:
                return {'error': f'Service {service_name} not found'}
        else:
            return {
                'services': {name: asdict(status) for name, status in self.service_status.items()},
                'monitoring': self.monitoring,
                'check_interval': self.check_interval,
                'total_services': len(self.services)
            }
    
    async def force_restart_service(self, service_name: str) -> bool:
        """Force restart a specific service"""
        if service_name not in self.services:
            return False
        
        await self._restart_service(service_name)
        return True

# Global service monitor instance
_service_monitor: Optional[ServiceMonitor] = None

def get_service_monitor() -> ServiceMonitor:
    """Get singleton service monitor instance"""
    global _service_monitor
    if _service_monitor is None:
        _service_monitor = ServiceMonitor()
    return _service_monitor

async def ensure_services_healthy() -> bool:
    """Ensure all services are healthy"""
    monitor = get_service_monitor()
    if not monitor.monitoring:
        return await monitor.start_monitoring()
    return True

async def stop_service_monitoring():
    """Stop service monitoring"""
    global _service_monitor
    if _service_monitor:
        await _service_monitor.stop_monitoring()

def main():
    """CLI entry point for service monitoring"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='NeuralSync Service Monitor')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'restart'])
    parser.add_argument('--service', help='Specific service name')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    async def run_action():
        monitor = get_service_monitor()
        
        if args.action == 'start':
            success = await monitor.start_monitoring()
            if args.json:
                print(json.dumps({'status': 'started' if success else 'failed'}))
            else:
                print(f"Service monitoring: {'started' if success else 'failed'}")
            
        elif args.action == 'stop':
            await monitor.stop_monitoring()
            if args.json:
                print(json.dumps({'status': 'stopped'}))
            else:
                print("Service monitoring stopped")
            
        elif args.action == 'status':
            status = monitor.get_service_status(args.service)
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                if args.service:
                    service_status = status
                    print(f"Service: {args.service}")
                    print(f"Status: {service_status.get('status', 'unknown')}")
                    print(f"PID: {service_status.get('pid', 'N/A')}")
                    print(f"Uptime: {service_status.get('uptime', 0):.1f}s")
                    print(f"Memory: {service_status.get('memory_mb', 0):.1f}MB")
                    print(f"CPU: {service_status.get('cpu_percent', 0):.1f}%")
                else:
                    print(f"Monitoring: {status['monitoring']}")
                    print(f"Total Services: {status['total_services']}")
                    print("\nServices:")
                    for name, svc_status in status['services'].items():
                        print(f"  {name}: {svc_status['status']} (PID: {svc_status['pid'] or 'N/A'})")
            
        elif args.action == 'restart':
            if args.service:
                success = await monitor.force_restart_service(args.service)
                if args.json:
                    print(json.dumps({'status': 'restarted' if success else 'failed'}))
                else:
                    print(f"Service restart: {'success' if success else 'failed'}")
            else:
                print("ERROR: --service required for restart action")
                sys.exit(1)
    
    # Set up signal handling
    async def shutdown():
        await stop_service_monitoring()
    
    if sys.platform != 'win32':
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda s, f: asyncio.create_task(shutdown()))
    
    try:
        asyncio.run(run_action())
    except KeyboardInterrupt:
        asyncio.run(shutdown())

if __name__ == "__main__":
    main()