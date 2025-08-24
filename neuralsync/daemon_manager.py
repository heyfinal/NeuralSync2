#!/usr/bin/env python3
"""
NeuralSync2 Auto-Launch Daemon Manager
Production-ready daemon management with auto-launch, health monitoring, and recovery
"""

import asyncio
import os
import signal
import sys
import time
import psutil
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import threading
import socket
import requests
from concurrent.futures import ThreadPoolExecutor

from .config import load_config, DEFAULT_HOME
from .ultra_comm import MessageBroker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DaemonStatus:
    """Daemon process status information"""
    pid: int
    running: bool
    uptime_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    port: int
    last_health_check: float
    health_status: str
    restart_count: int
    auto_restart: bool


@dataclass 
class ServiceConfig:
    """Service configuration for managed processes"""
    name: str
    command: List[str]
    cwd: str
    env: Dict[str, str]
    health_check_url: str
    health_check_interval: int
    restart_on_failure: bool
    max_restart_attempts: int
    startup_timeout: int


class DaemonManager:
    """Advanced daemon manager for NeuralSync2 services"""
    
    def __init__(self, config_dir: Path = DEFAULT_HOME):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.pid_dir = self.config_dir / "pids"
        self.log_dir = self.config_dir / "logs" 
        self.pid_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        self.services: Dict[str, ServiceConfig] = {}
        self.service_processes: Dict[str, subprocess.Popen] = {}
        self.service_status: Dict[str, DaemonStatus] = {}
        
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="daemon_mgr")
        self.running = False
        self.monitor_task = None
        
        # Load NeuralSync config
        self.ns_config = load_config()
        
        # Register core services
        self._register_core_services()
        
    def _register_core_services(self):
        """Register core NeuralSync services"""
        
        # Main NeuralSync server
        neuralsync_env = os.environ.copy()
        neuralsync_env.update({
            'NS_HOST': self.ns_config.bind_host,
            'NS_PORT': str(self.ns_config.bind_port),
            'NS_TOKEN': self.ns_config.token
        })
        
        self.services['neuralsync-server'] = ServiceConfig(
            name='neuralsync-server',
            command=[
                sys.executable, '-m', 'neuralsync.enhanced_server'
            ],
            cwd=str(Path(__file__).parent.parent),
            env=neuralsync_env,
            health_check_url=f'http://{self.ns_config.bind_host}:{self.ns_config.bind_port}/health',
            health_check_interval=10,
            restart_on_failure=True,
            max_restart_attempts=5,
            startup_timeout=30
        )
        
        # Message broker for inter-CLI communication  
        self.services['neuralsync-broker'] = ServiceConfig(
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
            startup_timeout=10
        )
        
    def register_service(self, service_config: ServiceConfig):
        """Register a new service to be managed"""
        self.services[service_config.name] = service_config
        logger.info(f"Registered service: {service_config.name}")
        
    async def start_service(self, service_name: str) -> bool:
        """Start a specific service"""
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False
            
        if self.is_service_running(service_name):
            logger.info(f"Service already running: {service_name}")
            return True
            
        service_config = self.services[service_name]
        
        try:
            logger.info(f"Starting service: {service_name}")
            
            # Prepare log files
            stdout_log = self.log_dir / f"{service_name}.out.log"
            stderr_log = self.log_dir / f"{service_name}.err.log"
            
            # Start process
            process = subprocess.Popen(
                service_config.command,
                cwd=service_config.cwd,
                env=service_config.env,
                stdout=open(stdout_log, 'a'),
                stderr=open(stderr_log, 'a'),
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Store process
            self.service_processes[service_name] = process
            
            # Write PID file
            pid_file = self.pid_dir / f"{service_name}.pid"
            with open(pid_file, 'w') as f:
                f.write(str(process.pid))
                
            # Wait for startup
            startup_start = time.time()
            while time.time() - startup_start < service_config.startup_timeout:
                if await self._health_check_service(service_name):
                    logger.info(f"Service started successfully: {service_name} (PID: {process.pid})")
                    
                    # Initialize status
                    self.service_status[service_name] = DaemonStatus(
                        pid=process.pid,
                        running=True,
                        uptime_seconds=0,
                        memory_usage_mb=0,
                        cpu_percent=0,
                        port=self._extract_port_from_config(service_config),
                        last_health_check=time.time(),
                        health_status='healthy',
                        restart_count=0,
                        auto_restart=service_config.restart_on_failure
                    )
                    
                    return True
                    
                await asyncio.sleep(1)
                
            # Startup timeout
            logger.error(f"Service startup timeout: {service_name}")
            await self.stop_service(service_name)
            return False
            
        except Exception as e:
            logger.error(f"Failed to start service {service_name}: {e}")
            return False
            
    async def stop_service(self, service_name: str, force: bool = False) -> bool:
        """Stop a specific service"""
        if service_name not in self.service_processes:
            # Try to stop by PID file
            pid_file = self.pid_dir / f"{service_name}.pid"
            if pid_file.exists():
                try:
                    with open(pid_file) as f:
                        pid = int(f.read().strip())
                    
                    process = psutil.Process(pid)
                    if force:
                        process.kill()
                    else:
                        process.terminate()
                        
                    # Wait for termination
                    try:
                        process.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        process.kill()
                        
                    pid_file.unlink()
                    
                except (ValueError, psutil.NoSuchProcess, FileNotFoundError):
                    pass
                    
            return True
            
        process = self.service_processes[service_name]
        
        try:
            if force:
                process.kill()
            else:
                process.terminate()
                
            # Wait for termination
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                
            # Cleanup
            del self.service_processes[service_name]
            if service_name in self.service_status:
                del self.service_status[service_name]
                
            # Remove PID file
            pid_file = self.pid_dir / f"{service_name}.pid"
            pid_file.unlink(missing_ok=True)
            
            logger.info(f"Service stopped: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop service {service_name}: {e}")
            return False
            
    def is_service_running(self, service_name: str) -> bool:
        """Check if a service is running"""
        # Check in-memory process
        if service_name in self.service_processes:
            process = self.service_processes[service_name]
            if process.poll() is None:
                return True
            else:
                # Process died, clean up
                del self.service_processes[service_name]
                
        # Check PID file
        pid_file = self.pid_dir / f"{service_name}.pid"
        if pid_file.exists():
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                    
                # Check if process exists
                try:
                    process = psutil.Process(pid)
                    return process.is_running()
                except psutil.NoSuchProcess:
                    # Stale PID file
                    pid_file.unlink()
                    return False
                    
            except (ValueError, FileNotFoundError):
                return False
                
        return False
        
    async def _health_check_service(self, service_name: str) -> bool:
        """Perform health check on a service"""
        if service_name not in self.services:
            return False
            
        service_config = self.services[service_name]
        
        try:
            if service_config.health_check_url.startswith('http'):
                # HTTP health check
                response = requests.get(
                    service_config.health_check_url,
                    timeout=5
                )
                return response.status_code == 200
                
            elif service_config.health_check_url.startswith('unix://'):
                # Unix socket health check
                socket_path = service_config.health_check_url[7:]
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    sock.settimeout(2)
                    sock.connect(socket_path)
                    return True
                except:
                    return False
                finally:
                    sock.close()
                    
            else:
                # Process-based check (just verify running)
                return self.is_service_running(service_name)
                
        except Exception as e:
            logger.debug(f"Health check failed for {service_name}: {e}")
            return False
            
    def _extract_port_from_config(self, service_config: ServiceConfig) -> int:
        """Extract port number from service config"""
        if 'NS_PORT' in service_config.env:
            return int(service_config.env['NS_PORT'])
        return 0
        
    async def ensure_core_services_running(self) -> bool:
        """Ensure all core services are running, start if needed"""
        all_healthy = True
        
        # Start services in order of dependency
        service_order = ['neuralsync-broker', 'neuralsync-server']
        
        for service_name in service_order:
            if not self.is_service_running(service_name):
                logger.info(f"Auto-starting core service: {service_name}")
                success = await self.start_service(service_name)
                if not success:
                    logger.error(f"Failed to auto-start core service: {service_name}")
                    all_healthy = False
                else:
                    # Small delay to allow service to fully initialize
                    await asyncio.sleep(2)
            else:
                # Verify health
                if not await self._health_check_service(service_name):
                    logger.warning(f"Core service unhealthy, restarting: {service_name}")
                    await self.stop_service(service_name)
                    success = await self.start_service(service_name)
                    if not success:
                        all_healthy = False
                        
        return all_healthy
        
    async def start_monitoring(self):
        """Start continuous monitoring of services"""
        if self.running:
            return
            
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Started daemon monitoring")
        
    async def stop_monitoring(self):
        """Stop monitoring and shutdown all services"""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        # Stop all services
        for service_name in list(self.service_processes.keys()):
            await self.stop_service(service_name)
            
        self.executor.shutdown(wait=True)
        logger.info("Stopped daemon monitoring")
        
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._update_service_status()
                await self._check_and_restart_failed_services()
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(10)
                
    async def _update_service_status(self):
        """Update status information for all services"""
        for service_name in list(self.service_status.keys()):
            try:
                status = self.service_status[service_name]
                
                # Check if process still exists
                if not self.is_service_running(service_name):
                    status.running = False
                    status.health_status = 'dead'
                    continue
                    
                # Get process info
                try:
                    process = psutil.Process(status.pid)
                    status.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                    status.cpu_percent = process.cpu_percent()
                    status.uptime_seconds = time.time() - process.create_time()
                except psutil.NoSuchProcess:
                    status.running = False
                    status.health_status = 'dead'
                    continue
                    
                # Health check
                service_config = self.services[service_name]
                if time.time() - status.last_health_check >= service_config.health_check_interval:
                    is_healthy = await self._health_check_service(service_name)
                    status.health_status = 'healthy' if is_healthy else 'unhealthy'
                    status.last_health_check = time.time()
                    
                    if not is_healthy:
                        logger.warning(f"Service health check failed: {service_name}")
                        
            except Exception as e:
                logger.error(f"Failed to update status for {service_name}: {e}")
                
    async def _check_and_restart_failed_services(self):
        """Check for failed services and restart if configured"""
        for service_name, service_config in self.services.items():
            if not service_config.restart_on_failure:
                continue
                
            status = self.service_status.get(service_name)
            if not status or status.health_status in ('dead', 'unhealthy'):
                
                if status and status.restart_count >= service_config.max_restart_attempts:
                    logger.error(f"Service {service_name} exceeded max restart attempts ({service_config.max_restart_attempts})")
                    continue
                    
                logger.info(f"Restarting failed service: {service_name}")
                
                # Stop if still running
                if self.is_service_running(service_name):
                    await self.stop_service(service_name)
                    
                # Restart
                success = await self.start_service(service_name)
                
                if success and service_name in self.service_status:
                    self.service_status[service_name].restart_count += 1
                    
    def get_service_status(self, service_name: str) -> Optional[DaemonStatus]:
        """Get status for a specific service"""
        return self.service_status.get(service_name)
        
    def get_all_service_status(self) -> Dict[str, DaemonStatus]:
        """Get status for all services"""
        return self.service_status.copy()
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        system_info = {
            'daemon_manager': {
                'running': self.running,
                'config_dir': str(self.config_dir),
                'services_count': len(self.services),
                'running_services_count': len([s for s in self.service_status.values() if s.running])
            },
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            'services': {
                name: asdict(status) for name, status in self.service_status.items()
            }
        }
        
        return system_info


# Global daemon manager instance
_daemon_manager: Optional[DaemonManager] = None


def get_daemon_manager() -> DaemonManager:
    """Get singleton daemon manager instance"""
    global _daemon_manager
    if _daemon_manager is None:
        _daemon_manager = DaemonManager()
    return _daemon_manager


async def ensure_neuralsync_running() -> bool:
    """Ensure NeuralSync core services are running (main entry point for wrappers)"""
    manager = get_daemon_manager()
    
    # Check if already running
    if (manager.is_service_running('neuralsync-server') and 
        manager.is_service_running('neuralsync-broker')):
        return True
        
    # Start monitoring if not already started
    if not manager.running:
        await manager.start_monitoring()
        
    # Ensure services are running
    return await manager.ensure_core_services_running()


async def graceful_shutdown():
    """Gracefully shutdown daemon manager"""
    global _daemon_manager
    if _daemon_manager:
        await _daemon_manager.stop_monitoring()
        

def main():
    """CLI entry point for daemon management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuralSync Daemon Manager')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status', 'monitor'])
    parser.add_argument('--service', help='Specific service name')
    parser.add_argument('--force', action='store_true', help='Force operation')
    
    args = parser.parse_args()
    
    async def run_action():
        manager = get_daemon_manager()
        
        if args.action == 'start':
            if args.service:
                success = await manager.start_service(args.service)
                print(f"Service {args.service}: {'started' if success else 'failed'}")
            else:
                await manager.start_monitoring()
                success = await manager.ensure_core_services_running()
                print(f"Core services: {'started' if success else 'failed'}")
                
        elif args.action == 'stop':
            if args.service:
                success = await manager.stop_service(args.service, args.force)
                print(f"Service {args.service}: {'stopped' if success else 'failed'}")
            else:
                await manager.stop_monitoring()
                print("All services stopped")
                
        elif args.action == 'restart':
            if args.service:
                await manager.stop_service(args.service, args.force)
                success = await manager.start_service(args.service)
                print(f"Service {args.service}: {'restarted' if success else 'failed'}")
            else:
                await manager.stop_monitoring()
                await manager.start_monitoring()
                success = await manager.ensure_core_services_running()
                print(f"Core services: {'restarted' if success else 'failed'}")
                
        elif args.action == 'status':
            if args.service:
                status = manager.get_service_status(args.service)
                if status:
                    print(json.dumps(asdict(status), indent=2))
                else:
                    print(f"Service {args.service} not found")
            else:
                system_info = manager.get_system_info()
                print(json.dumps(system_info, indent=2))
                
        elif args.action == 'monitor':
            await manager.start_monitoring()
            print("Monitoring started. Press Ctrl+C to stop.")
            try:
                while manager.running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                await manager.stop_monitoring()
                
    # Setup signal handlers
    async def signal_handler():
        await graceful_shutdown()
        
    if sys.platform != 'win32':
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda s, f: asyncio.create_task(signal_handler()))
            
    # Run the action
    try:
        asyncio.run(run_action())
    except KeyboardInterrupt:
        asyncio.run(graceful_shutdown())


if __name__ == "__main__":
    main()