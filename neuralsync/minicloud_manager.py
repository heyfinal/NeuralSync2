#!/usr/bin/env python3
"""
MiniCloud Management Module for NeuralSync2
Provides server health monitoring, auto-configuration, and remote management
"""

import asyncio
import json
import logging
import time
import subprocess
import socket
import ssl
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import paramiko
import psutil
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import aiofiles
import aiohttp
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ServerState(Enum):
    """MiniCloud server states"""
    UNKNOWN = "unknown"
    DISCOVERING = "discovering"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    ERROR = "error"

@dataclass
class MiniCloudServer:
    """MiniCloud server configuration and state"""
    server_id: str
    name: str
    hostname: str
    ip_address: str
    ssh_port: int = 22
    ssh_user: str = "root"
    ssh_key_path: Optional[str] = None
    ssh_password: Optional[str] = None
    
    # Server capabilities and status
    state: ServerState = ServerState.UNKNOWN
    last_seen: float = 0
    uptime: float = 0
    cpu_usage: float = 0
    memory_usage: float = 0
    disk_usage: float = 0
    network_latency: float = 0
    
    # Configuration
    auto_reboot_enabled: bool = True
    hibernation_disabled: bool = True
    memory_monitoring_enabled: bool = True
    encryption_unlocked: bool = False
    
    # Services and processes
    running_services: List[str] = None
    neuralsync_installed: bool = False
    neuralsync_version: Optional[str] = None
    
    # Security and authentication
    ssh_connection: Optional[paramiko.SSHClient] = None
    auth_token: Optional[str] = None
    last_backup: Optional[float] = None
    
    metadata: Dict[str, Any] = None

@dataclass
class ServerMetrics:
    """Comprehensive server metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: Tuple[float, float, float]
    process_count: int
    connection_count: int
    temperature: Optional[float] = None
    uptime_hours: float = 0

@dataclass
class ConfigurationTemplate:
    """Server configuration template"""
    name: str
    description: str
    target_os: str
    commands: List[str]
    required_packages: List[str]
    configuration_files: Dict[str, str]
    validation_checks: List[str]
    rollback_commands: List[str]

class MiniCloudSSHManager:
    """Secure SSH connection manager for MiniCloud servers"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.ssh_key_dir = config_dir / "ssh_keys"
        self.ssh_key_dir.mkdir(parents=True, exist_ok=True)
        self.connection_timeout = 30
        self.command_timeout = 300
        
    async def create_ssh_connection(self, server: MiniCloudServer) -> bool:
        """Create SSH connection to MiniCloud server"""
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connection parameters
            connect_params = {
                'hostname': server.ip_address,
                'port': server.ssh_port,
                'username': server.ssh_user,
                'timeout': self.connection_timeout,
                'banner_timeout': 30,
                'auth_timeout': 30
            }
            
            # Use SSH key if available
            if server.ssh_key_path and Path(server.ssh_key_path).exists():
                connect_params['key_filename'] = server.ssh_key_path
            elif server.ssh_password:
                connect_params['password'] = server.ssh_password
            else:
                # Try to use default keys
                default_key = self.ssh_key_dir / f"{server.server_id}_rsa"
                if default_key.exists():
                    connect_params['key_filename'] = str(default_key)
            
            # Establish connection
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.connect(**connect_params)
            )
            
            server.ssh_connection = client
            server.state = ServerState.CONNECTED
            logger.info(f"SSH connection established to {server.name}")
            return True
            
        except Exception as e:
            logger.error(f"SSH connection failed to {server.name}: {e}")
            server.state = ServerState.ERROR
            return False
    
    async def execute_command(self, server: MiniCloudServer, command: str) -> Tuple[int, str, str]:
        """Execute command on remote server via SSH"""
        if not server.ssh_connection:
            if not await self.create_ssh_connection(server):
                return -1, "", "No SSH connection"
        
        try:
            stdin, stdout, stderr = server.ssh_connection.exec_command(
                command, timeout=self.command_timeout
            )
            
            # Read output
            exit_code = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode('utf-8', errors='ignore')
            stderr_text = stderr.read().decode('utf-8', errors='ignore')
            
            return exit_code, stdout_text, stderr_text
            
        except Exception as e:
            logger.error(f"Command execution failed on {server.name}: {e}")
            return -1, "", str(e)
    
    async def upload_file(self, server: MiniCloudServer, local_path: str, remote_path: str) -> bool:
        """Upload file to remote server via SFTP"""
        if not server.ssh_connection:
            if not await self.create_ssh_connection(server):
                return False
        
        try:
            sftp = server.ssh_connection.open_sftp()
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: sftp.put(local_path, remote_path)
            )
            sftp.close()
            
            logger.debug(f"Uploaded {local_path} to {server.name}:{remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"File upload failed to {server.name}: {e}")
            return False
    
    async def download_file(self, server: MiniCloudServer, remote_path: str, local_path: str) -> bool:
        """Download file from remote server via SFTP"""
        if not server.ssh_connection:
            if not await self.create_ssh_connection(server):
                return False
        
        try:
            sftp = server.ssh_connection.open_sftp()
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: sftp.get(remote_path, local_path)
            )
            sftp.close()
            
            logger.debug(f"Downloaded {server.name}:{remote_path} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"File download failed from {server.name}: {e}")
            return False
    
    def close_connection(self, server: MiniCloudServer):
        """Close SSH connection"""
        if server.ssh_connection:
            server.ssh_connection.close()
            server.ssh_connection = None
            server.state = ServerState.OFFLINE

class MiniCloudHealthMonitor:
    """Health monitoring for MiniCloud servers"""
    
    def __init__(self, ssh_manager: MiniCloudSSHManager):
        self.ssh_manager = ssh_manager
        self.monitoring_interval = 60  # 1 minute
        self.health_thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 75.0,
            'memory_critical': 95.0,
            'memory_warning': 85.0,
            'disk_critical': 95.0,
            'disk_warning': 85.0,
            'temperature_critical': 80.0,
            'temperature_warning': 70.0
        }
    
    async def collect_server_metrics(self, server: MiniCloudServer) -> Optional[ServerMetrics]:
        """Collect comprehensive server metrics"""
        try:
            metrics_commands = {
                'cpu_usage': "top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | sed 's/%us,//'",
                'memory_info': "free -g | awk 'NR==2{printf \"%.2f %.2f\", $3/$2*100, $7}'",
                'disk_usage': "df -h / | awk 'NR==2{print $5 \" \" $4}'",
                'load_average': "uptime | awk -F'load average:' '{print $2}'",
                'process_count': "ps aux | wc -l",
                'network_stats': "cat /proc/net/dev | grep eth0 | awk '{print $2 \" \" $10}'",
                'uptime': "uptime -p",
                'temperature': "sensors | grep 'Core 0' | awk '{print $3}' | sed 's/+//g' | sed 's/°C//g'"
            }
            
            metrics_data = {}
            for metric, command in metrics_commands.items():
                exit_code, stdout, stderr = await self.ssh_manager.execute_command(server, command)
                if exit_code == 0:
                    metrics_data[metric] = stdout.strip()
            
            # Parse metrics
            timestamp = time.time()
            
            # CPU usage
            cpu_percent = 0.0
            if 'cpu_usage' in metrics_data:
                try:
                    cpu_percent = float(metrics_data['cpu_usage'])
                except:
                    pass
            
            # Memory usage
            memory_percent = 0.0
            memory_available_gb = 0.0
            if 'memory_info' in metrics_data:
                try:
                    mem_parts = metrics_data['memory_info'].split()
                    memory_percent = float(mem_parts[0])
                    memory_available_gb = float(mem_parts[1])
                except:
                    pass
            
            # Disk usage
            disk_usage_percent = 0.0
            disk_free_gb = 0.0
            if 'disk_usage' in metrics_data:
                try:
                    disk_parts = metrics_data['disk_usage'].split()
                    disk_usage_percent = float(disk_parts[0].replace('%', ''))
                    disk_free_str = disk_parts[1]
                    if disk_free_str.endswith('G'):
                        disk_free_gb = float(disk_free_str[:-1])
                    elif disk_free_str.endswith('T'):
                        disk_free_gb = float(disk_free_str[:-1]) * 1024
                except:
                    pass
            
            # Load average
            load_average = (0.0, 0.0, 0.0)
            if 'load_average' in metrics_data:
                try:
                    loads = [float(x.strip()) for x in metrics_data['load_average'].split(',')]
                    load_average = tuple(loads[:3])
                except:
                    pass
            
            # Process count
            process_count = 0
            if 'process_count' in metrics_data:
                try:
                    process_count = int(metrics_data['process_count'])
                except:
                    pass
            
            # Network stats
            network_bytes_sent = 0
            network_bytes_recv = 0
            if 'network_stats' in metrics_data:
                try:
                    net_parts = metrics_data['network_stats'].split()
                    network_bytes_recv = int(net_parts[0])
                    network_bytes_sent = int(net_parts[1])
                except:
                    pass
            
            # Temperature
            temperature = None
            if 'temperature' in metrics_data:
                try:
                    temperature = float(metrics_data['temperature'])
                except:
                    pass
            
            # Uptime
            uptime_hours = 0.0
            if 'uptime' in metrics_data:
                try:
                    uptime_str = metrics_data['uptime']
                    if 'hour' in uptime_str:
                        hours = int(uptime_str.split()[1])
                        uptime_hours = float(hours)
                    elif 'day' in uptime_str:
                        days = int(uptime_str.split()[1])
                        uptime_hours = float(days * 24)
                except:
                    pass
            
            # Update server state
            server.cpu_usage = cpu_percent
            server.memory_usage = memory_percent
            server.disk_usage = disk_usage_percent
            server.last_seen = timestamp
            
            # Create metrics object
            metrics = ServerMetrics(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                load_average=load_average,
                process_count=process_count,
                connection_count=0,
                temperature=temperature,
                uptime_hours=uptime_hours
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics from {server.name}: {e}")
            return None
    
    async def assess_server_health(self, server: MiniCloudServer, metrics: ServerMetrics) -> Dict[str, Any]:
        """Assess server health based on metrics"""
        health_status = {
            'overall': 'healthy',
            'score': 100,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # CPU health assessment
        if metrics.cpu_percent >= self.health_thresholds['cpu_critical']:
            health_status['issues'].append(f"Critical CPU usage: {metrics.cpu_percent:.1f}%")
            health_status['score'] -= 30
            health_status['overall'] = 'critical'
        elif metrics.cpu_percent >= self.health_thresholds['cpu_warning']:
            health_status['warnings'].append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            health_status['score'] -= 10
            if health_status['overall'] == 'healthy':
                health_status['overall'] = 'warning'
        
        # Memory health assessment
        if metrics.memory_percent >= self.health_thresholds['memory_critical']:
            health_status['issues'].append(f"Critical memory usage: {metrics.memory_percent:.1f}%")
            health_status['score'] -= 30
            health_status['overall'] = 'critical'
        elif metrics.memory_percent >= self.health_thresholds['memory_warning']:
            health_status['warnings'].append(f"High memory usage: {metrics.memory_percent:.1f}%")
            health_status['score'] -= 10
            if health_status['overall'] == 'healthy':
                health_status['overall'] = 'warning'
        
        # Disk health assessment
        if metrics.disk_usage_percent >= self.health_thresholds['disk_critical']:
            health_status['issues'].append(f"Critical disk usage: {metrics.disk_usage_percent:.1f}%")
            health_status['score'] -= 25
            health_status['overall'] = 'critical'
        elif metrics.disk_usage_percent >= self.health_thresholds['disk_warning']:
            health_status['warnings'].append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
            health_status['score'] -= 5
            if health_status['overall'] == 'healthy':
                health_status['overall'] = 'warning'
        
        # Temperature assessment
        if metrics.temperature and metrics.temperature >= self.health_thresholds['temperature_critical']:
            health_status['issues'].append(f"Critical temperature: {metrics.temperature:.1f}°C")
            health_status['score'] -= 20
            health_status['overall'] = 'critical'
        elif metrics.temperature and metrics.temperature >= self.health_thresholds['temperature_warning']:
            health_status['warnings'].append(f"High temperature: {metrics.temperature:.1f}°C")
            health_status['score'] -= 5
            if health_status['overall'] == 'healthy':
                health_status['overall'] = 'warning'
        
        # Load average assessment
        cpu_cores = max(1, metrics.process_count // 20)  # Rough estimate
        if max(metrics.load_average) > cpu_cores * 2:
            health_status['warnings'].append(f"High load average: {metrics.load_average[0]:.2f}")
            health_status['score'] -= 5
        
        # Generate recommendations
        if metrics.memory_percent > 80:
            health_status['recommendations'].append("Consider adding more RAM or reducing memory usage")
        
        if metrics.disk_usage_percent > 80:
            health_status['recommendations'].append("Clean up disk space or add more storage")
        
        if metrics.cpu_percent > 80:
            health_status['recommendations'].append("Check for CPU-intensive processes")
        
        # Update server state based on health
        if health_status['overall'] == 'critical':
            server.state = ServerState.DEGRADED
        elif health_status['overall'] == 'warning':
            server.state = ServerState.HEALTHY  # Still healthy but with warnings
        else:
            server.state = ServerState.HEALTHY
        
        return health_status

class MiniCloudConfigManager:
    """Configuration management for MiniCloud servers"""
    
    def __init__(self, ssh_manager: MiniCloudSSHManager, config_dir: Path):
        self.ssh_manager = ssh_manager
        self.config_dir = config_dir
        self.templates_dir = config_dir / "minicloud_templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Load built-in configuration templates
        self._create_builtin_templates()
    
    def _create_builtin_templates(self):
        """Create built-in configuration templates"""
        
        # Template for Ubuntu/Debian servers
        ubuntu_template = ConfigurationTemplate(
            name="ubuntu_optimization",
            description="Ubuntu/Debian server optimization and NeuralSync preparation",
            target_os="ubuntu",
            commands=[
                "apt-get update -y",
                "apt-get upgrade -y",
                "systemctl disable hibernate.target",
                "systemctl disable sleep.target",
                "systemctl disable suspend.target", 
                "systemctl disable hybrid-sleep.target",
                "echo 'vm.swappiness=10' >> /etc/sysctl.conf",
                "echo 'vm.vfs_cache_pressure=50' >> /etc/sysctl.conf",
                "sysctl -p",
                "apt-get install -y htop iotop nethogs python3 python3-pip git",
                "pip3 install psutil aiofiles",
                "systemctl enable ssh",
                "ufw allow ssh"
            ],
            required_packages=["python3", "python3-pip", "git", "htop"],
            configuration_files={
                "/etc/systemd/logind.conf": "[Login]\nHandleLidSwitch=ignore\nHandleLidSwitchDocked=ignore\n",
                "/etc/crontab": "0 */6 * * * root sync && echo 3 > /proc/sys/vm/drop_caches\n"
            },
            validation_checks=[
                "systemctl is-enabled ssh",
                "python3 --version",
                "pip3 --version"
            ],
            rollback_commands=[
                "systemctl enable hibernate.target",
                "systemctl enable sleep.target"
            ]
        )
        
        # Template for CentOS/RHEL servers
        centos_template = ConfigurationTemplate(
            name="centos_optimization",
            description="CentOS/RHEL server optimization and NeuralSync preparation",
            target_os="centos",
            commands=[
                "yum update -y",
                "systemctl disable hibernate.target",
                "systemctl disable sleep.target",
                "systemctl disable suspend.target",
                "echo 'vm.swappiness=10' >> /etc/sysctl.conf",
                "sysctl -p",
                "yum install -y htop iotop python3 python3-pip git",
                "pip3 install psutil aiofiles",
                "systemctl enable sshd",
                "firewall-cmd --permanent --add-service=ssh",
                "firewall-cmd --reload"
            ],
            required_packages=["python3", "python3-pip", "git", "htop"],
            configuration_files={
                "/etc/systemd/logind.conf": "[Login]\nHandleLidSwitch=ignore\n"
            },
            validation_checks=[
                "systemctl is-enabled sshd",
                "python3 --version"
            ],
            rollback_commands=[
                "systemctl enable hibernate.target"
            ]
        )
        
        # Save templates
        self._save_template(ubuntu_template)
        self._save_template(centos_template)
    
    def _save_template(self, template: ConfigurationTemplate):
        """Save configuration template to disk"""
        template_file = self.templates_dir / f"{template.name}.json"
        with open(template_file, 'w') as f:
            json.dump(asdict(template), f, indent=2)
        
        template_file.chmod(0o600)
    
    async def apply_configuration(self, server: MiniCloudServer, template_name: str) -> bool:
        """Apply configuration template to server"""
        try:
            # Load template
            template_file = self.templates_dir / f"{template_name}.json"
            if not template_file.exists():
                logger.error(f"Configuration template {template_name} not found")
                return False
            
            with open(template_file) as f:
                template_data = json.load(f)
            
            template = ConfigurationTemplate(**template_data)
            
            logger.info(f"Applying configuration {template.name} to {server.name}")
            
            # Backup current configuration
            backup_success = await self._create_configuration_backup(server)
            if not backup_success:
                logger.warning(f"Failed to create backup for {server.name}")
            
            # Execute configuration commands
            for command in template.commands:
                exit_code, stdout, stderr = await self.ssh_manager.execute_command(server, command)
                
                if exit_code != 0:
                    logger.error(f"Configuration command failed on {server.name}: {command}")
                    logger.error(f"Error: {stderr}")
                    
                    # Attempt rollback
                    await self._rollback_configuration(server, template)
                    return False
                
                logger.debug(f"Configuration command succeeded: {command}")
            
            # Apply configuration files
            for remote_path, content in template.configuration_files.items():
                temp_local_file = self.config_dir / "temp_config_file"
                with open(temp_local_file, 'w') as f:
                    f.write(content)
                
                upload_success = await self.ssh_manager.upload_file(
                    server, str(temp_local_file), remote_path
                )
                
                temp_local_file.unlink(missing_ok=True)
                
                if not upload_success:
                    logger.error(f"Failed to upload config file {remote_path} to {server.name}")
                    await self._rollback_configuration(server, template)
                    return False
            
            # Validate configuration
            validation_success = await self._validate_configuration(server, template)
            if not validation_success:
                logger.error(f"Configuration validation failed for {server.name}")
                await self._rollback_configuration(server, template)
                return False
            
            logger.info(f"Configuration {template.name} successfully applied to {server.name}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration application failed for {server.name}: {e}")
            return False
    
    async def _create_configuration_backup(self, server: MiniCloudServer) -> bool:
        """Create backup of current server configuration"""
        try:
            backup_dir = f"/tmp/neuralsync_backup_{int(time.time())}"
            backup_commands = [
                f"mkdir -p {backup_dir}",
                f"cp /etc/systemd/logind.conf {backup_dir}/ 2>/dev/null || true",
                f"cp /etc/sysctl.conf {backup_dir}/ 2>/dev/null || true",
                f"systemctl list-unit-files --state=enabled > {backup_dir}/enabled_services.txt"
            ]
            
            for command in backup_commands:
                exit_code, stdout, stderr = await self.ssh_manager.execute_command(server, command)
                if exit_code != 0:
                    logger.debug(f"Backup command warning: {command} - {stderr}")
            
            server.last_backup = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed for {server.name}: {e}")
            return False
    
    async def _validate_configuration(self, server: MiniCloudServer, template: ConfigurationTemplate) -> bool:
        """Validate applied configuration"""
        try:
            for check_command in template.validation_checks:
                exit_code, stdout, stderr = await self.ssh_manager.execute_command(server, check_command)
                
                if exit_code != 0:
                    logger.error(f"Validation check failed: {check_command}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    async def _rollback_configuration(self, server: MiniCloudServer, template: ConfigurationTemplate) -> bool:
        """Rollback configuration changes"""
        try:
            logger.info(f"Rolling back configuration for {server.name}")
            
            for command in template.rollback_commands:
                exit_code, stdout, stderr = await self.ssh_manager.execute_command(server, command)
                if exit_code != 0:
                    logger.warning(f"Rollback command failed: {command} - {stderr}")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration rollback failed: {e}")
            return False

class MiniCloudManager:
    """Main MiniCloud management system"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.servers_config_file = config_dir / "minicloud_servers.json"
        
        # Initialize components
        self.ssh_manager = MiniCloudSSHManager(config_dir)
        self.health_monitor = MiniCloudHealthMonitor(self.ssh_manager)
        self.config_manager = MiniCloudConfigManager(self.ssh_manager, config_dir)
        
        # Server registry
        self.servers: Dict[str, MiniCloudServer] = {}
        self.monitoring_active = False
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
    async def load_servers_config(self):
        """Load server configurations from file"""
        if not self.servers_config_file.exists():
            return
        
        try:
            with open(self.servers_config_file) as f:
                servers_data = json.load(f)
            
            for server_id, server_data in servers_data.items():
                # Convert state enum
                if 'state' in server_data:
                    server_data['state'] = ServerState(server_data['state'])
                
                server = MiniCloudServer(**server_data)
                self.servers[server_id] = server
            
            logger.info(f"Loaded {len(self.servers)} MiniCloud servers from configuration")
            
        except Exception as e:
            logger.error(f"Failed to load servers configuration: {e}")
    
    async def save_servers_config(self):
        """Save server configurations to file"""
        try:
            servers_data = {}
            for server_id, server in self.servers.items():
                server_dict = asdict(server)
                # Remove SSH connection object
                server_dict.pop('ssh_connection', None)
                # Convert enum to string
                server_dict['state'] = server.state.value
                servers_data[server_id] = server_dict
            
            with open(self.servers_config_file, 'w') as f:
                json.dump(servers_data, f, indent=2, default=str)
            
            self.servers_config_file.chmod(0o600)
            
        except Exception as e:
            logger.error(f"Failed to save servers configuration: {e}")
    
    async def add_server(self, server: MiniCloudServer) -> bool:
        """Add new MiniCloud server to management"""
        try:
            # Test connection
            connection_success = await self.ssh_manager.create_ssh_connection(server)
            if not connection_success:
                logger.error(f"Failed to connect to server {server.name}")
                return False
            
            # Collect initial metrics
            metrics = await self.health_monitor.collect_server_metrics(server)
            if metrics:
                health = await self.health_monitor.assess_server_health(server, metrics)
                logger.info(f"Server {server.name} health: {health['overall']}")
            
            # Add to registry
            self.servers[server.server_id] = server
            await self.save_servers_config()
            
            # Start monitoring
            if self.monitoring_active:
                await self._start_server_monitoring(server)
            
            logger.info(f"Added MiniCloud server: {server.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add server {server.name}: {e}")
            return False
    
    async def start_monitoring(self):
        """Start monitoring all servers"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting MiniCloud monitoring")
        
        # Start monitoring for all servers
        for server in self.servers.values():
            await self._start_server_monitoring(server)
    
    async def _start_server_monitoring(self, server: MiniCloudServer):
        """Start monitoring for specific server"""
        if server.server_id in self.monitoring_tasks:
            return
        
        task = asyncio.create_task(self._monitor_server_loop(server))
        self.monitoring_tasks[server.server_id] = task
        
        logger.debug(f"Started monitoring for {server.name}")
    
    async def _monitor_server_loop(self, server: MiniCloudServer):
        """Monitoring loop for individual server"""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = await self.health_monitor.collect_server_metrics(server)
                
                if metrics:
                    # Assess health
                    health = await self.health_monitor.assess_server_health(server, metrics)
                    
                    # Log health issues
                    if health['issues']:
                        logger.warning(f"Server {server.name} health issues: {health['issues']}")
                    
                    # Trigger automated responses if needed
                    await self._handle_health_issues(server, health)
                else:
                    logger.warning(f"Failed to collect metrics from {server.name}")
                    server.state = ServerState.ERROR
                
                await asyncio.sleep(self.health_monitor.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error for {server.name}: {e}")
                await asyncio.sleep(60)
    
    async def _handle_health_issues(self, server: MiniCloudServer, health: Dict[str, Any]):
        """Handle server health issues with automated responses"""
        if health['overall'] == 'critical':
            # Critical issues - take immediate action
            for issue in health['issues']:
                if 'memory' in issue.lower():
                    # High memory usage - restart memory-intensive services
                    logger.info(f"Attempting memory cleanup on {server.name}")
                    await self.ssh_manager.execute_command(server, "sync && echo 3 > /proc/sys/vm/drop_caches")
                
                elif 'disk' in issue.lower():
                    # High disk usage - cleanup temp files
                    logger.info(f"Attempting disk cleanup on {server.name}")
                    cleanup_commands = [
                        "apt-get clean 2>/dev/null || yum clean all 2>/dev/null || true",
                        "rm -rf /tmp/* 2>/dev/null || true",
                        "journalctl --vacuum-time=7d 2>/dev/null || true"
                    ]
                    for cmd in cleanup_commands:
                        await self.ssh_manager.execute_command(server, cmd)
                
                elif 'temperature' in issue.lower():
                    # High temperature - reduce CPU frequency if possible
                    logger.info(f"Attempting temperature control on {server.name}")
                    await self.ssh_manager.execute_command(server, "cpupower frequency-set -g powersave 2>/dev/null || true")
        
        # Save updated metrics
        await self.save_servers_config()
    
    async def apply_configuration_to_server(self, server_id: str, template_name: str) -> bool:
        """Apply configuration template to specific server"""
        server = self.servers.get(server_id)
        if not server:
            logger.error(f"Server {server_id} not found")
            return False
        
        return await self.config_manager.apply_configuration(server, template_name)
    
    async def get_server_status(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status for specific server"""
        server = self.servers.get(server_id)
        if not server:
            return None
        
        # Collect current metrics
        metrics = await self.health_monitor.collect_server_metrics(server)
        health = None
        
        if metrics:
            health = await self.health_monitor.assess_server_health(server, metrics)
        
        status = {
            'server_info': {
                'id': server.server_id,
                'name': server.name,
                'hostname': server.hostname,
                'ip_address': server.ip_address,
                'state': server.state.value,
                'last_seen': server.last_seen
            },
            'current_metrics': asdict(metrics) if metrics else None,
            'health_assessment': health,
            'configuration': {
                'auto_reboot_enabled': server.auto_reboot_enabled,
                'hibernation_disabled': server.hibernation_disabled,
                'memory_monitoring_enabled': server.memory_monitoring_enabled,
                'encryption_unlocked': server.encryption_unlocked
            },
            'services': {
                'running_services': server.running_services or [],
                'neuralsync_installed': server.neuralsync_installed,
                'neuralsync_version': server.neuralsync_version
            }
        }
        
        return status
    
    async def get_all_servers_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all managed servers"""
        all_status = {}
        
        for server_id in self.servers.keys():
            status = await self.get_server_status(server_id)
            if status:
                all_status[server_id] = status
        
        return all_status
    
    async def stop_monitoring(self):
        """Stop all server monitoring"""
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        
        self.monitoring_tasks.clear()
        
        # Close all SSH connections
        for server in self.servers.values():
            self.ssh_manager.close_connection(server)
        
        logger.info("MiniCloud monitoring stopped")

# Utility function for testing MiniCloud integration
async def test_minicloud_integration(hostname: str, ssh_user: str, ssh_password: str = None) -> bool:
    """Test MiniCloud integration with a server"""
    try:
        config_dir = Path.home() / '.neuralsync'
        manager = MiniCloudManager(config_dir)
        
        # Create test server
        test_server = MiniCloudServer(
            server_id=f"test_{int(time.time())}",
            name=f"Test Server {hostname}",
            hostname=hostname,
            ip_address=hostname,
            ssh_user=ssh_user,
            ssh_password=ssh_password
        )
        
        # Test connection and metrics
        success = await manager.add_server(test_server)
        if success:
            status = await manager.get_server_status(test_server.server_id)
            print(f"Server connection successful")
            print(f"Status: {status['server_info']['state']}")
            if status['current_metrics']:
                metrics = status['current_metrics']
                print(f"CPU: {metrics['cpu_percent']:.1f}%")
                print(f"Memory: {metrics['memory_percent']:.1f}%")
                print(f"Disk: {metrics['disk_usage_percent']:.1f}%")
            
            # Clean up test server
            manager.servers.pop(test_server.server_id, None)
        
        return success
        
    except Exception as e:
        print(f"MiniCloud integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Integration testing
    import sys
    if len(sys.argv) >= 3:
        hostname = sys.argv[1]
        ssh_user = sys.argv[2]
        ssh_password = sys.argv[3] if len(sys.argv) > 3 else None
        asyncio.run(test_minicloud_integration(hostname, ssh_user, ssh_password))
    else:
        print("Usage: python minicloud_manager.py <hostname> <ssh_user> [ssh_password]")