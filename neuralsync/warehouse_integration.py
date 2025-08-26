#!/usr/bin/env python3
"""
Cross-System Integration Module for NeuralSync2 and MCU Tool Warehouse
Provides seamless bidirectional integration with automatic discovery and health monitoring
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import grpc
from grpc import aio as aiogrpc
import socket
from urllib.parse import urlparse
import subprocess
import hashlib
import uuid

logger = logging.getLogger(__name__)

class WarehouseState(Enum):
    """Warehouse connection states"""
    UNKNOWN = "unknown"
    DISCOVERING = "discovering"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISCONNECTED = "disconnected"

@dataclass
class WarehouseEndpoint:
    """MCU Tool Warehouse endpoint information"""
    warehouse_id: str
    name: str
    grpc_endpoint: str
    rest_endpoint: str
    metrics_endpoint: str
    health_endpoint: str
    capabilities: List[str]
    version: str
    installation_path: str
    last_seen: float
    state: WarehouseState = WarehouseState.UNKNOWN
    auth_token: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class IntegrationMetrics:
    """Integration performance metrics"""
    discovery_time_ms: float
    connection_time_ms: float
    authentication_time_ms: float
    health_check_time_ms: float
    request_count: int
    error_count: int
    last_error: Optional[str] = None
    uptime_percentage: float = 100.0

class WarehouseDiscovery:
    """Automatic warehouse discovery and registration"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.discovery_interval = 300  # 5 minutes
        self.discovery_timeout = 10    # 10 seconds
        self.known_warehouses: Dict[str, WarehouseEndpoint] = {}
        self.discovery_running = False
        
        # Common warehouse discovery patterns
        self.discovery_patterns = [
            # Local installation patterns
            Path.home() / ".local/share/neuralsync_warehouse",
            Path.home() / ".neuralsync_warehouse", 
            Path("/opt/neuralsync_warehouse"),
            Path("/usr/local/share/neuralsync_warehouse"),
            
            # Service discovery endpoints
            "localhost:50051",  # Default gRPC
            "localhost:8080",   # Default REST
            "127.0.0.1:50051",
            "127.0.0.1:8080",
        ]
        
        # Network discovery ranges for enterprise deployments
        self.network_ranges = [
            "192.168.1.0/24",
            "10.0.0.0/8", 
            "172.16.0.0/12"
        ]
        
    async def start_discovery(self):
        """Start continuous warehouse discovery"""
        if self.discovery_running:
            logger.debug("Warehouse discovery already running")
            return
            
        self.discovery_running = True
        logger.info("Starting warehouse discovery service")
        
        # Initial discovery
        await self.discover_warehouses()
        
        # Continuous discovery loop
        asyncio.create_task(self._discovery_loop())
        
    async def _discovery_loop(self):
        """Continuous warehouse discovery loop"""
        while self.discovery_running:
            try:
                await asyncio.sleep(self.discovery_interval)
                await self.discover_warehouses()
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(60)  # Back off on errors
                
    async def discover_warehouses(self) -> List[WarehouseEndpoint]:
        """Discover MCU Tool Warehouses on local system and network"""
        discovered = []
        discovery_start = time.time()
        
        logger.debug("Starting warehouse discovery scan")
        
        # Local file-system discovery
        local_warehouses = await self._discover_local_warehouses()
        discovered.extend(local_warehouses)
        
        # Network service discovery
        network_warehouses = await self._discover_network_warehouses()
        discovered.extend(network_warehouses)
        
        # Process discovery results
        for warehouse in discovered:
            await self._register_discovered_warehouse(warehouse)
        
        discovery_time = (time.time() - discovery_start) * 1000
        logger.info(f"Discovered {len(discovered)} warehouses in {discovery_time:.2f}ms")
        
        return discovered
    
    async def _discover_local_warehouses(self) -> List[WarehouseEndpoint]:
        """Discover warehouses through local filesystem"""
        discovered = []
        
        for pattern in self.discovery_patterns:
            if isinstance(pattern, Path):
                # File-based discovery
                if pattern.exists():
                    manifest_file = pattern / "integration_manifest.json"
                    if manifest_file.exists():
                        try:
                            with open(manifest_file) as f:
                                manifest = json.load(f)
                            
                            warehouse = await self._create_warehouse_from_manifest(
                                manifest, str(pattern)
                            )
                            if warehouse:
                                discovered.append(warehouse)
                                
                        except Exception as e:
                            logger.debug(f"Failed to read manifest from {pattern}: {e}")
            
            elif isinstance(pattern, str):
                # Network endpoint discovery
                warehouse = await self._discover_network_endpoint(pattern)
                if warehouse:
                    discovered.append(warehouse)
        
        return discovered
    
    async def _discover_network_warehouses(self) -> List[WarehouseEndpoint]:
        """Discover warehouses on network through service discovery"""
        discovered = []
        
        # mDNS/Bonjour discovery for local network
        try:
            import zeroconf
            from zeroconf import ServiceBrowser, Zeroconf
            
            zc = Zeroconf()
            services = []
            
            class WarehouseListener:
                def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                    info = zc.get_service_info(type_, name)
                    if info and b'neuralsync' in info.properties.get(b'service', b''):
                        services.append(info)
                
                def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                    pass
                
                def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                    pass
            
            listener = WarehouseListener()
            browser = ServiceBrowser(zc, "_http._tcp.local.", listener)
            
            # Wait for discovery
            await asyncio.sleep(2)
            browser.cancel()
            zc.close()
            
            # Process discovered services
            for service_info in services:
                warehouse = await self._create_warehouse_from_mdns(service_info)
                if warehouse:
                    discovered.append(warehouse)
                    
        except ImportError:
            logger.debug("zeroconf not available for mDNS discovery")
        except Exception as e:
            logger.debug(f"mDNS discovery failed: {e}")
        
        return discovered
    
    async def _discover_network_endpoint(self, endpoint: str) -> Optional[WarehouseEndpoint]:
        """Discover warehouse at specific network endpoint"""
        try:
            host, port = endpoint.split(':')
            port = int(port)
            
            # Try to connect and get health info
            timeout = aiohttp.ClientTimeout(total=self.discovery_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                
                # Try REST endpoint first
                health_url = f"http://{endpoint}/health"
                try:
                    async with session.get(health_url) as resp:
                        if resp.status == 200:
                            health_data = await resp.json()
                            if health_data.get('service') == 'neuralsync_warehouse':
                                return await self._create_warehouse_from_health(
                                    endpoint, health_data
                                )
                except Exception:
                    pass
                
                # Try gRPC health check
                try:
                    channel = aiogrpc.insecure_channel(endpoint)
                    # Simple connection test
                    await asyncio.wait_for(channel.channel_ready(), timeout=2)
                    await channel.close()
                    
                    # Create basic warehouse info for gRPC-only services
                    return WarehouseEndpoint(
                        warehouse_id=f"grpc_{hashlib.md5(endpoint.encode()).hexdigest()[:8]}",
                        name=f"Warehouse at {endpoint}",
                        grpc_endpoint=f"{endpoint}",
                        rest_endpoint=f"http://{host}:{port+1}/api/v1" if port == 50051 else f"http://{endpoint}/api/v1",
                        metrics_endpoint=f"http://{host}:{port+2}/metrics" if port == 50051 else f"http://{endpoint}/metrics",
                        health_endpoint=f"http://{host}:{port+1}/health" if port == 50051 else f"http://{endpoint}/health",
                        capabilities=['grpc', 'tool_execution'],
                        version="unknown",
                        installation_path="network",
                        last_seen=time.time(),
                        state=WarehouseState.DISCOVERED,
                        metadata={'discovery_method': 'network_grpc'}
                    )
                    
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug(f"Failed to discover endpoint {endpoint}: {e}")
            
        return None
    
    async def _create_warehouse_from_manifest(
        self, manifest: Dict[str, Any], path: str
    ) -> Optional[WarehouseEndpoint]:
        """Create warehouse endpoint from integration manifest"""
        try:
            endpoints = manifest.get('endpoints', {})
            return WarehouseEndpoint(
                warehouse_id=manifest.get('installation_id', str(uuid.uuid4())),
                name=f"MCU Warehouse at {path}",
                grpc_endpoint=endpoints.get('grpc', 'localhost:50051'),
                rest_endpoint=endpoints.get('rest', 'http://localhost:8080/api/v1'),
                metrics_endpoint=endpoints.get('metrics', 'http://localhost:9090/metrics'),
                health_endpoint=endpoints.get('rest', 'http://localhost:8080') + '/health',
                capabilities=manifest.get('capabilities', []),
                version=manifest.get('warehouse_version', '1.0.0'),
                installation_path=path,
                last_seen=time.time(),
                state=WarehouseState.DISCOVERED,
                metadata={'discovery_method': 'manifest', 'manifest': manifest}
            )
        except Exception as e:
            logger.debug(f"Failed to create warehouse from manifest: {e}")
            return None
    
    async def _create_warehouse_from_health(
        self, endpoint: str, health_data: Dict[str, Any]
    ) -> Optional[WarehouseEndpoint]:
        """Create warehouse endpoint from health check response"""
        try:
            host, port = endpoint.split(':')
            base_url = f"http://{endpoint}"
            
            return WarehouseEndpoint(
                warehouse_id=health_data.get('instance_id', str(uuid.uuid4())),
                name=health_data.get('name', f"Warehouse at {endpoint}"),
                grpc_endpoint=health_data.get('endpoints', {}).get('grpc', f"{host}:50051"),
                rest_endpoint=health_data.get('endpoints', {}).get('rest', f"{base_url}/api/v1"),
                metrics_endpoint=health_data.get('endpoints', {}).get('metrics', f"{base_url}/metrics"),
                health_endpoint=f"{base_url}/health",
                capabilities=health_data.get('capabilities', []),
                version=health_data.get('version', '1.0.0'),
                installation_path="network",
                last_seen=time.time(),
                state=WarehouseState.DISCOVERED,
                metadata={'discovery_method': 'health_endpoint', 'health_data': health_data}
            )
        except Exception as e:
            logger.debug(f"Failed to create warehouse from health data: {e}")
            return None
    
    async def _register_discovered_warehouse(self, warehouse: WarehouseEndpoint):
        """Register a discovered warehouse"""
        existing = self.known_warehouses.get(warehouse.warehouse_id)
        
        if existing:
            # Update existing warehouse info
            existing.last_seen = time.time()
            existing.state = warehouse.state
            existing.capabilities = warehouse.capabilities
            existing.version = warehouse.version
            logger.debug(f"Updated warehouse {warehouse.name}")
        else:
            # New warehouse discovered
            self.known_warehouses[warehouse.warehouse_id] = warehouse
            logger.info(f"Registered new warehouse: {warehouse.name}")
            
            # Save discovery cache
            await self._save_discovery_cache()
    
    async def _save_discovery_cache(self):
        """Save discovered warehouses to cache"""
        cache_file = self.config_dir / "warehouse_discovery_cache.json"
        
        cache_data = {
            'last_updated': time.time(),
            'warehouses': {
                wid: asdict(warehouse) for wid, warehouse in self.known_warehouses.items()
            }
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        
        cache_file.chmod(0o600)
    
    async def load_discovery_cache(self):
        """Load cached warehouse discoveries"""
        cache_file = self.config_dir / "warehouse_discovery_cache.json"
        
        if not cache_file.exists():
            return
            
        try:
            with open(cache_file) as f:
                cache_data = json.load(f)
            
            # Load warehouses from cache
            for wid, warehouse_data in cache_data.get('warehouses', {}).items():
                # Convert back to enum
                warehouse_data['state'] = WarehouseState(warehouse_data['state'])
                warehouse = WarehouseEndpoint(**warehouse_data)
                self.known_warehouses[wid] = warehouse
                
            logger.info(f"Loaded {len(self.known_warehouses)} warehouses from cache")
            
        except Exception as e:
            logger.error(f"Failed to load discovery cache: {e}")

class WarehouseClient:
    """Client for communicating with MCU Tool Warehouses"""
    
    def __init__(self, endpoint: WarehouseEndpoint):
        self.endpoint = endpoint
        self.grpc_channel: Optional[aiogrpc.Channel] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.metrics = IntegrationMetrics(
            discovery_time_ms=0,
            connection_time_ms=0, 
            authentication_time_ms=0,
            health_check_time_ms=0,
            request_count=0,
            error_count=0
        )
        
    async def connect(self) -> bool:
        """Connect to the warehouse"""
        connect_start = time.time()
        
        try:
            # Setup gRPC connection
            self.grpc_channel = aiogrpc.insecure_channel(self.endpoint.grpc_endpoint)
            await asyncio.wait_for(self.grpc_channel.channel_ready(), timeout=5)
            
            # Setup HTTP session
            timeout = aiohttp.ClientTimeout(total=30)
            self.http_session = aiohttp.ClientSession(
                timeout=timeout,
                headers={'User-Agent': 'NeuralSync2-Integration/1.0'}
            )
            
            self.endpoint.state = WarehouseState.CONNECTED
            self.metrics.connection_time_ms = (time.time() - connect_start) * 1000
            
            logger.info(f"Connected to warehouse {self.endpoint.name}")
            return True
            
        except Exception as e:
            self.endpoint.state = WarehouseState.FAILED
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Failed to connect to warehouse {self.endpoint.name}: {e}")
            return False
    
    async def authenticate(self) -> bool:
        """Authenticate with the warehouse"""
        auth_start = time.time()
        
        try:
            if not self.http_session:
                return False
                
            # Try to get auth token
            auth_payload = {
                'client_id': 'neuralsync2',
                'client_type': 'daemon',
                'capabilities': ['tool_execution', 'health_monitoring', 'service_discovery']
            }
            
            async with self.http_session.post(
                f"{self.endpoint.rest_endpoint}/auth/token", 
                json=auth_payload
            ) as resp:
                if resp.status == 200:
                    auth_data = await resp.json()
                    self.endpoint.auth_token = auth_data.get('access_token')
                    self.endpoint.state = WarehouseState.AUTHENTICATED
                    
                    # Update session headers
                    self.http_session.headers.update({
                        'Authorization': f"Bearer {self.endpoint.auth_token}"
                    })
                    
                    self.metrics.authentication_time_ms = (time.time() - auth_start) * 1000
                    logger.debug(f"Authenticated with warehouse {self.endpoint.name}")
                    return True
                else:
                    logger.debug(f"Auth failed for {self.endpoint.name}: {resp.status}")
                    
        except Exception as e:
            logger.debug(f"Authentication error for {self.endpoint.name}: {e}")
            
        # Fallback - continue without auth for open warehouses
        self.endpoint.state = WarehouseState.CONNECTED
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on warehouse"""
        health_start = time.time()
        
        try:
            if not self.http_session:
                return {'status': 'error', 'message': 'No connection'}
                
            async with self.http_session.get(self.endpoint.health_endpoint) as resp:
                health_data = await resp.json() if resp.content_type == 'application/json' else {}
                
                self.metrics.health_check_time_ms = (time.time() - health_start) * 1000
                self.metrics.request_count += 1
                
                if resp.status == 200:
                    self.endpoint.state = WarehouseState.HEALTHY
                    return {'status': 'healthy', 'data': health_data}
                else:
                    self.endpoint.state = WarehouseState.DEGRADED
                    return {'status': 'degraded', 'code': resp.status, 'data': health_data}
                    
        except Exception as e:
            self.endpoint.state = WarehouseState.FAILED
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            return {'status': 'failed', 'error': str(e)}
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from warehouse"""
        try:
            if not self.http_session:
                return []
                
            async with self.http_session.get(f"{self.endpoint.rest_endpoint}/tools") as resp:
                if resp.status == 200:
                    tools_data = await resp.json()
                    self.metrics.request_count += 1
                    return tools_data.get('tools', [])
                    
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Failed to get tools from {self.endpoint.name}: {e}")
            
        return []
    
    async def register_neuralsync_daemon(self, daemon_info: Dict[str, Any]) -> bool:
        """Register NeuralSync daemon with warehouse"""
        try:
            if not self.http_session:
                return False
                
            registration_payload = {
                'service_type': 'neuralsync_daemon',
                'service_id': daemon_info.get('daemon_id'),
                'name': 'NeuralSync2 Daemon',
                'version': daemon_info.get('version', '2.0.0'),
                'endpoints': daemon_info.get('endpoints', {}),
                'capabilities': daemon_info.get('capabilities', []),
                'metadata': daemon_info.get('metadata', {})
            }
            
            async with self.http_session.post(
                f"{self.endpoint.rest_endpoint}/services/register",
                json=registration_payload
            ) as resp:
                if resp.status in [200, 201]:
                    self.metrics.request_count += 1
                    logger.info(f"Registered NeuralSync daemon with {self.endpoint.name}")
                    return True
                else:
                    logger.warning(f"Failed to register daemon with {self.endpoint.name}: {resp.status}")
                    
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Daemon registration failed for {self.endpoint.name}: {e}")
            
        return False
    
    async def close(self):
        """Close connections to warehouse"""
        if self.grpc_channel:
            await self.grpc_channel.close()
            self.grpc_channel = None
            
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
            
        self.endpoint.state = WarehouseState.DISCONNECTED

class CrossSystemIntegration:
    """Main cross-system integration manager"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.discovery = WarehouseDiscovery(config_dir)
        self.warehouse_clients: Dict[str, WarehouseClient] = {}
        self.integration_running = False
        self.health_check_interval = 60  # 1 minute
        
        # NeuralSync daemon info for registration
        self.daemon_info = {
            'daemon_id': f"neuralsync2_{uuid.uuid4().hex[:8]}",
            'version': '2.0.0',
            'endpoints': {
                'api': 'http://localhost:7950/api/v1',
                'websocket': 'ws://localhost:7950/ws',
                'health': 'http://localhost:7950/health'
            },
            'capabilities': [
                'memory_sync',
                'personality_management', 
                'cli_integration',
                'cross_tool_communication',
                'distributed_storage'
            ],
            'metadata': {
                'installation_type': 'enhanced',
                'features': ['crdt_sync', 'nas_integration', 'unleashed_mode']
            }
        }
    
    async def start_integration(self):
        """Start cross-system integration"""
        if self.integration_running:
            return
            
        self.integration_running = True
        logger.info("Starting cross-system integration")
        
        # Load cached discoveries
        await self.discovery.load_discovery_cache()
        
        # Start warehouse discovery
        await self.discovery.start_discovery()
        
        # Connect to discovered warehouses
        await self._connect_to_warehouses()
        
        # Start health monitoring
        asyncio.create_task(self._health_monitoring_loop())
        
        logger.info("Cross-system integration started successfully")
    
    async def _connect_to_warehouses(self):
        """Connect to all discovered warehouses"""
        for warehouse_id, endpoint in self.discovery.known_warehouses.items():
            try:
                client = WarehouseClient(endpoint)
                
                # Connect and authenticate
                if await client.connect():
                    if await client.authenticate():
                        # Register NeuralSync daemon
                        await client.register_neuralsync_daemon(self.daemon_info)
                        
                        self.warehouse_clients[warehouse_id] = client
                        logger.info(f"Successfully integrated with warehouse: {endpoint.name}")
                    else:
                        await client.close()
                        logger.warning(f"Authentication failed for warehouse: {endpoint.name}")
                else:
                    logger.warning(f"Connection failed for warehouse: {endpoint.name}")
                    
            except Exception as e:
                logger.error(f"Failed to integrate with warehouse {endpoint.name}: {e}")
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring of warehouse connections"""
        while self.integration_running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_warehouse_health()
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_warehouse_health(self):
        """Check health of all warehouse connections"""
        for warehouse_id, client in list(self.warehouse_clients.items()):
            try:
                health_result = await client.health_check()
                
                if health_result['status'] == 'failed':
                    # Try to reconnect
                    logger.info(f"Attempting to reconnect to warehouse {client.endpoint.name}")
                    await client.close()
                    
                    if await client.connect() and await client.authenticate():
                        await client.register_neuralsync_daemon(self.daemon_info)
                        logger.info(f"Successfully reconnected to {client.endpoint.name}")
                    else:
                        # Remove failed connection
                        del self.warehouse_clients[warehouse_id]
                        logger.warning(f"Removed failed warehouse connection: {client.endpoint.name}")
                        
            except Exception as e:
                logger.error(f"Health check failed for {client.endpoint.name}: {e}")
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        status = {
            'integration_running': self.integration_running,
            'discovered_warehouses': len(self.discovery.known_warehouses),
            'connected_warehouses': len(self.warehouse_clients),
            'daemon_info': self.daemon_info,
            'warehouses': {}
        }
        
        for warehouse_id, client in self.warehouse_clients.items():
            status['warehouses'][warehouse_id] = {
                'name': client.endpoint.name,
                'state': client.endpoint.state.value,
                'endpoints': {
                    'grpc': client.endpoint.grpc_endpoint,
                    'rest': client.endpoint.rest_endpoint,
                    'metrics': client.endpoint.metrics_endpoint
                },
                'capabilities': client.endpoint.capabilities,
                'version': client.endpoint.version,
                'metrics': asdict(client.metrics),
                'last_seen': client.endpoint.last_seen
            }
        
        return status
    
    async def get_available_tools_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get available tools from all connected warehouses"""
        all_tools = {}
        
        for warehouse_id, client in self.warehouse_clients.items():
            try:
                tools = await client.get_available_tools()
                all_tools[warehouse_id] = tools
            except Exception as e:
                logger.error(f"Failed to get tools from {client.endpoint.name}: {e}")
                all_tools[warehouse_id] = []
        
        return all_tools
    
    async def stop_integration(self):
        """Stop cross-system integration"""
        self.integration_running = False
        
        # Close all warehouse connections
        for client in self.warehouse_clients.values():
            await client.close()
        
        self.warehouse_clients.clear()
        logger.info("Cross-system integration stopped")

# Utility functions for integration testing
async def test_warehouse_integration() -> bool:
    """Test warehouse integration functionality"""
    try:
        config_dir = Path.home() / '.neuralsync'
        integration = CrossSystemIntegration(config_dir)
        
        # Test discovery
        warehouses = await integration.discovery.discover_warehouses()
        print(f"Discovered {len(warehouses)} warehouses")
        
        if warehouses:
            # Test connection to first warehouse
            client = WarehouseClient(warehouses[0])
            connected = await client.connect()
            
            if connected:
                authenticated = await client.authenticate()
                health = await client.health_check()
                tools = await client.get_available_tools()
                
                print(f"Connection: {connected}")
                print(f"Authentication: {authenticated}")
                print(f"Health: {health['status']}")
                print(f"Available tools: {len(tools)}")
                
                await client.close()
                return True
                
        return len(warehouses) > 0
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Integration testing
    asyncio.run(test_warehouse_integration())