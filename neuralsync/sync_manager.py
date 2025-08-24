#!/usr/bin/env python3
"""
Multi-Machine Sync Manager for NeuralSync2
Synchronizes core memory across local machines and CLI tools
"""

import asyncio
import json
import time
import hashlib
import socket
import ssl
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
import uuid
import aiofiles
import websockets
import sqlite3
from concurrent.futures import ThreadPoolExecutor

@dataclass
class SyncNode:
    """Represents a sync node (machine/CLI tool)"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    last_seen: float
    cli_tools: List[str]
    memory_version: int
    sync_enabled: bool = True

@dataclass 
class SyncOperation:
    """Represents a sync operation"""
    op_id: str
    node_id: str
    operation: str  # 'add', 'update', 'delete'
    item_id: str
    timestamp: float
    data: Dict[str, Any]
    lamport_clock: int

class MultiMachineSyncManager:
    """Manages memory synchronization across multiple machines"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or str(Path.home() / ".neuralsync" / "sync_config.json")
        self.node_id = self._generate_node_id()
        self.hostname = socket.gethostname()
        self.sync_port = 8374
        self.discovery_port = 8375
        
        self.known_nodes: Dict[str, SyncNode] = {}
        self.pending_operations: List[SyncOperation] = []
        self.lamport_clock = 0
        self.sync_enabled = True
        
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="sync")
        
        # WebSocket server for real-time sync
        self.sync_server = None
        self.discovery_server = None
        
        self._load_config()
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID based on machine characteristics"""
        # Use MAC address and hostname for consistent node ID
        try:
            import uuid as uuid_lib
            mac = uuid_lib.getnode()
            hostname = socket.gethostname()
            node_data = f"{mac}_{hostname}_{time.time()}"
            return hashlib.sha256(node_data.encode()).hexdigest()[:16]
        except Exception:
            return str(uuid.uuid4())[:16]
    
    def _load_config(self):
        """Load sync configuration"""
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                # Load known nodes
                for node_data in config.get('known_nodes', []):
                    node = SyncNode(**node_data)
                    self.known_nodes[node.node_id] = node
                    
                # Load settings
                self.sync_port = config.get('sync_port', 8374)
                self.discovery_port = config.get('discovery_port', 8375)
                self.sync_enabled = config.get('sync_enabled', True)
                self.lamport_clock = config.get('lamport_clock', 0)
                    
            except Exception as e:
                print(f"❌ Error loading sync config: {e}")
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default sync configuration"""
        config = {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'sync_port': self.sync_port,
            'discovery_port': self.discovery_port,
            'sync_enabled': True,
            'known_nodes': [],
            'lamport_clock': 0
        }
        
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    async def start_sync_services(self):
        """Start sync and discovery services"""
        if not self.sync_enabled:
            return
            
        try:
            # Start WebSocket sync server
            self.sync_server = await websockets.serve(
                self._handle_sync_connection,
                "0.0.0.0",
                self.sync_port
            )
            
            # Start UDP discovery server
            asyncio.create_task(self._start_discovery_server())
            
            # Start periodic tasks
            asyncio.create_task(self._periodic_sync())
            asyncio.create_task(self._periodic_discovery())
            asyncio.create_task(self._periodic_cleanup())
            
            print(f"🔄 NeuralSync multi-machine sync started on port {self.sync_port}")
            
        except Exception as e:
            print(f"❌ Failed to start sync services: {e}")
    
    async def _handle_sync_connection(self, websocket, path):
        """Handle incoming sync connections"""
        node_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type')
                
                if msg_type == 'handshake':
                    node_id = data.get('node_id')
                    await self._handle_handshake(websocket, data)
                elif msg_type == 'sync_operation':
                    await self._handle_sync_operation(data)
                elif msg_type == 'memory_request':
                    await self._handle_memory_request(websocket, data)
                elif msg_type == 'heartbeat':
                    await self._handle_heartbeat(data)
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"❌ Sync connection error: {e}")
        finally:
            if node_id and node_id in self.known_nodes:
                self.known_nodes[node_id].last_seen = time.time() - 3600  # Mark as offline
    
    async def _handle_handshake(self, websocket, data):
        """Handle handshake from remote node"""
        remote_node = SyncNode(
            node_id=data['node_id'],
            hostname=data['hostname'],
            ip_address=data['ip_address'],
            port=data['port'],
            last_seen=time.time(),
            cli_tools=data.get('cli_tools', []),
            memory_version=data.get('memory_version', 0),
            sync_enabled=True
        )
        
        self.known_nodes[remote_node.node_id] = remote_node
        
        # Send handshake response
        response = {
            'type': 'handshake_response',
            'node_id': self.node_id,
            'hostname': self.hostname,
            'ip_address': self._get_local_ip(),
            'port': self.sync_port,
            'cli_tools': await self._get_local_cli_tools(),
            'memory_version': await self._get_memory_version(),
            'lamport_clock': self.lamport_clock
        }
        
        await websocket.send(json.dumps(response))
        await self._save_config()
    
    async def _handle_sync_operation(self, data):
        """Handle incoming sync operation"""
        operation = SyncOperation(
            op_id=data['op_id'],
            node_id=data['node_id'],
            operation=data['operation'],
            item_id=data['item_id'],
            timestamp=data['timestamp'],
            data=data['data'],
            lamport_clock=data['lamport_clock']
        )
        
        # Update lamport clock
        self.lamport_clock = max(self.lamport_clock, operation.lamport_clock) + 1
        
        # Apply operation to local memory
        await self._apply_sync_operation(operation)
        
        # Broadcast to other nodes
        await self._broadcast_operation(operation)
    
    async def _apply_sync_operation(self, operation: SyncOperation):
        """Apply sync operation to local memory database"""
        try:
            memory_db_path = str(Path.home() / ".neuralsync" / "memory.db")
            
            from .storage import connect, upsert_item, delete_item
            con = connect(memory_db_path)
            
            if operation.operation == 'add' or operation.operation == 'update':
                upsert_item(con, operation.data)
            elif operation.operation == 'delete':
                delete_item(con, operation.item_id)
            
            con.close()
            
            print(f"🔄 Applied sync operation: {operation.operation} {operation.item_id}")
            
        except Exception as e:
            print(f"❌ Error applying sync operation: {e}")
    
    async def _broadcast_operation(self, operation: SyncOperation):
        """Broadcast operation to all known nodes"""
        message = {
            'type': 'sync_operation',
            'op_id': operation.op_id,
            'node_id': operation.node_id,
            'operation': operation.operation,
            'item_id': operation.item_id,
            'timestamp': operation.timestamp,
            'data': operation.data,
            'lamport_clock': operation.lamport_clock
        }
        
        for node in self.known_nodes.values():
            if node.sync_enabled and node.node_id != self.node_id:
                asyncio.create_task(self._send_to_node(node, message))
    
    async def _send_to_node(self, node: SyncNode, message: Dict[str, Any]):
        """Send message to specific node"""
        try:
            uri = f"ws://{node.ip_address}:{node.port}"
            
            async with websockets.connect(uri, timeout=10) as websocket:
                await websocket.send(json.dumps(message))
                
        except Exception as e:
            print(f"❌ Failed to send to node {node.hostname}: {e}")
    
    async def _start_discovery_server(self):
        """Start UDP discovery server"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind(('', self.discovery_port))
            
            while True:
                data, addr = sock.recvfrom(1024)
                asyncio.create_task(self._handle_discovery_message(data, addr))
                
        except Exception as e:
            print(f"❌ Discovery server error: {e}")
    
    async def _handle_discovery_message(self, data: bytes, addr: Tuple[str, int]):
        """Handle discovery message"""
        try:
            message = json.loads(data.decode())
            
            if message.get('type') == 'discovery_request':
                # Respond with our node info
                response = {
                    'type': 'discovery_response',
                    'node_id': self.node_id,
                    'hostname': self.hostname,
                    'ip_address': addr[0],
                    'sync_port': self.sync_port,
                    'cli_tools': await self._get_local_cli_tools(),
                    'memory_version': await self._get_memory_version()
                }
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(json.dumps(response).encode(), addr)
                sock.close()
                
            elif message.get('type') == 'discovery_response':
                # Add discovered node
                node = SyncNode(
                    node_id=message['node_id'],
                    hostname=message['hostname'],
                    ip_address=message['ip_address'],
                    port=message['sync_port'],
                    last_seen=time.time(),
                    cli_tools=message.get('cli_tools', []),
                    memory_version=message.get('memory_version', 0)
                )
                
                if node.node_id != self.node_id:
                    self.known_nodes[node.node_id] = node
                    await self._initiate_sync_with_node(node)
                    
        except Exception as e:
            print(f"❌ Discovery message error: {e}")
    
    async def _periodic_discovery(self):
        """Periodically broadcast discovery messages"""
        while True:
            try:
                await asyncio.sleep(30)  # Discovery every 30 seconds
                
                message = {
                    'type': 'discovery_request',
                    'node_id': self.node_id,
                    'hostname': self.hostname,
                    'sync_port': self.sync_port
                }
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.sendto(json.dumps(message).encode(), ('255.255.255.255', self.discovery_port))
                sock.close()
                
            except Exception as e:
                print(f"❌ Periodic discovery error: {e}")
                await asyncio.sleep(30)
    
    async def _periodic_sync(self):
        """Periodically sync with known nodes"""
        while True:
            try:
                await asyncio.sleep(60)  # Sync every minute
                
                for node in list(self.known_nodes.values()):
                    if node.sync_enabled and time.time() - node.last_seen < 300:  # 5 minute timeout
                        await self._sync_with_node(node)
                        
            except Exception as e:
                print(f"❌ Periodic sync error: {e}")
                await asyncio.sleep(60)
    
    async def _sync_with_node(self, node: SyncNode):
        """Perform incremental sync with specific node"""
        try:
            # Get our memory version
            local_version = await self._get_memory_version()
            
            if local_version > node.memory_version:
                # We have newer data, send updates
                await self._send_memory_updates(node, node.memory_version)
            elif node.memory_version > local_version:
                # They have newer data, request updates
                await self._request_memory_updates(node, local_version)
                
        except Exception as e:
            print(f"❌ Sync error with {node.hostname}: {e}")
    
    async def _send_memory_updates(self, node: SyncNode, since_version: int):
        """Send memory updates to node"""
        try:
            # Get items updated since version
            memory_db_path = str(Path.home() / ".neuralsync" / "memory.db")
            con = sqlite3.connect(memory_db_path)
            
            updates = []
            for row in con.execute('SELECT * FROM items WHERE lamport > ?', (since_version,)):
                # Convert row to dict
                cols = [desc[0] for desc in con.description]
                item = dict(zip(cols, row))
                updates.append(item)
            
            con.close()
            
            if updates:
                message = {
                    'type': 'memory_updates',
                    'updates': updates,
                    'from_version': since_version,
                    'to_version': await self._get_memory_version()
                }
                
                await self._send_to_node(node, message)
                
        except Exception as e:
            print(f"❌ Error sending updates to {node.hostname}: {e}")
    
    async def _request_memory_updates(self, node: SyncNode, our_version: int):
        """Request memory updates from node"""
        message = {
            'type': 'memory_request',
            'requesting_node': self.node_id,
            'our_version': our_version
        }
        
        await self._send_to_node(node, message)
    
    async def _handle_memory_request(self, websocket, data):
        """Handle request for memory updates"""
        requesting_version = data.get('our_version', 0)
        await self._send_memory_updates_via_websocket(websocket, requesting_version)
    
    async def _send_memory_updates_via_websocket(self, websocket, since_version: int):
        """Send memory updates via websocket"""
        try:
            memory_db_path = str(Path.home() / ".neuralsync" / "memory.db")
            con = sqlite3.connect(memory_db_path)
            
            updates = []
            for row in con.execute('SELECT * FROM items WHERE lamport > ?', (since_version,)):
                cols = [desc[0] for desc in con.description]
                item = dict(zip(cols, row))
                updates.append(item)
            
            con.close()
            
            message = {
                'type': 'memory_updates',
                'updates': updates,
                'from_version': since_version,
                'to_version': await self._get_memory_version()
            }
            
            await websocket.send(json.dumps(message))
            
        except Exception as e:
            print(f"❌ Error sending updates via websocket: {e}")
    
    async def _initiate_sync_with_node(self, node: SyncNode):
        """Initiate sync handshake with newly discovered node"""
        try:
            uri = f"ws://{node.ip_address}:{node.port}"
            
            async with websockets.connect(uri, timeout=10) as websocket:
                # Send handshake
                handshake = {
                    'type': 'handshake',
                    'node_id': self.node_id,
                    'hostname': self.hostname,
                    'ip_address': self._get_local_ip(),
                    'port': self.sync_port,
                    'cli_tools': await self._get_local_cli_tools(),
                    'memory_version': await self._get_memory_version()
                }
                
                await websocket.send(json.dumps(handshake))
                
                # Wait for response
                response = await websocket.recv()
                response_data = json.loads(response)
                
                if response_data.get('type') == 'handshake_response':
                    print(f"✅ Connected to node: {response_data['hostname']}")
                    
        except Exception as e:
            print(f"❌ Failed to initiate sync with {node.hostname}: {e}")
    
    async def _periodic_cleanup(self):
        """Clean up stale nodes and operations"""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                current_time = time.time()
                stale_nodes = []
                
                for node_id, node in self.known_nodes.items():
                    if current_time - node.last_seen > 1800:  # 30 minutes
                        stale_nodes.append(node_id)
                
                for node_id in stale_nodes:
                    del self.known_nodes[node_id]
                    print(f"🧹 Removed stale node: {self.known_nodes.get(node_id, {}).get('hostname', node_id)}")
                
                if stale_nodes:
                    await self._save_config()
                    
            except Exception as e:
                print(f"❌ Cleanup error: {e}")
                await asyncio.sleep(300)
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a dummy address to find local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    async def _get_local_cli_tools(self) -> List[str]:
        """Get list of local CLI tools"""
        # This would detect running CLI tools connected to NeuralSync
        return ["claude-code", "gemini", "codex-cli"]  # Simplified
    
    async def _get_memory_version(self) -> int:
        """Get current memory database version"""
        try:
            memory_db_path = str(Path.home() / ".neuralsync" / "memory.db")
            con = sqlite3.connect(memory_db_path)
            
            result = con.execute('SELECT MAX(lamport) FROM items').fetchone()
            version = result[0] if result and result[0] else 0
            
            con.close()
            return version
            
        except Exception:
            return 0
    
    async def _save_config(self):
        """Save configuration to disk"""
        config = {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'sync_port': self.sync_port,
            'discovery_port': self.discovery_port,
            'sync_enabled': self.sync_enabled,
            'lamport_clock': self.lamport_clock,
            'known_nodes': [asdict(node) for node in self.known_nodes.values()]
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    async def add_sync_operation(self, operation: str, item_id: str, data: Dict[str, Any]):
        """Add sync operation to be broadcast"""
        self.lamport_clock += 1
        
        sync_op = SyncOperation(
            op_id=str(uuid.uuid4()),
            node_id=self.node_id,
            operation=operation,
            item_id=item_id,
            timestamp=time.time(),
            data=data,
            lamport_clock=self.lamport_clock
        )
        
        await self._broadcast_operation(sync_op)
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """Get sync status and statistics"""
        return {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'sync_enabled': self.sync_enabled,
            'known_nodes': len(self.known_nodes),
            'active_nodes': len([n for n in self.known_nodes.values() if time.time() - n.last_seen < 300]),
            'memory_version': await self._get_memory_version(),
            'lamport_clock': self.lamport_clock,
            'nodes': [
                {
                    'hostname': node.hostname,
                    'ip_address': node.ip_address,
                    'last_seen': node.last_seen,
                    'cli_tools': node.cli_tools,
                    'memory_version': node.memory_version,
                    'online': time.time() - node.last_seen < 300
                }
                for node in self.known_nodes.values()
            ]
        }
    
    async def force_sync_with_all_nodes(self):
        """Force immediate sync with all known nodes"""
        for node in self.known_nodes.values():
            if node.sync_enabled:
                await self._sync_with_node(node)
    
    async def stop(self):
        """Stop sync services"""
        if self.sync_server:
            self.sync_server.close()
            await self.sync_server.wait_closed()
        
        self.executor.shutdown(wait=True)

# High-level API
class SyncAPI:
    """High-level sync API for CLI tools"""
    
    def __init__(self):
        self.manager = MultiMachineSyncManager()
    
    async def start(self):
        """Start sync services"""
        await self.manager.start_sync_services()
    
    async def sync_memory_item(self, operation: str, item_id: str, data: Dict[str, Any]):
        """Sync memory item across all nodes"""
        await self.manager.add_sync_operation(operation, item_id, data)
    
    async def get_sync_nodes(self) -> List[Dict[str, Any]]:
        """Get list of sync nodes"""
        status = await self.manager.get_sync_status()
        return status['nodes']
    
    async def force_full_sync(self):
        """Force full sync with all nodes"""
        await self.manager.force_sync_with_all_nodes()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get sync status"""
        return await self.manager.get_sync_status()

# Integration example
async def integrate_sync_with_neuralsync():
    """Example integration with NeuralSync memory system"""
    sync_api = SyncAPI()
    
    # Start sync services
    await sync_api.start()
    
    # Example: When a memory item is added locally
    async def on_memory_added(item_data):
        await sync_api.sync_memory_item('add', item_data['id'], item_data)
    
    # Example: When a memory item is updated locally
    async def on_memory_updated(item_data):
        await sync_api.sync_memory_item('update', item_data['id'], item_data)
    
    # Example: When a memory item is deleted locally
    async def on_memory_deleted(item_id):
        await sync_api.sync_memory_item('delete', item_id, {})
    
    return sync_api

if __name__ == "__main__":
    async def test_sync_system():
        sync_api = SyncAPI()
        
        # Start sync services
        await sync_api.start()
        
        # Get status
        status = await sync_api.get_status()
        print(f"Sync status: {status}")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(10)
                nodes = await sync_api.get_sync_nodes()
                print(f"Active nodes: {len([n for n in nodes if n['online']])}")
        except KeyboardInterrupt:
            print("🛑 Stopping sync services...")
    
    asyncio.run(test_sync_system())