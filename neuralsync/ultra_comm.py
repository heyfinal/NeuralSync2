#!/usr/bin/env python3
"""
Ultra-Low Latency Communication System for NeuralSync2
Sub-10ms inter-CLI communication via Unix domain sockets
"""

import asyncio
import json
import time
import uuid
import struct
from pathlib import Path
from typing import Dict, Set, Optional, Callable, Any, Union, List
from dataclasses import dataclass, asdict
from enum import Enum
import socket
import pickle
from concurrent.futures import ThreadPoolExecutor


class MessageTypes(Enum):
    """Enhanced message types for inter-agent communication"""
    # Core system messages
    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    ROUTE = "route"
    BROADCAST = "broadcast"
    ACK = "ack"
    
    # Memory and data synchronization
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"
    MEMORY_SYNC = "memory_sync"
    PERSONALITY_UPDATE = "personality_update"
    SYNC_RESPONSE = "sync_response"
    
    # Agent coordination
    AGENT_SPAWN = "agent_spawn"
    AGENT_STATUS = "agent_status"
    TASK_DELEGATE = "task_delegate"
    TASK_COMPLETE = "task_complete"
    
    # Specialized capabilities
    CODE_ANALYZE = "code_analyze"
    CODE_GENERATE = "code_generate"
    CODE_REVIEW = "code_review"
    CODE_REFACTOR = "code_refactor"
    RESEARCH_REQUEST = "research_request"
    CONTENT_GENERATE = "content_generate"
    PROBLEM_SOLVE = "problem_solve"
    
    # Security and unleashed mode
    UNLEASHED_MODE = "unleashed_mode"
    SECURITY_ALERT = "security_alert"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class Message:
    """Ultra-lightweight message structure"""
    id: str
    source: str
    target: str
    type: str
    payload: Dict[str, Any]
    timestamp: float = 0.0
    ack_required: bool = True

@dataclass
class CliRegistration:
    """CLI tool registration info"""
    name: str
    socket_path: str
    pid: int
    capabilities: Set[str]
    last_heartbeat: float

class MessageBroker:
    """Ultra-fast message broker using Unix domain sockets"""
    
    def __init__(self, socket_path: str = "/tmp/neuralsync_broker.sock"):
        self.socket_path = socket_path
        self.server_socket: Optional[socket.socket] = None
        self.clients: Dict[str, CliRegistration] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="ns_comm")
        
    async def start(self):
        """Start the message broker"""
        # Clean up existing socket
        Path(self.socket_path).unlink(missing_ok=True)
        
        # Create Unix domain socket server
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(50)  # High connection backlog
        self.server_socket.setblocking(False)
        
        self.running = True
        
        # Start accept loop
        asyncio.create_task(self._accept_loop())
        
        # Start health check loop
        asyncio.create_task(self._health_check_loop())
        
        print(f"🚀 NeuralSync MessageBroker started on {self.socket_path}")
        
    async def _accept_loop(self):
        """Accept new client connections"""
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                client_sock, _ = await loop.sock_accept(self.server_socket)
                client_sock.setblocking(False)
                
                # Handle client in separate task
                asyncio.create_task(self._handle_client(client_sock))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Error accepting connection: {e}")
                await asyncio.sleep(0.1)
                
    async def _handle_client(self, client_sock: socket.socket):
        """Handle individual client connection"""
        cli_name = None
        try:
            while self.running:
                # Read message length prefix (4 bytes)
                length_data = await self._recv_exact(client_sock, 4)
                if not length_data:
                    break
                    
                msg_length = struct.unpack('!I', length_data)[0]
                
                # Read message data
                msg_data = await self._recv_exact(client_sock, msg_length)
                if not msg_data:
                    break
                
                # Deserialize message (using pickle for speed)
                try:
                    message = pickle.loads(msg_data)
                except:
                    # Fallback to JSON
                    message = json.loads(msg_data.decode())
                    
                # Handle message
                await self._process_message(message, client_sock)
                
        except Exception as e:
            print(f"❌ Client error: {e}")
        finally:
            if cli_name and cli_name in self.clients:
                del self.clients[cli_name]
                print(f"📤 CLI tool disconnected: {cli_name}")
            client_sock.close()
            
    async def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from socket"""
        loop = asyncio.get_event_loop()
        data = b''
        while len(data) < n:
            chunk = await loop.sock_recv(sock, n - len(data))
            if not chunk:
                return b''
            data += chunk
        return data
        
    async def _process_message(self, message: Dict[str, Any], client_sock: socket.socket):
        """Process incoming message"""
        msg_type = message.get('type')
        
        if msg_type == 'register':
            await self._handle_registration(message, client_sock)
        elif msg_type == 'route':
            await self._handle_routing(message)
        elif msg_type == 'broadcast':
            await self._handle_broadcast(message)
        elif msg_type == 'heartbeat':
            await self._handle_heartbeat(message)
        elif msg_type == 'discovery':
            await self._handle_discovery(message, client_sock)
            
    async def _handle_registration(self, message: Dict[str, Any], client_sock: socket.socket):
        """Handle CLI tool registration"""
        payload = message.get('payload', {})
        cli_name = payload.get('name')
        
        if cli_name:
            registration = CliRegistration(
                name=cli_name,
                socket_path=payload.get('socket_path', ''),
                pid=payload.get('pid', 0),
                capabilities=set(payload.get('capabilities', [])),
                last_heartbeat=time.time()
            )
            
            self.clients[cli_name] = registration
            print(f"📥 CLI tool registered: {cli_name}")
            
            # Send acknowledgment
            ack = {
                'id': str(uuid.uuid4()),
                'type': 'ack',
                'source': 'broker',
                'target': cli_name,
                'payload': {'status': 'registered'},
                'timestamp': time.time()
            }
            
            await self._send_message(client_sock, ack)
            
    async def _handle_routing(self, message: Dict[str, Any]):
        """Route message to target CLI tool"""
        target = message.get('target')
        
        if target in self.clients:
            target_reg = self.clients[target]
            # Connect to target CLI tool and forward message
            try:
                await self._forward_to_cli(message, target_reg)
            except Exception as e:
                print(f"❌ Failed to route to {target}: {e}")
        else:
            print(f"⚠️ Target CLI tool not found: {target}")
            
    async def _forward_to_cli(self, message: Dict[str, Any], target_reg: CliRegistration):
        """Forward message to specific CLI tool"""
        if not target_reg.socket_path:
            return
            
        try:
            # Connect to target CLI
            cli_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            cli_sock.connect(target_reg.socket_path)
            cli_sock.setblocking(False)
            
            await self._send_message(cli_sock, message)
            
            # Wait for response if needed
            if message.get('ack_required', True):
                response = await self._receive_message(cli_sock)
                # Route response back to source
                if response:
                    source = message.get('source')
                    if source in self.clients:
                        source_reg = self.clients[source]
                        await self._forward_to_cli(response, source_reg)
                        
            cli_sock.close()
            
        except Exception as e:
            print(f"❌ Forward error: {e}")
            
    async def _send_message(self, sock: socket.socket, message: Dict[str, Any]):
        """Send message with length prefix"""
        loop = asyncio.get_event_loop()
        
        # Serialize message (try pickle first for speed)
        try:
            msg_data = pickle.dumps(message)
        except:
            msg_data = json.dumps(message).encode()
            
        # Send length prefix + message
        length_prefix = struct.pack('!I', len(msg_data))
        await loop.sock_sendall(sock, length_prefix + msg_data)
        
    async def _receive_message(self, sock: socket.socket) -> Optional[Dict[str, Any]]:
        """Receive message with timeout"""
        try:
            length_data = await asyncio.wait_for(
                self._recv_exact(sock, 4), 
                timeout=5.0
            )
            
            if not length_data:
                return None
                
            msg_length = struct.unpack('!I', length_data)[0]
            msg_data = await asyncio.wait_for(
                self._recv_exact(sock, msg_length),
                timeout=5.0
            )
            
            # Deserialize
            try:
                return pickle.loads(msg_data)
            except:
                return json.loads(msg_data.decode())
                
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"❌ Receive error: {e}")
            return None
            
    async def _handle_broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected CLI tools"""
        for cli_name, registration in self.clients.items():
            if cli_name != message.get('source'):  # Don't broadcast to sender
                await self._forward_to_cli(message, registration)
                
    async def _handle_heartbeat(self, message: Dict[str, Any]):
        """Handle heartbeat from CLI tool"""
        cli_name = message.get('source')
        if cli_name in self.clients:
            self.clients[cli_name].last_heartbeat = time.time()
            
    async def _handle_discovery(self, message: Dict[str, Any], client_sock: socket.socket):
        """Handle CLI tool discovery request"""
        capabilities_filter = message.get('payload', {}).get('capabilities', [])
        
        available_tools = []
        for cli_name, reg in self.clients.items():
            if not capabilities_filter or any(cap in reg.capabilities for cap in capabilities_filter):
                available_tools.append({
                    'name': cli_name,
                    'capabilities': list(reg.capabilities),
                    'last_seen': reg.last_heartbeat
                })
                
        response = {
            'id': str(uuid.uuid4()),
            'type': 'discovery_response',
            'source': 'broker',
            'target': message.get('source'),
            'payload': {'tools': available_tools},
            'timestamp': time.time()
        }
        
        await self._send_message(client_sock, response)
        
    async def _health_check_loop(self):
        """Remove stale CLI tool registrations"""
        while self.running:
            try:
                current_time = time.time()
                stale_clients = []
                
                for cli_name, registration in self.clients.items():
                    if current_time - registration.last_heartbeat > 30:  # 30 second timeout
                        stale_clients.append(cli_name)
                        
                for cli_name in stale_clients:
                    del self.clients[cli_name]
                    print(f"🧹 Removed stale CLI registration: {cli_name}")
                    
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"❌ Health check error: {e}")
                await asyncio.sleep(10)
                
    async def stop(self):
        """Stop the message broker"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        Path(self.socket_path).unlink(missing_ok=True)
        self.executor.shutdown(wait=True)


class CliCommunicator:
    """Client-side communicator for CLI tools"""
    
    def __init__(self, cli_name: str, capabilities: Set[str] = None):
        self.cli_name = cli_name
        self.capabilities = capabilities or set()
        self.broker_socket_path = "/tmp/neuralsync_broker.sock"
        self.my_socket_path = f"/tmp/neuralsync_cli_{cli_name}_{uuid.uuid4().hex[:8]}.sock"
        self.broker_sock: Optional[socket.socket] = None
        self.server_sock: Optional[socket.socket] = None
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        
    async def connect(self):
        """Connect to message broker and start listening"""
        try:
            # Connect to broker
            self.broker_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.broker_sock.connect(self.broker_socket_path)
            self.broker_sock.setblocking(False)
            
            # Create our own server socket for receiving messages
            Path(self.my_socket_path).unlink(missing_ok=True)
            self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_sock.bind(self.my_socket_path)
            self.server_sock.listen(10)
            self.server_sock.setblocking(False)
            
            # Register with broker
            registration_msg = {
                'id': str(uuid.uuid4()),
                'type': 'register',
                'source': self.cli_name,
                'target': 'broker',
                'payload': {
                    'name': self.cli_name,
                    'socket_path': self.my_socket_path,
                    'pid': os.getpid(),
                    'capabilities': list(self.capabilities)
                },
                'timestamp': time.time()
            }
            
            await self._send_to_broker(registration_msg)
            
            self.running = True
            
            # Start listening for incoming messages
            asyncio.create_task(self._message_listener())
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat_loop())
            
            print(f"✅ {self.cli_name} connected to NeuralSync communication system")
            
        except Exception as e:
            print(f"❌ Failed to connect to broker: {e}")
            raise
            
    async def _send_to_broker(self, message: Dict[str, Any]):
        """Send message to broker"""
        if not self.broker_sock:
            raise RuntimeError("Not connected to broker")
            
        loop = asyncio.get_event_loop()
        
        # Serialize with pickle for speed
        try:
            msg_data = pickle.dumps(message)
        except:
            msg_data = json.dumps(message).encode()
            
        length_prefix = struct.pack('!I', len(msg_data))
        await loop.sock_sendall(self.broker_sock, length_prefix + msg_data)
        
    async def send_message(self, target_cli: str, message_type: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message to another CLI tool"""
        message = {
            'id': str(uuid.uuid4()),
            'type': 'route',
            'source': self.cli_name,
            'target': target_cli,
            'payload': {
                'message_type': message_type,
                'data': payload
            },
            'timestamp': time.time(),
            'ack_required': True
        }
        
        await self._send_to_broker(message)
        
    async def broadcast_message(self, message_type: str, payload: Dict[str, Any]):
        """Broadcast message to all CLI tools"""
        message = {
            'id': str(uuid.uuid4()),
            'type': 'broadcast',
            'source': self.cli_name,
            'target': 'all',
            'payload': {
                'message_type': message_type,
                'data': payload
            },
            'timestamp': time.time(),
            'ack_required': False
        }
        
        await self._send_to_broker(message)
        
    async def discover_tools(self, capabilities: list = None) -> list:
        """Discover available CLI tools"""
        message = {
            'id': str(uuid.uuid4()),
            'type': 'discovery',
            'source': self.cli_name,
            'target': 'broker',
            'payload': {
                'capabilities': capabilities or []
            },
            'timestamp': time.time()
        }
        
        await self._send_to_broker(message)
        
        # This is a simplified version - in practice you'd wait for the response
        return []
        
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for incoming messages"""
        self.message_handlers[message_type] = handler
        
    async def _message_listener(self):
        """Listen for incoming messages on our socket"""
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                client_sock, _ = await loop.sock_accept(self.server_sock)
                client_sock.setblocking(False)
                
                # Handle message in separate task
                asyncio.create_task(self._handle_incoming_message(client_sock))
                
            except Exception as e:
                print(f"❌ Message listener error: {e}")
                await asyncio.sleep(0.1)
                
    async def _handle_incoming_message(self, client_sock: socket.socket):
        """Handle incoming message"""
        try:
            # Read message
            loop = asyncio.get_event_loop()
            length_data = await loop.sock_recv(client_sock, 4)
            
            if len(length_data) < 4:
                return
                
            msg_length = struct.unpack('!I', length_data)[0]
            msg_data = await loop.sock_recv(client_sock, msg_length)
            
            # Deserialize
            try:
                message = pickle.loads(msg_data)
            except:
                message = json.loads(msg_data.decode())
                
            # Process message
            payload = message.get('payload', {})
            message_type = payload.get('message_type')
            
            if message_type in self.message_handlers:
                response = await self.message_handlers[message_type](payload.get('data', {}))
                
                # Send response if required
                if message.get('ack_required') and response:
                    response_msg = {
                        'id': str(uuid.uuid4()),
                        'type': 'response',
                        'source': self.cli_name,
                        'target': message.get('source'),
                        'payload': response,
                        'timestamp': time.time()
                    }
                    
                    # Send response
                    response_data = pickle.dumps(response_msg)
                    length_prefix = struct.pack('!I', len(response_data))
                    await loop.sock_sendall(client_sock, length_prefix + response_data)
                    
        except Exception as e:
            print(f"❌ Message handling error: {e}")
        finally:
            client_sock.close()
            
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to broker"""
        while self.running:
            try:
                heartbeat = {
                    'id': str(uuid.uuid4()),
                    'type': 'heartbeat',
                    'source': self.cli_name,
                    'target': 'broker',
                    'payload': {},
                    'timestamp': time.time()
                }
                
                await self._send_to_broker(heartbeat)
                await asyncio.sleep(15)  # Heartbeat every 15 seconds
                
            except Exception as e:
                print(f"❌ Heartbeat error: {e}")
                await asyncio.sleep(15)
                
    async def disconnect(self):
        """Disconnect from communication system"""
        self.running = False
        
        if self.broker_sock:
            self.broker_sock.close()
            
        if self.server_sock:
            self.server_sock.close()
            
        Path(self.my_socket_path).unlink(missing_ok=True)


# CLI Integration Helpers
import os

async def claude_code_integration():
    """Integration example for claude-code CLI"""
    communicator = CliCommunicator("claude-code", {"code-generation", "analysis", "debugging"})
    
    # Register handlers
    async def handle_analyze_request(data):
        # Handle analysis request from other tools
        return {"status": "analyzed", "result": "analysis result here"}
        
    async def handle_review_request(data):
        # Handle code review request
        return {"status": "reviewed", "issues": []}
        
    communicator.register_message_handler("analyze_code", handle_analyze_request)
    communicator.register_message_handler("review_code", handle_review_request)
    
    await communicator.connect()
    return communicator

async def codex_cli_integration():
    """Integration example for codex-cli"""
    communicator = CliCommunicator("codex-cli", {"code-completion", "generation", "review"})
    
    async def handle_generation_request(data):
        # Handle code generation request
        return {"status": "generated", "code": "# Generated code here"}
        
    async def handle_completion_request(data):
        # Handle code completion request  
        return {"status": "completed", "suggestions": []}
        
    communicator.register_message_handler("generate_code", handle_generation_request)
    communicator.register_message_handler("complete_code", handle_completion_request)
    
    await communicator.connect()
    return communicator

# Main broker startup
async def start_message_broker():
    """Start the NeuralSync message broker"""
    broker = MessageBroker()
    await broker.start()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Shutting down message broker...")
        await broker.stop()

class CommunicationManager:
    """High-level communication manager for NeuralSync agents"""
    
    def __init__(self):
        self.message_broker: Optional[MessageBroker] = None
        self.communicators: Dict[str, CliCommunicator] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        
    async def start_system(self):
        """Start the complete communication system"""
        try:
            # Start message broker
            self.message_broker = MessageBroker()
            await self.message_broker.start()
            
            # Register global message handlers
            self._register_global_handlers()
            
            self.running = True
            logger.info("🚀 NeuralSync Communication System started")
            
        except Exception as e:
            logger.error(f"Failed to start communication system: {e}")
            raise
            
    async def stop_system(self):
        """Stop the complete communication system"""
        self.running = False
        
        # Disconnect all communicators
        for communicator in self.communicators.values():
            await communicator.disconnect()
            
        # Stop message broker
        if self.message_broker:
            await self.message_broker.stop()
            
        logger.info("🛑 NeuralSync Communication System stopped")
        
    def _register_global_handlers(self):
        """Register global message handlers"""
        
        async def handle_agent_coordination(message):
            """Handle agent coordination messages"""
            try:
                msg_type = message.get('type')
                payload = message.get('payload', {})
                
                if msg_type == MessageTypes.AGENT_SPAWN.value:
                    await self._handle_agent_spawn(message.get('source'), payload)
                elif msg_type == MessageTypes.TASK_DELEGATE.value:
                    await self._handle_task_delegation(message.get('source'), payload)
                elif msg_type == MessageTypes.AGENT_STATUS.value:
                    await self._handle_agent_status_request(message.get('source'), payload)
                    
            except Exception as e:
                logger.error(f"Agent coordination error: {e}")
                
        async def handle_security_messages(message):
            """Handle security-related messages"""
            try:
                msg_type = message.get('type')
                payload = message.get('payload', {})
                
                if msg_type == MessageTypes.SECURITY_ALERT.value:
                    await self._handle_security_alert(message.get('source'), payload)
                elif msg_type == MessageTypes.EMERGENCY_SHUTDOWN.value:
                    await self._handle_emergency_shutdown(message.get('source'), payload)
                    
            except Exception as e:
                logger.error(f"Security message error: {e}")
                
        # Register handlers with broker
        if self.message_broker:
            self.message_broker.message_handlers.update({
                'agent_coordination': handle_agent_coordination,
                'security': handle_security_messages
            })
            
    async def _handle_agent_spawn(self, source: str, payload: Dict[str, Any]):
        """Handle agent spawn request"""
        agent_type = payload.get('agent_type')
        task_description = payload.get('task')
        context = payload.get('context', {})
        
        logger.info(f"Agent spawn request from {source}: {agent_type} for task '{task_description}'")
        
        # Implementation would spawn the requested agent
        # For now, log the request
        
    async def _handle_task_delegation(self, source: str, payload: Dict[str, Any]):
        """Handle task delegation between agents"""
        target_agent = payload.get('target_agent')
        task_type = payload.get('task_type')
        task_data = payload.get('task_data', {})
        
        logger.info(f"Task delegation from {source} to {target_agent}: {task_type}")
        
        # Forward task to target agent if connected
        if target_agent in self.communicators:
            communicator = self.communicators[target_agent]
            await communicator.send_message(target_agent, MessageTypes.TASK_DELEGATE.value, {
                'task_type': task_type,
                'task_data': task_data,
                'delegated_by': source,
                'timestamp': time.time()
            })
            
    async def _handle_agent_status_request(self, source: str, payload: Dict[str, Any]):
        """Handle agent status request"""
        logger.info(f"Agent status request from {source}")
        
        # Collect status from all connected agents
        status_info = {
            'connected_agents': list(self.communicators.keys()),
            'system_running': self.running,
            'timestamp': time.time()
        }
        
        # Send response back to source
        if source in self.communicators:
            communicator = self.communicators[source]
            await communicator.send_message(source, MessageTypes.AGENT_STATUS.value, status_info)
            
    async def _handle_security_alert(self, source: str, payload: Dict[str, Any]):
        """Handle security alert"""
        alert_level = payload.get('level', 'medium')
        alert_message = payload.get('message', '')
        
        logger.warning(f"🚨 Security alert from {source} ({alert_level}): {alert_message}")
        
        # Broadcast to all agents if high severity
        if alert_level in ['high', 'critical']:
            for communicator in self.communicators.values():
                await communicator.broadcast_message(MessageTypes.SECURITY_ALERT.value, payload)
                
    async def _handle_emergency_shutdown(self, source: str, payload: Dict[str, Any]):
        """Handle emergency shutdown request"""
        reason = payload.get('reason', 'Unknown')
        
        logger.critical(f"🚨 Emergency shutdown requested by {source}: {reason}")
        
        # Initiate emergency shutdown
        await self.stop_system()
        
    async def register_agent(self, agent_name: str, capabilities: Set[str]) -> CliCommunicator:
        """Register a new agent with the communication system"""
        try:
            communicator = CliCommunicator(agent_name, capabilities)
            await communicator.connect()
            
            self.communicators[agent_name] = communicator
            logger.info(f"✅ Registered agent: {agent_name} with capabilities: {capabilities}")
            
            return communicator
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_name}: {e}")
            raise
            
    async def unregister_agent(self, agent_name: str):
        """Unregister an agent from the communication system"""
        if agent_name in self.communicators:
            communicator = self.communicators[agent_name]
            await communicator.disconnect()
            del self.communicators[agent_name]
            logger.info(f"❌ Unregistered agent: {agent_name}")
            
    def get_connected_agents(self) -> List[str]:
        """Get list of currently connected agents"""
        return list(self.communicators.keys())
        
    def get_agent_capabilities(self, agent_name: str) -> Set[str]:
        """Get capabilities of a specific agent"""
        if agent_name in self.communicators:
            return self.communicators[agent_name].capabilities
        return set()
        
    async def broadcast_to_agents(self, message_type: str, payload: Dict[str, Any], exclude: Set[str] = None):
        """Broadcast message to all connected agents"""
        exclude = exclude or set()
        
        for agent_name, communicator in self.communicators.items():
            if agent_name not in exclude:
                try:
                    await communicator.broadcast_message(message_type, payload)
                except Exception as e:
                    logger.error(f"Failed to broadcast to {agent_name}: {e}")
                    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive communication system statistics"""
        return {
            'running': self.running,
            'connected_agents': len(self.communicators),
            'agent_details': {
                name: {
                    'capabilities': list(comm.capabilities),
                    'connected': True
                }
                for name, comm in self.communicators.items()
            },
            'broker_active': self.message_broker is not None and hasattr(self.message_broker, 'running') and self.message_broker.running,
            'timestamp': time.time()
        }


# Global communication manager instance
_comm_manager: Optional[CommunicationManager] = None


def get_comm_manager(site_id: str = None) -> CommunicationManager:
    """Get singleton communication manager instance"""
    global _comm_manager
    if _comm_manager is None:
        _comm_manager = CommunicationManager()
    return _comm_manager


async def ensure_communication_system() -> bool:
    """Ensure the communication system is running"""
    try:
        manager = get_comm_manager()
        if not manager.running:
            await manager.start_system()
        return True
    except Exception as e:
        logger.error(f"Failed to ensure communication system: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(start_message_broker())