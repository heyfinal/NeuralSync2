"""
Advanced Message Broker for Inter-CLI Communication
Provides sub-10ms routing, discovery, and delegation between CLI tools
"""

import asyncio
import json
import time
import logging
import threading
import hashlib
import weakref
from typing import Dict, Any, Optional, List, Callable, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from functools import wraps
import signal
import os
from pathlib import Path

from .ultra_comm import UltraLowLatencyComm, Message, MessageTypes

logger = logging.getLogger(__name__)


@dataclass 
class CLINodeInfo:
    """Information about a connected CLI tool"""
    node_id: str
    tool_name: str
    capabilities: List[str]
    status: str  # 'active', 'idle', 'busy', 'disconnected'
    last_heartbeat: float
    pid: int
    working_directory: str
    session_data: Dict[str, Any]
    connection_time: float
    message_count: int = 0
    avg_response_time: float = 0.0


@dataclass
class RoutingRule:
    """Message routing rule"""
    source_pattern: str
    target_pattern: str
    message_type: str
    priority: int = 0
    transform_func: Optional[Callable] = None
    condition_func: Optional[Callable] = None


@dataclass
class DelegationRequest:
    """CLI tool delegation request"""
    request_id: str
    source_cli: str
    target_cli: str
    command: str
    parameters: Dict[str, Any]
    timeout: float
    callback_on_complete: bool = True
    stream_updates: bool = False
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class MessageBroker:
    """High-performance message broker for CLI tool communication"""
    
    def __init__(self, broker_id: str = "neuralsync_broker"):
        self.broker_id = broker_id
        self.nodes: Dict[str, CLINodeInfo] = {}
        self.routing_rules: List[RoutingRule] = []
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.active_delegations: Dict[str, DelegationRequest] = {}
        self.message_history: deque = deque(maxlen=10000)
        
        # Performance tracking
        self.stats = {
            'messages_routed': 0,
            'avg_routing_latency': 0.0,
            'failed_deliveries': 0,
            'active_connections': 0,
            'delegation_requests': 0
        }
        
        # Communication layer
        self.comm = UltraLowLatencyComm(broker_id)
        self._setup_handlers()
        
        # Background tasks
        self.heartbeat_task = None
        self.cleanup_task = None
        self.running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
    def _setup_handlers(self):
        """Setup message handlers for broker operations"""
        
        # Core broker operations
        self.comm.register_handler("node_register", self._handle_node_register)
        self.comm.register_handler("node_unregister", self._handle_node_unregister)
        self.comm.register_handler("node_discovery", self._handle_node_discovery)
        self.comm.register_handler("heartbeat", self._handle_heartbeat)
        
        # Message routing
        self.comm.register_handler("route_message", self._handle_route_message)
        self.comm.register_handler("broadcast", self._handle_broadcast)
        
        # CLI delegation
        self.comm.register_handler("delegation_request", self._handle_delegation_request)
        self.comm.register_handler("delegation_response", self._handle_delegation_response)
        self.comm.register_handler("delegation_stream", self._handle_delegation_stream)
        
        # Memory and status updates
        self.comm.register_handler("memory_update", self._handle_memory_update)
        self.comm.register_handler("status_update", self._handle_status_update)
        
    async def start(self):
        """Start the message broker"""
        if self.running:
            return
            
        logger.info(f"Starting message broker: {self.broker_id}")
        
        # Start communication layer
        await self.comm.start_server()
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        self.cleanup_task = asyncio.create_task(self._cleanup_monitor())
        
        self.running = True
        
        # Register default routing rules
        self._setup_default_routing()
        
        logger.info("Message broker started successfully")
        
    async def stop(self):
        """Stop the message broker"""
        if not self.running:
            return
            
        logger.info("Stopping message broker")
        
        self.running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
            
        # Notify all nodes of shutdown
        await self._broadcast_shutdown()
        
        # Cleanup communication
        await self.comm.cleanup()
        
        logger.info("Message broker stopped")
        
    def _setup_default_routing(self):
        """Setup default routing rules"""
        
        # Route memory operations to any capable node
        self.add_routing_rule(
            source_pattern="*",
            target_pattern="capability:memory_storage",
            message_type="memory_store",
            priority=1
        )
        
        # Route delegation requests to specific CLI tools
        self.add_routing_rule(
            source_pattern="*",
            target_pattern="tool:*",
            message_type="delegation_request",
            priority=2
        )
        
        # Broadcast status updates to all nodes
        self.add_routing_rule(
            source_pattern="*", 
            target_pattern="*",
            message_type="status_update",
            priority=0
        )
        
    def add_routing_rule(self, 
                        source_pattern: str,
                        target_pattern: str, 
                        message_type: str,
                        priority: int = 0,
                        transform_func: Optional[Callable] = None,
                        condition_func: Optional[Callable] = None):
        """Add message routing rule"""
        
        rule = RoutingRule(
            source_pattern=source_pattern,
            target_pattern=target_pattern,
            message_type=message_type,
            priority=priority,
            transform_func=transform_func,
            condition_func=condition_func
        )
        
        with self.lock:
            self.routing_rules.append(rule)
            # Sort by priority (higher first)
            self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
            
        logger.debug(f"Added routing rule: {source_pattern} -> {target_pattern} ({message_type})")
        
    async def register_node(self, 
                           node_id: str,
                           tool_name: str, 
                           capabilities: List[str],
                           session_data: Dict[str, Any] = None) -> bool:
        """Register a CLI tool node"""
        
        node_info = CLINodeInfo(
            node_id=node_id,
            tool_name=tool_name,
            capabilities=capabilities or [],
            status="active",
            last_heartbeat=time.time(),
            pid=session_data.get('pid', 0) if session_data else 0,
            working_directory=session_data.get('working_directory', '') if session_data else '',
            session_data=session_data or {},
            connection_time=time.time()
        )
        
        with self.lock:
            self.nodes[node_id] = node_info
            self.stats['active_connections'] = len(self.nodes)
            
        logger.info(f"Registered CLI node: {tool_name} ({node_id})")
        
        # Notify other nodes of new connection
        await self._broadcast_node_update("node_connected", node_info)
        
        return True
        
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a CLI tool node"""
        
        with self.lock:
            if node_id not in self.nodes:
                return False
                
            node_info = self.nodes[node_id]
            del self.nodes[node_id] 
            self.stats['active_connections'] = len(self.nodes)
            
        logger.info(f"Unregistered CLI node: {node_info.tool_name} ({node_id})")
        
        # Notify other nodes
        await self._broadcast_node_update("node_disconnected", node_info)
        
        return True
        
    async def discover_nodes(self, 
                           tool_name: Optional[str] = None,
                           capability: Optional[str] = None) -> List[CLINodeInfo]:
        """Discover available CLI tool nodes"""
        
        with self.lock:
            nodes = list(self.nodes.values())
            
        # Filter by tool name
        if tool_name:
            nodes = [n for n in nodes if n.tool_name == tool_name]
            
        # Filter by capability
        if capability:
            nodes = [n for n in nodes if capability in n.capabilities]
            
        return nodes
        
    async def route_message(self,
                           source: str,
                           message_type: str, 
                           payload: Dict[str, Any],
                           target: Optional[str] = None) -> bool:
        """Route message between CLI tools"""
        
        start_time = time.perf_counter()
        
        try:
            # Find routing targets
            targets = await self._resolve_targets(source, target, message_type, payload)
            
            if not targets:
                logger.warning(f"No routing targets found for {message_type} from {source}")
                self.stats['failed_deliveries'] += 1
                return False
                
            # Send to all targets
            success_count = 0
            for target_node in targets:
                try:
                    # Apply transformation if needed
                    transformed_payload = await self._transform_payload(
                        source, target_node, message_type, payload
                    )
                    
                    # Send message
                    success = await self.comm.send_message(
                        target_node,
                        message_type,
                        json.dumps(transformed_payload).encode()
                    )
                    
                    if success:
                        success_count += 1
                        
                        # Update node stats
                        with self.lock:
                            if target_node in self.nodes:
                                self.nodes[target_node].message_count += 1
                                
                except Exception as e:
                    logger.error(f"Failed to route to {target_node}: {e}")
                    
            # Update broker stats
            self.stats['messages_routed'] += 1
            latency = (time.perf_counter() - start_time) * 1000
            self.stats['avg_routing_latency'] = (
                self.stats['avg_routing_latency'] * 0.9 + latency * 0.1
            )
            
            # Log message for history
            self.message_history.append({
                'timestamp': time.time(),
                'source': source,
                'targets': targets,
                'message_type': message_type,
                'success_count': success_count,
                'latency_ms': latency
            })
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Message routing error: {e}")
            self.stats['failed_deliveries'] += 1
            return False
            
    async def delegate_command(self,
                             source_cli: str,
                             target_cli: str,
                             command: str,
                             parameters: Dict[str, Any] = None,
                             timeout: float = 30.0,
                             stream_updates: bool = False) -> str:
        """Delegate command execution to another CLI tool"""
        
        request_id = hashlib.sha256(
            f"{source_cli}{target_cli}{command}{time.time()}".encode()
        ).hexdigest()[:16]
        
        delegation = DelegationRequest(
            request_id=request_id,
            source_cli=source_cli,
            target_cli=target_cli,
            command=command,
            parameters=parameters or {},
            timeout=timeout,
            stream_updates=stream_updates
        )
        
        with self.lock:
            self.active_delegations[request_id] = delegation
            self.stats['delegation_requests'] += 1
            
        logger.info(f"Creating delegation: {source_cli} -> {target_cli}: {command}")
        
        # Send delegation request
        success = await self.comm.send_message(
            target_cli,
            "delegation_request",
            json.dumps(asdict(delegation)).encode()
        )
        
        if not success:
            with self.lock:
                del self.active_delegations[request_id]
            raise RuntimeError(f"Failed to send delegation request to {target_cli}")
            
        return request_id
        
    async def _resolve_targets(self,
                              source: str,
                              target: Optional[str],
                              message_type: str,
                              payload: Dict[str, Any]) -> List[str]:
        """Resolve message routing targets based on rules"""
        
        targets = []
        
        # Direct target specified
        if target and target != "*":
            if target.startswith("tool:"):
                # Route to specific tool type
                tool_name = target[5:]
                with self.lock:
                    targets = [
                        node_id for node_id, info in self.nodes.items()
                        if info.tool_name == tool_name and info.status == "active"
                    ]
            elif target.startswith("capability:"):
                # Route to nodes with specific capability
                capability = target[11:]
                with self.lock:
                    targets = [
                        node_id for node_id, info in self.nodes.items()
                        if capability in info.capabilities and info.status == "active"
                    ]
            else:
                # Direct node target
                targets = [target] if target in self.nodes else []
                
        else:
            # Apply routing rules
            with self.lock:
                for rule in self.routing_rules:
                    if (rule.message_type == message_type and
                        self._matches_pattern(source, rule.source_pattern)):
                        
                        # Check condition if specified
                        if rule.condition_func and not rule.condition_func(payload):
                            continue
                            
                        # Resolve target pattern
                        rule_targets = self._resolve_target_pattern(rule.target_pattern)
                        targets.extend(rule_targets)
                        
                        # Stop at first matching rule with targets
                        if targets:
                            break
                            
        return list(set(targets))  # Remove duplicates
        
    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Check if value matches pattern (supports wildcards)"""
        if pattern == "*":
            return True
        return value == pattern
        
    def _resolve_target_pattern(self, pattern: str) -> List[str]:
        """Resolve target pattern to actual node IDs"""
        if pattern == "*":
            return [node_id for node_id, info in self.nodes.items() if info.status == "active"]
        elif pattern.startswith("tool:"):
            tool_name = pattern[5:]
            return [
                node_id for node_id, info in self.nodes.items()
                if info.tool_name == tool_name and info.status == "active"
            ]
        elif pattern.startswith("capability:"):
            capability = pattern[11:]
            return [
                node_id for node_id, info in self.nodes.items()
                if capability in info.capabilities and info.status == "active"
            ]
        else:
            return [pattern] if pattern in self.nodes else []
            
    async def _transform_payload(self,
                               source: str,
                               target: str, 
                               message_type: str,
                               payload: Dict[str, Any]) -> Dict[str, Any]:
        """Transform payload based on routing rules"""
        
        # Find applicable transformation rule
        for rule in self.routing_rules:
            if (rule.message_type == message_type and
                rule.transform_func and
                self._matches_pattern(source, rule.source_pattern)):
                
                try:
                    return rule.transform_func(payload)
                except Exception as e:
                    logger.error(f"Payload transformation error: {e}")
                    break
                    
        return payload
        
    # Message Handlers
    
    async def _handle_node_register(self, message: Message):
        """Handle node registration request"""
        try:
            data = json.loads(message.payload.decode())
            await self.register_node(
                node_id=message.sender,
                tool_name=data.get('tool_name', 'unknown'),
                capabilities=data.get('capabilities', []),
                session_data=data.get('session_data', {})
            )
            
            # Send acknowledgment
            await self.comm.send_message(
                message.sender,
                "registration_ack",
                json.dumps({'success': True}).encode()
            )
            
        except Exception as e:
            logger.error(f"Node registration error: {e}")
            
    async def _handle_node_unregister(self, message: Message):
        """Handle node unregistration"""
        await self.unregister_node(message.sender)
        
    async def _handle_node_discovery(self, message: Message):
        """Handle node discovery request"""
        try:
            data = json.loads(message.payload.decode())
            nodes = await self.discover_nodes(
                tool_name=data.get('tool_name'),
                capability=data.get('capability')
            )
            
            response = {
                'nodes': [asdict(node) for node in nodes]
            }
            
            await self.comm.send_message(
                message.sender,
                "discovery_response",
                json.dumps(response).encode()
            )
            
        except Exception as e:
            logger.error(f"Node discovery error: {e}")
            
    async def _handle_heartbeat(self, message: Message):
        """Handle heartbeat from node"""
        with self.lock:
            if message.sender in self.nodes:
                self.nodes[message.sender].last_heartbeat = time.time()
                
    async def _handle_route_message(self, message: Message):
        """Handle message routing request"""
        try:
            data = json.loads(message.payload.decode())
            await self.route_message(
                source=message.sender,
                message_type=data['message_type'],
                payload=data['payload'],
                target=data.get('target')
            )
        except Exception as e:
            logger.error(f"Message routing error: {e}")
            
    async def _handle_broadcast(self, message: Message):
        """Handle broadcast message"""
        try:
            data = json.loads(message.payload.decode())
            
            # Send to all active nodes except sender
            with self.lock:
                targets = [
                    node_id for node_id, info in self.nodes.items()
                    if node_id != message.sender and info.status == "active"
                ]
                
            for target in targets:
                await self.comm.send_message(
                    target,
                    data['message_type'],
                    json.dumps(data['payload']).encode()
                )
                
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            
    async def _handle_delegation_request(self, message: Message):
        """Handle CLI delegation request"""
        try:
            data = json.loads(message.payload.decode())
            delegation = DelegationRequest(**data)
            
            # Route to target CLI tool
            await self.comm.send_message(
                delegation.target_cli,
                "execute_delegation", 
                message.payload
            )
            
            logger.info(f"Routed delegation {delegation.request_id} to {delegation.target_cli}")
            
        except Exception as e:
            logger.error(f"Delegation routing error: {e}")
            
    async def _handle_delegation_response(self, message: Message):
        """Handle delegation response"""
        try:
            data = json.loads(message.payload.decode())
            request_id = data.get('request_id')
            
            with self.lock:
                if request_id in self.active_delegations:
                    delegation = self.active_delegations[request_id]
                    
                    # Route response back to source CLI
                    await self.comm.send_message(
                        delegation.source_cli,
                        "delegation_complete",
                        message.payload
                    )
                    
                    # Cleanup completed delegation
                    if data.get('final', True):
                        del self.active_delegations[request_id]
                        
        except Exception as e:
            logger.error(f"Delegation response error: {e}")
            
    async def _handle_delegation_stream(self, message: Message):
        """Handle streaming delegation update"""
        try:
            data = json.loads(message.payload.decode())
            request_id = data.get('request_id')
            
            with self.lock:
                if request_id in self.active_delegations:
                    delegation = self.active_delegations[request_id]
                    
                    # Stream update to source CLI
                    await self.comm.send_message(
                        delegation.source_cli,
                        "delegation_stream_update",
                        message.payload
                    )
                    
        except Exception as e:
            logger.error(f"Delegation stream error: {e}")
            
    async def _handle_memory_update(self, message: Message):
        """Handle memory update broadcast"""
        # Broadcast to all nodes with memory capability
        targets = await self._resolve_targets(
            message.sender,
            "capability:memory_storage",
            "memory_update",
            {}
        )
        
        for target in targets:
            if target != message.sender:
                await self.comm.send_message(
                    target,
                    "memory_sync",
                    message.payload
                )
                
    async def _handle_status_update(self, message: Message):
        """Handle status update from CLI tool"""
        try:
            data = json.loads(message.payload.decode())
            
            with self.lock:
                if message.sender in self.nodes:
                    self.nodes[message.sender].status = data.get('status', 'active')
                    
            # Broadcast status change to interested nodes
            await self._broadcast_node_update("status_changed", self.nodes[message.sender])
            
        except Exception as e:
            logger.error(f"Status update error: {e}")
            
    # Background Tasks
    
    async def _heartbeat_monitor(self):
        """Monitor node heartbeats and cleanup stale connections"""
        while self.running:
            try:
                current_time = time.time()
                stale_nodes = []
                
                with self.lock:
                    for node_id, info in self.nodes.items():
                        # Mark nodes as stale after 30 seconds
                        if current_time - info.last_heartbeat > 30:
                            stale_nodes.append(node_id)
                            
                # Cleanup stale nodes
                for node_id in stale_nodes:
                    logger.warning(f"Node {node_id} heartbeat timeout, removing")
                    await self.unregister_node(node_id)
                    
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(5)
                
    async def _cleanup_monitor(self):
        """Cleanup expired delegations and old message history"""
        while self.running:
            try:
                current_time = time.time()
                
                # Cleanup expired delegations
                expired_delegations = []
                with self.lock:
                    for request_id, delegation in self.active_delegations.items():
                        if current_time - delegation.created_at > delegation.timeout:
                            expired_delegations.append(request_id)
                            
                for request_id in expired_delegations:
                    logger.warning(f"Delegation {request_id} timed out")
                    
                    with self.lock:
                        delegation = self.active_delegations[request_id]
                        del self.active_delegations[request_id]
                        
                    # Notify source CLI of timeout
                    await self.comm.send_message(
                        delegation.source_cli,
                        "delegation_timeout",
                        json.dumps({'request_id': request_id}).encode()
                    )
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Cleanup monitor error: {e}")
                await asyncio.sleep(30)
                
    async def _broadcast_node_update(self, update_type: str, node_info: CLINodeInfo):
        """Broadcast node status updates to all connected nodes"""
        try:
            update_data = {
                'update_type': update_type,
                'node_info': asdict(node_info)
            }
            
            with self.lock:
                targets = [
                    node_id for node_id in self.nodes.keys()
                    if node_id != node_info.node_id
                ]
                
            for target in targets:
                await self.comm.send_message(
                    target,
                    "node_update",
                    json.dumps(update_data).encode()
                )
                
        except Exception as e:
            logger.error(f"Node update broadcast error: {e}")
            
    async def _broadcast_shutdown(self):
        """Notify all nodes of broker shutdown"""
        try:
            with self.lock:
                targets = list(self.nodes.keys())
                
            for target in targets:
                await self.comm.send_message(
                    target,
                    "broker_shutdown",
                    json.dumps({'timestamp': time.time()}).encode()
                )
                
        except Exception as e:
            logger.error(f"Shutdown broadcast error: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get broker status and statistics"""
        with self.lock:
            return {
                'broker_id': self.broker_id,
                'running': self.running,
                'stats': self.stats,
                'active_nodes': len(self.nodes),
                'nodes': {
                    node_id: {
                        'tool_name': info.tool_name,
                        'status': info.status,
                        'capabilities': info.capabilities,
                        'message_count': info.message_count,
                        'connection_time': info.connection_time
                    }
                    for node_id, info in self.nodes.items()
                },
                'active_delegations': len(self.active_delegations),
                'routing_rules': len(self.routing_rules),
                'recent_messages': list(self.message_history)[-10:]  # Last 10 messages
            }


# Global broker instance
_global_broker: Optional[MessageBroker] = None

def get_message_broker(broker_id: str = None) -> MessageBroker:
    """Get global message broker instance"""
    global _global_broker
    if _global_broker is None:
        _global_broker = MessageBroker(broker_id or "neuralsync_broker")
    return _global_broker


# Convenience functions for CLI tools
async def register_cli_tool(tool_name: str, 
                           capabilities: List[str],
                           session_data: Dict[str, Any] = None) -> str:
    """Register CLI tool with message broker"""
    broker = get_message_broker()
    node_id = f"{tool_name}_{os.getpid()}_{int(time.time())}"
    
    success = await broker.register_node(node_id, tool_name, capabilities, session_data)
    if success:
        logger.info(f"Registered CLI tool: {tool_name} as {node_id}")
        return node_id
    else:
        raise RuntimeError(f"Failed to register CLI tool: {tool_name}")


async def send_to_cli(target_cli: str,
                     message_type: str,
                     payload: Dict[str, Any],
                     source_node_id: str = None) -> bool:
    """Send message to specific CLI tool"""
    broker = get_message_broker()
    source = source_node_id or f"unknown_{os.getpid()}"
    
    return await broker.route_message(source, message_type, payload, f"tool:{target_cli}")


async def delegate_to_cli(source_cli: str,
                         target_cli: str, 
                         command: str,
                         parameters: Dict[str, Any] = None,
                         timeout: float = 30.0) -> str:
    """Delegate command to another CLI tool"""
    broker = get_message_broker()
    
    return await broker.delegate_command(
        source_cli, target_cli, command, parameters, timeout
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        broker = get_message_broker()
        
        try:
            await broker.start()
            print("Message broker started. Press Ctrl+C to stop.")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            await broker.stop()
            
    asyncio.run(main())