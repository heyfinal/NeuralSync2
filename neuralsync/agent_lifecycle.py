#!/usr/bin/env python3
"""
NeuralSync Agent Lifecycle Management
Agent spawning, task delegation, lifecycle monitoring, and coordination
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import logging
import psutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from contextlib import asynccontextmanager

from .config import load_config, DEFAULT_HOME
from .ultra_comm import get_comm_manager, MessageTypes
from .agent_sync import get_agent_synchronizer
from .daemon_manager import get_daemon_manager

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status states"""
    SPAWNING = "spawning"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentInfo:
    """Agent instance information"""
    agent_id: str
    agent_type: str  # claude, codex, gemini
    cli_tool: str
    capabilities: Set[str]
    process_id: int
    status: AgentStatus
    spawn_time: float
    last_heartbeat: float
    current_task: Optional[str]
    completed_tasks: List[str]
    resource_usage: Dict[str, float]
    parent_agent: Optional[str]
    spawn_context: Dict[str, Any]


@dataclass  
class TaskDefinition:
    """Task definition for agent execution"""
    task_id: str
    task_type: str
    description: str
    target_agent_type: str
    required_capabilities: Set[str]
    priority: int
    timeout_seconds: int
    context: Dict[str, Any]
    input_data: Any
    expected_output_type: str
    created_by: str
    created_at: float


@dataclass
class TaskExecution:
    """Task execution tracking"""
    task_id: str
    assigned_agent: str
    status: TaskStatus
    start_time: Optional[float]
    end_time: Optional[float]
    result: Any
    error_message: Optional[str]
    resource_usage: Dict[str, float]
    intermediate_outputs: List[Dict[str, Any]]


class AgentLifecycleManager:
    """Advanced agent lifecycle and task management system"""
    
    def __init__(self, config_dir: Path = DEFAULT_HOME):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Agent registry
        self.active_agents: Dict[str, AgentInfo] = {}
        self.agent_processes: Dict[str, subprocess.Popen] = {}
        
        # Task management
        self.pending_tasks: Dict[str, TaskDefinition] = {}
        self.active_tasks: Dict[str, TaskExecution] = {}
        self.completed_tasks: Dict[str, TaskExecution] = {}
        
        # Agent type mappings
        self.agent_commands = {
            'claude': 'claude-ns',
            'codex': 'codex-ns', 
            'gemini': 'gemini-ns'
        }
        
        # Resource limits
        self.max_agents_per_type = {
            'claude': 3,
            'codex': 2,
            'gemini': 2
        }
        self.max_total_agents = 5
        
        # Dependencies
        self.ns_config = load_config()
        self.comm_manager = None
        self.synchronizer = None
        self.daemon_manager = None
        
        # Background tasks
        self.monitor_task = None
        self.task_scheduler_task = None
        self.cleanup_task = None
        self.running = False
        
        # Threading
        self.lifecycle_lock = threading.RLock()
        
    async def start_lifecycle_management(self):
        """Start the agent lifecycle management system"""
        if self.running:
            return
            
        try:
            # Initialize dependencies
            self.comm_manager = get_comm_manager()
            self.synchronizer = get_agent_synchronizer()
            self.daemon_manager = get_daemon_manager()
            
            # Ensure core systems are running
            if not self.comm_manager.running:
                await self.comm_manager.start_system()
            if not self.synchronizer.running:
                await self.synchronizer.start_synchronization()
                
            # Register lifecycle message handlers
            await self._register_lifecycle_handlers()
            
            # Start background tasks
            self.monitor_task = asyncio.create_task(self._agent_monitor_loop())
            self.task_scheduler_task = asyncio.create_task(self._task_scheduler_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.running = True
            logger.info("🚀 Agent lifecycle management started")
            
        except Exception as e:
            logger.error(f"Failed to start lifecycle management: {e}")
            raise
            
    async def stop_lifecycle_management(self):
        """Stop the lifecycle management system"""
        self.running = False
        
        # Terminate all managed agents
        await self._terminate_all_agents()
        
        # Cancel background tasks
        tasks = [self.monitor_task, self.task_scheduler_task, self.cleanup_task]
        for task in tasks:
            if task:
                task.cancel()
                
        # Wait for tasks to complete
        if any(tasks):
            await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
            
        logger.info("🛑 Agent lifecycle management stopped")
        
    async def _register_lifecycle_handlers(self):
        """Register lifecycle-related message handlers"""
        
        async def handle_spawn_request(data):
            """Handle agent spawn request"""
            try:
                agent_type = data.get('agent_type', '')
                task = data.get('task', '')
                context = data.get('context', {})
                parent_agent = data.get('parent_agent', '')
                
                agent_id = await self.spawn_agent(
                    agent_type=agent_type,
                    task_description=task,
                    context=context,
                    parent_agent=parent_agent
                )
                
                return {'agent_id': agent_id, 'status': 'spawned' if agent_id else 'failed'}
                
            except Exception as e:
                logger.error(f"Spawn request handler error: {e}")
                return {'status': 'error', 'message': str(e)}
                
        async def handle_task_delegation(data):
            """Handle task delegation"""
            try:
                task_def = TaskDefinition(
                    task_id=data.get('task_id', str(uuid.uuid4())),
                    task_type=data.get('task_type', ''),
                    description=data.get('description', ''),
                    target_agent_type=data.get('target_agent_type', ''),
                    required_capabilities=set(data.get('required_capabilities', [])),
                    priority=data.get('priority', 5),
                    timeout_seconds=data.get('timeout_seconds', 300),
                    context=data.get('context', {}),
                    input_data=data.get('input_data'),
                    expected_output_type=data.get('expected_output_type', 'json'),
                    created_by=data.get('created_by', ''),
                    created_at=time.time()
                )
                
                success = await self.delegate_task(task_def)
                return {'task_id': task_def.task_id, 'status': 'delegated' if success else 'failed'}
                
            except Exception as e:
                logger.error(f"Task delegation handler error: {e}")
                return {'status': 'error', 'message': str(e)}
                
        async def handle_agent_heartbeat(data):
            """Handle agent heartbeat"""
            try:
                agent_id = data.get('agent_id', '')
                status = data.get('status', 'active')
                resource_usage = data.get('resource_usage', {})
                
                await self._update_agent_heartbeat(agent_id, status, resource_usage)
                return {'status': 'acknowledged'}
                
            except Exception as e:
                logger.error(f"Heartbeat handler error: {e}")
                return {'status': 'error', 'message': str(e)}
                
        # Register with communication manager (simplified)
        # In practice, would register these with the message broker
        
    async def spawn_agent(self, agent_type: str, task_description: str = "", 
                         context: Dict[str, Any] = None, parent_agent: str = None,
                         priority: int = 5) -> Optional[str]:
        """Spawn a new agent instance"""
        try:
            with self.lifecycle_lock:
                # Validate agent type
                if agent_type not in self.agent_commands:
                    logger.error(f"Unknown agent type: {agent_type}")
                    return None
                    
                # Check resource limits
                if not self._check_spawn_limits(agent_type):
                    logger.warning(f"Agent spawn blocked by resource limits: {agent_type}")
                    return None
                    
                # Generate agent ID
                agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
                
                # Prepare spawn command
                cmd = [self.agent_commands[agent_type]]
                
                # Add task if specified
                if task_description:
                    task_context = context or {}
                    task_data = {
                        'task': task_description,
                        'context': task_context,
                        'agent_id': agent_id,
                        'parent_agent': parent_agent
                    }
                    cmd.extend(['--task', json.dumps(task_data)])
                    
                # Setup environment
                env = os.environ.copy()
                env.update({
                    'NEURALSYNC_AGENT_ID': agent_id,
                    'NEURALSYNC_PARENT': parent_agent or '',
                    'NEURALSYNC_PRIORITY': str(priority)
                })
                
                # Spawn process
                logger.info(f"Spawning {agent_type} agent: {agent_id}")
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=os.setsid if os.name != 'nt' else None
                )
                
                # Create agent info
                agent_info = AgentInfo(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    cli_tool=self.agent_commands[agent_type],
                    capabilities=self._get_agent_capabilities(agent_type),
                    process_id=process.pid,
                    status=AgentStatus.SPAWNING,
                    spawn_time=time.time(),
                    last_heartbeat=time.time(),
                    current_task=None,
                    completed_tasks=[],
                    resource_usage={},
                    parent_agent=parent_agent,
                    spawn_context=context or {}
                )
                
                # Register agent
                self.active_agents[agent_id] = agent_info
                self.agent_processes[agent_id] = process
                
                # Register with synchronizer
                await self.synchronizer.register_agent_session(
                    agent_id, 
                    agent_type, 
                    agent_info.capabilities
                )
                
                # Wait for initialization (with timeout)
                init_timeout = 30
                start_time = time.time()
                
                while time.time() - start_time < init_timeout:
                    if process.poll() is not None:
                        # Process died during startup
                        logger.error(f"Agent {agent_id} died during startup")
                        await self._cleanup_failed_agent(agent_id)
                        return None
                        
                    # Check if agent is responding
                    if await self._check_agent_health(agent_id):
                        agent_info.status = AgentStatus.ACTIVE
                        logger.info(f"✅ Agent {agent_id} spawned successfully")
                        
                        # Notify other agents
                        await self._broadcast_agent_event('agent_spawned', {
                            'agent_id': agent_id,
                            'agent_type': agent_type,
                            'parent': parent_agent,
                            'task': task_description
                        })
                        
                        return agent_id
                        
                    await asyncio.sleep(1)
                    
                # Spawn timeout
                logger.error(f"Agent {agent_id} spawn timeout")
                await self._cleanup_failed_agent(agent_id)
                return None
                
        except Exception as e:
            logger.error(f"Failed to spawn {agent_type} agent: {e}")
            return None
            
    async def terminate_agent(self, agent_id: str, force: bool = False) -> bool:
        """Terminate a specific agent"""
        try:
            with self.lifecycle_lock:
                if agent_id not in self.active_agents:
                    logger.warning(f"Agent not found for termination: {agent_id}")
                    return False
                    
                agent_info = self.active_agents[agent_id]
                process = self.agent_processes.get(agent_id)
                
                if not process:
                    logger.warning(f"No process found for agent: {agent_id}")
                    return False
                    
                # Update status
                agent_info.status = AgentStatus.TERMINATING
                
                # Cancel current task if any
                if agent_info.current_task and agent_info.current_task in self.active_tasks:
                    task_exec = self.active_tasks[agent_info.current_task]
                    task_exec.status = TaskStatus.CANCELLED
                    task_exec.end_time = time.time()
                    task_exec.error_message = f"Agent {agent_id} terminated"
                    
                # Terminate process
                try:
                    if force:
                        process.kill()
                    else:
                        process.terminate()
                        
                    # Wait for termination
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Agent {agent_id} did not terminate gracefully, killing")
                        process.kill()
                        process.wait(timeout=5)
                        
                except Exception as e:
                    logger.error(f"Error terminating process for agent {agent_id}: {e}")
                    
                # Cleanup
                agent_info.status = AgentStatus.TERMINATED
                
                # Unregister from synchronizer
                await self.synchronizer.unregister_agent_session(agent_id)
                
                # Notify other agents
                await self._broadcast_agent_event('agent_terminated', {
                    'agent_id': agent_id,
                    'agent_type': agent_info.agent_type
                })
                
                logger.info(f"✅ Agent {agent_id} terminated")
                return True
                
        except Exception as e:
            logger.error(f"Failed to terminate agent {agent_id}: {e}")
            return False
            
    async def delegate_task(self, task_def: TaskDefinition) -> bool:
        """Delegate a task to an appropriate agent"""
        try:
            # Find suitable agent
            agent_id = await self._find_suitable_agent(task_def)
            
            if not agent_id:
                # No suitable agent available, spawn one
                agent_id = await self.spawn_agent(
                    agent_type=task_def.target_agent_type,
                    task_description=task_def.description,
                    context=task_def.context,
                    priority=task_def.priority
                )
                
                if not agent_id:
                    logger.error(f"Failed to delegate task {task_def.task_id} - no agent available")
                    return False
                    
            # Create task execution
            task_exec = TaskExecution(
                task_id=task_def.task_id,
                assigned_agent=agent_id,
                status=TaskStatus.ASSIGNED,
                start_time=None,
                end_time=None,
                result=None,
                error_message=None,
                resource_usage={},
                intermediate_outputs=[]
            )
            
            # Register task
            self.pending_tasks[task_def.task_id] = task_def
            self.active_tasks[task_def.task_id] = task_exec
            
            # Update agent
            agent_info = self.active_agents[agent_id]
            agent_info.current_task = task_def.task_id
            agent_info.status = AgentStatus.BUSY
            
            # Send task to agent (via communication system)
            if self.comm_manager and agent_id in self.comm_manager.communicators:
                await self.comm_manager.communicators[agent_id].send_message(
                    agent_id,
                    MessageTypes.TASK_DELEGATE.value,
                    {
                        'task_definition': asdict(task_def),
                        'execution_id': task_def.task_id
                    }
                )
                
            task_exec.status = TaskStatus.IN_PROGRESS
            task_exec.start_time = time.time()
            
            logger.info(f"✅ Task {task_def.task_id} delegated to agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delegate task {task_def.task_id}: {e}")
            return False
            
    async def _find_suitable_agent(self, task_def: TaskDefinition) -> Optional[str]:
        """Find the most suitable agent for a task"""
        suitable_agents = []
        
        for agent_id, agent_info in self.active_agents.items():
            if (agent_info.status == AgentStatus.IDLE and
                agent_info.agent_type == task_def.target_agent_type and
                task_def.required_capabilities.issubset(agent_info.capabilities)):
                
                # Calculate suitability score
                score = self._calculate_agent_suitability(agent_info, task_def)
                suitable_agents.append((agent_id, score))
                
        if suitable_agents:
            # Return agent with highest suitability score
            suitable_agents.sort(key=lambda x: x[1], reverse=True)
            return suitable_agents[0][0]
            
        return None
        
    def _calculate_agent_suitability(self, agent_info: AgentInfo, task_def: TaskDefinition) -> float:
        """Calculate how suitable an agent is for a task"""
        score = 0.0
        
        # Base score for correct type and capabilities
        score += 1.0
        
        # Bonus for having completed similar tasks
        similar_tasks = len([t for t in agent_info.completed_tasks if task_def.task_type in t])
        score += min(similar_tasks * 0.1, 0.5)
        
        # Penalty for high resource usage
        cpu_usage = agent_info.resource_usage.get('cpu_percent', 0)
        memory_usage = agent_info.resource_usage.get('memory_percent', 0)
        score -= (cpu_usage + memory_usage) / 200  # Normalize to 0-1 range
        
        # Bonus for recent activity (freshly spawned agents might be faster)
        age_minutes = (time.time() - agent_info.spawn_time) / 60
        if age_minutes < 10:  # Recently spawned
            score += 0.2
        elif age_minutes > 60:  # Old agent might be sluggish
            score -= 0.1
            
        return max(score, 0.0)
        
    async def _check_agent_health(self, agent_id: str) -> bool:
        """Check if an agent is healthy and responsive"""
        try:
            if agent_id not in self.active_agents:
                return False
                
            agent_info = self.active_agents[agent_id]
            process = self.agent_processes.get(agent_id)
            
            if not process or process.poll() is not None:
                return False
                
            # Check heartbeat recency
            if time.time() - agent_info.last_heartbeat > 30:  # 30 seconds
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for agent {agent_id}: {e}")
            return False
            
    async def _update_agent_heartbeat(self, agent_id: str, status: str, resource_usage: Dict[str, float]):
        """Update agent heartbeat information"""
        if agent_id in self.active_agents:
            agent_info = self.active_agents[agent_id]
            agent_info.last_heartbeat = time.time()
            agent_info.resource_usage = resource_usage
            
            # Update status if provided
            try:
                agent_info.status = AgentStatus(status)
            except ValueError:
                pass  # Invalid status, ignore
                
    def _check_spawn_limits(self, agent_type: str) -> bool:
        """Check if agent spawn is within resource limits"""
        # Count agents of this type
        type_count = len([a for a in self.active_agents.values() 
                         if a.agent_type == agent_type and a.status != AgentStatus.TERMINATED])
        
        if type_count >= self.max_agents_per_type.get(agent_type, 2):
            return False
            
        # Check total agent count
        total_active = len([a for a in self.active_agents.values() 
                          if a.status != AgentStatus.TERMINATED])
        
        return total_active < self.max_total_agents
        
    def _get_agent_capabilities(self, agent_type: str) -> Set[str]:
        """Get capabilities for an agent type"""
        capabilities_map = {
            'claude': {'code-generation', 'analysis', 'debugging', 'documentation', 'refactoring'},
            'codex': {'code-completion', 'generation', 'optimization', 'translation'},
            'gemini': {'reasoning', 'analysis', 'research', 'summarization', 'explanation'}
        }
        return capabilities_map.get(agent_type, set())
        
    async def _cleanup_failed_agent(self, agent_id: str):
        """Clean up a failed agent"""
        try:
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
            if agent_id in self.agent_processes:
                process = self.agent_processes[agent_id]
                try:
                    process.kill()
                except:
                    pass
                del self.agent_processes[agent_id]
        except Exception as e:
            logger.error(f"Error cleaning up failed agent {agent_id}: {e}")
            
    async def _terminate_all_agents(self):
        """Terminate all managed agents"""
        agent_ids = list(self.active_agents.keys())
        for agent_id in agent_ids:
            await self.terminate_agent(agent_id, force=True)
            
    async def _broadcast_agent_event(self, event_type: str, event_data: Dict[str, Any]):
        """Broadcast agent lifecycle event"""
        if self.comm_manager:
            await self.comm_manager.broadcast_to_agents(
                MessageTypes.AGENT_STATUS.value,
                {
                    'event': event_type,
                    'data': event_data,
                    'timestamp': time.time()
                }
            )
            
    async def _agent_monitor_loop(self):
        """Background agent monitoring loop"""
        while self.running:
            try:
                current_time = time.time()
                failed_agents = []
                
                for agent_id, agent_info in self.active_agents.items():
                    if agent_info.status == AgentStatus.TERMINATED:
                        continue
                        
                    # Check if agent is still alive
                    if not await self._check_agent_health(agent_id):
                        logger.warning(f"Agent {agent_id} failed health check")
                        agent_info.status = AgentStatus.ERROR
                        failed_agents.append(agent_id)
                        continue
                        
                    # Update resource usage
                    process = self.agent_processes.get(agent_id)
                    if process:
                        try:
                            proc_info = psutil.Process(process.pid)
                            agent_info.resource_usage = {
                                'cpu_percent': proc_info.cpu_percent(),
                                'memory_percent': proc_info.memory_percent(),
                                'memory_mb': proc_info.memory_info().rss / 1024 / 1024
                            }
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            failed_agents.append(agent_id)
                            
                # Clean up failed agents
                for agent_id in failed_agents:
                    await self.terminate_agent(agent_id, force=True)
                    
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Agent monitor loop error: {e}")
                await asyncio.sleep(30)
                
    async def _task_scheduler_loop(self):
        """Background task scheduling loop"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check for timed out tasks
                timeout_tasks = []
                for task_id, task_exec in self.active_tasks.items():
                    if task_exec.status == TaskStatus.IN_PROGRESS:
                        task_def = self.pending_tasks.get(task_id)
                        if (task_def and task_exec.start_time and 
                            current_time - task_exec.start_time > task_def.timeout_seconds):
                            timeout_tasks.append(task_id)
                            
                # Handle timeouts
                for task_id in timeout_tasks:
                    task_exec = self.active_tasks[task_id]
                    task_exec.status = TaskStatus.FAILED
                    task_exec.end_time = current_time
                    task_exec.error_message = "Task timeout"
                    
                    # Free up the agent
                    agent_id = task_exec.assigned_agent
                    if agent_id in self.active_agents:
                        agent_info = self.active_agents[agent_id]
                        agent_info.current_task = None
                        agent_info.status = AgentStatus.IDLE
                        
                    logger.warning(f"Task {task_id} timed out")
                    
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task scheduler loop error: {e}")
                await asyncio.sleep(60)
                
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.running:
            try:
                current_time = time.time()
                
                # Move old completed tasks to history
                old_tasks = []
                for task_id, task_exec in self.completed_tasks.items():
                    if current_time - (task_exec.end_time or 0) > 3600:  # 1 hour
                        old_tasks.append(task_id)
                        
                for task_id in old_tasks:
                    del self.completed_tasks[task_id]
                    if task_id in self.pending_tasks:
                        del self.pending_tasks[task_id]
                        
                await asyncio.sleep(600)  # Cleanup every 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(600)
                
    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle statistics"""
        stats = {
            'running': self.running,
            'agents': {
                'total': len(self.active_agents),
                'by_type': {},
                'by_status': {},
                'resource_usage': {}
            },
            'tasks': {
                'pending': len(self.pending_tasks),
                'active': len(self.active_tasks),
                'completed': len(self.completed_tasks)
            },
            'limits': {
                'max_total_agents': self.max_total_agents,
                'max_agents_per_type': self.max_agents_per_type
            },
            'timestamp': time.time()
        }
        
        # Agent statistics
        for agent_info in self.active_agents.values():
            agent_type = agent_info.agent_type
            status = agent_info.status.value
            
            stats['agents']['by_type'][agent_type] = stats['agents']['by_type'].get(agent_type, 0) + 1
            stats['agents']['by_status'][status] = stats['agents']['by_status'].get(status, 0) + 1
            
        # Resource usage
        total_cpu = sum(a.resource_usage.get('cpu_percent', 0) for a in self.active_agents.values())
        total_memory = sum(a.resource_usage.get('memory_mb', 0) for a in self.active_agents.values())
        
        stats['agents']['resource_usage'] = {
            'total_cpu_percent': total_cpu,
            'total_memory_mb': total_memory,
            'average_cpu_percent': total_cpu / len(self.active_agents) if self.active_agents else 0,
            'average_memory_mb': total_memory / len(self.active_agents) if self.active_agents else 0
        }
        
        return stats


# Global lifecycle manager instance
_lifecycle_manager: Optional[AgentLifecycleManager] = None


def get_lifecycle_manager() -> AgentLifecycleManager:
    """Get singleton lifecycle manager instance"""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = AgentLifecycleManager()
    return _lifecycle_manager


async def ensure_lifecycle_management() -> bool:
    """Ensure the lifecycle management system is running"""
    try:
        manager = get_lifecycle_manager()
        if not manager.running:
            await manager.start_lifecycle_management()
        return True
    except Exception as e:
        logger.error(f"Failed to ensure lifecycle management: {e}")
        return False