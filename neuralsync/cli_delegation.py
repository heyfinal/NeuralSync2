"""
CLI Tool Command Delegation System
Enables CLI tools to launch, control, and delegate commands to each other
"""

import asyncio
import json
import time
import logging
import threading
import hashlib
import uuid
from typing import Dict, Any, Optional, List, Callable, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
from functools import wraps
import subprocess
import tempfile
import os

from .message_broker import MessageBroker, get_message_broker
from .cli_discovery import CLIDiscoveryAgent, get_discovery_agent, DiscoveredCLI
from .ultra_comm import UltraLowLatencyComm, Message

logger = logging.getLogger(__name__)


class DelegationStatus(Enum):
    """Status of command delegation"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    RUNNING = "running"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class DelegationCommand:
    """Command to be delegated to another CLI tool"""
    command_id: str
    source_cli: str
    target_cli: str
    command: str
    parameters: Dict[str, Any]
    context: Dict[str, Any]
    timeout: float
    priority: int = 0
    stream_output: bool = False
    require_user_consent: bool = False
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class DelegationResult:
    """Result of command delegation"""
    command_id: str
    status: DelegationStatus
    output: Optional[str] = None
    error: Optional[str] = None
    exit_code: Optional[int] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    completed_at: float = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = time.time()


@dataclass
class DelegationStream:
    """Streaming output from delegation"""
    command_id: str
    stream_type: str  # 'stdout', 'stderr', 'status', 'progress'
    data: str
    timestamp: float
    sequence: int


class DelegationExecutor:
    """Executes delegated commands within a CLI tool"""
    
    def __init__(self, tool_name: str, broker: MessageBroker = None):
        self.tool_name = tool_name
        self.broker = broker or get_message_broker()
        
        self.command_handlers: Dict[str, Callable] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.execution_history: deque = deque(maxlen=1000)
        
        # Capability configuration
        self.max_concurrent_executions = 5
        self.allowed_commands: Set[str] = set()
        self.restricted_commands: Set[str] = {"rm", "rmdir", "sudo", "su", "chmod", "chown"}
        self.require_consent_commands: Set[str] = {"git", "npm", "pip", "docker"}
        
        # Security settings
        self.sandbox_mode = True
        self.allowed_directories: List[str] = []
        self.max_execution_time = 300.0  # 5 minutes default
        
        # Statistics
        self.stats = {
            'commands_executed': 0,
            'commands_failed': 0,
            'avg_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        
        # Setup message handlers
        self._setup_message_handlers()
        
    def _setup_message_handlers(self):
        """Setup message handlers for delegation"""
        self.broker.comm.register_handler("execute_delegation", self._handle_execute_delegation)
        self.broker.comm.register_handler("cancel_delegation", self._handle_cancel_delegation)
        self.broker.comm.register_handler("delegation_status_request", self._handle_status_request)
        
    def register_command_handler(self, command: str, handler: Callable):
        """Register custom command handler"""
        self.command_handlers[command] = handler
        self.allowed_commands.add(command)
        logger.debug(f"Registered command handler: {command}")
        
    def configure_security(self,
                          sandbox_mode: bool = True,
                          allowed_directories: List[str] = None,
                          max_execution_time: float = 300.0,
                          allowed_commands: List[str] = None,
                          restricted_commands: List[str] = None):
        """Configure security settings"""
        
        self.sandbox_mode = sandbox_mode
        self.allowed_directories = allowed_directories or []
        self.max_execution_time = max_execution_time
        
        if allowed_commands:
            self.allowed_commands.update(allowed_commands)
        if restricted_commands:
            self.restricted_commands.update(restricted_commands)
            
        logger.info(f"Security configured for {self.tool_name}: sandbox={sandbox_mode}")
        
    async def _handle_execute_delegation(self, message: Message):
        """Handle delegation execution request"""
        try:
            data = json.loads(message.payload.decode())
            command = DelegationCommand(**data)
            
            # Validate and execute
            if await self._validate_delegation(command):
                # Create execution task
                task = asyncio.create_task(self._execute_delegation(command))
                self.active_executions[command.command_id] = task
                
                # Send acceptance
                await self._send_delegation_response(
                    command.command_id,
                    DelegationStatus.ACCEPTED,
                    message.sender
                )
            else:
                # Send rejection
                await self._send_delegation_response(
                    command.command_id,
                    DelegationStatus.FAILED,
                    message.sender,
                    error="Command validation failed"
                )
                
        except Exception as e:
            logger.error(f"Delegation execution error: {e}")
            
    async def _handle_cancel_delegation(self, message: Message):
        """Handle delegation cancellation request"""
        try:
            data = json.loads(message.payload.decode())
            command_id = data.get('command_id')
            
            if command_id in self.active_executions:
                self.active_executions[command_id].cancel()
                del self.active_executions[command_id]
                
                await self._send_delegation_response(
                    command_id,
                    DelegationStatus.CANCELLED,
                    message.sender
                )
                
        except Exception as e:
            logger.error(f"Delegation cancellation error: {e}")
            
    async def _handle_status_request(self, message: Message):
        """Handle delegation status request"""
        try:
            data = json.loads(message.payload.decode())
            command_id = data.get('command_id')
            
            status = DelegationStatus.PENDING
            if command_id in self.active_executions:
                task = self.active_executions[command_id]
                if task.done():
                    status = DelegationStatus.COMPLETED if not task.cancelled() else DelegationStatus.CANCELLED
                else:
                    status = DelegationStatus.RUNNING
                    
            response = {
                'command_id': command_id,
                'status': status.value,
                'active_executions': len(self.active_executions),
                'stats': self.stats
            }
            
            await self.broker.comm.send_message(
                message.sender,
                "delegation_status_response",
                json.dumps(response).encode()
            )
            
        except Exception as e:
            logger.error(f"Status request error: {e}")
            
    async def _validate_delegation(self, command: DelegationCommand) -> bool:
        """Validate delegation command for security and capability"""
        
        # Check if too many concurrent executions
        if len(self.active_executions) >= self.max_concurrent_executions:
            logger.warning(f"Max concurrent executions reached: {self.max_concurrent_executions}")
            return False
            
        # Check command restrictions
        base_command = command.command.split()[0] if command.command else ""
        
        if base_command in self.restricted_commands:
            logger.warning(f"Restricted command attempted: {base_command}")
            return False
            
        if self.allowed_commands and base_command not in self.allowed_commands:
            logger.warning(f"Command not in allowed list: {base_command}")
            return False
            
        # Check execution time limit
        if command.timeout > self.max_execution_time:
            logger.warning(f"Command timeout exceeds limit: {command.timeout} > {self.max_execution_time}")
            return False
            
        # Sandbox validation
        if self.sandbox_mode:
            if not await self._validate_sandbox_command(command):
                return False
                
        return True
        
    async def _validate_sandbox_command(self, command: DelegationCommand) -> bool:
        """Validate command for sandbox execution"""
        
        # Check working directory restrictions
        working_dir = command.context.get('working_directory', os.getcwd())
        
        if self.allowed_directories:
            allowed = any(
                working_dir.startswith(allowed_dir)
                for allowed_dir in self.allowed_directories
            )
            if not allowed:
                logger.warning(f"Working directory not allowed: {working_dir}")
                return False
                
        # Additional sandbox checks can be added here
        return True
        
    async def _execute_delegation(self, command: DelegationCommand):
        """Execute delegated command"""
        start_time = time.time()
        
        try:
            # Send running status
            await self._send_delegation_response(
                command.command_id,
                DelegationStatus.RUNNING,
                command.source_cli
            )
            
            # Check for custom handler first
            if command.command in self.command_handlers:
                result = await self._execute_custom_handler(command)
            else:
                result = await self._execute_shell_command(command)
                
            # Update statistics
            execution_time = time.time() - start_time
            self.stats['commands_executed'] += 1
            self.stats['total_execution_time'] += execution_time
            self.stats['avg_execution_time'] = (
                self.stats['total_execution_time'] / self.stats['commands_executed']
            )
            
            result.execution_time = execution_time
            
            # Send final result
            await self._send_delegation_response(
                command.command_id,
                result.status,
                command.source_cli,
                output=result.output,
                error=result.error,
                exit_code=result.exit_code,
                execution_time=result.execution_time,
                metadata=result.metadata
            )
            
            # Store in history
            self.execution_history.append({
                'command': command,
                'result': result,
                'timestamp': time.time()
            })
            
        except asyncio.CancelledError:
            # Command was cancelled
            await self._send_delegation_response(
                command.command_id,
                DelegationStatus.CANCELLED,
                command.source_cli
            )
            
        except Exception as e:
            # Execution failed
            logger.error(f"Command execution failed: {e}")
            
            self.stats['commands_failed'] += 1
            
            await self._send_delegation_response(
                command.command_id,
                DelegationStatus.FAILED,
                command.source_cli,
                error=str(e)
            )
            
        finally:
            # Cleanup
            if command.command_id in self.active_executions:
                del self.active_executions[command.command_id]
                
    async def _execute_custom_handler(self, command: DelegationCommand) -> DelegationResult:
        """Execute command using custom handler"""
        try:
            handler = self.command_handlers[command.command]
            
            if asyncio.iscoroutinefunction(handler):
                result = await handler(command)
            else:
                result = handler(command)
                
            if isinstance(result, DelegationResult):
                return result
            else:
                # Wrap simple results
                return DelegationResult(
                    command_id=command.command_id,
                    status=DelegationStatus.COMPLETED,
                    output=str(result) if result is not None else None
                )
                
        except Exception as e:
            return DelegationResult(
                command_id=command.command_id,
                status=DelegationStatus.FAILED,
                error=str(e)
            )
            
    async def _execute_shell_command(self, command: DelegationCommand) -> DelegationResult:
        """Execute shell command"""
        try:
            # Setup environment
            env = dict(os.environ)
            env.update(command.context.get('environment', {}))
            
            # Setup working directory
            cwd = command.context.get('working_directory', os.getcwd())
            
            # Create subprocess
            if command.stream_output:
                return await self._execute_streaming_command(command, env, cwd)
            else:
                return await self._execute_buffered_command(command, env, cwd)
                
        except Exception as e:
            return DelegationResult(
                command_id=command.command_id,
                status=DelegationStatus.FAILED,
                error=str(e)
            )
            
    async def _execute_buffered_command(self, command: DelegationCommand, env: Dict[str, str], cwd: str) -> DelegationResult:
        """Execute command with buffered output"""
        
        process = await asyncio.create_subprocess_shell(
            command.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=command.timeout
            )
            
            return DelegationResult(
                command_id=command.command_id,
                status=DelegationStatus.COMPLETED if process.returncode == 0 else DelegationStatus.FAILED,
                output=stdout.decode() if stdout else None,
                error=stderr.decode() if stderr else None,
                exit_code=process.returncode
            )
            
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            
            return DelegationResult(
                command_id=command.command_id,
                status=DelegationStatus.TIMEOUT,
                error="Command execution timed out"
            )
            
    async def _execute_streaming_command(self, command: DelegationCommand, env: Dict[str, str], cwd: str) -> DelegationResult:
        """Execute command with streaming output"""
        
        process = await asyncio.create_subprocess_shell(
            command.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd
        )
        
        # Send streaming status
        await self._send_delegation_response(
            command.command_id,
            DelegationStatus.STREAMING,
            command.source_cli
        )
        
        # Stream output
        stdout_task = asyncio.create_task(
            self._stream_output(command.command_id, command.source_cli, process.stdout, 'stdout')
        )
        stderr_task = asyncio.create_task(
            self._stream_output(command.command_id, command.source_cli, process.stderr, 'stderr')
        )
        
        try:
            # Wait for completion with timeout
            await asyncio.wait_for(process.wait(), timeout=command.timeout)
            
            # Wait for streaming to complete
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            
            return DelegationResult(
                command_id=command.command_id,
                status=DelegationStatus.COMPLETED if process.returncode == 0 else DelegationStatus.FAILED,
                exit_code=process.returncode
            )
            
        except asyncio.TimeoutError:
            process.kill()
            stdout_task.cancel()
            stderr_task.cancel()
            
            return DelegationResult(
                command_id=command.command_id,
                status=DelegationStatus.TIMEOUT,
                error="Command execution timed out"
            )
            
    async def _stream_output(self, command_id: str, target: str, stream: asyncio.StreamReader, stream_type: str):
        """Stream command output in real-time"""
        sequence = 0
        
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                    
                # Send stream data
                stream_data = DelegationStream(
                    command_id=command_id,
                    stream_type=stream_type,
                    data=line.decode().rstrip('\n'),
                    timestamp=time.time(),
                    sequence=sequence
                )
                
                await self.broker.comm.send_message(
                    target,
                    "delegation_stream",
                    json.dumps(asdict(stream_data)).encode()
                )
                
                sequence += 1
                
        except Exception as e:
            logger.error(f"Stream output error: {e}")
            
    async def _send_delegation_response(self,
                                      command_id: str,
                                      status: DelegationStatus,
                                      target: str,
                                      output: str = None,
                                      error: str = None,
                                      exit_code: int = None,
                                      execution_time: float = None,
                                      metadata: Dict[str, Any] = None):
        """Send delegation response"""
        
        response = {
            'command_id': command_id,
            'status': status.value,
            'output': output,
            'error': error,
            'exit_code': exit_code,
            'execution_time': execution_time,
            'metadata': metadata or {},
            'executor': self.tool_name,
            'timestamp': time.time()
        }
        
        await self.broker.comm.send_message(
            target,
            "delegation_response",
            json.dumps(response).encode()
        )


class DelegationClient:
    """Client for delegating commands to other CLI tools"""
    
    def __init__(self, tool_name: str, broker: MessageBroker = None, discovery: CLIDiscoveryAgent = None):
        self.tool_name = tool_name
        self.broker = broker or get_message_broker()
        self.discovery = discovery or get_discovery_agent()
        
        self.pending_commands: Dict[str, DelegationCommand] = {}
        self.command_results: Dict[str, DelegationResult] = {}
        self.stream_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'commands_sent': 0,
            'commands_completed': 0,
            'commands_failed': 0,
            'avg_response_time': 0.0
        }
        
        # Setup message handlers
        self._setup_message_handlers()
        
    def _setup_message_handlers(self):
        """Setup message handlers for delegation responses"""
        self.broker.comm.register_handler("delegation_response", self._handle_delegation_response)
        self.broker.comm.register_handler("delegation_stream", self._handle_delegation_stream)
        self.broker.comm.register_handler("delegation_timeout", self._handle_delegation_timeout)
        
    async def delegate_command(self,
                              target_cli: str,
                              command: str,
                              parameters: Dict[str, Any] = None,
                              context: Dict[str, Any] = None,
                              timeout: float = 30.0,
                              stream_output: bool = False,
                              priority: int = 0) -> str:
        """Delegate command to another CLI tool"""
        
        command_id = str(uuid.uuid4())
        
        delegation = DelegationCommand(
            command_id=command_id,
            source_cli=self.tool_name,
            target_cli=target_cli,
            command=command,
            parameters=parameters or {},
            context=context or {},
            timeout=timeout,
            stream_output=stream_output,
            priority=priority
        )
        
        # Store pending command
        self.pending_commands[command_id] = delegation
        
        # Send delegation request
        success = await self.broker.comm.send_message(
            target_cli,
            "execute_delegation",
            json.dumps(asdict(delegation)).encode()
        )
        
        if not success:
            del self.pending_commands[command_id]
            raise RuntimeError(f"Failed to send delegation to {target_cli}")
            
        self.stats['commands_sent'] += 1
        
        logger.info(f"Delegated command to {target_cli}: {command}")
        return command_id
        
    async def wait_for_completion(self, command_id: str, timeout: float = None) -> DelegationResult:
        """Wait for delegation completion"""
        
        if command_id not in self.pending_commands:
            raise ValueError(f"Unknown command ID: {command_id}")
            
        command = self.pending_commands[command_id]
        effective_timeout = timeout or command.timeout
        
        start_time = time.time()
        
        while time.time() - start_time < effective_timeout:
            if command_id in self.command_results:
                result = self.command_results[command_id]
                
                # Cleanup
                del self.pending_commands[command_id]
                del self.command_results[command_id]
                
                # Update stats
                if result.status == DelegationStatus.COMPLETED:
                    self.stats['commands_completed'] += 1
                else:
                    self.stats['commands_failed'] += 1
                    
                response_time = time.time() - command.created_at
                self.stats['avg_response_time'] = (
                    self.stats['avg_response_time'] * 0.9 + response_time * 0.1
                )
                
                return result
                
            await asyncio.sleep(0.1)
            
        # Timeout
        await self._cancel_command(command_id)
        
        raise asyncio.TimeoutError(f"Command {command_id} timed out")
        
    async def _cancel_command(self, command_id: str):
        """Cancel pending command"""
        
        if command_id not in self.pending_commands:
            return
            
        command = self.pending_commands[command_id]
        
        cancel_request = {
            'command_id': command_id,
            'source_cli': self.tool_name
        }
        
        await self.broker.comm.send_message(
            command.target_cli,
            "cancel_delegation",
            json.dumps(cancel_request).encode()
        )
        
        # Cleanup
        del self.pending_commands[command_id]
        
    def register_stream_handler(self, command_id: str, handler: Callable[[DelegationStream], None]):
        """Register handler for streaming output"""
        self.stream_handlers[command_id].append(handler)
        
    async def _handle_delegation_response(self, message: Message):
        """Handle delegation response"""
        try:
            data = json.loads(message.payload.decode())
            command_id = data.get('command_id')
            
            if command_id not in self.pending_commands:
                return
                
            result = DelegationResult(
                command_id=command_id,
                status=DelegationStatus(data.get('status')),
                output=data.get('output'),
                error=data.get('error'),
                exit_code=data.get('exit_code'),
                execution_time=data.get('execution_time'),
                metadata=data.get('metadata', {}),
                completed_at=data.get('timestamp', time.time())
            )
            
            self.command_results[command_id] = result
            
        except Exception as e:
            logger.error(f"Delegation response error: {e}")
            
    async def _handle_delegation_stream(self, message: Message):
        """Handle streaming delegation output"""
        try:
            data = json.loads(message.payload.decode())
            stream = DelegationStream(**data)
            
            # Call registered handlers
            for handler in self.stream_handlers.get(stream.command_id, []):
                try:
                    handler(stream)
                except Exception as e:
                    logger.error(f"Stream handler error: {e}")
                    
        except Exception as e:
            logger.error(f"Delegation stream error: {e}")
            
    async def _handle_delegation_timeout(self, message: Message):
        """Handle delegation timeout"""
        try:
            data = json.loads(message.payload.decode())
            command_id = data.get('command_id')
            
            if command_id in self.pending_commands:
                result = DelegationResult(
                    command_id=command_id,
                    status=DelegationStatus.TIMEOUT,
                    error="Command timed out"
                )
                self.command_results[command_id] = result
                
        except Exception as e:
            logger.error(f"Delegation timeout error: {e}")


# High-level convenience functions
async def ask_cli_tool(target_cli: str,
                      question: str,
                      context: Dict[str, Any] = None,
                      timeout: float = 30.0) -> str:
    """Ask another CLI tool a question"""
    
    client = DelegationClient("neuralsync_query")
    
    command_id = await client.delegate_command(
        target_cli=target_cli,
        command="query",
        parameters={'question': question},
        context=context or {},
        timeout=timeout
    )
    
    result = await client.wait_for_completion(command_id)
    
    if result.status == DelegationStatus.COMPLETED:
        return result.output or ""
    else:
        raise RuntimeError(f"Query failed: {result.error}")


async def delegate_code_review(target_cli: str,
                             file_path: str,
                             review_type: str = "general",
                             timeout: float = 60.0) -> str:
    """Delegate code review to another CLI tool"""
    
    client = DelegationClient("neuralsync_review")
    
    command_id = await client.delegate_command(
        target_cli=target_cli,
        command="review_code",
        parameters={
            'file_path': file_path,
            'review_type': review_type
        },
        timeout=timeout
    )
    
    result = await client.wait_for_completion(command_id)
    
    if result.status == DelegationStatus.COMPLETED:
        return result.output or ""
    else:
        raise RuntimeError(f"Code review failed: {result.error}")


async def execute_on_cli(target_cli: str,
                        command: str,
                        working_dir: str = None,
                        timeout: float = 30.0,
                        stream: bool = False) -> DelegationResult:
    """Execute shell command on another CLI tool"""
    
    client = DelegationClient("neuralsync_exec")
    
    context = {}
    if working_dir:
        context['working_directory'] = working_dir
        
    command_id = await client.delegate_command(
        target_cli=target_cli,
        command=command,
        context=context,
        timeout=timeout,
        stream_output=stream
    )
    
    return await client.wait_for_completion(command_id)


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Setup executor for this CLI tool
        executor = DelegationExecutor("test-cli")
        
        # Register custom command handler
        async def handle_greeting(command: DelegationCommand) -> DelegationResult:
            name = command.parameters.get('name', 'World')
            return DelegationResult(
                command_id=command.command_id,
                status=DelegationStatus.COMPLETED,
                output=f"Hello, {name}!"
            )
            
        executor.register_command_handler("greet", handle_greeting)
        
        print("Delegation system ready. Example usage:")
        print("  await ask_cli_tool('claude-code', 'What files are in the current directory?')")
        print("  await delegate_code_review('codex-cli', 'main.py')")
        print("  await execute_on_cli('gemini', 'ls -la')")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    asyncio.run(main())