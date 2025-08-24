"""
CLI Tool Discovery and Registration System
Automatic discovery, registration, and management of CLI tools
"""

import asyncio
import os
import json
import time
import logging
import signal
import psutil
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import tempfile
import hashlib
from functools import wraps, lru_cache

from .message_broker import MessageBroker, get_message_broker, CLINodeInfo
from .ultra_comm import UltraLowLatencyComm

logger = logging.getLogger(__name__)


@dataclass
class CLIToolDefinition:
    """Definition of a CLI tool and its capabilities"""
    name: str
    executable: str
    capabilities: List[str]
    discovery_methods: List[str]  # process, env_var, socket, binary_search
    env_vars: Dict[str, str]
    process_patterns: List[str]
    socket_paths: List[str]
    launch_command: Optional[str] = None
    config_file: Optional[str] = None
    version_command: Optional[str] = None
    health_check_command: Optional[str] = None


@dataclass
class DiscoveredCLI:
    """Information about a discovered CLI tool"""
    definition: CLIToolDefinition
    pid: Optional[int]
    socket_path: Optional[str]
    env_context: Dict[str, str]
    version: Optional[str]
    status: str  # 'running', 'available', 'installed', 'not_found'
    discovery_method: str
    last_seen: float
    working_directory: Optional[str] = None
    connection_established: bool = False


class CLIDiscoveryAgent:
    """Automatic CLI tool discovery and registration agent"""
    
    def __init__(self, broker: MessageBroker = None):
        self.broker = broker or get_message_broker()
        self.discovered_tools: Dict[str, DiscoveredCLI] = {}
        self.tool_definitions: Dict[str, CLIToolDefinition] = {}
        self.discovery_callbacks: Dict[str, List[Callable]] = {}
        
        # Discovery settings
        self.discovery_interval = 5.0  # seconds
        self.connection_timeout = 3.0
        self.max_retry_attempts = 3
        
        # Background tasks
        self.discovery_task = None
        self.monitoring_task = None
        self.running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Setup built-in tool definitions
        self._setup_builtin_definitions()
        
    def _setup_builtin_definitions(self):
        """Setup built-in CLI tool definitions"""
        
        # Claude Code
        self.add_tool_definition(CLIToolDefinition(
            name="claude-code",
            executable="claude-code",
            capabilities=[
                "code_generation", "file_editing", "project_analysis",
                "memory_storage", "memory_recall", "real_time_collaboration"
            ],
            discovery_methods=["process", "env_var", "socket"],
            env_vars={"CLAUDE_CODE_SESSION": "*"},
            process_patterns=["claude-code", "claude_code"],
            socket_paths=["/tmp/claude-code.sock", "/tmp/neuralsync2_claude-code*.sock"],
            launch_command="claude-code",
            version_command="claude-code --version",
            health_check_command="claude-code --health"
        ))
        
        # Codex CLI
        self.add_tool_definition(CLIToolDefinition(
            name="codex-cli", 
            executable="codex",
            capabilities=[
                "code_completion", "bug_fixing", "code_review",
                "memory_storage", "syntax_analysis"
            ],
            discovery_methods=["process", "env_var", "socket"],
            env_vars={"CODEX_SESSION": "*"},
            process_patterns=["codex", "codex-cli"],
            socket_paths=["/tmp/codex.sock", "/tmp/neuralsync2_codex*.sock"],
            launch_command="codex",
            version_command="codex --version"
        ))
        
        # Gemini CLI
        self.add_tool_definition(CLIToolDefinition(
            name="gemini",
            executable="gemini",
            capabilities=[
                "creative_writing", "research", "multi_modal",
                "memory_storage", "image_analysis"
            ],
            discovery_methods=["process", "env_var", "socket"],
            env_vars={"GEMINI_SESSION": "*"},
            process_patterns=["gemini", "bard-cli"],
            socket_paths=["/tmp/gemini.sock", "/tmp/neuralsync2_gemini*.sock"],
            launch_command="gemini",
            version_command="gemini --version"
        ))
        
        # Cursor
        self.add_tool_definition(CLIToolDefinition(
            name="cursor",
            executable="cursor",
            capabilities=[
                "editor_integration", "real_time_assistance",
                "code_editing", "project_navigation"
            ],
            discovery_methods=["process", "socket"],
            env_vars={"CURSOR_SESSION": "*"},
            process_patterns=["cursor", "cursor-server"],
            socket_paths=["/tmp/cursor.sock"],
            launch_command="cursor"
        ))
        
        # GitHub Copilot CLI
        self.add_tool_definition(CLIToolDefinition(
            name="copilot-cli",
            executable="copilot",
            capabilities=[
                "command_suggestions", "terminal_assistance",
                "git_integration", "shell_completion"
            ],
            discovery_methods=["process", "binary_search"],
            env_vars={"COPILOT_SESSION": "*"},
            process_patterns=["copilot", "gh-copilot"],
            socket_paths=[],
            launch_command="gh copilot"
        ))
        
        # Continue.dev
        self.add_tool_definition(CLIToolDefinition(
            name="continue",
            executable="continue",
            capabilities=[
                "code_assistance", "autocomplete", "chat_interface"
            ],
            discovery_methods=["process", "socket"],
            env_vars={"CONTINUE_SESSION": "*"},
            process_patterns=["continue", "continue-server"],
            socket_paths=["/tmp/continue.sock"],
            launch_command="continue"
        ))
        
    def add_tool_definition(self, definition: CLIToolDefinition):
        """Add CLI tool definition for discovery"""
        with self.lock:
            self.tool_definitions[definition.name] = definition
            
        logger.debug(f"Added CLI tool definition: {definition.name}")
        
    def register_discovery_callback(self, tool_name: str, callback: Callable[[DiscoveredCLI], None]):
        """Register callback for when specific CLI tool is discovered"""
        with self.lock:
            if tool_name not in self.discovery_callbacks:
                self.discovery_callbacks[tool_name] = []
            self.discovery_callbacks[tool_name].append(callback)
            
    async def start_discovery(self):
        """Start automatic CLI tool discovery"""
        if self.running:
            return
            
        logger.info("Starting CLI tool discovery agent")
        
        self.running = True
        
        # Start background tasks
        self.discovery_task = asyncio.create_task(self._discovery_loop())
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Initial discovery sweep
        await self._discover_all_tools()
        
        logger.info("CLI discovery agent started")
        
    async def stop_discovery(self):
        """Stop CLI tool discovery"""
        if not self.running:
            return
            
        logger.info("Stopping CLI discovery agent")
        
        self.running = False
        
        # Cancel background tasks
        if self.discovery_task:
            self.discovery_task.cancel()
        if self.monitoring_task:
            self.monitoring_task.cancel()
            
        # Disconnect from discovered tools
        await self._disconnect_all_tools()
        
        logger.info("CLI discovery agent stopped")
        
    async def _discovery_loop(self):
        """Main discovery loop"""
        while self.running:
            try:
                await self._discover_all_tools()
                await asyncio.sleep(self.discovery_interval)
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(5)
                
    async def _monitoring_loop(self):
        """Monitor discovered tools for status changes"""
        while self.running:
            try:
                await self._monitor_discovered_tools()
                await asyncio.sleep(2)  # More frequent monitoring
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
                
    async def _discover_all_tools(self):
        """Discover all configured CLI tools"""
        with self.lock:
            definitions = list(self.tool_definitions.values())
            
        for definition in definitions:
            try:
                discovered = await self._discover_tool(definition)
                
                if discovered:
                    with self.lock:
                        old_tool = self.discovered_tools.get(definition.name)
                        self.discovered_tools[definition.name] = discovered
                        
                    # Handle new discoveries
                    if not old_tool or old_tool.status != discovered.status:
                        await self._handle_tool_discovered(discovered)
                        
            except Exception as e:
                logger.error(f"Error discovering {definition.name}: {e}")
                
    async def _discover_tool(self, definition: CLIToolDefinition) -> Optional[DiscoveredCLI]:
        """Discover specific CLI tool using multiple methods"""
        
        for method in definition.discovery_methods:
            try:
                if method == "process":
                    result = await self._discover_by_process(definition)
                elif method == "env_var":
                    result = await self._discover_by_env_var(definition)
                elif method == "socket":
                    result = await self._discover_by_socket(definition)
                elif method == "binary_search":
                    result = await self._discover_by_binary_search(definition)
                else:
                    continue
                    
                if result:
                    result.discovery_method = method
                    return result
                    
            except Exception as e:
                logger.debug(f"Discovery method {method} failed for {definition.name}: {e}")
                continue
                
        return None
        
    async def _discover_by_process(self, definition: CLIToolDefinition) -> Optional[DiscoveredCLI]:
        """Discover CLI tool by running processes"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd', 'environ']):
                try:
                    proc_info = proc.info
                    
                    # Check process name
                    if any(pattern in proc_info['name'].lower() 
                          for pattern in definition.process_patterns):
                        
                        # Get version if possible
                        version = await self._get_tool_version(definition)
                        
                        return DiscoveredCLI(
                            definition=definition,
                            pid=proc_info['pid'],
                            socket_path=None,
                            env_context=proc_info.get('environ', {}),
                            version=version,
                            status="running",
                            discovery_method="process",
                            last_seen=time.time(),
                            working_directory=proc_info.get('cwd')
                        )
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.debug(f"Process discovery error for {definition.name}: {e}")
            
        return None
        
    async def _discover_by_env_var(self, definition: CLIToolDefinition) -> Optional[DiscoveredCLI]:
        """Discover CLI tool by environment variables"""
        try:
            for env_var, pattern in definition.env_vars.items():
                env_value = os.getenv(env_var)
                
                if env_value and (pattern == "*" or pattern in env_value):
                    version = await self._get_tool_version(definition)
                    
                    return DiscoveredCLI(
                        definition=definition,
                        pid=None,
                        socket_path=None,
                        env_context={env_var: env_value},
                        version=version,
                        status="available",
                        discovery_method="env_var",
                        last_seen=time.time()
                    )
                    
        except Exception as e:
            logger.debug(f"Environment discovery error for {definition.name}: {e}")
            
        return None
        
    async def _discover_by_socket(self, definition: CLIToolDefinition) -> Optional[DiscoveredCLI]:
        """Discover CLI tool by Unix socket"""
        import glob
        
        try:
            for socket_pattern in definition.socket_paths:
                # Handle glob patterns
                if '*' in socket_pattern:
                    socket_files = glob.glob(socket_pattern)
                else:
                    socket_files = [socket_pattern] if Path(socket_pattern).exists() else []
                    
                for socket_path in socket_files:
                    if Path(socket_path).exists():
                        # Try to connect to verify it's active
                        if await self._test_socket_connection(socket_path):
                            version = await self._get_tool_version(definition)
                            
                            return DiscoveredCLI(
                                definition=definition,
                                pid=None,
                                socket_path=socket_path,
                                env_context={},
                                version=version,
                                status="running",
                                discovery_method="socket",
                                last_seen=time.time()
                            )
                            
        except Exception as e:
            logger.debug(f"Socket discovery error for {definition.name}: {e}")
            
        return None
        
    async def _discover_by_binary_search(self, definition: CLIToolDefinition) -> Optional[DiscoveredCLI]:
        """Discover CLI tool by searching for binary"""
        try:
            # Check if executable is in PATH
            result = await asyncio.create_subprocess_shell(
                f"which {definition.executable}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            
            if result.returncode == 0 and stdout:
                executable_path = stdout.decode().strip()
                version = await self._get_tool_version(definition)
                
                return DiscoveredCLI(
                    definition=definition,
                    pid=None,
                    socket_path=None,
                    env_context={"executable_path": executable_path},
                    version=version,
                    status="installed",
                    discovery_method="binary_search",
                    last_seen=time.time()
                )
                
        except Exception as e:
            logger.debug(f"Binary search error for {definition.name}: {e}")
            
        return None
        
    async def _get_tool_version(self, definition: CLIToolDefinition) -> Optional[str]:
        """Get CLI tool version"""
        if not definition.version_command:
            return None
            
        try:
            result = await asyncio.create_subprocess_shell(
                definition.version_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=5
            )
            stdout, _ = await result.communicate()
            
            if result.returncode == 0:
                return stdout.decode().strip()
                
        except Exception as e:
            logger.debug(f"Version check error for {definition.name}: {e}")
            
        return None
        
    async def _test_socket_connection(self, socket_path: str) -> bool:
        """Test if socket is responsive"""
        try:
            import socket
            
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.connection_timeout)
            
            try:
                sock.connect(socket_path)
                return True
            finally:
                sock.close()
                
        except Exception:
            return False
            
    async def _handle_tool_discovered(self, discovered: DiscoveredCLI):
        """Handle newly discovered CLI tool"""
        logger.info(f"Discovered CLI tool: {discovered.definition.name} ({discovered.status})")
        
        # Try to establish communication
        if discovered.status in ["running", "available"]:
            await self._establish_communication(discovered)
            
        # Call registered callbacks
        with self.lock:
            callbacks = self.discovery_callbacks.get(discovered.definition.name, [])
            
        for callback in callbacks:
            try:
                callback(discovered)
            except Exception as e:
                logger.error(f"Discovery callback error: {e}")
                
    async def _establish_communication(self, discovered: DiscoveredCLI):
        """Establish communication with discovered CLI tool"""
        try:
            if discovered.socket_path:
                # Direct socket communication
                success = await self._connect_via_socket(discovered)
            else:
                # Try to create communication channel
                success = await self._create_communication_channel(discovered)
                
            discovered.connection_established = success
            
            if success:
                # Register with message broker
                node_id = f"{discovered.definition.name}_{discovered.pid or int(time.time())}"
                
                session_data = {
                    'pid': discovered.pid,
                    'working_directory': discovered.working_directory,
                    'version': discovered.version,
                    'discovery_method': discovered.discovery_method
                }
                
                await self.broker.register_node(
                    node_id=node_id,
                    tool_name=discovered.definition.name,
                    capabilities=discovered.definition.capabilities,
                    session_data=session_data
                )
                
                logger.info(f"Established communication with {discovered.definition.name}")
                
        except Exception as e:
            logger.error(f"Communication setup error for {discovered.definition.name}: {e}")
            
    async def _connect_via_socket(self, discovered: DiscoveredCLI) -> bool:
        """Connect to CLI tool via existing socket"""
        try:
            # Create communication instance
            comm = UltraLowLatencyComm(f"discovery_{discovered.definition.name}")
            
            # Test connection by sending ping
            test_message = json.dumps({"type": "ping", "timestamp": time.time()})
            
            success = await comm.send_message(
                discovered.definition.name,
                "ping",
                test_message.encode()
            )
            
            await comm.cleanup()
            return success
            
        except Exception as e:
            logger.debug(f"Socket connection test failed: {e}")
            return False
            
    async def _create_communication_channel(self, discovered: DiscoveredCLI) -> bool:
        """Create new communication channel with CLI tool"""
        try:
            # Check if tool supports NeuralSync integration
            if discovered.definition.launch_command:
                # Try to launch with NeuralSync wrapper
                wrapper_path = self._create_neuralsync_wrapper(discovered)
                
                if wrapper_path:
                    # Launch wrapped version
                    process = await asyncio.create_subprocess_exec(
                        wrapper_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    # Give it time to initialize
                    await asyncio.sleep(2)
                    
                    # Check if communication socket was created
                    socket_path = f"/tmp/neuralsync2_{discovered.definition.name}.sock"
                    if Path(socket_path).exists():
                        discovered.socket_path = socket_path
                        return True
                        
        except Exception as e:
            logger.debug(f"Communication channel creation failed: {e}")
            
        return False
        
    def _create_neuralsync_wrapper(self, discovered: DiscoveredCLI) -> Optional[str]:
        """Create NeuralSync wrapper script for CLI tool"""
        try:
            wrapper_content = f'''#!/usr/bin/env bash
# NeuralSync2 CLI Integration Wrapper for {discovered.definition.name}

# Set NeuralSync environment
export NEURALSYNC_CLI_TOOL="{discovered.definition.name}"
export NEURALSYNC_BROKER_SOCKET="/tmp/neuralsync2_broker.sock"
export NEURALSYNC_DISCOVERY_MODE="true"

# Set tool-specific environment
{chr(10).join(f'export {k}="{v}"' for k, v in discovered.env_context.items())}

# Launch original command with NeuralSync integration
exec "{discovered.definition.launch_command}" "$@"
'''
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(wrapper_content)
                wrapper_path = f.name
                
            # Make executable
            os.chmod(wrapper_path, 0o755)
            
            return wrapper_path
            
        except Exception as e:
            logger.error(f"Wrapper creation error: {e}")
            return None
            
    async def _monitor_discovered_tools(self):
        """Monitor status of discovered CLI tools"""
        with self.lock:
            tools_to_check = list(self.discovered_tools.items())
            
        for tool_name, discovered in tools_to_check:
            try:
                # Check if tool is still running/available
                current_status = await self._check_tool_status(discovered)
                
                if current_status != discovered.status:
                    logger.info(f"CLI tool status changed: {tool_name} {discovered.status} -> {current_status}")
                    
                    discovered.status = current_status
                    discovered.last_seen = time.time()
                    
                    # Handle status change
                    if current_status in ["running", "available"] and not discovered.connection_established:
                        await self._establish_communication(discovered)
                    elif current_status in ["not_found", "stopped"]:
                        await self._handle_tool_disconnected(discovered)
                        
            except Exception as e:
                logger.error(f"Tool monitoring error for {tool_name}: {e}")
                
    async def _check_tool_status(self, discovered: DiscoveredCLI) -> str:
        """Check current status of discovered CLI tool"""
        
        # Check by original discovery method
        if discovered.discovery_method == "process" and discovered.pid:
            try:
                proc = psutil.Process(discovered.pid)
                return "running" if proc.is_running() else "stopped"
            except psutil.NoSuchProcess:
                return "stopped"
                
        elif discovered.discovery_method == "socket" and discovered.socket_path:
            if Path(discovered.socket_path).exists():
                if await self._test_socket_connection(discovered.socket_path):
                    return "running"
                else:
                    return "available"  # Socket exists but not responsive
            else:
                return "stopped"
                
        elif discovered.discovery_method in ["env_var", "binary_search"]:
            # Re-run discovery to check availability
            rediscovered = await self._discover_tool(discovered.definition)
            return rediscovered.status if rediscovered else "not_found"
            
        return "unknown"
        
    async def _handle_tool_disconnected(self, discovered: DiscoveredCLI):
        """Handle CLI tool disconnection"""
        logger.info(f"CLI tool disconnected: {discovered.definition.name}")
        
        # Unregister from message broker
        try:
            node_id = f"{discovered.definition.name}_{discovered.pid or int(time.time())}"
            await self.broker.unregister_node(node_id)
        except Exception as e:
            logger.debug(f"Unregistration error: {e}")
            
        discovered.connection_established = False
        
    async def _disconnect_all_tools(self):
        """Disconnect from all discovered CLI tools"""
        with self.lock:
            tools = list(self.discovered_tools.values())
            
        for discovered in tools:
            if discovered.connection_established:
                await self._handle_tool_disconnected(discovered)
                
    def get_discovered_tools(self) -> Dict[str, DiscoveredCLI]:
        """Get all discovered CLI tools"""
        with self.lock:
            return dict(self.discovered_tools)
            
    def get_tool_status(self, tool_name: str) -> Optional[DiscoveredCLI]:
        """Get status of specific CLI tool"""
        with self.lock:
            return self.discovered_tools.get(tool_name)
            
    async def launch_tool(self, tool_name: str, args: List[str] = None) -> Optional[subprocess.Popen]:
        """Launch CLI tool with NeuralSync integration"""
        with self.lock:
            definition = self.tool_definitions.get(tool_name)
            
        if not definition or not definition.launch_command:
            logger.error(f"Cannot launch {tool_name}: no launch command defined")
            return None
            
        try:
            # Create wrapper
            wrapper_path = self._create_launch_wrapper(definition, args or [])
            
            # Launch process
            process = await asyncio.create_subprocess_exec(
                wrapper_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            logger.info(f"Launched CLI tool: {tool_name} (PID: {process.pid})")
            
            # Wait for initialization
            await asyncio.sleep(3)
            
            # Re-discover to pick up new instance
            await self._discover_all_tools()
            
            return process
            
        except Exception as e:
            logger.error(f"Launch error for {tool_name}: {e}")
            return None
            
    def _create_launch_wrapper(self, definition: CLIToolDefinition, args: List[str]) -> str:
        """Create launch wrapper with NeuralSync integration"""
        
        wrapper_content = f'''#!/usr/bin/env bash
# NeuralSync2 Launch Wrapper for {definition.name}

# Set NeuralSync environment
export NEURALSYNC_CLI_TOOL="{definition.name}"
export NEURALSYNC_BROKER_SOCKET="/tmp/neuralsync2_broker.sock"
export NEURALSYNC_AUTO_REGISTER="true"

# Launch with NeuralSync integration
exec "{definition.launch_command}" {' '.join(args)}
'''
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(wrapper_content)
            wrapper_path = f.name
            
        # Make executable
        os.chmod(wrapper_path, 0o755)
        
        return wrapper_path


# Global discovery agent instance
_global_discovery_agent: Optional[CLIDiscoveryAgent] = None

def get_discovery_agent(broker: MessageBroker = None) -> CLIDiscoveryAgent:
    """Get global CLI discovery agent"""
    global _global_discovery_agent
    if _global_discovery_agent is None:
        _global_discovery_agent = CLIDiscoveryAgent(broker)
    return _global_discovery_agent


# Convenience functions
async def discover_cli_tools() -> Dict[str, DiscoveredCLI]:
    """Discover all available CLI tools"""
    agent = get_discovery_agent()
    
    if not agent.running:
        await agent.start_discovery()
        await asyncio.sleep(2)  # Give time for initial discovery
        
    return agent.get_discovered_tools()


async def find_cli_tool(tool_name: str) -> Optional[DiscoveredCLI]:
    """Find specific CLI tool"""
    agent = get_discovery_agent()
    return agent.get_tool_status(tool_name)


async def launch_cli_tool(tool_name: str, args: List[str] = None) -> Optional[subprocess.Popen]:
    """Launch CLI tool with NeuralSync integration"""
    agent = get_discovery_agent()
    return await agent.launch_tool(tool_name, args)


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        agent = get_discovery_agent()
        
        try:
            await agent.start_discovery()
            
            print("Discovering CLI tools...")
            await asyncio.sleep(5)
            
            tools = agent.get_discovered_tools()
            print(f"\nDiscovered {len(tools)} CLI tools:")
            
            for name, discovered in tools.items():
                print(f"  {name}: {discovered.status} ({discovered.discovery_method})")
                if discovered.version:
                    print(f"    Version: {discovered.version}")
                if discovered.socket_path:
                    print(f"    Socket: {discovered.socket_path}")
                print(f"    Capabilities: {', '.join(discovered.definition.capabilities)}")
                print()
                
            print("Monitoring for changes... Press Ctrl+C to stop.")
            
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            await agent.stop_discovery()
            
    asyncio.run(main())