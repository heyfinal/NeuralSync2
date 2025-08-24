#!/usr/bin/env python3
"""
CLI Tool Integration for NeuralSync2
Enhanced compatibility for claude-code, gemini, codex-cli
"""

import asyncio
import json
import os
import sys
import subprocess
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import shutil
import shlex
import signal
import threading

from .ultra_comm import CliCommunicator
from .research_dedup import ResearchAPI
from .unleashed_mode import UnleashedAPI
from .unified_memory_api import UnifiedMemoryAPI
from .sync_manager import SyncAPI

@dataclass
class CLIToolConfig:
    """Configuration for CLI tool integration"""
    name: str
    executable_path: str
    wrapper_script: str
    capabilities: List[str]
    unleashed_enabled: bool = False
    memory_integration: bool = True
    research_dedup: bool = True
    sync_enabled: bool = True
    personality_sync: bool = True

class NeuralSyncCLIIntegration:
    """Main integration system for CLI tools"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or str(Path.home() / ".neuralsync" / "cli_integration.json")
        self.cli_tools: Dict[str, CLIToolConfig] = {}
        
        # Initialize subsystems
        self.memory_api = UnifiedMemoryAPI()
        self.research_api = ResearchAPI()
        self.unleashed_api = UnleashedAPI()
        self.sync_api = SyncAPI()
        self.communicator = None
        
        # Runtime state
        self.active_sessions: Dict[str, Dict] = {}
        self.personality_data = ""
        
        self._load_config()
        self._setup_cli_tools()
        
    def _load_config(self):
        """Load CLI integration configuration"""
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    
                for tool_data in data.get('cli_tools', []):
                    config = CLIToolConfig(**tool_data)
                    self.cli_tools[config.name] = config
                    
                self.personality_data = data.get('personality_data', "")
                    
            except Exception as e:
                print(f"❌ Error loading CLI integration config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default CLI tool configuration"""
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Default CLI tool configurations
        default_tools = [
            CLIToolConfig(
                name="claude-code",
                executable_path="claude",
                wrapper_script="claude_wrapper.py",
                capabilities=["code-generation", "analysis", "debugging", "refactoring"],
                unleashed_enabled=True,
                memory_integration=True,
                research_dedup=True,
                sync_enabled=True,
                personality_sync=True
            ),
            CLIToolConfig(
                name="gemini",
                executable_path="gemini",
                wrapper_script="gemini_wrapper.py", 
                capabilities=["multimodal", "reasoning", "creative-writing", "analysis"],
                unleashed_enabled=True,
                memory_integration=True,
                research_dedup=True,
                sync_enabled=True,
                personality_sync=True
            ),
            CLIToolConfig(
                name="codex-cli",
                executable_path="codex",
                wrapper_script="codex_wrapper.py",
                capabilities=["code-completion", "generation", "translation", "optimization"],
                unleashed_enabled=True,
                memory_integration=True,
                research_dedup=True,
                sync_enabled=True,
                personality_sync=True
            )
        ]
        
        for tool in default_tools:
            self.cli_tools[tool.name] = tool
        
        self._save_config()
    
    def _save_config(self):
        """Save configuration"""
        config = {
            'personality_data': self.personality_data,
            'cli_tools': [
                {
                    'name': tool.name,
                    'executable_path': tool.executable_path,
                    'wrapper_script': tool.wrapper_script,
                    'capabilities': tool.capabilities,
                    'unleashed_enabled': tool.unleashed_enabled,
                    'memory_integration': tool.memory_integration,
                    'research_dedup': tool.research_dedup,
                    'sync_enabled': tool.sync_enabled,
                    'personality_sync': tool.personality_sync
                }
                for tool in self.cli_tools.values()
            ]
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _setup_cli_tools(self):
        """Setup wrapper scripts for all CLI tools"""
        wrapper_dir = Path.home() / ".neuralsync" / "wrappers"
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        
        for tool in self.cli_tools.values():
            wrapper_path = wrapper_dir / tool.wrapper_script
            self._create_wrapper_script(tool, wrapper_path)
            
            # Make executable
            wrapper_path.chmod(0o755)
    
    def _create_wrapper_script(self, tool: CLIToolConfig, wrapper_path: Path):
        """Create enhanced wrapper script for CLI tool"""
        wrapper_content = f'''#!/usr/bin/env python3
"""
Enhanced NeuralSync2 wrapper for {tool.name}
Provides unified memory, research dedup, communication, and unleashed mode
"""

import asyncio
import sys
import os
import json
import time
import subprocess
from pathlib import Path

# Add NeuralSync to Python path
neuralsync_path = str(Path.home() / ".neuralsync")
sys.path.insert(0, neuralsync_path)

try:
    from neuralsync.cli_integration import EnhancedCLISession
except ImportError:
    print("❌ NeuralSync2 not found. Please ensure it's properly installed.")
    sys.exit(1)

async def main():
    session = EnhancedCLISession("{tool.name}")
    
    # Initialize session
    await session.start()
    
    try:
        # Execute original CLI with enhanced capabilities
        result = await session.execute_with_enhancements(sys.argv[1:])
        sys.exit(result)
    except KeyboardInterrupt:
        print("\\n🛑 Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {{e}}")
        sys.exit(1)
    finally:
        await session.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        wrapper_path.write_text(wrapper_content)
    
    async def start_integration(self):
        """Start the CLI integration system"""
        try:
            # Initialize memory system
            await self.memory_api.initialize()
            
            # Start sync services
            if any(tool.sync_enabled for tool in self.cli_tools.values()):
                await self.sync_api.start()
            
            # Setup communication system
            capabilities = set()
            for tool in self.cli_tools.values():
                capabilities.update(tool.capabilities)
            
            self.communicator = CliCommunicator("neuralsync-integration", capabilities)
            await self.communicator.connect()
            
            # Register message handlers
            self._register_message_handlers()
            
            print("🚀 NeuralSync2 CLI integration started")
            
        except Exception as e:
            print(f"❌ Failed to start CLI integration: {e}")
            raise
    
    def _register_message_handlers(self):
        """Register message handlers for inter-CLI communication"""
        
        async def handle_memory_query(data):
            """Handle memory query from other CLI tools"""
            query = data.get('query', '')
            scope = data.get('scope', 'any')
            
            results = await self.memory_api.recall(query, top_k=5, scope=scope)
            
            return {
                'status': 'success',
                'results': results,
                'query': query
            }
        
        async def handle_research_check(data):
            """Handle research deduplication check"""
            query = data.get('query', '')
            cli_tool = data.get('cli_tool', 'unknown')
            
            result = await self.research_api.before_api_call(query, cli_tool)
            
            return {
                'status': 'success',
                'found_similar': result.get('found_similar', False),
                'response': result.get('response', ''),
                'cost_saved': result.get('cost_saved', 0.0)
            }
        
        async def handle_unleashed_request(data):
            """Handle unleashed mode permission request"""
            cli_tool = data.get('cli_tool', '')
            command = data.get('command', '')
            justification = data.get('justification', '')
            
            approved = await self.unleashed_api.manager.request_permission(
                cli_tool, 'execute', command, justification
            )
            
            return {
                'status': 'success',
                'approved': approved
            }
        
        async def handle_personality_sync(data):
            """Handle personality synchronization"""
            return {
                'status': 'success',
                'personality': self.personality_data,
                'timestamp': time.time()
            }
        
        # Register handlers
        self.communicator.register_message_handler('memory_query', handle_memory_query)
        self.communicator.register_message_handler('research_check', handle_research_check)
        self.communicator.register_message_handler('unleashed_request', handle_unleashed_request)
        self.communicator.register_message_handler('personality_sync', handle_personality_sync)
    
    async def set_unified_personality(self, personality: str):
        """Set unified personality across all CLI tools"""
        self.personality_data = personality
        
        # Store in memory system
        await self.memory_api.remember(
            text=f"Personality: {personality}",
            kind="personality",
            scope="global",
            tool="neuralsync-integration"
        )
        
        # Sync across machines
        if self.sync_api:
            await self.sync_api.sync_memory_item('update', 'personality', {'text': personality})
        
        # Broadcast to all connected CLI tools
        if self.communicator:
            await self.communicator.broadcast_message('personality_update', {'personality': personality})
        
        self._save_config()
        print(f"🎭 Updated unified personality across all CLI tools")
    
    def create_enhanced_wrapper(self, cli_tool_name: str):
        """Create enhanced wrapper function for CLI tool"""
        if cli_tool_name not in self.cli_tools:
            raise ValueError(f"Unknown CLI tool: {cli_tool_name}")
        
        return EnhancedCLISession(cli_tool_name, self)


class EnhancedCLISession:
    """Enhanced session for individual CLI tool instances"""
    
    def __init__(self, cli_tool_name: str, integration: NeuralSyncCLIIntegration = None):
        self.cli_tool_name = cli_tool_name
        self.integration = integration or NeuralSyncCLIIntegration()
        self.config = self.integration.cli_tools.get(cli_tool_name)
        
        if not self.config:
            raise ValueError(f"CLI tool {cli_tool_name} not configured")
        
        # Session state
        self.session_id = f"{cli_tool_name}_{int(time.time())}"
        self.start_time = time.time()
        self.memory_context = []
        self.personality = ""
        
    async def start(self):
        """Start enhanced CLI session"""
        try:
            # Get personality if enabled
            if self.config.personality_sync:
                self.personality = self.integration.personality_data
                
                if not self.personality:
                    # Try to get from memory
                    results = await self.integration.memory_api.recall("personality", top_k=1, scope="global")
                    if results:
                        self.personality = results[0].get('text', '')
            
            # Load relevant memory context
            if self.config.memory_integration:
                # Get recent context for this tool
                context_results = await self.integration.memory_api.recall(
                    f"tool:{self.cli_tool_name}",
                    top_k=10,
                    scope="any",
                    tool=self.cli_tool_name
                )
                self.memory_context = context_results
            
            print(f"✅ Enhanced {self.cli_tool_name} session started")
            
        except Exception as e:
            print(f"❌ Failed to start enhanced session: {e}")
            raise
    
    async def execute_with_enhancements(self, args: List[str]) -> int:
        """Execute CLI tool with all enhancements"""
        try:
            # Build enhanced input with personality and memory context
            enhanced_input = await self._build_enhanced_input(args)
            
            # Check for research deduplication if enabled
            if self.config.research_dedup and len(args) > 0:
                query = " ".join(args)
                research_result = await self.integration.research_api.before_api_call(
                    query, self.cli_tool_name
                )
                
                if research_result.get('found_similar'):
                    print(f"🎯 Found similar research (saved ${research_result.get('cost_saved', 0):.4f})")
                    print(f"📝 Using cached response:")
                    print(research_result.get('response', ''))
                    return 0
            
            # Execute the original CLI tool
            cmd = [self.config.executable_path] + args
            
            if self.config.unleashed_enabled:
                # Request permission for execution
                command_str = " ".join(shlex.quote(arg) for arg in cmd)
                approved = await self.integration.unleashed_api.manager.request_permission(
                    self.cli_tool_name, 'execute', command_str, f"Executing {self.cli_tool_name}"
                )
                
                if not approved:
                    print("❌ Execution not approved by unleashed mode")
                    return 1
            
            # Run with enhanced environment
            env = os.environ.copy()
            if self.personality:
                env['NEURALSYNC_PERSONALITY'] = self.personality
            
            env['NEURALSYNC_TOOL'] = self.cli_tool_name
            env['NEURALSYNC_SESSION'] = self.session_id
            
            # Execute
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            stdout, stderr = process.communicate(input=enhanced_input)
            
            # Process output
            if stdout:
                print(stdout, end='')
            if stderr:
                print(stderr, end='', file=sys.stderr)
            
            # Store interaction in memory if enabled
            if self.config.memory_integration and args:
                await self._store_interaction(args, stdout, stderr)
            
            # Store research result if enabled
            if self.config.research_dedup and stdout and len(args) > 0:
                query = " ".join(args)
                await self.integration.research_api.after_api_call(
                    query, stdout, self.cli_tool_name, "unknown", 
                    len(stdout.split()), 0.01 * len(stdout.split())
                )
            
            return process.returncode
            
        except Exception as e:
            print(f"❌ Execution error: {e}")
            return 1
    
    async def _build_enhanced_input(self, args: List[str]) -> str:
        """Build enhanced input with personality and context"""
        parts = []
        
        # Add personality if available
        if self.personality:
            parts.append(f"Personality: {self.personality}")
            parts.append("")
        
        # Add relevant memory context
        if self.memory_context:
            parts.append("Relevant Context:")
            for i, context in enumerate(self.memory_context[:5], 1):
                parts.append(f"[C{i}] {context.get('text', '')}")
            parts.append("")
        
        # Add original user input
        if args:
            parts.append("User Input:")
            parts.append(" ".join(args))
        
        return "\\n".join(parts)
    
    async def _store_interaction(self, args: List[str], stdout: str, stderr: str):
        """Store CLI interaction in memory"""
        try:
            interaction_text = f"Command: {' '.join(args)}\\nOutput: {stdout[:500]}"
            if stderr:
                interaction_text += f"\\nErrors: {stderr[:200]}"
            
            await self.integration.memory_api.remember(
                text=interaction_text,
                kind="interaction",
                scope="tool",
                tool=self.cli_tool_name,
                tags=[self.cli_tool_name, "cli-interaction"],
                confidence=0.8
            )
            
        except Exception as e:
            print(f"❌ Failed to store interaction: {e}")
    
    async def communicate_with_tool(self, target_tool: str, message_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message to another CLI tool"""
        if self.integration.communicator:
            await self.integration.communicator.send_message(target_tool, message_type, data)
        return None
    
    async def cleanup(self):
        """Clean up session resources"""
        try:
            # Store session summary
            if self.config.memory_integration:
                session_duration = time.time() - self.start_time
                await self.integration.memory_api.remember(
                    text=f"Session completed. Duration: {session_duration:.1f}s",
                    kind="session",
                    scope="tool",
                    tool=self.cli_tool_name,
                    tags=[self.cli_tool_name, "session-end"]
                )
            
        except Exception as e:
            print(f"❌ Cleanup error: {e}")


# Utility functions for CLI tool setup
async def setup_claude_code_integration():
    """Setup integration for claude-code"""
    integration = NeuralSyncCLIIntegration()
    
    # Ensure claude-code is configured
    if 'claude-code' not in integration.cli_tools:
        config = CLIToolConfig(
            name="claude-code",
            executable_path="/opt/homebrew/bin/claude",  # Adjust path as needed
            wrapper_script="claude_wrapper.py",
            capabilities=["code-generation", "analysis", "debugging"],
            unleashed_enabled=True
        )
        integration.cli_tools['claude-code'] = config
        integration._save_config()
        integration._setup_cli_tools()
    
    await integration.start_integration()
    return integration

async def setup_gemini_integration():
    """Setup integration for gemini"""
    integration = NeuralSyncCLIIntegration()
    
    if 'gemini' not in integration.cli_tools:
        config = CLIToolConfig(
            name="gemini",
            executable_path="gemini",  # Adjust path as needed
            wrapper_script="gemini_wrapper.py", 
            capabilities=["multimodal", "reasoning", "analysis"],
            unleashed_enabled=True
        )
        integration.cli_tools['gemini'] = config
        integration._save_config()
        integration._setup_cli_tools()
    
    await integration.start_integration()
    return integration

async def setup_codex_cli_integration():
    """Setup integration for codex-cli"""
    integration = NeuralSyncCLIIntegration()
    
    if 'codex-cli' not in integration.cli_tools:
        config = CLIToolConfig(
            name="codex-cli",
            executable_path="codex",  # Adjust path as needed
            wrapper_script="codex_wrapper.py",
            capabilities=["code-completion", "generation", "optimization"],
            unleashed_enabled=True
        )
        integration.cli_tools['codex-cli'] = config
        integration._save_config()
        integration._setup_cli_tools()
    
    await integration.start_integration()
    return integration

# Main setup function
async def setup_all_cli_integrations():
    """Setup all CLI tool integrations"""
    integration = NeuralSyncCLIIntegration()
    await integration.start_integration()
    
    # Set a default unified personality
    await integration.set_unified_personality(
        "You are a helpful, knowledgeable AI assistant that provides clear, "
        "accurate, and actionable responses. You maintain context across "
        "different CLI tools and sessions, ensuring consistency in personality "
        "and memory retention."
    )
    
    print("🎯 All CLI integrations setup complete")
    return integration

if __name__ == "__main__":
    async def test_integration():
        integration = await setup_all_cli_integrations()
        
        # Test creating an enhanced session
        session = EnhancedCLISession("claude-code", integration)
        await session.start()
        
        print("✅ Integration test successful")
    
    asyncio.run(test_integration())