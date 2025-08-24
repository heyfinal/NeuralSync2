#!/usr/bin/env python3
"""
Unleashed Mode for NeuralSync2
Permission bypassing system for enhanced CLI tool capabilities
"""

import asyncio
import os
import json
import subprocess
import sys
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
import hashlib
import tempfile
import shlex

@dataclass
class UnleashedConfig:
    """Configuration for unleashed mode"""
    enabled: bool = False
    allowed_cli_tools: Set[str] = None
    security_level: str = "medium"  # low, medium, high
    audit_enabled: bool = True
    auto_approve_safe_operations: bool = True
    elevated_permissions: bool = False
    
    def __post_init__(self):
        if self.allowed_cli_tools is None:
            self.allowed_cli_tools = {"claude-code", "gemini", "codex-cli"}

@dataclass
class PermissionRequest:
    """Permission request for CLI tools"""
    request_id: str
    cli_tool: str
    operation: str
    command: str
    risk_level: str
    justification: str
    timestamp: float
    approved: Optional[bool] = None

class UnleashedModeManager:
    """Manages unleashed mode permissions and capabilities"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or str(Path.home() / ".neuralsync" / "unleashed_config.json")
        self.config = UnleashedConfig()
        self.audit_log_path = str(Path.home() / ".neuralsync" / "unleashed_audit.log")
        
        # Command risk classification
        self.risk_patterns = {
            'safe': [
                r'^ls\b', r'^cat\b', r'^grep\b', r'^find\b', r'^head\b', r'^tail\b',
                r'^echo\b', r'^which\b', r'^whoami\b', r'^pwd\b', r'^date\b',
                r'^python3?\b.*\.py$', r'^node\b.*\.js$', r'^ruby\b.*\.rb$'
            ],
            'medium': [
                r'^git\b', r'^npm\b', r'^pip\b', r'^curl\b', r'^wget\b',
                r'^mkdir\b', r'^cp\b', r'^mv\b', r'^touch\b', r'^chmod\b',
                r'^kill\b', r'^pkill\b', r'^ps\b', r'^top\b', r'^htop\b'
            ],
            'high': [
                r'^sudo\b', r'^su\b', r'^rm\b.*-r', r'^dd\b', r'^mkfs\b',
                r'^fdisk\b', r'^mount\b', r'^umount\b', r'^iptables\b',
                r'^systemctl\b', r'^service\b', r'^crontab\b'
            ]
        }
        
        self.pending_requests: Dict[str, PermissionRequest] = {}
        self.approved_commands: Set[str] = set()
        
        self._load_config()
        
        # Setup signal handler for emergency disable
        signal.signal(signal.SIGUSR1, self._emergency_disable)
    
    def _load_config(self):
        """Load unleashed mode configuration"""
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    
                # Convert allowed_cli_tools back to set
                if 'allowed_cli_tools' in data:
                    data['allowed_cli_tools'] = set(data['allowed_cli_tools'])
                    
                self.config = UnleashedConfig(**data)
                
            except Exception as e:
                print(f"❌ Error loading unleashed config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default unleashed configuration"""
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        
        config_data = asdict(self.config)
        config_data['allowed_cli_tools'] = list(self.config.allowed_cli_tools)
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def _save_config(self):
        """Save configuration to disk"""
        config_data = asdict(self.config)
        config_data['allowed_cli_tools'] = list(self.config.allowed_cli_tools)
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def _emergency_disable(self, signum, frame):
        """Emergency disable handler (SIGUSR1)"""
        self.config.enabled = False
        self._save_config()
        self._audit_log("EMERGENCY_DISABLE", "Unleashed mode disabled via signal", "system")
        print("🚨 Unleashed mode EMERGENCY DISABLED")
    
    def _audit_log(self, operation: str, details: str, cli_tool: str = "unknown"):
        """Log audit event"""
        if not self.config.audit_enabled:
            return
            
        audit_entry = {
            'timestamp': time.time(),
            'operation': operation,
            'details': details,
            'cli_tool': cli_tool,
            'user': os.getenv('USER', 'unknown'),
            'pid': os.getpid()
        }
        
        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            print(f"❌ Audit log error: {e}")
    
    def _classify_risk(self, command: str) -> str:
        """Classify command risk level"""
        import re
        
        for risk_level, patterns in self.risk_patterns.items():
            for pattern in patterns:
                if re.match(pattern, command, re.IGNORECASE):
                    return risk_level
        
        # Default to medium risk for unknown commands
        return 'medium'
    
    def _is_safe_operation(self, command: str) -> bool:
        """Check if operation is considered safe"""
        risk = self._classify_risk(command)
        return risk == 'safe'
    
    async def request_permission(
        self, 
        cli_tool: str, 
        operation: str, 
        command: str, 
        justification: str = ""
    ) -> bool:
        \"\"\"Request permission for CLI tool operation\"\"\"
        
        if not self.config.enabled:
            self._audit_log("PERMISSION_DENIED", f"Unleashed mode disabled: {command}", cli_tool)
            return False
        
        if cli_tool not in self.config.allowed_cli_tools:
            self._audit_log("PERMISSION_DENIED", f"CLI tool not allowed: {cli_tool}", cli_tool)
            return False
        
        risk_level = self._classify_risk(command)
        request_id = hashlib.md5(f"{cli_tool}_{operation}_{command}_{time.time()}".encode()).hexdigest()[:8]
        
        request = PermissionRequest(
            request_id=request_id,
            cli_tool=cli_tool,
            operation=operation,
            command=command,
            risk_level=risk_level,
            justification=justification,
            timestamp=time.time()
        )
        
        # Auto-approve based on security level and risk
        if self._should_auto_approve(request):
            request.approved = True
            self._audit_log("AUTO_APPROVED", f"{operation}: {command}", cli_tool)
            print(f"✅ Auto-approved {risk_level} risk operation for {cli_tool}")
            return True
        
        # Store pending request
        self.pending_requests[request_id] = request
        
        # Interactive approval for medium/high risk
        if risk_level in ['medium', 'high']:
            approval = await self._request_interactive_approval(request)
            request.approved = approval
            
            if approval:
                self._audit_log("USER_APPROVED", f"{operation}: {command}", cli_tool)
                self.approved_commands.add(command)
                print(f"✅ Approved {risk_level} risk operation for {cli_tool}")
            else:
                self._audit_log("USER_DENIED", f"{operation}: {command}", cli_tool)
                print(f"❌ Denied {risk_level} risk operation for {cli_tool}")
            
            return approval
        
        # Default deny for unknown cases
        self._audit_log("DEFAULT_DENIED", f"{operation}: {command}", cli_tool)
        return False
    
    def _should_auto_approve(self, request: PermissionRequest) -> bool:
        \"\"\"Determine if request should be auto-approved\"\"\"
        if not self.config.auto_approve_safe_operations:
            return False
        
        # Auto-approve safe operations
        if request.risk_level == 'safe':
            return True
        
        # Auto-approve previously approved commands
        if request.command in self.approved_commands:
            return True
        
        # Auto-approve based on security level
        if self.config.security_level == 'low':
            return request.risk_level in ['safe', 'medium']
        elif self.config.security_level == 'medium':
            return request.risk_level == 'safe'
        
        # High security level only auto-approves safe operations
        return False
    
    async def _request_interactive_approval(self, request: PermissionRequest) -> bool:
        \"\"\"Request interactive approval from user\"\"\"
        print(f"\\n🔐 Permission Request [{request.request_id}]")
        print(f"CLI Tool: {request.cli_tool}")
        print(f"Operation: {request.operation}")
        print(f"Command: {request.command}")
        print(f"Risk Level: {request.risk_level.upper()}")
        if request.justification:
            print(f"Justification: {request.justification}")
        
        # Show command analysis
        await self._show_command_analysis(request.command)
        
        print(f"\\nOptions:")
        print(f"[a] Approve once")
        print(f"[A] Approve and remember")
        print(f"[d] Deny")
        print(f"[D] Deny and block CLI tool")
        print(f"[s] Show more details")
        
        while True:
            try:
                choice = input("Choice [a/A/d/D/s]: ").strip().lower()
                
                if choice == 'a':
                    return True
                elif choice == 'A':
                    self.approved_commands.add(request.command)
                    return True
                elif choice == 'd':
                    return False
                elif choice == 'D':
                    self.config.allowed_cli_tools.discard(request.cli_tool)
                    self._save_config()
                    print(f"🚫 Blocked CLI tool: {request.cli_tool}")
                    return False
                elif choice == 's':
                    await self._show_detailed_analysis(request)
                else:
                    print("Invalid choice. Please enter a, A, d, D, or s.")
                    
            except (EOFError, KeyboardInterrupt):
                print("\\n❌ Permission denied (interrupted)")
                return False
    
    async def _show_command_analysis(self, command: str):
        \"\"\"Show basic command analysis\"\"\"
        print(f"\\n📊 Command Analysis:")
        
        # Parse command components
        try:
            parts = shlex.split(command)
            if parts:
                print(f"  Executable: {parts[0]}")
                if len(parts) > 1:
                    print(f"  Arguments: {' '.join(parts[1:])}")
        except ValueError:
            print(f"  Raw command: {command}")
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            ('sudo', 'Elevated privileges'),
            ('rm -r', 'Recursive deletion'),
            ('>', 'File redirection'),
            ('|', 'Command piping'),
            ('&', 'Background execution'),
            (';', 'Command chaining')
        ]
        
        warnings = []
        for pattern, desc in dangerous_patterns:
            if pattern in command:
                warnings.append(desc)
        
        if warnings:
            print(f"  ⚠️ Warnings: {', '.join(warnings)}")
    
    async def _show_detailed_analysis(self, request: PermissionRequest):
        \"\"\"Show detailed command analysis\"\"\"
        print(f"\\n🔍 Detailed Analysis:")
        print(f"  Request ID: {request.request_id}")
        print(f"  Timestamp: {time.ctime(request.timestamp)}")
        
        # Show command breakdown
        try:
            import shutil
            command_parts = shlex.split(request.command)
            if command_parts:
                executable = command_parts[0]
                executable_path = shutil.which(executable)
                if executable_path:
                    print(f"  Executable path: {executable_path}")
                    
                    # Get file info
                    stat = os.stat(executable_path)
                    print(f"  Permissions: {oct(stat.st_mode)[-3:]}")
                    print(f"  Owner: UID {stat.st_uid}")
        except Exception:
            pass
        
        # Show environment impact
        print(f"  Current directory: {os.getcwd()}")
        print(f"  User: {os.getenv('USER', 'unknown')}")
        print(f"  Shell: {os.getenv('SHELL', 'unknown')}")
    
    async def execute_with_permission(
        self, 
        cli_tool: str, 
        command: str, 
        operation: str = "execute",
        justification: str = "",
        capture_output: bool = True
    ) -> Optional[subprocess.CompletedProcess]:
        \"\"\"Execute command with permission checking\"\"\"
        
        # Request permission
        if not await self.request_permission(cli_tool, operation, command, justification):
            return None
        
        try:
            self._audit_log("EXECUTE_START", f"Executing: {command}", cli_tool)
            
            if capture_output:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    timeout=300
                )
            
            self._audit_log("EXECUTE_END", f"Exit code: {result.returncode}", cli_tool)
            return result
            
        except subprocess.TimeoutExpired:
            self._audit_log("EXECUTE_TIMEOUT", f"Command timed out: {command}", cli_tool)
            print("⏰ Command execution timed out")
            return None
        except Exception as e:
            self._audit_log("EXECUTE_ERROR", f"Error: {str(e)}", cli_tool)
            print(f"❌ Execution error: {e}")
            return None
    
    def enable_unleashed_mode(self, security_level: str = "medium") -> bool:
        \"\"\"Enable unleashed mode with specified security level\"\"\"
        if security_level not in ['low', 'medium', 'high']:
            print("❌ Invalid security level. Use: low, medium, high")
            return False
        
        self.config.enabled = True
        self.config.security_level = security_level
        self._save_config()
        
        self._audit_log("MODE_ENABLED", f"Security level: {security_level}", "system")
        print(f"🚀 Unleashed mode ENABLED with {security_level} security")
        print(f"   Emergency disable: kill -USR1 {os.getpid()}")
        
        return True
    
    def disable_unleashed_mode(self) -> bool:
        \"\"\"Disable unleashed mode\"\"\"
        self.config.enabled = False
        self._save_config()
        
        self._audit_log("MODE_DISABLED", "Disabled by user", "system")
        print("🔒 Unleashed mode DISABLED")
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        \"\"\"Get unleashed mode status\"\"\"
        return {
            'enabled': self.config.enabled,
            'security_level': self.config.security_level,
            'allowed_cli_tools': list(self.config.allowed_cli_tools),
            'audit_enabled': self.config.audit_enabled,
            'pending_requests': len(self.pending_requests),
            'approved_commands': len(self.approved_commands),
            'auto_approve_safe': self.config.auto_approve_safe_operations
        }
    
    def add_allowed_cli_tool(self, cli_tool: str) -> bool:
        \"\"\"Add CLI tool to allowed list\"\"\"
        self.config.allowed_cli_tools.add(cli_tool)
        self._save_config()
        self._audit_log("CLI_TOOL_ADDED", f"Added: {cli_tool}", "system")
        return True
    
    def remove_allowed_cli_tool(self, cli_tool: str) -> bool:
        \"\"\"Remove CLI tool from allowed list\"\"\"
        self.config.allowed_cli_tools.discard(cli_tool)
        self._save_config()
        self._audit_log("CLI_TOOL_REMOVED", f"Removed: {cli_tool}", "system")
        return True
    
    async def get_audit_log(self, hours: int = 24) -> List[Dict[str, Any]]:
        \"\"\"Get audit log entries\"\"\"
        if not Path(self.audit_log_path).exists():
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        entries = []
        
        try:
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry['timestamp'] >= cutoff_time:
                        entries.append(entry)
        except Exception as e:
            print(f"❌ Error reading audit log: {e}")
        
        return sorted(entries, key=lambda x: x['timestamp'], reverse=True)

# High-level API for CLI tools
class UnleashedAPI:
    \"\"\"High-level unleashed mode API\"\"\"
    
    def __init__(self):
        self.manager = UnleashedModeManager()
    
    async def execute_command(
        self, 
        cli_tool: str, 
        command: str, 
        justification: str = ""
    ) -> Optional[subprocess.CompletedProcess]:
        \"\"\"Execute command with permission checking\"\"\"
        return await self.manager.execute_with_permission(
            cli_tool, command, "execute", justification
        )
    
    async def request_file_access(self, cli_tool: str, file_path: str, mode: str = "read") -> bool:
        \"\"\"Request file access permission\"\"\"
        operation = f"file_{mode}"
        command = f"access {file_path} ({mode})"
        return await self.manager.request_permission(cli_tool, operation, command)
    
    async def request_network_access(self, cli_tool: str, url: str, method: str = "GET") -> bool:
        \"\"\"Request network access permission\"\"\"
        operation = "network_access"
        command = f"{method} {url}"
        return await self.manager.request_permission(cli_tool, operation, command)
    
    def enable_unleashed(self, security_level: str = "medium") -> bool:
        \"\"\"Enable unleashed mode\"\"\"
        return self.manager.enable_unleashed_mode(security_level)
    
    def disable_unleashed(self) -> bool:
        \"\"\"Disable unleashed mode\"\"\"
        return self.manager.disable_unleashed_mode()
    
    def get_status(self) -> Dict[str, Any]:
        \"\"\"Get status\"\"\"
        return self.manager.get_status()

# CLI tool wrapper functions
def create_unleashed_wrapper(cli_tool_name: str):
    \"\"\"Create unleashed wrapper for CLI tool\"\"\"
    api = UnleashedAPI()
    
    async def execute_with_unleashed(command: str, justification: str = "") -> Optional[subprocess.CompletedProcess]:
        return await api.execute_command(cli_tool_name, command, justification)
    
    return execute_with_unleashed

# Example integrations
async def claude_code_unleashed():
    \"\"\"Unleashed mode integration for claude-code\"\"\"
    return create_unleashed_wrapper("claude-code")

async def gemini_unleashed():
    \"\"\"Unleashed mode integration for gemini\"\"\"
    return create_unleashed_wrapper("gemini")

async def codex_cli_unleashed():
    \"\"\"Unleashed mode integration for codex-cli\"\"\"
    return create_unleashed_wrapper("codex-cli")

if __name__ == "__main__":
    async def test_unleashed_mode():
        api = UnleashedAPI()
        
        # Enable unleashed mode
        api.enable_unleashed("medium")
        
        # Test command execution
        result = await api.execute_command(
            "claude-code", 
            "ls -la", 
            "Testing directory listing"
        )
        
        if result:
            print(f"Command output: {result.stdout}")
        
        # Get status
        status = api.get_status()
        print(f"Unleashed status: {status}")
    
    asyncio.run(test_unleashed_mode())