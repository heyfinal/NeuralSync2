#!/usr/bin/env python3
"""
NeuralSync2 Auto-Launch Integration System Installer
Production-ready installer with dependency management, auto-configuration, and validation
"""

import asyncio
import os
import sys
import subprocess
import shutil
import json
import logging
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import getpass
import tempfile
import urllib.request
import stat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuralSyncInstaller:
    """Production-ready NeuralSync auto-launch integration installer"""
    
    def __init__(self):
        self.install_dir = Path.cwd()
        self.bin_dir = Path.home() / ".local" / "bin"
        self.system_info = self._get_system_info()
        self.sudo_password = None
        self.sudo_cached = False
        
        # Installation configuration
        self.required_commands = ['python3', 'pip', 'git']
        self.optional_commands = ['claude-code', 'codexcli', 'gemini']
        
        # Enhanced daemon management enabled by default
        self.enhanced_daemon_enabled = True
        
        # Dependencies
        self.python_deps = [
            'fastapi>=0.115.0',
            'uvicorn[standard]>=0.30.6',
            'pydantic>=2.8.2',
            'typer>=0.12.5',
            'numpy>=2.0.1',
            'sqlite-utils>=3.36',
            'PyYAML>=6.0.2',
            'requests>=2.32.3',
            'xxhash>=3.4.1',
            'zstandard>=0.22.0',
            'psutil>=5.9.5',
            'aiofiles>=23.2.1',
            'redis>=5.0.1',
            'lz4>=4.3.2',
            'msgpack>=1.0.7',
            'orjson>=3.9.10',
            'uvloop>=0.19.0',
            'cython>=3.0.5'
        ]
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information"""
        return {
            'os': platform.system(),
            'arch': platform.machine(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': platform.platform(),
            'home': str(Path.home())
        }
        
    async def run_with_sudo(self, cmd: List[str], description: str = "") -> Tuple[bool, str, str]:
        """Run command with sudo, caching password"""
        try:
            if not self.sudo_cached:
                if not self.sudo_password:
                    print(f"\n🔐 Administrator access required{': ' + description if description else ''}")
                    self.sudo_password = getpass.getpass("Password: ")
                    
            # Test sudo with cached password
            test_process = await asyncio.create_subprocess_exec(
                'sudo', '-S', 'true',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await test_process.communicate(
                input=f"{self.sudo_password}\n".encode()
            )
            
            if test_process.returncode != 0:
                print("❌ Invalid password")
                self.sudo_password = None
                return await self.run_with_sudo(cmd, description)  # Retry
                
            self.sudo_cached = True
            
            # Run actual command with sudo
            full_cmd = ['sudo', '-S'] + cmd
            process = await asyncio.create_subprocess_exec(
                *full_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(
                input=f"{self.sudo_password}\n".encode()
            )
            
            return (
                process.returncode == 0,
                stdout.decode() if stdout else "",
                stderr.decode() if stderr else ""
            )
            
        except Exception as e:
            logger.error(f"Sudo command failed: {e}")
            return False, "", str(e)
            
    async def run_command(self, cmd: List[str], description: str = "", 
                         use_sudo: bool = False, cwd: Optional[Path] = None) -> Tuple[bool, str, str]:
        """Run a command with proper error handling"""
        try:
            if use_sudo:
                return await self.run_with_sudo(cmd, description)
                
            logger.info(f"Running: {' '.join(cmd)}" + (f" ({description})" if description else ""))
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
            
            if not success:
                logger.error(f"Command failed: {' '.join(cmd)}")
                logger.error(f"Return code: {process.returncode}")
                if stderr_str:
                    logger.error(f"STDERR: {stderr_str}")
                    
            return success, stdout_str, stderr_str
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False, "", str(e)
            
    def check_command_available(self, cmd: str) -> bool:
        """Check if a command is available in PATH"""
        return shutil.which(cmd) is not None
        
    async def check_dependencies(self) -> Dict[str, Any]:
        """Check all system dependencies"""
        status = {
            'system_requirements': {},
            'python_packages': {},
            'cli_tools': {},
            'overall_status': 'checking'
        }
        
        print("🔍 Checking system dependencies...")
        
        # Check required system commands
        for cmd in self.required_commands:
            available = self.check_command_available(cmd)
            status['system_requirements'][cmd] = {
                'available': available,
                'required': True
            }
            if available:
                print(f"✅ {cmd} - Available")
            else:
                print(f"❌ {cmd} - Missing (Required)")
                
        # Check optional CLI tools
        for cmd in self.optional_commands:
            available = self.check_command_available(cmd)
            status['cli_tools'][cmd] = {
                'available': available,
                'required': False
            }
            if available:
                print(f"✅ {cmd} - Available")
            else:
                print(f"⚠️  {cmd} - Not found (Optional - install for full functionality)")
                
        # Check Python packages
        print("🔍 Checking Python packages...")
        missing_packages = []
        
        for package in self.python_deps:
            package_name = package.split('>=')[0].split('==')[0]
            try:
                __import__(package_name.replace('-', '_'))
                status['python_packages'][package_name] = {'available': True}
                print(f"✅ {package_name} - Available")
            except ImportError:
                status['python_packages'][package_name] = {'available': False}
                missing_packages.append(package)
                print(f"❌ {package_name} - Missing")
                
        # Determine overall status
        required_missing = [cmd for cmd, info in status['system_requirements'].items() 
                          if not info['available'] and info['required']]
        
        if required_missing:
            status['overall_status'] = 'missing_requirements'
            print(f"❌ Missing required dependencies: {', '.join(required_missing)}")
        elif missing_packages:
            status['overall_status'] = 'missing_packages'
            print(f"⚠️  Missing Python packages: {len(missing_packages)} packages")
        else:
            status['overall_status'] = 'ready'
            print("✅ All dependencies available")
            
        return status
        
    async def install_system_dependencies(self) -> bool:
        """Install missing system dependencies"""
        print("📦 Installing system dependencies...")
        
        system = self.system_info['os'].lower()
        
        if system == 'darwin':  # macOS
            # Check for Homebrew
            if not self.check_command_available('brew'):
                print("Installing Homebrew...")
                install_cmd = [
                    '/bin/bash', '-c',
                    '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)'
                ]
                success, stdout, stderr = await self.run_command(install_cmd, "Installing Homebrew")
                if not success:
                    print("❌ Failed to install Homebrew")
                    return False
                    
            # Install missing packages with brew
            missing_packages = []
            if not self.check_command_available('python3'):
                missing_packages.append('python@3.11')
            if not self.check_command_available('git'):
                missing_packages.append('git')
                
            if missing_packages:
                for package in missing_packages:
                    success, stdout, stderr = await self.run_command(
                        ['brew', 'install', package],
                        f"Installing {package}"
                    )
                    if not success:
                        print(f"❌ Failed to install {package}")
                        return False
                        
        elif system == 'linux':
            # Detect package manager
            if self.check_command_available('apt'):
                # Ubuntu/Debian
                print("Updating package lists...")
                success, stdout, stderr = await self.run_command(
                    ['apt', 'update'],
                    "Updating package lists",
                    use_sudo=True
                )
                if not success:
                    print("⚠️  Failed to update package lists")
                    
                missing_packages = []
                if not self.check_command_available('python3'):
                    missing_packages.extend(['python3', 'python3-pip', 'python3-venv'])
                if not self.check_command_available('git'):
                    missing_packages.append('git')
                    
                if missing_packages:
                    success, stdout, stderr = await self.run_command(
                        ['apt', 'install', '-y'] + missing_packages,
                        f"Installing {', '.join(missing_packages)}",
                        use_sudo=True
                    )
                    if not success:
                        print(f"❌ Failed to install packages")
                        return False
                        
            elif self.check_command_available('yum'):
                # RHEL/CentOS
                missing_packages = []
                if not self.check_command_available('python3'):
                    missing_packages.extend(['python3', 'python3-pip'])
                if not self.check_command_available('git'):
                    missing_packages.append('git')
                    
                if missing_packages:
                    success, stdout, stderr = await self.run_command(
                        ['yum', 'install', '-y'] + missing_packages,
                        f"Installing {', '.join(missing_packages)}",
                        use_sudo=True
                    )
                    if not success:
                        print(f"❌ Failed to install packages")
                        return False
                        
            else:
                print("❌ Unsupported Linux distribution. Please install python3, pip, and git manually.")
                return False
                
        else:
            print("❌ Unsupported operating system")
            return False
            
        print("✅ System dependencies installed")
        return True
        
    async def setup_python_environment(self) -> bool:
        """Setup Python virtual environment and install dependencies"""
        print("🐍 Setting up Python environment...")
        
        venv_path = self.install_dir / ".venv"
        
        # Create virtual environment if it doesn't exist
        if not venv_path.exists():
            print("Creating Python virtual environment...")
            success, stdout, stderr = await self.run_command(
                ['python3', '-m', 'venv', str(venv_path)],
                "Creating virtual environment"
            )
            if not success:
                print("❌ Failed to create virtual environment")
                return False
                
        # Determine activation script
        if self.system_info['os'] == 'Windows':
            python_exe = venv_path / "Scripts" / "python.exe"
            pip_exe = venv_path / "Scripts" / "pip.exe"
        else:
            python_exe = venv_path / "bin" / "python"
            pip_exe = venv_path / "bin" / "pip"
            
        # Upgrade pip
        print("Upgrading pip...")
        success, stdout, stderr = await self.run_command(
            [str(python_exe), '-m', 'pip', 'install', '--upgrade', 'pip'],
            "Upgrading pip"
        )
        if not success:
            print("⚠️  Failed to upgrade pip")
            
        # Install Python dependencies
        print("Installing Python dependencies...")
        
        # Create requirements file
        requirements_content = "\n".join(self.python_deps)
        requirements_file = self.install_dir / "requirements_auto.txt"
        
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
            
        success, stdout, stderr = await self.run_command(
            [str(pip_exe), 'install', '-r', str(requirements_file)],
            "Installing Python dependencies"
        )
        
        # Clean up temporary requirements file
        requirements_file.unlink()
        
        if not success:
            print("❌ Failed to install Python dependencies")
            return False
            
        print("✅ Python environment setup complete")
        return True
        
    async def setup_bin_directory(self) -> bool:
        """Setup local bin directory for wrapper scripts"""
        print("📁 Setting up bin directory...")
        
        # Ensure .local/bin exists
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if .local/bin is in PATH
        path_env = os.environ.get('PATH', '')
        if str(self.bin_dir) not in path_env:
            print(f"⚠️  {self.bin_dir} is not in PATH")
            
            # Add to shell profile
            shell = os.environ.get('SHELL', '/bin/bash')
            if 'zsh' in shell:
                profile_file = Path.home() / '.zshrc'
            elif 'fish' in shell:
                profile_file = Path.home() / '.config' / 'fish' / 'config.fish'
            else:
                profile_file = Path.home() / '.bashrc'
                
            export_line = f'export PATH="{self.bin_dir}:$PATH"'
            
            # Check if already added
            if profile_file.exists():
                content = profile_file.read_text()
                if export_line not in content:
                    print(f"Adding {self.bin_dir} to PATH in {profile_file}")
                    with open(profile_file, 'a') as f:
                        f.write(f'\n# NeuralSync auto-launch integration\n')
                        f.write(f'{export_line}\n')
            else:
                print(f"Creating {profile_file} and adding PATH")
                profile_file.parent.mkdir(parents=True, exist_ok=True)
                with open(profile_file, 'w') as f:
                    f.write(f'# NeuralSync auto-launch integration\n')
                    f.write(f'{export_line}\n')
                    
        print("✅ Bin directory setup complete")
        return True
        
    async def install_wrapper_scripts(self) -> bool:
        """Install and configure wrapper scripts"""
        print("🔧 Installing wrapper scripts...")
        
        wrapper_scripts = [
            ('bin/claude-ns-fixed', 'claude-ns'),
            ('bin/codex-ns-fixed', 'codex-ns'),
            ('bin/gemini-ns-fixed', 'gemini-ns')
        ]
        
        for source_path, script_name in wrapper_scripts:
            source = self.install_dir / source_path
            target = self.bin_dir / script_name
            
            if not source.exists():
                print(f"❌ Source script not found: {source}")
                continue
                
            # Copy and make executable
            try:
                shutil.copy2(source, target)
                target.chmod(target.stat().st_mode | stat.S_IEXEC)
                print(f"✅ Installed {script_name}")
            except Exception as e:
                print(f"❌ Failed to install {script_name}: {e}")
                return False
                
        # Install nswrap
        nswrap_source = self.install_dir / "nswrap"
        nswrap_target = self.bin_dir / "nswrap"
        
        if nswrap_source.exists():
            try:
                shutil.copy2(nswrap_source, nswrap_target)
                nswrap_target.chmod(nswrap_target.stat().st_mode | stat.S_IEXEC)
                print(f"✅ Installed nswrap")
            except Exception as e:
                print(f"❌ Failed to install nswrap: {e}")
                return False
        else:
            print(f"❌ nswrap not found at {nswrap_source}")
            
        print("✅ Wrapper scripts installed")
        return True
        
    async def setup_neuralsync_config(self) -> bool:
        """Setup NeuralSync configuration"""
        print("⚙️  Setting up NeuralSync configuration...")
        
        config_dir = Path.home() / '.neuralsync'
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / 'config.yaml'
        
        # Generate configuration if it doesn't exist
        if not config_file.exists():
            import uuid
            config = {
                'site_id': str(uuid.uuid4()),
                'db_path': str(config_dir / 'memory.db'),
                'oplog_path': str(config_dir / 'oplog.jsonl'),
                'vector_dim': 512,
                'bind_host': '127.0.0.1',
                'bind_port': 8373,
                'token': ''
            }
            
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            print(f"✅ Created configuration: {config_file}")
        else:
            print(f"✅ Configuration exists: {config_file}")
            
        # Setup enhanced daemon management if enabled
        if self.enhanced_daemon_enabled:
            enhanced_setup_success = await self.setup_enhanced_daemon_management()
            if not enhanced_setup_success:
                print("⚠️  Enhanced daemon management setup failed - continuing with basic setup")
            
        return True
    
    async def setup_enhanced_daemon_management(self) -> bool:
        """Setup enhanced daemon management with performance optimizations"""
        print("🚀 Setting up enhanced daemon management...")
        
        try:
            # Validate enhanced daemon management modules exist
            neuralsync_dir = self.install_dir / 'neuralsync'
            required_modules = [
                'enhanced_daemon_manager.py',
                'robust_service_detector.py', 
                'smart_process_discovery.py',
                'configuration_validator.py',
                'performance_optimizer.py'
            ]
            
            missing_modules = []
            for module in required_modules:
                if not (neuralsync_dir / module).exists():
                    missing_modules.append(module)
                    
            if missing_modules:
                print(f"⚠️  Enhanced daemon management modules missing: {missing_modules}")
                return False
                
            # Update daemon_manager to use enhanced version
            daemon_manager_path = neuralsync_dir / 'daemon_manager.py'
            if daemon_manager_path.exists():
                # Backup original
                backup_path = daemon_manager_path.with_suffix('.py.backup')
                if not backup_path.exists():
                    shutil.copy2(daemon_manager_path, backup_path)
                    print(f"✅ Backed up original daemon_manager.py")
                
                # Update daemon_manager to integrate enhanced features
                self._update_daemon_manager_integration(daemon_manager_path)
                print(f"✅ Updated daemon_manager.py with enhanced features")
            
            # Test enhanced daemon manager functionality
            venv_python = self.install_dir / ".venv" / ("Scripts" if self.system_info['os'] == 'Windows' else "bin") / "python"
            if venv_python.exists():
                test_script = '''
import sys
sys.path.insert(0, ".")
try:
    from neuralsync.enhanced_daemon_manager import EnhancedDaemonManager
    from neuralsync.robust_service_detector import RobustServiceDetector
    from neuralsync.performance_optimizer import PerformanceOptimizer
    print("✅ Enhanced daemon management modules imported successfully")
    
    # Test basic functionality
    config_dir = str(__import__('pathlib').Path.home() / '.neuralsync')
    detector = RobustServiceDetector(config_dir)
    optimizer = PerformanceOptimizer()
    print("✅ Enhanced daemon management objects created successfully")
    
except Exception as e:
    print(f"❌ Enhanced daemon management test failed: {e}")
    sys.exit(1)
'''
                
                success, stdout, stderr = await self.run_command(
                    [str(venv_python), '-c', test_script],
                    "Testing enhanced daemon management",
                    cwd=self.install_dir
                )
                
                if success:
                    print("✅ Enhanced daemon management - Functional")
                    return True
                else:
                    print(f"❌ Enhanced daemon management test failed: {stderr}")
                    return False
            
            print("✅ Enhanced daemon management setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced daemon management setup failed: {e}")
            return False
    
    def _update_daemon_manager_integration(self, daemon_manager_path: Path):
        """Update daemon_manager.py to integrate enhanced features"""
        try:
            # Read original daemon manager
            with open(daemon_manager_path, 'r') as f:
                content = f.read()
            
            # Add enhanced daemon manager import if not present
            if 'from .enhanced_daemon_manager import' not in content:
                # Find imports section and add enhanced import
                lines = content.split('\n')
                import_inserted = False
                
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        continue
                    else:
                        # Insert enhanced import before first non-import line
                        lines.insert(i, 'try:')
                        lines.insert(i+1, '    from .enhanced_daemon_manager import ensure_neuralsync_running_enhanced, EnhancedDaemonManager')
                        lines.insert(i+2, '    _ENHANCED_AVAILABLE = True')
                        lines.insert(i+3, 'except ImportError:')
                        lines.insert(i+4, '    _ENHANCED_AVAILABLE = False')
                        lines.insert(i+5, '')
                        import_inserted = True
                        break
                
                if import_inserted:
                    # Update ensure_neuralsync_running function to use enhanced version when available
                    content = '\n'.join(lines)
                    
                    # Find ensure_neuralsync_running function and modify it
                    if 'async def ensure_neuralsync_running(' in content:
                        enhanced_wrapper = '''async def ensure_neuralsync_running():
    """Ensure NeuralSync services are running with enhanced daemon management"""
    if _ENHANCED_AVAILABLE:
        try:
            return await ensure_neuralsync_running_enhanced()
        except Exception as e:
            logger.warning(f"Enhanced daemon management failed, falling back to basic: {e}")
    
    # Fallback to original implementation
    return await _ensure_neuralsync_running_basic()

async def _ensure_neuralsync_running_basic():'''
                        
                        # Replace the function signature
                        content = content.replace(
                            'async def ensure_neuralsync_running(',
                            enhanced_wrapper + '\n    # Original ensure_neuralsync_running implementation\nasync def _ensure_neuralsync_running_basic('
                        )
                
                # Write updated content
                with open(daemon_manager_path, 'w') as f:
                    f.write(content)
                    
        except Exception as e:
            logger.error(f"Failed to update daemon_manager integration: {e}")
            raise
        
    async def validate_installation(self) -> bool:
        """Validate the complete installation"""
        print("🔍 Validating installation...")
        
        all_valid = True
        
        # Check wrapper scripts
        for script_name in ['claude-ns', 'codex-ns', 'gemini-ns']:
            script_path = self.bin_dir / script_name
            if script_path.exists() and os.access(script_path, os.X_OK):
                print(f"✅ {script_name} - Installed and executable")
                
                # Test script (basic syntax check)
                success, stdout, stderr = await self.run_command(
                    [str(script_path), '--neuralsync-status'],
                    f"Testing {script_name}",
                    # Don't fail on this - services might not be running yet
                )
                # Just log the result, don't fail validation
                if success:
                    print(f"✅ {script_name} - Basic functionality test passed")
                else:
                    print(f"⚠️  {script_name} - Basic test failed (services may not be running)")
                    
            else:
                print(f"❌ {script_name} - Not found or not executable")
                all_valid = False
                
        # Test Python environment
        venv_python = self.install_dir / ".venv" / ("Scripts" if self.system_info['os'] == 'Windows' else "bin") / "python"
        if venv_python.exists():
            success, stdout, stderr = await self.run_command(
                [str(venv_python), '-c', 'import neuralsync; print("NeuralSync module loaded")'],
                "Testing Python environment",
                cwd=self.install_dir
            )
            if success:
                print("✅ Python environment - NeuralSync modules accessible")
            else:
                print("❌ Python environment - NeuralSync modules not accessible")
                all_valid = False
        else:
            print("❌ Python virtual environment not found")
            all_valid = False
            
        # Check configuration
        config_file = Path.home() / '.neuralsync' / 'config.yaml'
        if config_file.exists():
            print("✅ NeuralSync configuration - Present")
        else:
            print("❌ NeuralSync configuration - Missing")
            all_valid = False
            
        return all_valid
        
    async def run_post_install_tests(self) -> bool:
        """Run comprehensive post-installation tests"""
        print("\n🧪 Running post-installation tests...")
        
        test_results = []
        
        # Test 1: Basic import test
        print("Test 1: Module imports...")
        venv_python = self.install_dir / ".venv" / ("Scripts" if self.system_info['os'] == 'Windows' else "bin") / "python"
        
        test_script = '''
import sys
sys.path.insert(0, ".")

try:
    from neuralsync.daemon_manager import get_daemon_manager
    from neuralsync.ultra_comm import get_comm_manager  
    from neuralsync.agent_sync import get_agent_synchronizer
    from neuralsync.agent_lifecycle import get_lifecycle_manager
    print("✅ All modules imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            test_file = f.name
            
        try:
            success, stdout, stderr = await self.run_command(
                [str(venv_python), test_file],
                "Module import test",
                cwd=self.install_dir
            )
            test_results.append(('Module imports', success, stdout + stderr))
        finally:
            os.unlink(test_file)
            
        # Test 2: Configuration load test
        print("Test 2: Configuration loading...")
        config_test = '''
import sys
sys.path.insert(0, ".")

try:
    from neuralsync.config import load_config
    config = load_config()
    print(f"✅ Configuration loaded: site_id={config.site_id}")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    sys.exit(1)
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(config_test)
            test_file = f.name
            
        try:
            success, stdout, stderr = await self.run_command(
                [str(venv_python), test_file],
                "Configuration test",
                cwd=self.install_dir
            )
            test_results.append(('Configuration loading', success, stdout + stderr))
        finally:
            os.unlink(test_file)
            
        # Print test summary
        print("\n📋 Test Summary:")
        all_passed = True
        for test_name, passed, output in test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name}")
            if not passed:
                print(f"   Output: {output[:200]}...")
                all_passed = False
                
        return all_passed
        
    def print_completion_message(self):
        """Print installation completion message with usage instructions"""
        print("\n" + "="*80)
        print("🎉 NeuralSync Auto-Launch Integration Installation Complete!")
        print("="*80)
        print()
        print("📋 What was installed:")
        print("  • NeuralSync2 daemon management system")
        print("  • Auto-launch integration for CLI tools")
        print("  • Inter-agent communication system")
        print("  • Shared memory and persona synchronization")
        print("  • Agent spawning and lifecycle management")
        print()
        print("🔧 Available commands:")
        print("  • claude-ns   - Claude Code with NeuralSync integration")
        print("  • codex-ns    - CodexCLI with NeuralSync integration")
        print("  • gemini-ns   - Gemini CLI with NeuralSync integration")
        print()
        print("📖 Usage examples:")
        print("  claude-ns --help                    # Get help")
        print("  claude-ns                           # Run with auto-launch")
        print("  claude-ns --neuralsync-status       # Check system status")
        print("  claude-ns --spawn-agent codex 'generate a function'")
        print()
        print("⚙️  Configuration:")
        print(f"  Config file: {Path.home() / '.neuralsync' / 'config.yaml'}")
        print(f"  Log directory: {Path.home() / '.neuralsync' / 'logs'}")
        print(f"  Wrappers: {self.bin_dir}")
        print()
        print("🔄 First run:")
        print("  The NeuralSync services will auto-start on first use.")
        print("  Status can be checked with: claude-ns --neuralsync-status")
        print()
        if str(self.bin_dir) not in os.environ.get('PATH', ''):
            print("⚠️  IMPORTANT: Restart your terminal or run:")
            print(f"  export PATH=\"{self.bin_dir}:$PATH\"")
            print()
        print("🎯 Next steps:")
        print("  1. Restart terminal or reload shell profile")
        print("  2. Test: claude-ns --version")
        print("  3. Verify status: claude-ns --neuralsync-status")
        print("  4. Start using: claude-ns, codex-ns, gemini-ns")
        print()
        print("="*80)
        
    async def install(self, skip_deps: bool = False, skip_tests: bool = False) -> bool:
        """Run complete installation process"""
        print("🚀 Starting NeuralSync Auto-Launch Integration Installation")
        print(f"📍 Installation directory: {self.install_dir}")
        print(f"📍 System: {self.system_info['os']} {self.system_info['arch']}")
        print(f"📍 Python: {self.system_info['python_version']}")
        print()
        
        try:
            # Step 1: Check dependencies
            dep_status = await self.check_dependencies()
            
            if not skip_deps and dep_status['overall_status'] in ['missing_requirements', 'missing_packages']:
                # Install system dependencies if needed
                if dep_status['overall_status'] == 'missing_requirements':
                    success = await self.install_system_dependencies()
                    if not success:
                        print("❌ Failed to install system dependencies")
                        return False
                        
                # Setup Python environment
                success = await self.setup_python_environment()
                if not success:
                    print("❌ Failed to setup Python environment")
                    return False
                    
            # Step 2: Setup directories and paths
            success = await self.setup_bin_directory()
            if not success:
                print("❌ Failed to setup bin directory")
                return False
                
            # Step 3: Install wrapper scripts
            success = await self.install_wrapper_scripts()
            if not success:
                print("❌ Failed to install wrapper scripts")
                return False
                
            # Step 4: Setup configuration
            success = await self.setup_neuralsync_config()
            if not success:
                print("❌ Failed to setup configuration")
                return False
                
            # Step 5: Validate installation
            success = await self.validate_installation()
            if not success:
                print("⚠️  Installation validation failed - some features may not work")
                
            # Step 6: Run tests (optional)
            if not skip_tests:
                test_success = await self.run_post_install_tests()
                if not test_success:
                    print("⚠️  Some tests failed - installation may have issues")
                    
            # Success!
            self.print_completion_message()
            return True
            
        except KeyboardInterrupt:
            print("\n❌ Installation cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            print(f"❌ Installation failed: {e}")
            return False


async def main():
    """Main installer entry point"""
    parser = argparse.ArgumentParser(description='NeuralSync Auto-Launch Integration Installer')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-tests', action='store_true', help='Skip post-installation tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    installer = NeuralSyncInstaller()
    success = await installer.install(skip_deps=args.skip_deps, skip_tests=args.skip_tests)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())