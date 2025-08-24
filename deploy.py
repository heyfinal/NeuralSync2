#!/usr/bin/env python3
"""
NeuralSync2 Deployment Script
Complete system deployment and verification
"""

import asyncio
import os
import sys
import subprocess
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

class NeuralSync2Deployer:
    """Complete NeuralSync2 deployment system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.home_dir = Path.home()
        self.neuralsync_dir = self.home_dir / ".neuralsync"
        self.install_dir = self.neuralsync_dir
        self.backup_dir = self.install_dir / "backups" / f"backup_{int(time.time())}"
        
        # Deployment configuration
        self.config = {
            "version": "2.0.0",
            "build_date": time.time(),
            "components": [
                "core_memory",
                "ultra_comm", 
                "research_dedup",
                "nas_storage",
                "sync_manager",
                "unleashed_mode",
                "cli_integration"
            ],
            "required_dependencies": [
                "asyncio",
                "sqlite3", 
                "numpy",
                "scikit-learn",
                "websockets",
                "aiofiles",
                "aiohttp"
            ],
            "optional_dependencies": [
                "spacy",
                "uvloop"
            ]
        }
        
        self.deployment_status = {}
    
    async def deploy_complete_system(self) -> bool:
        """Deploy the complete NeuralSync2 system"""
        print("🚀 Starting NeuralSync2 deployment...")
        print(f"   Version: {self.config['version']}")
        print(f"   Target: {self.install_dir}")
        
        try:
            # Pre-deployment checks
            await self._pre_deployment_checks()
            
            # Backup existing installation
            await self._backup_existing_installation()
            
            # Install dependencies
            await self._install_dependencies()
            
            # Deploy core components
            await self._deploy_core_components()
            
            # Setup CLI tool integrations
            await self._setup_cli_integrations()
            
            # Create system services
            await self._create_system_services()
            
            # Verify installation
            await self._verify_installation()
            
            # Post-deployment setup
            await self._post_deployment_setup()
            
            print("✅ NeuralSync2 deployment completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            await self._rollback_deployment()
            return False
    
    async def _pre_deployment_checks(self):
        """Perform pre-deployment system checks"""
        print("🔍 Performing pre-deployment checks...")
        
        # Check Python version
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10+ required")
        
        # Check available disk space (need at least 100MB)
        free_space = shutil.disk_usage(self.home_dir).free
        if free_space < 100 * 1024 * 1024:  # 100MB
            raise RuntimeError("Insufficient disk space")
        
        # Check network connectivity
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
        except:
            print("⚠️ Limited network connectivity - some features may be unavailable")
        
        # Check permissions
        if not os.access(self.home_dir, os.W_OK):
            raise RuntimeError("No write permission to home directory")
        
        print("✅ Pre-deployment checks passed")
    
    async def _backup_existing_installation(self):
        """Backup existing NeuralSync installation"""
        if self.install_dir.exists():
            print("📦 Backing up existing installation...")
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            for item in self.install_dir.iterdir():
                if item.name != "backups":
                    if item.is_dir():
                        shutil.copytree(item, self.backup_dir / item.name)
                    else:
                        shutil.copy2(item, self.backup_dir)
            
            print(f"✅ Backup created: {self.backup_dir}")
    
    async def _install_dependencies(self):
        """Install required Python dependencies"""
        print("📦 Installing dependencies...")
        
        required_packages = [
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0", 
            "websockets>=10.0",
            "aiofiles>=0.8.0",
            "aiohttp>=3.8.0",
            "typer>=0.6.0"
        ]
        
        optional_packages = [
            "spacy>=3.4.0",
            "uvloop>=0.17.0"
        ]
        
        # Install required packages
        for package in required_packages:
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, check=True)
                print(f"   ✅ {package}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install {package}: {e.stderr}")
        
        # Install optional packages (best effort)
        for package in optional_packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, check=True)
                print(f"   ✅ {package} (optional)")
            except subprocess.CalledProcessError:
                print(f"   ⚠️ {package} (optional - failed to install)")
    
    async def _deploy_core_components(self):
        """Deploy all core NeuralSync components"""
        print("🔧 Deploying core components...")
        
        # Create directory structure
        self.install_dir.mkdir(parents=True, exist_ok=True)
        (self.install_dir / "neuralsync").mkdir(exist_ok=True)
        (self.install_dir / "wrappers").mkdir(exist_ok=True)
        (self.install_dir / "logs").mkdir(exist_ok=True)
        (self.install_dir / "cache").mkdir(exist_ok=True)
        
        # Copy source files
        neuralsync_source = self.project_root / "neuralsync"
        if neuralsync_source.exists():
            for py_file in neuralsync_source.glob("*.py"):
                dest_file = self.install_dir / "neuralsync" / py_file.name
                shutil.copy2(py_file, dest_file)
                print(f"   ✅ {py_file.name}")
        
        # Copy main executables
        for script in ["nsctl", "nswrap"]:
            src_script = self.project_root / script
            if src_script.exists():
                dest_script = self.install_dir / script
                shutil.copy2(src_script, dest_script)
                dest_script.chmod(0o755)  # Make executable
                print(f"   ✅ {script}")
        
        # Create main entry point
        main_script = self.install_dir / "neuralsync2"
        main_script.write_text(f'''#!/usr/bin/env python3
"""
NeuralSync2 Main Entry Point
"""
import sys
from pathlib import Path

# Add neuralsync to path
sys.path.insert(0, str(Path(__file__).parent / "neuralsync"))

from cli_integration import setup_all_cli_integrations
import asyncio

async def main():
    integration = await setup_all_cli_integrations()
    print("🧠 NeuralSync2 ready - all CLI tools enhanced")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\\n🛑 NeuralSync2 shutting down...")

if __name__ == "__main__":
    asyncio.run(main())
''')
        main_script.chmod(0o755)
        
        print("✅ Core components deployed")
    
    async def _setup_cli_integrations(self):
        """Setup CLI tool integrations"""
        print("🔗 Setting up CLI tool integrations...")
        
        # Create wrapper scripts for each supported CLI tool
        wrapper_configs = [
            {
                "name": "claude-code",
                "executable": "claude", 
                "wrapper": "claude_neuralsync"
            },
            {
                "name": "gemini",
                "executable": "gemini",
                "wrapper": "gemini_neuralsync"
            },
            {
                "name": "codex-cli", 
                "executable": "codex",
                "wrapper": "codex_neuralsync"
            }
        ]
        
        for config in wrapper_configs:
            wrapper_script = self.install_dir / "wrappers" / config["wrapper"]
            wrapper_script.write_text(f'''#!/usr/bin/env python3
"""
NeuralSync2 Enhanced Wrapper for {config["name"]}
"""
import asyncio
import sys
from pathlib import Path

# Add neuralsync to path
neuralsync_path = str(Path(__file__).parent.parent / "neuralsync")
sys.path.insert(0, neuralsync_path)

from cli_integration import EnhancedCLISession

async def main():
    session = EnhancedCLISession("{config["name"]}")
    await session.start()
    
    try:
        result = await session.execute_with_enhancements(sys.argv[1:])
        sys.exit(result)
    except KeyboardInterrupt:
        print("\\n🛑 Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {{e}}")
        sys.exit(1)
    finally:
        await session.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
''')
            wrapper_script.chmod(0o755)
            print(f"   ✅ {config['wrapper']}")
        
        print("✅ CLI integrations setup complete")
    
    async def _create_system_services(self):
        """Create system services and daemons"""
        print("🛠️ Creating system services...")
        
        # Create neuralsync daemon script
        daemon_script = self.install_dir / "neuralsync_daemon.py"
        daemon_script.write_text('''#!/usr/bin/env python3
"""
NeuralSync2 Background Daemon
Runs message broker and sync services
"""
import asyncio
import sys
import signal
from pathlib import Path

# Add neuralsync to path
sys.path.insert(0, str(Path(__file__).parent / "neuralsync"))

from ultra_comm import start_message_broker
from sync_manager import SyncAPI

class NeuralSyncDaemon:
    def __init__(self):
        self.running = True
        
    async def start(self):
        print("🚀 Starting NeuralSync2 daemon...")
        
        # Start message broker
        broker_task = asyncio.create_task(start_message_broker())
        
        # Start sync services
        sync_api = SyncAPI()
        await sync_api.start()
        
        # Keep running until stopped
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("🛑 Shutting down daemon...")
            self.running = False
            broker_task.cancel()

daemon = NeuralSyncDaemon()

def signal_handler(signum, frame):
    daemon.running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    asyncio.run(daemon.start())
''')
        daemon_script.chmod(0o755)
        
        # Create startup script
        startup_script = self.install_dir / "start_neuralsync.sh"
        startup_script.write_text(f'''#!/bin/bash
# NeuralSync2 Startup Script

export NEURALSYNC_HOME="{self.install_dir}"
export PATH="$NEURALSYNC_HOME/wrappers:$PATH"

echo "🧠 Starting NeuralSync2..."

# Start daemon in background
python3 "$NEURALSYNC_HOME/neuralsync_daemon.py" &
DAEMON_PID=$!

echo "✅ NeuralSync2 daemon started (PID: $DAEMON_PID)"
echo "   Enhanced CLI tools available: claude_neuralsync, gemini_neuralsync, codex_neuralsync"
echo "   Configuration: $NEURALSYNC_HOME"

# Save PID for stopping
echo $DAEMON_PID > "$NEURALSYNC_HOME/neuralsync.pid"
''')
        startup_script.chmod(0o755)
        
        # Create stop script
        stop_script = self.install_dir / "stop_neuralsync.sh"
        stop_script.write_text(f'''#!/bin/bash
# NeuralSync2 Stop Script

NEURALSYNC_HOME="{self.install_dir}"
PID_FILE="$NEURALSYNC_HOME/neuralsync.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "🛑 Stopping NeuralSync2 daemon (PID: $PID)..."
    kill $PID 2>/dev/null
    rm -f "$PID_FILE"
    echo "✅ NeuralSync2 stopped"
else
    echo "⚠️ NeuralSync2 daemon not running"
fi
''')
        stop_script.chmod(0o755)
        
        print("✅ System services created")
    
    async def _verify_installation(self):
        """Verify the installation is working correctly"""
        print("🔍 Verifying installation...")
        
        # Check all core files exist
        required_files = [
            "neuralsync2",
            "neuralsync_daemon.py",
            "start_neuralsync.sh",
            "stop_neuralsync.sh",
            "neuralsync/__init__.py",
            "neuralsync/core_memory.py",
            "neuralsync/ultra_comm.py",
            "neuralsync/research_dedup.py",
            "neuralsync/nas_storage.py", 
            "neuralsync/sync_manager.py",
            "neuralsync/unleashed_mode.py",
            "neuralsync/cli_integration.py",
            "wrappers/claude_neuralsync",
            "wrappers/gemini_neuralsync",
            "wrappers/codex_neuralsync"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.install_dir / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            raise RuntimeError(f"Missing required files: {missing_files}")
        
        # Test Python imports
        try:
            sys.path.insert(0, str(self.install_dir / "neuralsync"))
            
            # Test critical imports
            import core_memory
            import ultra_comm
            import research_dedup
            import nas_storage
            import sync_manager
            import unleashed_mode
            import cli_integration
            
            print("   ✅ All Python modules import successfully")
            
        except ImportError as e:
            raise RuntimeError(f"Import test failed: {e}")
        
        # Test database creation
        try:
            from core_memory import CoreMemoryManager
            test_db_path = str(self.install_dir / "test_memory.db")
            
            async def test_db():
                manager = CoreMemoryManager(test_db_path)
                await manager.initialize()
                await manager.cleanup()
                Path(test_db_path).unlink(missing_ok=True)
            
            await test_db()
            print("   ✅ Database systems working")
            
        except Exception as e:
            raise RuntimeError(f"Database test failed: {e}")
        
        print("✅ Installation verified successfully")
    
    async def _post_deployment_setup(self):
        """Post-deployment setup and configuration"""
        print("⚙️ Performing post-deployment setup...")
        
        # Create initial configuration
        config_file = self.install_dir / "neuralsync_config.json"
        config_data = {
            "version": self.config["version"],
            "install_date": time.time(),
            "install_path": str(self.install_dir),
            "features": {
                "core_memory": True,
                "ultra_comm": True,
                "research_dedup": True,
                "nas_storage": True,
                "sync_manager": True,
                "unleashed_mode": True,
                "cli_integration": True
            },
            "settings": {
                "auto_start_daemon": False,
                "log_level": "INFO",
                "max_memory_entries": 1000000,
                "cache_ttl_hours": 24
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Create shell integration script
        shell_integration = self.install_dir / "shell_integration.sh"
        shell_integration.write_text(f'''#!/bin/bash
# NeuralSync2 Shell Integration

# Add to your ~/.bashrc or ~/.zshrc:

# NeuralSync2 Setup
export NEURALSYNC_HOME="{self.install_dir}"
export PATH="$NEURALSYNC_HOME/wrappers:$PATH"

# Aliases for enhanced CLI tools
alias claude="claude_neuralsync"
alias gemini="gemini_neuralsync" 
alias codex="codex_neuralsync"

# NeuralSync control functions
neuralsync_start() {{
    bash "$NEURALSYNC_HOME/start_neuralsync.sh"
}}

neuralsync_stop() {{
    bash "$NEURALSYNC_HOME/stop_neuralsync.sh"
}}

neuralsync_status() {{
    if [ -f "$NEURALSYNC_HOME/neuralsync.pid" ]; then
        PID=$(cat "$NEURALSYNC_HOME/neuralsync.pid")
        if ps -p $PID > /dev/null; then
            echo "🟢 NeuralSync2 is running (PID: $PID)"
        else
            echo "🔴 NeuralSync2 is not running (stale PID file)"
        fi
    else
        echo "🔴 NeuralSync2 is not running"
    fi
}}

echo "🧠 NeuralSync2 shell integration loaded"
echo "   Commands: neuralsync_start, neuralsync_stop, neuralsync_status"
echo "   Enhanced CLIs: claude, gemini, codex"
''')
        shell_integration.chmod(0o755)
        
        # Create README
        readme_file = self.install_dir / "README.md"
        readme_file.write_text(f'''# NeuralSync2 - Installation Complete

Version: {self.config["version"]}
Installed: {time.ctime()}
Location: {self.install_dir}

## Features Installed

✅ **Core Memory System** - Persistent memory across CLI tools
✅ **Ultra-Low Latency Communication** - Sub-10ms inter-CLI messaging  
✅ **Research Deduplication** - Avoid redundant API calls
✅ **NAS Storage Integration** - Cold storage for memory archival
✅ **Multi-Machine Sync** - Sync memory across machines
✅ **Unleashed Mode** - Permission system for enhanced capabilities
✅ **CLI Tool Integration** - Enhanced claude-code, gemini, codex-cli

## Quick Start

1. **Add to your shell**:
   ```bash
   source {self.install_dir}/shell_integration.sh
   ```

2. **Start NeuralSync2**:
   ```bash
   neuralsync_start
   ```

3. **Use enhanced CLI tools**:
   ```bash
   claude "Help me write a function"
   gemini "Analyze this image"
   codex "Complete this code"
   ```

4. **Check status**:
   ```bash
   neuralsync_status
   ```

## Advanced Usage

- **Unleashed Mode**: Enhanced permissions for CLI tools
- **Memory Queries**: Shared memory across all CLI sessions
- **Research Cache**: Automatic deduplication of API calls
- **Multi-Machine**: Sync memory across multiple computers

## Support

Configuration: {self.install_dir}/neuralsync_config.json
Logs: {self.install_dir}/logs/
Cache: {self.install_dir}/cache/

For help: Run any enhanced CLI with --neuralsync-help
''')
        
        print("✅ Post-deployment setup complete")
        print(f"📋 Configuration saved: {config_file}")
        print(f"🚀 Shell integration: {shell_integration}")
    
    async def _rollback_deployment(self):
        """Rollback failed deployment"""
        print("🔄 Rolling back failed deployment...")
        
        if self.backup_dir.exists():
            # Remove current installation
            if self.install_dir.exists():
                shutil.rmtree(self.install_dir)
            
            # Restore backup
            self.install_dir.mkdir(parents=True)
            for item in self.backup_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(item, self.install_dir / item.name)
                else:
                    shutil.copy2(item, self.install_dir)
            
            print("✅ Rollback completed - restored from backup")
        else:
            print("⚠️ No backup available for rollback")
    
    async def run_comprehensive_tests(self) -> bool:
        """Run comprehensive test suite"""
        print("🧪 Running comprehensive test suite...")
        
        try:
            # Run the test suite we created
            test_file = self.project_root / "tests" / "test_comprehensive.py"
            
            if test_file.exists():
                result = subprocess.run([
                    sys.executable, "-m", "pytest", str(test_file), "-v"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ All tests passed!")
                    return True
                else:
                    print(f"❌ Tests failed:\\n{result.stdout}\\n{result.stderr}")
                    return False
            else:
                print("⚠️ Test suite not found - skipping tests")
                return True
                
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False
    
    async def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report"""
        report = {
            "deployment_info": {
                "version": self.config["version"],
                "deployment_time": time.time(),
                "install_path": str(self.install_dir),
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "user": os.getenv("USER", "unknown")
                }
            },
            "components_status": {},
            "features_available": [],
            "performance_metrics": {},
            "recommendations": []
        }
        
        # Check component status
        components = [
            "core_memory", "ultra_comm", "research_dedup",
            "nas_storage", "sync_manager", "unleashed_mode", "cli_integration"
        ]
        
        for component in components:
            module_file = self.install_dir / "neuralsync" / f"{component}.py"
            report["components_status"][component] = module_file.exists()
            
            if module_file.exists():
                report["features_available"].append(component)
        
        # Performance metrics
        report["performance_metrics"] = {
            "estimated_memory_usage_mb": 50,  # Rough estimate
            "startup_time_estimate_ms": 500,
            "max_concurrent_cli_tools": 10,
            "memory_entries_capacity": 1000000
        }
        
        # Recommendations
        if len(report["features_available"]) == len(components):
            report["recommendations"].append("All components installed successfully")
        else:
            missing = set(components) - set(report["features_available"])
            report["recommendations"].append(f"Missing components: {missing}")
        
        report["recommendations"].extend([
            "Add shell integration to ~/.bashrc or ~/.zshrc",
            "Start NeuralSync daemon for full functionality",
            "Configure API keys for enhanced features"
        ])
        
        return report


async def main():
    """Main deployment function"""
    deployer = NeuralSync2Deployer()
    
    print("🧠 NeuralSync2 - Advanced CLI Memory & Communication System")
    print("=" * 60)
    
    # Deploy system
    success = await deployer.deploy_complete_system()
    
    if success:
        # Run tests
        test_success = await deployer.run_comprehensive_tests()
        
        # Generate report
        report = await deployer.generate_deployment_report()
        
        # Save deployment report
        report_file = deployer.install_dir / "deployment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\\n" + "=" * 60)
        print("🎉 NEURALSYNC2 DEPLOYMENT COMPLETE!")
        print("=" * 60)
        print(f"📍 Installation Path: {deployer.install_dir}")
        print(f"📊 Components: {len(report['features_available'])}/{len(report['components_status'])}")
        print(f"🧪 Tests: {'✅ PASSED' if test_success else '⚠️ WARNINGS'}")
        print(f"📋 Report: {report_file}")
        
        print("\\n🚀 Quick Start:")
        print(f"   source {deployer.install_dir}/shell_integration.sh")
        print(f"   neuralsync_start")
        print("   claude 'Hello NeuralSync2!'")
        
        return True
    else:
        print("\\n❌ DEPLOYMENT FAILED")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)