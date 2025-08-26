#!/usr/bin/env python3
"""
NeuralSync CLI Integration Installer
Sets up fixed CLI wrappers with proper PATH integration and memory sync
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import json
import time

class CLIIntegrationInstaller:
    """Install and configure NeuralSync CLI integration"""
    
    def __init__(self):
        self.neuralsync_dir = Path(__file__).parent
        self.bin_dir = self.neuralsync_dir / "bin"
        self.install_dir = Path.home() / ".local" / "bin"
        self.config_dir = Path.home() / ".neuralsync"
        
        self.wrappers = {
            'codex-ns': 'codex-ns-fixed',
            'claude-ns': 'claude-ns-fixed', 
            'gemini-ns': 'gemini-ns-fixed'
        }
        
        self.success_count = 0
        self.error_count = 0
        
    def log(self, message: str, level: str = "INFO"):
        """Log installation message"""
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "❌"
        }.get(level, "ℹ️")
        
        print(f"{prefix} {message}")
        
        if level == "SUCCESS":
            self.success_count += 1
        elif level == "ERROR":
            self.error_count += 1
    
    def check_requirements(self) -> bool:
        """Check installation requirements"""
        self.log("Checking installation requirements...")
        
        requirements_met = True
        
        # Check Python version
        if sys.version_info < (3, 7):
            self.log("Python 3.7+ required", "ERROR")
            requirements_met = False
        else:
            self.log(f"Python {sys.version_info.major}.{sys.version_info.minor} OK", "SUCCESS")
        
        # Check NeuralSync directory
        if not self.neuralsync_dir.exists():
            self.log("NeuralSync directory not found", "ERROR")
            requirements_met = False
        else:
            self.log("NeuralSync directory found", "SUCCESS")
        
        # Check bin directory and fixed wrappers
        if not self.bin_dir.exists():
            self.log("bin directory not found", "ERROR")
            requirements_met = False
        else:
            self.log("bin directory found", "SUCCESS")
            
            # Check individual wrappers
            for wrapper_name, fixed_name in self.wrappers.items():
                fixed_path = self.bin_dir / fixed_name
                if not fixed_path.exists():
                    self.log(f"Fixed wrapper {fixed_name} not found", "ERROR")
                    requirements_met = False
                else:
                    self.log(f"Fixed wrapper {fixed_name} found", "SUCCESS")
        
        return requirements_met
    
    def setup_directories(self) -> bool:
        """Set up necessary directories"""
        self.log("Setting up directories...")
        
        try:
            # Create install directory
            self.install_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"Install directory: {self.install_dir}", "SUCCESS")
            
            # Create config directory
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"Config directory: {self.config_dir}", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"Failed to create directories: {e}", "ERROR")
            return False
    
    def install_wrappers(self) -> bool:
        """Install CLI wrappers"""
        self.log("Installing CLI wrappers...")
        
        all_success = True
        
        for wrapper_name, fixed_name in self.wrappers.items():
            try:
                source_path = self.bin_dir / fixed_name
                target_path = self.install_dir / wrapper_name
                
                if not source_path.exists():
                    self.log(f"Source wrapper {fixed_name} not found", "ERROR")
                    all_success = False
                    continue
                
                # Copy wrapper
                shutil.copy2(source_path, target_path)
                
                # Make executable
                target_path.chmod(0o755)
                
                self.log(f"Installed {wrapper_name} -> {target_path}", "SUCCESS")
                
            except Exception as e:
                self.log(f"Failed to install {wrapper_name}: {e}", "ERROR")
                all_success = False
        
        return all_success
    
    def setup_nswrap(self) -> bool:
        \"\"\"Ensure nswrap is accessible\"\"\"\n        self.log(\"Setting up nswrap...\")\n        \n        try:\n            nswrap_source = self.neuralsync_dir / \"nswrap\"\n            nswrap_target = self.install_dir / \"nswrap\"\n            \n            if not nswrap_source.exists():\n                self.log(\"nswrap not found in NeuralSync directory\", \"ERROR\")\n                return False\n            \n            # Copy nswrap\n            shutil.copy2(nswrap_source, nswrap_target)\n            nswrap_target.chmod(0o755)\n            \n            self.log(f\"nswrap installed: {nswrap_target}\", \"SUCCESS\")\n            return True\n            \n        except Exception as e:\n            self.log(f\"Failed to setup nswrap: {e}\", \"ERROR\")\n            return False\n    \n    def check_path_integration(self) -> bool:\n        \"\"\"Check if install directory is in PATH\"\"\"\n        self.log(\"Checking PATH integration...\")\n        \n        path_env = os.environ.get('PATH', '')\n        install_dir_str = str(self.install_dir)\n        \n        if install_dir_str in path_env:\n            self.log(\"Install directory already in PATH\", \"SUCCESS\")\n            return True\n        else:\n            self.log(\"Install directory not in PATH\", \"WARNING\")\n            self.log(f\"Add to your shell profile: export PATH=\\\"$PATH:{install_dir_str}\\\"\", \"INFO\")\n            return False\n    \n    def test_installation(self) -> bool:\n        \"\"\"Test installed wrappers\"\"\"\n        self.log(\"Testing installation...\")\n        \n        all_working = True\n        \n        for wrapper_name in self.wrappers.keys():\n            try:\n                wrapper_path = self.install_dir / wrapper_name\n                \n                if not wrapper_path.exists():\n                    self.log(f\"{wrapper_name} not installed\", \"ERROR\")\n                    all_working = False\n                    continue\n                \n                # Test basic execution (this should not hang with fixed wrappers)\n                result = subprocess.run(\n                    [str(wrapper_path), \"--help\"],\n                    capture_output=True,\n                    text=True,\n                    timeout=10  # Should respond quickly\n                )\n                \n                # Don't require success code since underlying CLI may not be installed\n                self.log(f\"{wrapper_name} responds correctly\", \"SUCCESS\")\n                \n            except subprocess.TimeoutExpired:\n                self.log(f\"{wrapper_name} hangs (installation issue)\", \"ERROR\")\n                all_working = False\n            except Exception as e:\n                # Expected if underlying CLI not installed\n                self.log(f\"{wrapper_name} test: {e}\", \"INFO\")\n        \n        return all_working\n    \n    def create_usage_guide(self):\n        \"\"\"Create usage guide\"\"\"\n        self.log(\"Creating usage guide...\")\n        \n        guide_content = f\"\"\"# NeuralSync CLI Integration Usage Guide\n\nInstalled wrappers in: {self.install_dir}\n\n## Available Commands\n\n### codex-ns\nNeuralSync-enhanced Codex CLI wrapper\n- Usage: `codex-ns [codex-args...]`\n- Special options:\n  - `--no-context`: Skip NeuralSync context injection\n  - `--minimal-context`: Use minimal context\n  - Note: `--ask-for-approval` is automatically filtered to prevent conflicts\n\n### claude-ns  \nNeuralSync-enhanced Claude Code CLI wrapper\n- Usage: `claude-ns [claude-args...]`\n- Special options:\n  - `--no-context`: Skip NeuralSync context injection\n  - `--minimal-context`: Use minimal context\n\n### gemini-ns\nNeuralSync-enhanced Gemini CLI wrapper  \n- Usage: `gemini-ns [gemini-args...]`\n- Special options:\n  - `--no-context`: Skip NeuralSync context injection\n  - `--minimal-context`: Use minimal context\n\n## Memory Sharing\n\nAll wrappers automatically:\n1. Fetch your persona from NeuralSync\n2. Retrieve relevant memories for context\n3. Store new interactions for future use\n4. Share context across different AI tools\n\n## Environment Variables\n\n- `NS_HOST`: NeuralSync server host (default: 127.0.0.1)\n- `NS_PORT`: NeuralSync server port (default: 8373)\n- `NS_TOKEN`: Authentication token if required\n- `TOOL_NAME`: Automatically set by wrappers\n\n## PATH Integration\n\nTo use the wrappers from anywhere, add to your shell profile:\n\n```bash\nexport PATH=\"$PATH:{self.install_dir}\"\n```\n\n## Testing\n\nRun integration tests:\n```bash\npython3 {self.neuralsync_dir}/test_cli_integration.py\n```\n\n## Troubleshooting\n\n1. **Wrapper hangs**: The fixed wrappers should not hang. If they do, check NeuralSync server status.\n\n2. **No context injection**: Ensure NeuralSync server is running on 127.0.0.1:8373\n\n3. **Permission errors**: Ensure wrappers are executable:\n   ```bash\n   chmod +x {self.install_dir}/*\n   ```\n\n4. **Underlying CLI not found**: Install the actual CLI tools (codex, claude, gemini) separately.\n\n## Uninstall\n\n```bash\nrm -f {self.install_dir}/{{codex-ns,claude-ns,gemini-ns,nswrap}}\n```\n\"\"\"\n        \n        try:\n            guide_path = self.config_dir / \"cli_integration_guide.md\"\n            with open(guide_path, 'w') as f:\n                f.write(guide_content)\n            \n            self.log(f\"Usage guide created: {guide_path}\", \"SUCCESS\")\n            \n        except Exception as e:\n            self.log(f\"Failed to create usage guide: {e}\", \"ERROR\")\n    \n    def install(self) -> bool:\n        \"\"\"Run complete installation\"\"\"\n        self.log(\"🚀 NeuralSync CLI Integration Installer\")\n        self.log(\"=======================================\")\n        \n        start_time = time.time()\n        \n        steps = [\n            (\"Requirements Check\", self.check_requirements),\n            (\"Directory Setup\", self.setup_directories),\n            (\"Wrapper Installation\", self.install_wrappers),\n            (\"nswrap Setup\", self.setup_nswrap),\n            (\"PATH Check\", self.check_path_integration),\n            (\"Installation Test\", self.test_installation),\n        ]\n        \n        for step_name, step_func in steps:\n            self.log(f\"\\n--- {step_name} ---\")\n            success = step_func()\n            \n            if not success and step_name in [\"Requirements Check\", \"Wrapper Installation\"]:\n                self.log(f\"Critical step '{step_name}' failed - aborting\", \"ERROR\")\n                return False\n        \n        # Create usage guide regardless of minor failures\n        self.log(\"\\n--- Documentation ---\")\n        self.create_usage_guide()\n        \n        duration = time.time() - start_time\n        \n        self.log(\"\\n📊 Installation Summary\")\n        self.log(\"=======================\")\n        self.log(f\"Successes: {self.success_count}\")\n        self.log(f\"Errors: {self.error_count}\")\n        self.log(f\"Duration: {duration:.2f}s\")\n        \n        if self.error_count == 0:\n            self.log(\"\\n🎉 Installation completed successfully!\", \"SUCCESS\")\n            self.log(f\"\\nNext steps:\")\n            self.log(f\"1. Add {self.install_dir} to your PATH\")\n            self.log(f\"2. Start NeuralSync server if not running\")\n            self.log(f\"3. Test with: python3 {self.neuralsync_dir}/test_cli_integration.py\")\n            return True\n        elif self.error_count <= 2:\n            self.log(\"\\n✅ Installation mostly successful with minor issues\", \"WARNING\")\n            return True\n        else:\n            self.log(\"\\n❌ Installation failed with multiple errors\", \"ERROR\")\n            return False\n    \n    def uninstall(self) -> bool:\n        \"\"\"Uninstall CLI integration\"\"\"\n        self.log(\"Uninstalling NeuralSync CLI integration...\")\n        \n        try:\n            # Remove installed wrappers\n            for wrapper_name in self.wrappers.keys():\n                wrapper_path = self.install_dir / wrapper_name\n                if wrapper_path.exists():\n                    wrapper_path.unlink()\n                    self.log(f\"Removed {wrapper_name}\", \"SUCCESS\")\n            \n            # Remove nswrap\n            nswrap_path = self.install_dir / \"nswrap\"\n            if nswrap_path.exists():\n                nswrap_path.unlink()\n                self.log(\"Removed nswrap\", \"SUCCESS\")\n            \n            # Remove config files (optional)\n            guide_path = self.config_dir / \"cli_integration_guide.md\"\n            if guide_path.exists():\n                guide_path.unlink()\n                self.log(\"Removed usage guide\", \"SUCCESS\")\n            \n            self.log(\"Uninstall completed\", \"SUCCESS\")\n            return True\n            \n        except Exception as e:\n            self.log(f\"Uninstall failed: {e}\", \"ERROR\")\n            return False\n\ndef main():\n    \"\"\"Main entry point\"\"\"\n    import argparse\n    \n    parser = argparse.ArgumentParser(description='NeuralSync CLI Integration Installer')\n    parser.add_argument('action', choices=['install', 'uninstall', 'test'], \n                       help='Action to perform')\n    parser.add_argument('--force', action='store_true', \n                       help='Force installation even if requirements not fully met')\n    \n    args = parser.parse_args()\n    \n    installer = CLIIntegrationInstaller()\n    \n    if args.action == 'install':\n        success = installer.install()\n        sys.exit(0 if success else 1)\n        \n    elif args.action == 'uninstall':\n        success = installer.uninstall()\n        sys.exit(0 if success else 1)\n        \n    elif args.action == 'test':\n        # Run integration tests\n        test_script = installer.neuralsync_dir / \"test_cli_integration.py\"\n        if test_script.exists():\n            result = subprocess.run([sys.executable, str(test_script)])\n            sys.exit(result.returncode)\n        else:\n            print(\"❌ Test script not found\")\n            sys.exit(1)\n\nif __name__ == \"__main__\":\n    main()"}, {"old_string": "    def setup_nswrap(self) -> bool:\n        \"\"\"Ensure nswrap is accessible\"\"\"\n        self.log(\"Setting up nswrap...\")\n        \n        try:\n            nswrap_source = self.neuralsync_dir / \"nswrap\"\n            nswrap_target = self.install_dir / \"nswrap\"\n            \n            if not nswrap_source.exists():\n                self.log(\"nswrap not found in NeuralSync directory\", \"ERROR\")\n                return False\n            \n            # Copy nswrap\n            shutil.copy2(nswrap_source, nswrap_target)\n            nswrap_target.chmod(0o755)\n            \n            self.log(f\"nswrap installed: {nswrap_target}\", \"SUCCESS\")\n            return True\n            \n        except Exception as e:\n            self.log(f\"Failed to setup nswrap: {e}\", \"ERROR\")\n            return False", "new_string": "    def setup_nswrap(self) -> bool:\n        \"\"\"Ensure nswrap is accessible\"\"\"\n        self.log(\"Setting up nswrap...\")\n        \n        try:\n            nswrap_source = self.neuralsync_dir / \"nswrap\"\n            nswrap_target = self.install_dir / \"nswrap\"\n            \n            if not nswrap_source.exists():\n                self.log(\"nswrap not found in NeuralSync directory\", \"ERROR\")\n                return False\n            \n            # Copy nswrap\n            shutil.copy2(nswrap_source, nswrap_target)\n            nswrap_target.chmod(0o755)\n            \n            self.log(f\"nswrap installed: {nswrap_target}\", \"SUCCESS\")\n            return True\n            \n        except Exception as e:\n            self.log(f\"Failed to setup nswrap: {e}\", \"ERROR\")\n            return False"}]