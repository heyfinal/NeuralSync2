#!/usr/bin/env python3
"""
NeuralSync Auto-Launch Integration Installation Validator
Quick validation script to verify installation completeness and functionality
"""

import asyncio
import json
import os
import sys
import subprocess
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class InstallationValidator:
    """Quick installation validation and health check"""
    
    def __init__(self):
        self.install_dir = Path.cwd()
        self.bin_dir = Path.home() / ".local" / "bin"
        self.config_dir = Path.home() / ".neuralsync"
        
        self.wrapper_commands = ['claude-ns', 'codex-ns', 'gemini-ns']
        self.validation_results = []
        
    def record_result(self, check_name: str, success: bool, message: str = "", details: Any = None):
        """Record validation result"""
        self.validation_results.append({
            'check': check_name,
            'success': success,
            'message': message,
            'details': details
        })
        
        status = "✅" if success else "❌"
        print(f"{status} {check_name}" + (f": {message}" if message else ""))
        
    def check_wrapper_scripts(self) -> bool:
        """Check if wrapper scripts are installed and executable"""
        print("🔍 Checking wrapper scripts...")
        all_good = True
        
        for cmd in self.wrapper_commands:
            script_path = self.bin_dir / cmd
            
            # Check file exists
            if not script_path.exists():
                self.record_result(
                    f"wrapper.{cmd}.exists", 
                    False, 
                    f"Not found at {script_path}"
                )
                all_good = False
                continue
                
            # Check executable
            if not os.access(script_path, os.X_OK):
                self.record_result(
                    f"wrapper.{cmd}.executable", 
                    False, 
                    "Not executable"
                )
                all_good = False
                continue
                
            # Check in PATH
            in_path = shutil.which(cmd) is not None
            if not in_path:
                self.record_result(
                    f"wrapper.{cmd}.path", 
                    False, 
                    "Not in PATH - may need terminal restart"
                )
                # Not a failure, just a warning
            else:
                self.record_result(f"wrapper.{cmd}.path", True, "Available in PATH")
                
            self.record_result(f"wrapper.{cmd}.installed", True, "Installed correctly")
            
        return all_good
        
    def check_configuration(self) -> bool:
        """Check NeuralSync configuration"""
        print("🔍 Checking configuration...")
        
        config_file = self.config_dir / "config.yaml"
        
        if not config_file.exists():
            self.record_result("config.exists", False, f"Config file not found: {config_file}")
            return False
            
        try:
            with open(config_file, 'r') as f:
                import yaml
                config = yaml.safe_load(f)
                
            required_keys = ['site_id', 'db_path', 'oplog_path', 'bind_host', 'bind_port']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                self.record_result(
                    "config.completeness", 
                    False, 
                    f"Missing keys: {missing_keys}"
                )
                return False
                
            self.record_result("config.valid", True, "Configuration file valid")
            
            # Check directories exist
            db_dir = Path(config['db_path']).parent
            oplog_dir = Path(config['oplog_path']).parent
            
            db_dir.mkdir(parents=True, exist_ok=True)
            oplog_dir.mkdir(parents=True, exist_ok=True)
            
            self.record_result("config.directories", True, "Data directories ready")
            
            return True
            
        except Exception as e:
            self.record_result("config.parse", False, f"Config parse error: {e}")
            return False
            
    def check_python_environment(self) -> bool:
        """Check Python environment and dependencies"""
        print("🔍 Checking Python environment...")
        
        venv_path = self.install_dir / ".venv"
        if not venv_path.exists():
            self.record_result("python.venv", False, "Virtual environment not found")
            return False
            
        # Check Python executable
        if os.name == 'nt':  # Windows
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
            
        if not python_exe.exists():
            self.record_result("python.executable", False, "Python executable not found in venv")
            return False
            
        self.record_result("python.venv", True, "Virtual environment found")
        
        # Test key imports
        test_script = '''
import sys
sys.path.insert(0, ".")

try:
    import neuralsync
    from neuralsync.daemon_manager import get_daemon_manager
    from neuralsync.ultra_comm import get_comm_manager
    from neuralsync.agent_sync import get_agent_synchronizer
    print("SUCCESS: All key modules imported")
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    sys.exit(1)
'''
        
        try:
            result = subprocess.run(
                [str(python_exe), '-c', test_script],
                cwd=self.install_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.record_result("python.imports", True, "Core modules import successfully")
                return True
            else:
                self.record_result(
                    "python.imports", 
                    False, 
                    f"Import test failed: {result.stderr}"
                )
                return False
                
        except subprocess.TimeoutExpired:
            self.record_result("python.imports", False, "Import test timed out")
            return False
        except Exception as e:
            self.record_result("python.imports", False, f"Import test error: {e}")
            return False
            
    async def check_basic_functionality(self) -> bool:
        """Check basic functionality of wrapper scripts"""
        print("🔍 Checking basic functionality...")
        
        all_good = True
        
        for cmd in self.wrapper_commands:
            # Skip if command not in PATH
            if not shutil.which(cmd):
                self.record_result(
                    f"function.{cmd}.skip", 
                    True, 
                    "Skipped - not in PATH (restart terminal)"
                )
                continue
                
            try:
                # Test status command with timeout
                process = await asyncio.create_subprocess_exec(
                    cmd, '--neuralsync-status',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), 
                        timeout=15
                    )
                    
                    if process.returncode == 0:
                        # Try to parse JSON response
                        try:
                            status_data = json.loads(stdout.decode())
                            self.record_result(
                                f"function.{cmd}.status", 
                                True, 
                                "Status command works"
                            )
                        except json.JSONDecodeError:
                            self.record_result(
                                f"function.{cmd}.status", 
                                False, 
                                "Invalid JSON response"
                            )
                            all_good = False
                    else:
                        self.record_result(
                            f"function.{cmd}.status", 
                            False, 
                            f"Command failed: {stderr.decode()[:100]}"
                        )
                        all_good = False
                        
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    self.record_result(
                        f"function.{cmd}.status", 
                        False, 
                        "Command timed out"
                    )
                    all_good = False
                    
            except Exception as e:
                self.record_result(
                    f"function.{cmd}.status", 
                    False, 
                    f"Test failed: {e}"
                )
                all_good = False
                
        return all_good
        
    def check_system_requirements(self) -> bool:
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        
        all_good = True
        
        # Check Python version
        python_version = sys.version_info
        min_version = (3, 8)
        
        if python_version >= min_version:
            self.record_result(
                "system.python_version", 
                True, 
                f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
            )
        else:
            self.record_result(
                "system.python_version", 
                False, 
                f"Python {python_version.major}.{python_version.minor} < required 3.8+"
            )
            all_good = False
            
        # Check required commands
        required_commands = ['python3', 'pip']
        for cmd in required_commands:
            if shutil.which(cmd):
                self.record_result(f"system.{cmd}", True, "Available")
            else:
                self.record_result(f"system.{cmd}", False, "Not found")
                all_good = False
                
        return all_good
        
    def check_optional_tools(self) -> bool:
        """Check optional CLI tools that can be wrapped"""
        print("🔍 Checking optional CLI tools...")
        
        optional_tools = ['claude-code', 'codexcli', 'gemini']
        found_tools = []
        
        for tool in optional_tools:
            if shutil.which(tool):
                self.record_result(f"optional.{tool}", True, "Available")
                found_tools.append(tool)
            else:
                self.record_result(f"optional.{tool}", False, "Not found")
                
        if found_tools:
            self.record_result(
                "optional.summary", 
                True, 
                f"Found {len(found_tools)} tools: {', '.join(found_tools)}"
            )
        else:
            self.record_result(
                "optional.summary", 
                False, 
                "No optional tools found - install for full functionality"
            )
            
        return len(found_tools) > 0
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        total_checks = len(self.validation_results)
        passed_checks = len([r for r in self.validation_results if r['success']])
        failed_checks = total_checks - passed_checks
        
        # Categorize results
        categories = {}
        critical_failures = 0
        
        for result in self.validation_results:
            category = result['check'].split('.')[0]
            if category not in categories:
                categories[category] = {'passed': 0, 'failed': 0, 'checks': []}
                
            if result['success']:
                categories[category]['passed'] += 1
            else:
                categories[category]['failed'] += 1
                
                # Count critical failures (not optional)
                if category not in ['optional']:
                    critical_failures += 1
                    
            categories[category]['checks'].append(result)
            
        # Determine overall status
        if critical_failures == 0:
            if failed_checks == 0:
                overall_status = "EXCELLENT"
            else:
                overall_status = "GOOD"  # Only optional failures
        elif critical_failures <= 2:
            overall_status = "ISSUES"
        else:
            overall_status = "POOR"
            
        return {
            'overall_status': overall_status,
            'summary': {
                'total_checks': total_checks,
                'passed': passed_checks,
                'failed': failed_checks,
                'critical_failures': critical_failures,
                'success_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0
            },
            'categories': categories,
            'detailed_results': self.validation_results
        }
        
    def print_validation_report(self, report: Dict[str, Any]):
        """Print formatted validation report"""
        summary = report['summary']
        status = report['overall_status']
        
        # Status emoji
        status_emoji = {
            'EXCELLENT': '🎉',
            'GOOD': '✅', 
            'ISSUES': '⚠️',
            'POOR': '❌'
        }
        
        print("\n" + "="*80)
        print("🔍 NeuralSync Installation Validation Report")
        print("="*80)
        print(f"{status_emoji.get(status, '❓')} Overall Status: {status}")
        print(f"📊 Success Rate: {summary['success_rate']:.1f}%")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"🚨 Critical Issues: {summary['critical_failures']}")
        print()
        
        # Category breakdown
        print("📋 Validation Categories:")
        for category, stats in report['categories'].items():
            total = stats['passed'] + stats['failed']
            rate = (stats['passed'] / total * 100) if total > 0 else 0
            category_status = "✅" if stats['failed'] == 0 else "❌"
            print(f"  {category_status} {category.title()}: {stats['passed']}/{total} ({rate:.1f}%)")
            
        # Failed checks
        failed_checks = [r for r in self.validation_results if not r['success']]
        if failed_checks:
            print("\n❌ Failed Checks:")
            for check in failed_checks:
                print(f"  • {check['check']}: {check['message']}")
                
        # Recommendations
        print("\n💡 Recommendations:")
        
        if status == "EXCELLENT":
            print("  🎉 Perfect! Your installation is ready to use.")
            print("  🚀 Try: claude-ns --neuralsync-status")
            
        elif status == "GOOD":
            print("  ✅ Installation is functional with minor issues.")
            optional_failures = [r for r in failed_checks if r['check'].startswith('optional.')]
            if optional_failures:
                print("  📦 Consider installing optional CLI tools for full functionality:")
                print("     - claude-code: Claude Code CLI")
                print("     - codexcli: OpenAI Codex CLI") 
                print("     - gemini: Google Gemini CLI")
                
        elif status == "ISSUES":
            print("  ⚠️  Installation has issues that should be addressed:")
            critical_failures = [r for r in failed_checks 
                               if not r['check'].startswith('optional.')]
            for failure in critical_failures[:3]:  # Show top 3
                print(f"     - {failure['check']}: {failure['message']}")
            print("  🔧 Run: python3 install_neuralsync.py")
            
        elif status == "POOR":
            print("  ❌ Installation has major issues. Reinstallation recommended:")
            print("  🗑️  Run: python3 uninstall_neuralsync.py")
            print("  📦 Run: python3 install_neuralsync.py")
            
        print("\n📖 For help and documentation:")
        print("  • Check README.md for usage instructions")
        print("  • Run validation again: python3 validate_installation.py")
        print("  • Full test suite: python3 tests/integration_test.py")
        print()
        print("="*80)
        
    async def run_validation(self) -> bool:
        """Run all validation checks"""
        print("🔍 Starting NeuralSync Installation Validation")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Run validation checks
            checks = [
                ("System Requirements", self.check_system_requirements),
                ("Configuration", self.check_configuration),
                ("Python Environment", self.check_python_environment),
                ("Wrapper Scripts", self.check_wrapper_scripts),
                ("Optional Tools", self.check_optional_tools),
                ("Basic Functionality", self.check_basic_functionality)
            ]
            
            for check_name, check_func in checks:
                print(f"\n🔍 {check_name}...")
                try:
                    if asyncio.iscoroutinefunction(check_func):
                        await check_func()
                    else:
                        check_func()
                except Exception as e:
                    logger.error(f"Validation check {check_name} failed: {e}")
                    self.record_result(
                        f"{check_name.lower().replace(' ', '_')}.error",
                        False,
                        f"Check failed: {e}"
                    )
                    
            # Generate and print report
            report = self.generate_validation_report()
            self.print_validation_report(report)
            
            # Timing
            duration = time.time() - start_time
            print(f"⏱️  Validation completed in {duration:.2f} seconds")
            
            return report['overall_status'] in ['EXCELLENT', 'GOOD']
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False


async def main():
    """Main validation entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuralSync Installation Validator')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--json', action='store_true', help='Output JSON report')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        
    validator = InstallationValidator()
    success = await validator.run_validation()
    
    if args.json:
        report = validator.generate_validation_report()
        print(json.dumps(report, indent=2))
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())