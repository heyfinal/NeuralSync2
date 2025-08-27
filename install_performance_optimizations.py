#!/usr/bin/env python3
"""
Performance Optimization Installer for NeuralSync v2
Comprehensive installer for all performance enhancements and monitoring
"""

import os
import sys
import shutil
import subprocess
import logging
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceOptimizationInstaller:
    """Comprehensive installer for NeuralSync v2 performance optimizations"""
    
    def __init__(self, install_dir: str = None, sudo_password: str = None):
        self.system_type = platform.system().lower()
        self.python_executable = sys.executable
        
        # Installation paths
        if install_dir is None:
            if self.system_type == 'darwin':
                install_dir = '/usr/local/bin'
            else:
                install_dir = '/usr/local/bin'
        
        self.install_dir = Path(install_dir)
        self.source_dir = Path(__file__).parent
        self.config_dir = Path.home() / '.neuralsync'
        self.cache_dir = Path('/tmp/neuralsync_cache')
        
        # Sudo password for system operations
        self.sudo_password = sudo_password
        self.sudo_cached = False
        
        # Installation components
        self.components = {
            'core_optimizations': {
                'description': 'Core performance optimization modules',
                'files': [
                    'intelligent_cache.py',
                    'async_network.py', 
                    'context_prewarmer.py',
                    'fast_recall.py',
                    'lazy_loader.py',
                    'cli_performance_integration.py',
                    'optimized_server.py'
                ],
                'required': True
            },
            'optimized_cli': {
                'description': 'Optimized CLI wrapper (nswrap)',
                'files': ['nswrap_optimized'],
                'executable': True,
                'required': True
            },
            'performance_monitoring': {
                'description': 'Enhanced performance monitoring',
                'files': ['performance_monitor.py'],
                'required': False
            },
            'dependencies': {
                'description': 'Performance optimization dependencies',
                'packages': [
                    'aiohttp>=3.8.0',
                    'lz4>=4.0.0',
                    'psutil>=5.9.0',
                    'numpy>=1.21.0',
                    'httptools>=0.4.0'
                ],
                'required': True
            },
            'system_configs': {
                'description': 'System-level performance configurations',
                'configs': [
                    ('NS_FAST_MODE', '1'),
                    ('NS_PRELOAD', '1'),
                    ('NS_MAX_WAIT_MS', '800'),
                    ('NS_LOADING_MODE', 'adaptive')
                ],
                'required': False
            }
        }
        
        # Installation status
        self.installation_log = []
        self.errors = []
        self.warnings = []
        
        logger.info("PerformanceOptimizationInstaller initialized")
    
    def run_with_sudo(self, command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run command with sudo, using cached password if available"""
        
        if self.system_type == 'darwin' or self.system_type == 'linux':
            if not self.sudo_cached and self.sudo_password:
                # Cache sudo password
                cache_cmd = ['sudo', '-S', 'true']
                result = subprocess.run(
                    cache_cmd,
                    input=f"{self.sudo_password}\n",
                    text=True,
                    capture_output=True
                )
                if result.returncode == 0:
                    self.sudo_cached = True
                    logger.info("Sudo access cached successfully")
                else:
                    raise RuntimeError(f"Sudo authentication failed: {result.stderr}")
            
            sudo_command = ['sudo'] + command
        else:
            sudo_command = command
        
        return subprocess.run(sudo_command, capture_output=capture_output, text=True)
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements for performance optimizations"""
        
        logger.info("Checking system requirements...")
        
        requirements = {
            'python_version': {
                'required': '3.8.0',
                'current': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'satisfied': sys.version_info >= (3, 8, 0)
            },
            'disk_space': {
                'required_mb': 100,
                'available_mb': 0,
                'satisfied': False
            },
            'memory': {
                'required_mb': 512,
                'available_mb': 0,
                'satisfied': False
            },
            'network_access': {
                'required': True,
                'satisfied': False
            }
        }
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage(self.install_dir.parent)
            available_mb = disk_usage.free // (1024 * 1024)
            requirements['disk_space']['available_mb'] = available_mb
            requirements['disk_space']['satisfied'] = available_mb >= requirements['disk_space']['required_mb']
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_mb = memory.available // (1024 * 1024)
            requirements['memory']['available_mb'] = available_mb
            requirements['memory']['satisfied'] = available_mb >= requirements['memory']['required_mb']
        except ImportError:
            logger.warning("psutil not available for memory check")
            requirements['memory']['satisfied'] = True  # Assume satisfied
        
        # Check network access
        try:
            import urllib.request
            urllib.request.urlopen('https://pypi.org', timeout=5)
            requirements['network_access']['satisfied'] = True
        except Exception:
            logger.warning("Network access check failed")
        
        # Log results
        for requirement, details in requirements.items():
            if details['satisfied']:
                logger.info(f"✓ {requirement}: {details.get('current', 'OK')}")
            else:
                logger.error(f"✗ {requirement}: {details}")
                self.errors.append(f"System requirement not met: {requirement}")
        
        return requirements
    
    def install_dependencies(self) -> bool:
        """Install required Python dependencies"""
        
        logger.info("Installing performance optimization dependencies...")
        
        packages = self.components['dependencies']['packages']
        
        try:
            # Upgrade pip first
            subprocess.check_call([
                self.python_executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ])
            
            # Install packages
            install_cmd = [
                self.python_executable, '-m', 'pip', 'install', '--upgrade'
            ] + packages
            
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Dependencies installed successfully")
                self.installation_log.append("Dependencies installed")
                return True
            else:
                logger.error(f"Dependency installation failed: {result.stderr}")
                self.errors.append(f"Dependency installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Dependency installation error: {e}")
            self.errors.append(f"Dependency installation error: {e}")
            return False
    
    def install_core_optimizations(self) -> bool:
        """Install core optimization modules"""
        
        logger.info("Installing core performance optimization modules...")
        
        try:
            # Ensure neuralsync module directory exists
            neuralsync_dir = self.source_dir / 'neuralsync'
            if not neuralsync_dir.exists():
                self.errors.append("Source neuralsync directory not found")
                return False
            
            # Files are already in place, just validate they exist
            core_files = self.components['core_optimizations']['files']
            missing_files = []
            
            for filename in core_files:
                file_path = neuralsync_dir / filename
                if not file_path.exists():
                    missing_files.append(filename)
            
            if missing_files:
                self.errors.append(f"Missing core optimization files: {missing_files}")
                return False
            
            logger.info("Core optimization modules validated")
            self.installation_log.append("Core optimizations validated")
            return True
            
        except Exception as e:
            logger.error(f"Core optimization installation error: {e}")
            self.errors.append(f"Core optimization installation error: {e}")
            return False
    
    def install_optimized_cli(self) -> bool:
        """Install optimized CLI wrapper"""
        
        logger.info("Installing optimized CLI wrapper...")
        
        try:
            source_file = self.source_dir / 'nswrap_optimized'
            target_file = self.install_dir / 'nswrap'
            
            if not source_file.exists():
                self.errors.append("nswrap_optimized source file not found")
                return False
            
            # Create install directory if it doesn't exist
            self.install_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file with sudo if needed
            try:
                if self.install_dir.is_dir() and os.access(self.install_dir, os.W_OK):
                    # Directory is writable
                    shutil.copy2(source_file, target_file)
                    os.chmod(target_file, 0o755)
                else:
                    # Need sudo
                    result = self.run_with_sudo(['cp', str(source_file), str(target_file)])
                    if result.returncode != 0:
                        raise RuntimeError(f"Copy failed: {result.stderr}")
                    
                    # Make executable
                    result = self.run_with_sudo(['chmod', '755', str(target_file)])
                    if result.returncode != 0:
                        logger.warning(f"Could not make executable: {result.stderr}")
                
                logger.info(f"Optimized CLI installed to {target_file}")
                self.installation_log.append(f"CLI installed to {target_file}")
                
                # Create backup of original nswrap if it exists
                original_nswrap = self.source_dir / 'nswrap'
                if original_nswrap.exists():
                    backup_file = self.source_dir / 'nswrap_original'
                    if not backup_file.exists():
                        shutil.copy2(original_nswrap, backup_file)
                        logger.info("Original nswrap backed up")
                
                return True
                
            except Exception as e:
                logger.error(f"CLI installation failed: {e}")
                self.errors.append(f"CLI installation failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"CLI installation error: {e}")
            self.errors.append(f"CLI installation error: {e}")
            return False
    
    def setup_configuration(self) -> bool:
        """Setup configuration files and environment"""
        
        logger.info("Setting up performance configuration...")
        
        try:
            # Create config directory
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create performance configuration file
            config_file = self.config_dir / 'performance.json'
            performance_config = {
                'version': '2.0.0',
                'optimizations_enabled': True,
                'fast_mode': True,
                'preloading_enabled': True,
                'max_wait_ms': 800,
                'loading_mode': 'adaptive',
                'cache_settings': {
                    'persona_ttl_ms': 600000,
                    'memory_ttl_ms': 300000,
                    'context_ttl_ms': 180000,
                    'max_cache_size_mb': 256
                },
                'performance_thresholds': {
                    'excellent_ms': 200,
                    'good_ms': 500,
                    'acceptable_ms': 1000
                },
                'installed_components': list(self.components.keys())
            }
            
            with open(config_file, 'w') as f:
                json.dump(performance_config, f, indent=2)
            
            logger.info(f"Performance configuration created: {config_file}")
            self.installation_log.append("Configuration files created")
            
            # Setup shell environment suggestions
            shell_config = self._generate_shell_config()
            
            shell_config_file = self.config_dir / 'shell_config.sh'
            with open(shell_config_file, 'w') as f:
                f.write(shell_config)
            
            logger.info(f"Shell configuration generated: {shell_config_file}")
            self.installation_log.append("Shell configuration generated")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration setup error: {e}")
            self.errors.append(f"Configuration setup error: {e}")
            return False
    
    def _generate_shell_config(self) -> str:
        """Generate shell configuration for performance optimizations"""
        
        config_lines = [
            "#!/bin/bash",
            "# NeuralSync v2 Performance Optimization Configuration",
            "# Source this file in your shell profile (.bashrc, .zshrc, etc.)",
            "",
            "# Performance optimization environment variables",
        ]
        
        for key, value in self.components['system_configs']['configs']:
            config_lines.append(f"export {key}={value}")
        
        config_lines.extend([
            "",
            "# Additional optimization settings",
            "export NS_CACHE_DIR=/tmp/neuralsync_cache",
            f"export NS_CONFIG_DIR={self.config_dir}",
            "export NS_DEBUG_PERF=0  # Set to 1 for debug output",
            "",
            "# Ensure nswrap is in PATH",
            f"export PATH=\"{self.install_dir}:$PATH\"",
            "",
            "# Optional: Create aliases for common operations",
            "alias ns-stats='python3 -c \"from neuralsync.cli_performance_integration import get_performance_integration; import json; print(json.dumps(get_performance_integration().get_performance_summary(), indent=2))\"'",
            "alias ns-cache-clear='python3 -c \"from neuralsync.intelligent_cache import get_neuralsync_cache; import asyncio; asyncio.run(get_neuralsync_cache().persona_cache.clear())\"'",
            "",
            "echo '✅ NeuralSync v2 Performance Optimizations Loaded'"
        ])
        
        return "\n".join(config_lines)
    
    def validate_installation(self) -> Dict[str, bool]:
        """Validate that all components are properly installed"""
        
        logger.info("Validating installation...")
        
        validation_results = {}
        
        # Check CLI wrapper
        nswrap_path = self.install_dir / 'nswrap'
        validation_results['cli_wrapper'] = nswrap_path.exists() and os.access(nswrap_path, os.X_OK)
        
        # Check core modules
        core_modules_valid = True
        try:
            import neuralsync.intelligent_cache
            import neuralsync.async_network
            import neuralsync.context_prewarmer
            import neuralsync.fast_recall
            import neuralsync.lazy_loader
            import neuralsync.cli_performance_integration
        except ImportError as e:
            core_modules_valid = False
            self.errors.append(f"Module import failed: {e}")
        
        validation_results['core_modules'] = core_modules_valid
        
        # Check dependencies
        dependencies_valid = True
        try:
            import aiohttp
            import lz4
            import psutil
            import numpy
        except ImportError as e:
            dependencies_valid = False
            self.errors.append(f"Dependency import failed: {e}")
        
        validation_results['dependencies'] = dependencies_valid
        
        # Check configuration
        config_file = self.config_dir / 'performance.json'
        validation_results['configuration'] = config_file.exists()
        
        # Test basic functionality
        functionality_valid = True
        try:
            # Quick functionality test
            result = subprocess.run([
                str(nswrap_path), '--', 'echo', 'test'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0 or 'test' not in result.stdout:
                functionality_valid = False
                self.errors.append(f"Functionality test failed: {result.stderr}")
                
        except Exception as e:
            functionality_valid = False
            self.errors.append(f"Functionality test error: {e}")
        
        validation_results['functionality'] = functionality_valid
        
        # Log validation results
        for component, is_valid in validation_results.items():
            if is_valid:
                logger.info(f"✓ {component} validation passed")
            else:
                logger.error(f"✗ {component} validation failed")
        
        return validation_results
    
    def create_uninstaller(self) -> bool:
        """Create uninstaller script"""
        
        logger.info("Creating uninstaller...")
        
        try:
            uninstaller_content = f'''#!/usr/bin/env python3
"""
NeuralSync v2 Performance Optimizations Uninstaller
Auto-generated uninstaller script
"""

import os
import shutil
import subprocess
from pathlib import Path

def uninstall():
    print("🗑️  Uninstalling NeuralSync v2 Performance Optimizations...")
    
    items_removed = []
    
    # Remove CLI wrapper
    cli_file = Path("{self.install_dir}/nswrap")
    if cli_file.exists():
        try:
            if os.access(cli_file.parent, os.W_OK):
                cli_file.unlink()
            else:
                subprocess.run(["sudo", "rm", str(cli_file)], check=True)
            items_removed.append(str(cli_file))
        except Exception as e:
            print(f"Warning: Could not remove {{cli_file}}: {{e}}")
    
    # Remove configuration directory
    config_dir = Path("{self.config_dir}")
    if config_dir.exists():
        try:
            shutil.rmtree(config_dir)
            items_removed.append(str(config_dir))
        except Exception as e:
            print(f"Warning: Could not remove {{config_dir}}: {{e}}")
    
    # Remove cache directory
    cache_dir = Path("{self.cache_dir}")
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            items_removed.append(str(cache_dir))
        except Exception as e:
            print(f"Warning: Could not remove {{cache_dir}}: {{e}}")
    
    # Restore original nswrap if backup exists
    original_backup = Path("{self.source_dir}/nswrap_original")
    original_nswrap = Path("{self.source_dir}/nswrap")
    if original_backup.exists():
        try:
            shutil.copy2(original_backup, original_nswrap)
            original_backup.unlink()
            items_removed.append("Restored original nswrap")
        except Exception as e:
            print(f"Warning: Could not restore original nswrap: {{e}}")
    
    print(f"✅ Uninstallation complete. Removed {{len(items_removed)}} items:")
    for item in items_removed:
        print(f"   - {{item}}")
    
    print("\\n📝 Manual cleanup needed:")
    print("   - Remove environment variables from shell profile")
    print("   - Uninstall Python dependencies if no longer needed:")
    print("     pip uninstall aiohttp lz4 psutil httptools")

if __name__ == "__main__":
    uninstall()
'''
            
            uninstaller_path = self.source_dir / 'uninstall_performance_optimizations.py'
            with open(uninstaller_path, 'w') as f:
                f.write(uninstaller_content)
            
            os.chmod(uninstaller_path, 0o755)
            
            logger.info(f"Uninstaller created: {uninstaller_path}")
            self.installation_log.append("Uninstaller created")
            return True
            
        except Exception as e:
            logger.error(f"Uninstaller creation failed: {e}")
            self.warnings.append(f"Uninstaller creation failed: {e}")
            return False
    
    def run_performance_test(self) -> Dict[str, Any]:
        """Run quick performance test to verify optimizations"""
        
        logger.info("Running performance validation test...")
        
        test_results = {
            'cli_response_times': [],
            'cache_performance': {},
            'overall_health': 'unknown'
        }
        
        try:
            nswrap_path = self.install_dir / 'nswrap'
            
            # Test CLI response times
            test_commands = [
                ['echo', 'hello'],
                ['pwd'],
                ['date']
            ]
            
            for cmd in test_commands:
                start_time = time.perf_counter()
                
                result = subprocess.run([
                    str(nswrap_path), '--'
                ] + cmd, capture_output=True, text=True, timeout=5)
                
                response_time_ms = (time.perf_counter() - start_time) * 1000
                
                test_results['cli_response_times'].append({
                    'command': ' '.join(cmd),
                    'response_time_ms': response_time_ms,
                    'success': result.returncode == 0
                })
            
            # Calculate average response time
            successful_times = [
                t['response_time_ms'] for t in test_results['cli_response_times']
                if t['success']
            ]
            
            if successful_times:
                avg_time = sum(successful_times) / len(successful_times)
                
                if avg_time < 200:
                    test_results['overall_health'] = 'excellent'
                elif avg_time < 500:
                    test_results['overall_health'] = 'good'
                elif avg_time < 1000:
                    test_results['overall_health'] = 'acceptable'
                else:
                    test_results['overall_health'] = 'needs_improvement'
                
                logger.info(f"Average CLI response time: {avg_time:.1f}ms ({test_results['overall_health']})")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            test_results['error'] = str(e)
            return test_results
    
    def install(self, skip_sudo_prompt: bool = False) -> bool:
        """Run complete installation process"""
        
        logger.info("🚀 Starting NeuralSync v2 Performance Optimization Installation")
        
        # Request sudo password if needed
        if not skip_sudo_prompt and self.sudo_password is None and self.system_type in ['darwin', 'linux']:
            import getpass
            try:
                self.sudo_password = getpass.getpass("Enter sudo password (for system installation): ")
            except KeyboardInterrupt:
                logger.info("Installation cancelled by user")
                return False
        
        success = True
        
        # Step 1: Check system requirements
        logger.info("Step 1/7: Checking system requirements...")
        requirements = self.check_system_requirements()
        if not all(req['satisfied'] for req in requirements.values()):
            logger.error("System requirements not met")
            success = False
        
        # Step 2: Install dependencies
        if success:
            logger.info("Step 2/7: Installing dependencies...")
            if not self.install_dependencies():
                success = False
        
        # Step 3: Install core optimizations
        if success:
            logger.info("Step 3/7: Validating core optimizations...")
            if not self.install_core_optimizations():
                success = False
        
        # Step 4: Install CLI wrapper
        if success:
            logger.info("Step 4/7: Installing optimized CLI...")
            if not self.install_optimized_cli():
                success = False
        
        # Step 5: Setup configuration
        if success:
            logger.info("Step 5/7: Setting up configuration...")
            if not self.setup_configuration():
                success = False
        
        # Step 6: Validate installation
        if success:
            logger.info("Step 6/7: Validating installation...")
            validation_results = self.validate_installation()
            if not all(validation_results.values()):
                logger.warning("Some validation checks failed, but continuing...")
                for component, valid in validation_results.items():
                    if not valid:
                        self.warnings.append(f"{component} validation failed")
        
        # Step 7: Create uninstaller and run tests
        logger.info("Step 7/7: Finalizing installation...")
        self.create_uninstaller()
        
        if success:
            test_results = self.run_performance_test()
            
            # Generate installation report
            self._generate_installation_report(success, test_results, validation_results)
        
        return success
    
    def _generate_installation_report(self, success: bool, test_results: Dict[str, Any], validation_results: Dict[str, bool]):
        """Generate comprehensive installation report"""
        
        report_file = self.config_dir / 'installation_report.json'
        
        report = {
            'installation_timestamp': time.time(),
            'installation_success': success,
            'system_info': {
                'platform': platform.platform(),
                'python_version': sys.version,
                'install_dir': str(self.install_dir),
                'config_dir': str(self.config_dir)
            },
            'components_installed': list(self.components.keys()),
            'installation_log': self.installation_log,
            'errors': self.errors,
            'warnings': self.warnings,
            'validation_results': validation_results,
            'performance_test': test_results,
            'next_steps': self._generate_next_steps()
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Installation report saved: {report_file}")
        except Exception as e:
            logger.warning(f"Could not save installation report: {e}")
        
        # Print summary
        self._print_installation_summary(success, test_results)
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps for user"""
        
        steps = [
            f"Source the shell configuration: source {self.config_dir}/shell_config.sh",
            "Add the shell configuration to your profile (.bashrc, .zshrc, etc.)",
            "Test the installation with: nswrap -- echo 'Hello NeuralSync v2!'",
            "Monitor performance with: ns-stats (after sourcing shell config)",
            f"View detailed configuration at: {self.config_dir}/performance.json"
        ]
        
        if self.errors:
            steps.append("Review and resolve any installation errors listed above")
        
        if self.warnings:
            steps.append("Review installation warnings for potential improvements")
        
        return steps
    
    def _print_installation_summary(self, success: bool, test_results: Dict[str, Any]):
        """Print installation summary"""
        
        print("\n" + "="*70)
        print("🎯 NEURALSYNC v2 PERFORMANCE OPTIMIZATION INSTALLATION SUMMARY")
        print("="*70)
        
        if success:
            print("✅ Installation completed successfully!")
        else:
            print("❌ Installation completed with errors")
        
        print(f"\n📊 INSTALLATION STATISTICS:")
        print(f"   Components installed: {len(self.installation_log)}")
        print(f"   Errors encountered: {len(self.errors)}")
        print(f"   Warnings: {len(self.warnings)}")
        
        if test_results.get('cli_response_times'):
            successful_times = [
                t['response_time_ms'] for t in test_results['cli_response_times']
                if t['success']
            ]
            if successful_times:
                avg_time = sum(successful_times) / len(successful_times)
                print(f"   Average CLI response: {avg_time:.1f}ms ({test_results.get('overall_health', 'unknown')})")
        
        if self.errors:
            print(f"\n❌ ERRORS:")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        print(f"\n📁 INSTALLATION PATHS:")
        print(f"   CLI Wrapper: {self.install_dir}/nswrap")
        print(f"   Configuration: {self.config_dir}")
        print(f"   Cache Directory: {self.cache_dir}")
        
        print(f"\n🚀 NEXT STEPS:")
        for i, step in enumerate(self._generate_next_steps(), 1):
            print(f"   {i}. {step}")
        
        print(f"\n🗑️  TO UNINSTALL:")
        print(f"   Run: python3 {self.source_dir}/uninstall_performance_optimizations.py")
        
        print("\n" + "="*70)
        
        if success and test_results.get('overall_health') in ['excellent', 'good']:
            print("🎉 NeuralSync v2 Performance Optimizations are ready!")
            print("   Your CLI should now respond in sub-second times.")
        elif success:
            print("✅ Installation complete - some performance tuning may be needed.")
        else:
            print("❌ Installation had issues - please review errors and try again.")
        
        print("="*70)

def main():
    """Main installer function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Install NeuralSync v2 Performance Optimizations"
    )
    parser.add_argument(
        '--install-dir',
        default=None,
        help='Installation directory for CLI tools (default: /usr/local/bin)'
    )
    parser.add_argument(
        '--skip-sudo',
        action='store_true',
        help='Skip sudo password prompt (may cause installation failures)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    installer = PerformanceOptimizationInstaller(
        install_dir=args.install_dir
    )
    
    try:
        success = installer.install(skip_sudo_prompt=args.skip_sudo)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Installation failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()