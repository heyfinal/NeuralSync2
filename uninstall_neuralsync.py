#!/usr/bin/env python3
"""
NeuralSync2 Auto-Launch Integration System Uninstaller
Clean removal of all NeuralSync components and configurations
"""

import asyncio
import os
import sys
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse
import psutil
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuralSyncUninstaller:
    """Clean NeuralSync uninstaller"""
    
    def __init__(self):
        self.install_dir = Path.cwd()
        self.bin_dir = Path.home() / ".local" / "bin"
        self.config_dir = Path.home() / ".neuralsync"
        
        # Components to remove
        self.wrapper_scripts = ['claude-ns', 'codex-ns', 'gemini-ns']
        self.process_names = ['neuralsync', 'claude-ns', 'codex-ns', 'gemini-ns']
        
    async def stop_all_processes(self) -> bool:
        """Stop all running NeuralSync processes"""
        print("🛑 Stopping NeuralSync processes...")
        
        stopped_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    cmdline = proc_info.get('cmdline', [])
                    
                    # Check if process is related to NeuralSync
                    if any(name in ' '.join(cmdline) for name in self.process_names):
                        print(f"  Stopping process: {proc_info['name']} (PID: {proc_info['pid']})")
                        
                        # Try graceful termination first
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                            stopped_processes.append(proc_info['pid'])
                        except psutil.TimeoutExpired:
                            # Force kill if graceful termination fails
                            print(f"  Force killing process: {proc_info['pid']}")
                            proc.kill()
                            proc.wait(timeout=5)
                            stopped_processes.append(proc_info['pid'])
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process already gone or access denied
                    continue
                    
        except Exception as e:
            logger.error(f"Error stopping processes: {e}")
            
        if stopped_processes:
            print(f"✅ Stopped {len(stopped_processes)} processes")
        else:
            print("✅ No NeuralSync processes found running")
            
        return True
        
    def remove_wrapper_scripts(self) -> bool:
        """Remove wrapper scripts from bin directory"""
        print("🗑️  Removing wrapper scripts...")
        
        removed_count = 0
        
        for script_name in self.wrapper_scripts:
            script_path = self.bin_dir / script_name
            if script_path.exists():
                try:
                    script_path.unlink()
                    print(f"  ✅ Removed: {script_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ❌ Failed to remove {script_path}: {e}")
                    
        if removed_count == 0:
            print("  ✅ No wrapper scripts found to remove")
            
        return True
        
    def remove_configuration(self, keep_data: bool = False) -> bool:
        """Remove NeuralSync configuration and data"""
        if keep_data:
            print("⚙️  Keeping configuration and data (--keep-data specified)")
            return True
            
        print("🗑️  Removing configuration and data...")
        
        if not self.config_dir.exists():
            print("  ✅ No configuration directory found")
            return True
            
        try:
            # List what we're about to remove
            items_to_remove = []
            for item in self.config_dir.rglob('*'):
                if item.is_file():
                    items_to_remove.append(item)
                    
            if items_to_remove:
                print(f"  Removing {len(items_to_remove)} files from {self.config_dir}")
                
                # Remove the entire directory
                shutil.rmtree(self.config_dir)
                print(f"  ✅ Removed: {self.config_dir}")
            else:
                print(f"  ✅ Configuration directory empty: {self.config_dir}")
                
        except Exception as e:
            print(f"  ❌ Failed to remove configuration: {e}")
            return False
            
        return True
        
    def clean_shell_profiles(self) -> bool:
        """Remove NeuralSync entries from shell profiles"""
        print("🧹 Cleaning shell profiles...")
        
        shell_profiles = [
            Path.home() / '.bashrc',
            Path.home() / '.zshrc',
            Path.home() / '.config' / 'fish' / 'config.fish'
        ]
        
        neuralsync_markers = [
            '# NeuralSync auto-launch integration',
            'export PATH="' + str(self.bin_dir) + ':$PATH"',
            str(self.bin_dir)
        ]
        
        cleaned_files = 0
        
        for profile in shell_profiles:
            if not profile.exists():
                continue
                
            try:
                with open(profile, 'r') as f:
                    lines = f.readlines()
                    
                # Filter out NeuralSync-related lines
                cleaned_lines = []
                skip_next = False
                
                for line in lines:
                    # Check if line contains NeuralSync content
                    if any(marker in line for marker in neuralsync_markers):
                        # Skip this line
                        continue
                    elif skip_next:
                        # Skip line after marker
                        skip_next = False
                        continue
                    elif '# NeuralSync auto-launch integration' in line:
                        # Skip this line and potentially the next
                        skip_next = True
                        continue
                    else:
                        cleaned_lines.append(line)
                        
                # Write back if changes were made
                if len(cleaned_lines) != len(lines):
                    with open(profile, 'w') as f:
                        f.writelines(cleaned_lines)
                    print(f"  ✅ Cleaned: {profile}")
                    cleaned_files += 1
                    
            except Exception as e:
                print(f"  ⚠️  Failed to clean {profile}: {e}")
                
        if cleaned_files == 0:
            print("  ✅ No shell profiles needed cleaning")
            
        return True
        
    def remove_python_environment(self, force: bool = False) -> bool:
        """Remove Python virtual environment"""
        venv_path = self.install_dir / ".venv"
        
        if not force:
            print("⚠️  Python virtual environment preserved (use --remove-venv to remove)")
            return True
            
        print("🗑️  Removing Python virtual environment...")
        
        if not venv_path.exists():
            print("  ✅ No virtual environment found")
            return True
            
        try:
            shutil.rmtree(venv_path)
            print(f"  ✅ Removed: {venv_path}")
        except Exception as e:
            print(f"  ❌ Failed to remove virtual environment: {e}")
            return False
            
        return True
        
    def remove_temp_files(self) -> bool:
        """Remove temporary files and caches"""
        print("🧹 Removing temporary files...")
        
        temp_patterns = [
            '/tmp/neuralsync_*',
            '/tmp/claude_ns_*',
            '/tmp/codex_ns_*',
            '/tmp/gemini_ns_*'
        ]
        
        removed_count = 0
        
        for pattern in temp_patterns:
            try:
                import glob
                for temp_file in glob.glob(pattern):
                    temp_path = Path(temp_file)
                    if temp_path.exists():
                        if temp_path.is_file():
                            temp_path.unlink()
                        else:
                            shutil.rmtree(temp_path)
                        removed_count += 1
                        
            except Exception as e:
                logger.debug(f"Error removing temp files for pattern {pattern}: {e}")
                
        if removed_count > 0:
            print(f"  ✅ Removed {removed_count} temporary files")
        else:
            print("  ✅ No temporary files found")
            
        return True
        
    async def verify_removal(self) -> Dict[str, Any]:
        """Verify that all components have been removed"""
        print("🔍 Verifying removal...")
        
        verification = {
            'processes': {'running': 0, 'clean': True},
            'scripts': {'remaining': 0, 'clean': True},
            'config': {'exists': self.config_dir.exists(), 'clean': True},
            'overall': 'success'
        }
        
        # Check for running processes
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info.get('cmdline', []))
                    if any(name in cmdline for name in self.process_names):
                        verification['processes']['running'] += 1
                        verification['processes']['clean'] = False
                        print(f"  ⚠️  Process still running: {proc.info['name']} (PID: {proc.info['pid']})")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.error(f"Error checking processes: {e}")
            
        # Check for remaining scripts
        for script_name in self.wrapper_scripts:
            script_path = self.bin_dir / script_name
            if script_path.exists():
                verification['scripts']['remaining'] += 1
                verification['scripts']['clean'] = False
                print(f"  ⚠️  Script still exists: {script_path}")
                
        # Check configuration
        if verification['config']['exists']:
            verification['config']['clean'] = False
            print(f"  ⚠️  Configuration still exists: {self.config_dir}")
            
        # Overall status
        if not all([verification['processes']['clean'], 
                   verification['scripts']['clean'], 
                   verification['config']['clean']]):
            verification['overall'] = 'partial'
            print("  ⚠️  Partial removal - some components remain")
        else:
            verification['overall'] = 'complete'
            print("  ✅ Complete removal verified")
            
        return verification
        
    def print_completion_message(self, verification: Dict[str, Any]):
        """Print uninstallation completion message"""
        print("\n" + "="*80)
        
        if verification['overall'] == 'complete':
            print("✅ NeuralSync Auto-Launch Integration Uninstalled Successfully!")
        else:
            print("⚠️  NeuralSync Auto-Launch Integration Partially Uninstalled")
            
        print("="*80)
        print()
        print("📋 What was removed:")
        print("  • Wrapper scripts (claude-ns, codex-ns, gemini-ns)")
        print("  • Running processes")
        print("  • Shell profile entries")
        print("  • Temporary files and caches")
        
        if verification['config']['clean']:
            print("  • Configuration and data files")
        else:
            print("  ⚠️  Configuration preserved (--keep-data used)")
            
        if verification['overall'] == 'partial':
            print("\n⚠️  Manual cleanup may be required:")
            if not verification['processes']['clean']:
                print(f"  • {verification['processes']['running']} processes still running")
            if not verification['scripts']['clean']:
                print(f"  • {verification['scripts']['remaining']} scripts still exist")
            if not verification['config']['clean']:
                print(f"  • Configuration directory: {self.config_dir}")
                
        print("\n🔄 To complete removal:")
        print("  1. Restart your terminal")
        print("  2. Check that commands are no longer available:")
        print("     which claude-ns codex-ns gemini-ns")
        print("  3. Verify no processes running:")
        print("     ps aux | grep neuralsync")
        
        if verification['overall'] == 'complete':
            print("\n🎉 NeuralSync has been completely removed from your system!")
        
        print("\n" + "="*80)
        
    async def uninstall(self, keep_data: bool = False, remove_venv: bool = False) -> bool:
        """Run complete uninstallation process"""
        print("🗑️  Starting NeuralSync Auto-Launch Integration Uninstallation")
        print(f"📍 Installation directory: {self.install_dir}")
        print(f"📍 Configuration directory: {self.config_dir}")
        print()
        
        if not keep_data:
            response = input("⚠️  This will remove ALL NeuralSync data and configuration. Continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("❌ Uninstallation cancelled")
                return False
                
        try:
            # Step 1: Stop all running processes
            success = await self.stop_all_processes()
            if not success:
                print("⚠️  Some processes may still be running")
                
            # Step 2: Remove wrapper scripts
            success = self.remove_wrapper_scripts()
            if not success:
                print("⚠️  Some scripts may not have been removed")
                
            # Step 3: Remove configuration and data
            success = self.remove_configuration(keep_data=keep_data)
            if not success:
                print("⚠️  Configuration removal failed")
                
            # Step 4: Clean shell profiles
            success = self.clean_shell_profiles()
            if not success:
                print("⚠️  Shell profile cleaning failed")
                
            # Step 5: Remove Python environment (optional)
            success = self.remove_python_environment(force=remove_venv)
            if not success:
                print("⚠️  Python environment removal failed")
                
            # Step 6: Remove temporary files
            success = self.remove_temp_files()
            if not success:
                print("⚠️  Temporary file cleanup failed")
                
            # Step 7: Verify removal
            verification = await self.verify_removal()
            
            # Success!
            self.print_completion_message(verification)
            return verification['overall'] in ['complete', 'partial']
            
        except KeyboardInterrupt:
            print("\n❌ Uninstallation cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Uninstallation failed: {e}")
            print(f"❌ Uninstallation failed: {e}")
            return False


async def main():
    """Main uninstaller entry point"""
    parser = argparse.ArgumentParser(description='NeuralSync Auto-Launch Integration Uninstaller')
    parser.add_argument('--keep-data', action='store_true', help='Keep configuration and data files')
    parser.add_argument('--remove-venv', action='store_true', help='Remove Python virtual environment')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    uninstaller = NeuralSyncUninstaller()
    success = await uninstaller.uninstall(keep_data=args.keep_data, remove_venv=args.remove_venv)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())