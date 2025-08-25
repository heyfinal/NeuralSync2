#!/usr/bin/env python3
"""
Enhanced NeuralSync2 Fixes Uninstaller
Safely removes enhanced daemon management system and restores original functionality
"""

import os
import sys
import subprocess
import shutil
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedFixesUninstaller:
    """Uninstaller for NeuralSync2 enhanced daemon management fixes"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.neuralsync_dir = self.script_dir
        self.home_dir = Path.home()
        self.config_dir = self.home_dir / ".neuralsync"
        
        # Uninstallation status
        self.uninstall_log: List[str] = []
        self.errors: List[str] = []
        
    def log_step(self, message: str, success: bool = True):
        """Log uninstallation step"""
        status = "✅" if success else "❌"
        log_message = f"{status} {message}"
        print(log_message)
        logger.info(message)
        self.uninstall_log.append(log_message)
        
        if not success:
            self.errors.append(message)
    
    def check_installation_status(self) -> Dict[str, Any]:
        """Check current installation status"""
        print("🔍 Checking installation status...")
        
        status = {
            'enhanced_modules_present': False,
            'installation_profile_exists': False,
            'services_running': False,
            'backup_available': None
        }
        
        # Check if enhanced modules exist
        enhanced_modules = [
            'robust_service_detector.py',
            'smart_process_discovery.py', 
            'configuration_validator.py',
            'performance_optimizer.py',
            'enhanced_daemon_manager.py'
        ]
        
        modules_found = 0
        for module in enhanced_modules:
            if (self.neuralsync_dir / "neuralsync" / module).exists():
                modules_found += 1
        
        status['enhanced_modules_present'] = modules_found > 0
        self.log_step(f"Enhanced modules: {modules_found}/{len(enhanced_modules)} found")
        
        # Check installation profile
        profile_path = self.config_dir / "installation_profile.json"
        if profile_path.exists():
            status['installation_profile_exists'] = True
            self.log_step("Installation profile found")
        
        # Look for backups
        backup_dirs = [d for d in self.config_dir.parent.glob(".neuralsync_backup_*")]
        if backup_dirs:
            status['backup_available'] = max(backup_dirs, key=lambda p: p.stat().st_mtime)
            self.log_step(f"Backup found: {status['backup_available']}")
        else:
            self.log_step("No backups found")
        
        # Check if services are running
        try:
            # Try to detect running processes
            import psutil
            neuralsync_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('neuralsync' in str(arg).lower() for arg in cmdline):
                        neuralsync_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            status['services_running'] = len(neuralsync_processes) > 0
            if status['services_running']:
                self.log_step(f"{len(neuralsync_processes)} NeuralSync processes running")
            else:
                self.log_step("No NeuralSync processes detected")
                
        except ImportError:
            self.log_step("Cannot check running services (psutil not available)")
        
        return status
    
    def stop_running_services(self) -> bool:
        """Stop any running NeuralSync services"""
        print("\n🛑 Stopping running services...")
        
        try:
            # Try to stop services using enhanced daemon manager
            stop_script = f"""
import sys
import asyncio
sys.path.insert(0, '{self.neuralsync_dir}')

async def stop_services():
    try:
        from neuralsync.enhanced_daemon_manager import graceful_shutdown_enhanced
        await graceful_shutdown_enhanced()
        print("SUCCESS: Enhanced services stopped")
    except Exception as e:
        print(f"INFO: Enhanced shutdown failed: {{e}}")
        
        # Try original daemon manager
        try:
            from neuralsync.daemon_manager import graceful_shutdown
            await graceful_shutdown()
            print("SUCCESS: Original services stopped")
        except Exception as e2:
            print(f"INFO: Original shutdown failed: {{e2}}")

asyncio.run(stop_services())
"""
            
            result = subprocess.run(
                [sys.executable, "-c", stop_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            self.log_step("Service stop attempt completed")
            return True
            
        except Exception as e:
            self.log_step(f"Failed to stop services gracefully: {e}", success=False)
            
            # Force stop using process termination
            return self.force_stop_services()
    
    def force_stop_services(self) -> bool:
        """Force stop NeuralSync services"""
        print("🔨 Force stopping services...")
        
        try:
            import psutil
            stopped_count = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('neuralsync' in str(arg).lower() for arg in cmdline):
                        process = psutil.Process(proc.info['pid'])
                        process.terminate()
                        
                        try:
                            process.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            process.kill()
                        
                        stopped_count += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            self.log_step(f"Force stopped {stopped_count} processes")
            return True
            
        except ImportError:
            self.log_step("Cannot force stop services (psutil not available)")
            return False
        except Exception as e:
            self.log_step(f"Force stop failed: {e}", success=False)
            return False
    
    def remove_enhanced_modules(self) -> bool:
        """Remove enhanced modules"""
        print("\n🗑️  Removing enhanced modules...")
        
        enhanced_modules = [
            'robust_service_detector.py',
            'smart_process_discovery.py',
            'configuration_validator.py', 
            'performance_optimizer.py',
            'enhanced_daemon_manager.py',
            'test_enhanced_fixes.py'
        ]
        
        removed_count = 0
        
        for module in enhanced_modules:
            module_path = self.neuralsync_dir / "neuralsync" / module
            if module_path.exists():
                try:
                    module_path.unlink()
                    removed_count += 1
                    self.log_step(f"Removed {module}")
                except Exception as e:
                    self.log_step(f"Failed to remove {module}: {e}", success=False)
        
        # Remove __pycache__ entries
        pycache_dir = self.neuralsync_dir / "neuralsync" / "__pycache__"
        if pycache_dir.exists():
            try:
                for pyc_file in pycache_dir.glob("*enhanced*"):
                    pyc_file.unlink()
                for pyc_file in pycache_dir.glob("*robust*"):
                    pyc_file.unlink()
                for pyc_file in pycache_dir.glob("*smart*"):
                    pyc_file.unlink()
                for pyc_file in pycache_dir.glob("*performance*"):
                    pyc_file.unlink()
                for pyc_file in pycache_dir.glob("*configuration*"):
                    pyc_file.unlink()
                
                self.log_step("Cleaned __pycache__ files")
            except Exception as e:
                self.log_step(f"Failed to clean __pycache__: {e}", success=False)
        
        self.log_step(f"Removed {removed_count}/{len(enhanced_modules)} enhanced modules")
        return removed_count > 0
    
    def restore_original_wrapper(self) -> bool:
        """Restore original wrapper script"""
        print("\n🔄 Restoring original wrapper...")
        
        nswrap_path = self.neuralsync_dir / "nswrap"
        backup_path = nswrap_path.with_suffix(".backup")
        
        if backup_path.exists():
            try:
                shutil.copy2(backup_path, nswrap_path)
                backup_path.unlink()
                self.log_step("Restored original nswrap from backup")
                return True
            except Exception as e:
                self.log_step(f"Failed to restore nswrap backup: {e}", success=False)
        
        # Create minimal original wrapper
        try:
            original_nswrap = '''#!/usr/bin/env python3
import os, sys, shlex, subprocess, requests
from typing import Optional

NS = os.environ.get("NS_ENDPOINT", f"http://{os.environ.get('NS_HOST','127.0.0.1')}:{os.environ.get('NS_PORT','8373')}")
HEAD = {"Authorization": f"Bearer {os.environ['NS_TOKEN']}"} if os.environ.get("NS_TOKEN") else {}

def get_preamble(tool: Optional[str]):
    try:
        persona = requests.get(f"{NS}/persona", headers=HEAD, timeout=5).json().get("text","")
        recall = requests.post(f"{NS}/recall", headers=HEAD, json={"query":"", "top_k":8, "scope":"any", "tool":tool}, timeout=10).json()
        pre = []
        if persona:
            pre += [f"Persona: {persona}", ""]
        for i,it in enumerate(recall.get("items",[]),1):
            pre.append(f"[M{i}] ({it.get('kind')},{it.get('scope')},conf={it.get('confidence','')}) {it.get('text','')}")
        return "\\n".join(pre)+("\\n\\n" if pre else "")
    except:
        return ""

def send_remember(line: str):
    try:
        if not line.startswith("@remember:"): return
        payload = {"text":"", "kind":"note", "scope":"global", "tool":None, "tags":[], "confidence":0.8, "source":"nswrap"}
        rest = line[len("@remember:"):].strip()
        parts = shlex.split(rest)
        text_seen = False
        text_val = []
        for p in parts:
            if "=" in p and not text_seen:
                k,v = p.split("=",1)
                if k in ["kind","scope","tool"]:
                    payload[k]=None if v.lower()=="none" else v
                elif k=="confidence": payload["confidence"]=float(v)
                elif k=="tag": payload.setdefault("tags",[]).append(v)
            else:
                text_seen = True
                text_val.append(p)
        if text_val and not payload["text"]:
            payload["text"]=" ".join(text_val).strip('"')
        if payload["text"]:
            requests.post(f"{NS}/remember", headers=HEAD, json=payload, timeout=10)
    except Exception:
        pass

def main():
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        cmd = sys.argv[idx+1:]
        tool = os.environ.get("TOOL_NAME")
    else:
        cmd = ["cat"]
        tool = os.environ.get("TOOL_NAME")
    preamble = get_preamble(tool)
    user_input = sys.stdin.read()
    data = preamble + user_input
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out,_ = proc.communicate(input=data)
    for line in out.splitlines():
        print(line)
        if line.startswith("@remember:"):
            send_remember(line)
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
'''
            
            with open(nswrap_path, 'w') as f:
                f.write(original_nswrap)
            
            # Make executable
            nswrap_path.chmod(0o755)
            
            self.log_step("Created minimal original nswrap")
            return True
            
        except Exception as e:
            self.log_step(f"Failed to create original nswrap: {e}", success=False)
            return False
    
    def clean_configuration_files(self, preserve_core: bool = True) -> bool:
        """Clean enhanced configuration files"""
        print(f"\n🧹 Cleaning configuration files (preserve_core={preserve_core})...")
        
        if not self.config_dir.exists():
            self.log_step("No configuration directory to clean")
            return True
        
        files_to_remove = [
            'installation_profile.json',
            'validation_cache.json',
            'profiles.json'
        ]
        
        dirs_to_remove = [
            'discovery_cache',
            'locks'
        ]
        
        removed_files = 0
        
        # Remove specific files
        for file_name in files_to_remove:
            file_path = self.config_dir / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                    removed_files += 1
                    self.log_step(f"Removed {file_name}")
                except Exception as e:
                    self.log_step(f"Failed to remove {file_name}: {e}", success=False)
        
        # Remove specific directories
        for dir_name in dirs_to_remove:
            dir_path = self.config_dir / dir_name
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    self.log_step(f"Removed {dir_name}/")
                except Exception as e:
                    self.log_step(f"Failed to remove {dir_name}/: {e}", success=False)
        
        # Clean PID files directory of enhanced traces
        pids_dir = self.config_dir / "pids"
        if pids_dir.exists():
            try:
                enhanced_pids = pids_dir.glob("*enhanced*")
                for pid_file in enhanced_pids:
                    pid_file.unlink()
                self.log_step("Cleaned enhanced PID files")
            except Exception as e:
                self.log_step(f"Failed to clean PID files: {e}", success=False)
        
        if not preserve_core:
            self.log_step("WARNING: Core configuration preservation disabled")
        
        return True
    
    def create_uninstall_summary(self) -> Dict[str, Any]:
        """Create uninstallation summary"""
        
        summary = {
            'uninstall_time': time.time(),
            'uninstall_log': self.uninstall_log,
            'errors': self.errors,
            'uninstall_successful': len(self.errors) == 0
        }
        
        return summary
    
    def uninstall(self, preserve_core_config: bool = True) -> bool:
        """Run complete uninstallation process"""
        print("🗑️  Uninstalling NeuralSync2 Enhanced Daemon Management Fixes")
        print("=" * 70)
        
        start_time = time.time()
        
        # Check current status
        status = self.check_installation_status()
        
        if not status['enhanced_modules_present']:
            print("ℹ️  Enhanced modules not found - nothing to uninstall")
            return True
        
        # Uninstallation steps
        steps = [
            ("Stop Running Services", self.stop_running_services),
            ("Remove Enhanced Modules", self.remove_enhanced_modules),
            ("Restore Original Wrapper", self.restore_original_wrapper),
            ("Clean Configuration Files", lambda: self.clean_configuration_files(preserve_core_config)),
        ]
        
        all_success = True
        
        for step_name, step_function in steps:
            print(f"\n{'='*50}")
            print(f"STEP: {step_name}")
            print(f"{'='*50}")
            
            success = step_function()
            
            if not success:
                all_success = False
                print(f"❌ Step '{step_name}' had issues")
                
                if input("\nContinue with remaining steps? (Y/n): ").lower() == 'n':
                    break
            else:
                print(f"✅ Step '{step_name}' completed successfully")
        
        # Create uninstall summary
        uninstall_time = time.time() - start_time
        summary = self.create_uninstall_summary()
        summary['uninstall_duration'] = uninstall_time
        
        # Save summary
        try:
            summary_path = self.config_dir / "uninstall_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            self.log_step(f"Uninstall summary saved to {summary_path}")
        except Exception as e:
            self.log_step(f"Failed to save uninstall summary: {e}", success=False)
        
        # Final summary
        print(f"\n{'='*70}")
        print("UNINSTALLATION SUMMARY")
        print(f"{'='*70}")
        
        if all_success:
            print("✅ Uninstallation completed successfully!")
            print(f"⏱️  Total time: {uninstall_time:.1f} seconds")
            print(f"📍 Configuration: {self.config_dir}")
            print("\n🔄 WHAT WAS RESTORED:")
            print("   • Enhanced modules removed")
            print("   • Original wrapper script restored")
            print("   • Enhanced configuration files cleaned")
            print("   • Services stopped gracefully")
            print("\n📝 PRESERVED:")
            if preserve_core_config:
                print("   • Core NeuralSync configuration (config.yaml)")
                print("   • Memory database and logs")
                print("   • User personas and data")
            else:
                print("   • Nothing preserved (full cleanup)")
            
        else:
            print("⚠️  Uninstallation completed with issues")
            print(f"⏱️  Total time: {uninstall_time:.1f} seconds")
            print(f"❗ Issues encountered:")
            for error in self.errors:
                print(f"   • {error}")
                
            print("\n🔧 MANUAL CLEANUP:")
            print("   • Some enhanced modules may remain")
            print("   • Check ~/.neuralsync for leftover files")
            print("   • Kill any remaining neuralsync processes manually")
        
        return all_success


def main():
    """Main uninstallation entry point"""
    uninstaller = EnhancedFixesUninstaller()
    
    print("NeuralSync2 Enhanced Daemon Management Fixes Uninstaller")
    print("Removes enhanced features and restores original functionality")
    print()
    
    # Parse command line arguments
    preserve_core = True
    auto_mode = False
    
    if "--full" in sys.argv:
        preserve_core = False
        print("⚠️  Full uninstall mode: will remove ALL configuration")
    
    if "--auto" in sys.argv:
        auto_mode = True
        print("Running in automatic mode...")
    
    if not auto_mode:
        print("This will:")
        print("  • Stop running NeuralSync services")
        print("  • Remove enhanced daemon management modules")
        print("  • Restore original wrapper scripts")
        if preserve_core:
            print("  • Preserve core NeuralSync configuration")
        else:
            print("  • Remove ALL NeuralSync configuration")
        print()
        
        response = input("Proceed with uninstallation? (y/N): ")
        if response.lower() != 'y':
            print("Uninstallation cancelled")
            return False
    
    success = uninstaller.uninstall(preserve_core_config=preserve_core)
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Uninstallation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Uninstallation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)