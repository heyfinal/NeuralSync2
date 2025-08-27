#!/usr/bin/env python3
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
    cli_file = Path("/usr/local/bin/nswrap")
    if cli_file.exists():
        try:
            if os.access(cli_file.parent, os.W_OK):
                cli_file.unlink()
            else:
                subprocess.run(["sudo", "rm", str(cli_file)], check=True)
            items_removed.append(str(cli_file))
        except Exception as e:
            print(f"Warning: Could not remove {cli_file}: {e}")
    
    # Remove configuration directory
    config_dir = Path("/Users/daniel/.neuralsync")
    if config_dir.exists():
        try:
            shutil.rmtree(config_dir)
            items_removed.append(str(config_dir))
        except Exception as e:
            print(f"Warning: Could not remove {config_dir}: {e}")
    
    # Remove cache directory
    cache_dir = Path("/tmp/neuralsync_cache")
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            items_removed.append(str(cache_dir))
        except Exception as e:
            print(f"Warning: Could not remove {cache_dir}: {e}")
    
    # Restore original nswrap if backup exists
    original_backup = Path("/Users/daniel/NeuralSync2/nswrap_original")
    original_nswrap = Path("/Users/daniel/NeuralSync2/nswrap")
    if original_backup.exists():
        try:
            shutil.copy2(original_backup, original_nswrap)
            original_backup.unlink()
            items_removed.append("Restored original nswrap")
        except Exception as e:
            print(f"Warning: Could not restore original nswrap: {e}")
    
    print(f"✅ Uninstallation complete. Removed {len(items_removed)} items:")
    for item in items_removed:
        print(f"   - {item}")
    
    print("\n📝 Manual cleanup needed:")
    print("   - Remove environment variables from shell profile")
    print("   - Uninstall Python dependencies if no longer needed:")
    print("     pip uninstall aiohttp lz4 psutil httptools")

if __name__ == "__main__":
    uninstall()
