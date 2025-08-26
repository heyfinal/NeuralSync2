#!/usr/bin/env python3
"""
ViralForge Uninstaller
Removes ViralForge and cleans up system
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    print("🗑️  ViralForge Uninstaller")
    print("=" * 40)
    
    response = input("Are you sure you want to uninstall ViralForge? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Uninstallation cancelled")
        return
        
    print("🧹 Cleaning up ViralForge...")
    
    # Remove generated content
    dirs_to_remove = [
        "viral_output",
        "logs", 
        "config",
        "__pycache__"
    ]
    
    for directory in dirs_to_remove:
        dir_path = Path(directory)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"✅ Removed {directory}")
    
    # Remove files
    files_to_remove = [
        "viralforge.db",
        "viralforge.log",
        ".env",
        "requirements.txt"
    ]
    
    for filename in files_to_remove:
        file_path = Path(filename)
        if file_path.exists():
            file_path.unlink()
            print(f"✅ Removed {filename}")
    
    print("✅ ViralForge uninstalled successfully")
    
    # Self-delete
    try:
        os.remove(__file__)
    except:
        pass

if __name__ == "__main__":
    main()
