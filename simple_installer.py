#!/usr/bin/env python3
"""
Simple NeuralSync CLI Integration Installer
"""

import os
import shutil
from pathlib import Path

def main():
    neuralsync_dir = Path(__file__).parent
    bin_dir = neuralsync_dir / "bin"
    install_dir = Path.home() / ".local" / "bin"
    
    print("🚀 Installing NeuralSync CLI Integration")
    print("=" * 40)
    
    # Create install directory
    install_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created install directory: {install_dir}")
    
    # Install wrappers
    wrappers = {
        'codex-ns': 'codex-ns-fixed',
        'claude-ns': 'claude-ns-fixed', 
        'gemini-ns': 'gemini-ns-fixed'
    }
    
    for wrapper_name, fixed_name in wrappers.items():
        source_path = bin_dir / fixed_name
        target_path = install_dir / wrapper_name
        
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            target_path.chmod(0o755)
            print(f"✅ Installed {wrapper_name}")
        else:
            print(f"❌ {fixed_name} not found")
    
    # Install nswrap
    nswrap_source = neuralsync_dir / "nswrap"
    nswrap_target = install_dir / "nswrap"
    
    if nswrap_source.exists():
        shutil.copy2(nswrap_source, nswrap_target)
        nswrap_target.chmod(0o755)
        print(f"✅ Installed nswrap")
    else:
        print(f"❌ nswrap not found")
    
    print(f"\n📊 Installation Complete!")
    print(f"Installed to: {install_dir}")
    print(f"\nNext steps:")
    print(f"1. Add to PATH: export PATH=\"$PATH:{install_dir}\"")
    print(f"2. Test with: {install_dir}/codex-ns --version")

if __name__ == "__main__":
    main()