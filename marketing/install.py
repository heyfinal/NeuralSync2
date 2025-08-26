#!/usr/bin/env python3
"""
ViralForge Installer
Autonomous viral marketing system installer for NeuralSync2
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import platform
import urllib.request
import zipfile
import tempfile

class ViralForgeInstaller:
    """
    Self-contained installer for ViralForge autonomous viral marketing system
    
    Handles dependency detection, installation, configuration, and verification.
    """
    
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.install_dir = Path.cwd()
        self.config = self._load_default_config()
        self.sudo_password = None
        
        # Installation requirements
        self.required_python = (3, 8)
        self.required_packages = [
            "aiohttp>=3.8.0",
            "requests>=2.28.0", 
            "beautifulsoup4>=4.11.0",
            "jinja2>=3.1.0",
            "yake>=0.4.8",
            "pandas>=1.5.0",
            "pygments>=2.13.0",
            "markdown>=3.4.0",
            "streamlit>=1.28.0",
            "feedparser>=6.0.0",
            "schedule>=1.2.0"
        ]
        
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            "neuralsync2_repo": "https://github.com/heyfinal/NeuralSync2",
            "target_stars": 10000,
            "campaign_duration": 30,
            "output_directory": "./viral_output",
            "database_path": "./viralforge.db",
            "log_level": "INFO"
        }
        
    def print_header(self):
        """Print installation header"""
        print("=" * 70)
        print("🚀 VIRALFORGE INSTALLER")
        print("Autonomous Viral Marketing System for NeuralSync2")
        print("=" * 70)
        print()
        
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements"""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        if self.python_version < self.required_python:
            print(f"❌ Python {self.required_python[0]}.{self.required_python[1]}+ required")
            print(f"   Current version: {sys.version}")
            return False
        print(f"✅ Python {sys.version} (compatible)")
        
        # Check pip availability
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                          capture_output=True, check=True)
            print("✅ pip available")
        except subprocess.CalledProcessError:
            print("❌ pip not available")
            return False
            
        # Check internet connectivity
        try:
            urllib.request.urlopen('https://pypi.org/', timeout=10)
            print("✅ Internet connectivity")
        except Exception:
            print("❌ Internet connectivity required for installation")
            return False
            
        print()
        return True
        
    def install_dependencies(self) -> bool:
        """Install required Python packages"""
        print("📦 Installing dependencies...")
        
        try:
            # Upgrade pip first
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, capture_output=True)
            print("✅ pip upgraded")
            
            # Install required packages
            for i, package in enumerate(self.required_packages, 1):
                print(f"   Installing {package} ({i}/{len(self.required_packages)})...")
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install {package}")
                    print(f"   Error: {e}")
                    return False
                    
            print("✅ All dependencies installed successfully")
            print()
            return True
            
        except Exception as e:
            print(f"❌ Dependency installation failed: {e}")
            return False
            
    def setup_directory_structure(self) -> bool:
        """Create necessary directory structure"""
        print("📁 Setting up directory structure...")
        
        directories = [
            "viral_output",
            "viral_output/content",
            "viral_output/showcases", 
            "viral_output/influencer_magnets",
            "viral_output/community_content",
            "logs",
            "config"
        ]
        
        try:
            for directory in directories:
                dir_path = self.install_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created {directory}")
                
            print()
            return True
            
        except Exception as e:
            print(f"❌ Directory setup failed: {e}")
            return False
            
    def create_configuration_files(self) -> bool:
        """Create configuration files"""
        print("⚙️  Creating configuration files...")
        
        try:
            # Main configuration
            config_file = self.install_dir / "config" / "viralforge_config.json"
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print("✅ Main configuration created")
            
            # Environment template
            env_template = """# ViralForge Environment Variables
# Copy this to .env and fill in your values

# Optional: GitHub token for enhanced API access
# GITHUB_TOKEN=your_github_token_here

# Optional: Custom configuration
# VIRALFORGE_CONFIG_PATH=/path/to/custom/config.json

# Optional: Custom output directory
# VIRALFORGE_OUTPUT_DIR=/path/to/output

# Logging level
VIRALFORGE_LOG_LEVEL=INFO
"""
            
            env_file = self.install_dir / ".env.template"
            with open(env_file, 'w') as f:
                f.write(env_template)
            print("✅ Environment template created")
            
            print()
            return True
            
        except Exception as e:
            print(f"❌ Configuration creation failed: {e}")
            return False
            
    def create_launch_scripts(self) -> bool:
        """Create convenient launch scripts"""
        print("🚀 Creating launch scripts...")
        
        try:
            # Main launch script
            launch_script = f"""#!/usr/bin/env python3
\"\"\"
ViralForge Launcher
Convenient script to launch viral marketing campaigns
\"\"\"

import asyncio
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from viralforge_core import ViralForgeCore, CampaignStrategy

def main():
    print("🚀 Starting ViralForge Autonomous Viral Marketing System")
    print("Target: 10,000+ GitHub stars for NeuralSync2")
    print("-" * 60)
    
    # Create campaign strategy
    strategy = CampaignStrategy(
        name="NeuralSync2_Viral_Launch",
        target_stars=10000,
        duration_days=30,
        content_types=["demo", "showcase", "challenge", "article"],
        viral_mechanisms=[
            "content_multiplication", 
            "seo_amplification",
            "technical_showcases",
            "community_seeding", 
            "influencer_magnetism"
        ],
        success_metrics={{
            "stars": 10000, 
            "viral_coefficient": 2.0, 
            "discovery_rate": 85.0
        }}
    )
    
    # Initialize and start campaign
    forge = ViralForgeCore()
    
    try:
        asyncio.run(forge.start_autonomous_campaign(strategy))
    except KeyboardInterrupt:
        print("\\n🛑 Campaign stopped by user")
        forge.stop_campaign()
    except Exception as e:
        print(f"\\n❌ Campaign error: {{e}}")
    finally:
        # Show final status
        status = forge.get_campaign_status()
        print(f"\\n📊 Final Status: {{status}}")

if __name__ == "__main__":
    main()
"""
            
            launcher_file = self.install_dir / "launch_viralforge.py"
            with open(launcher_file, 'w') as f:
                f.write(launch_script)
            os.chmod(launcher_file, 0o755)
            print("✅ Main launcher created")
            
            # Quick test script
            test_script = """#!/usr/bin/env python3
\"\"\"
ViralForge Quick Test
Test installation and generate sample content
\"\"\"

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from viralforge_core import ViralForgeCore

async def quick_test():
    print("🧪 ViralForge Quick Test")
    print("=" * 40)
    
    forge = ViralForgeCore()
    
    # Test content generation
    print("Testing content generation...")
    content = await forge.content_engine.generate_batch(2)
    print(f"✅ Generated {len(content)} content pieces")
    
    # Test discovery optimization
    print("Testing discovery optimization...")
    optimized = await forge.discovery_amplifier.optimize_all_content()
    print(f"✅ Optimized {optimized} pieces for discovery")
    
    # Test technical showcases
    print("Testing technical showcases...")
    showcases = await forge.technical_showcase.create_showcases(1)
    print(f"✅ Created {len(showcases)} technical showcases")
    
    # Test community seeding
    print("Testing community seeding...")
    seeded = await forge.community_seeder.plant_discovery_content()
    print(f"✅ Seeded {seeded} community content pieces")
    
    # Test influencer magnetism
    print("Testing influencer magnetism...")
    magnets = await forge.influencer_magnetism.create_attraction_content(1)
    print(f"✅ Created {len(magnets)} influencer magnets")
    
    print()
    print("🎯 All systems operational!")
    print("Run 'python launch_viralforge.py' to start full campaign")

if __name__ == "__main__":
    asyncio.run(quick_test())
"""
            
            test_file = self.install_dir / "test_viralforge.py"
            with open(test_file, 'w') as f:
                f.write(test_script)
            os.chmod(test_file, 0o755)
            print("✅ Test script created")
            
            # Shell launcher (Unix/Mac)
            if self.system in ['linux', 'darwin']:
                shell_script = """#!/bin/bash
# ViralForge Shell Launcher

echo "🚀 ViralForge - Autonomous Viral Marketing for NeuralSync2"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt 2>/dev/null || echo "Using system packages"

# Launch ViralForge
echo "🚀 Launching ViralForge..."
python launch_viralforge.py
"""
                shell_file = self.install_dir / "viralforge.sh"
                with open(shell_file, 'w') as f:
                    f.write(shell_script)
                os.chmod(shell_file, 0o755)
                print("✅ Shell launcher created")
            
            print()
            return True
            
        except Exception as e:
            print(f"❌ Launch script creation failed: {e}")
            return False
            
    def create_requirements_file(self) -> bool:
        """Create requirements.txt file"""
        print("📋 Creating requirements.txt...")
        
        try:
            requirements_content = "\n".join(self.required_packages)
            requirements_file = self.install_dir / "requirements.txt"
            
            with open(requirements_file, 'w') as f:
                f.write("# ViralForge Dependencies\n")
                f.write("# Autonomous viral marketing system for NeuralSync2\n\n")
                f.write(requirements_content)
                f.write("\n")
                
            print("✅ requirements.txt created")
            print()
            return True
            
        except Exception as e:
            print(f"❌ Requirements file creation failed: {e}")
            return False
            
    def create_uninstaller(self) -> bool:
        """Create uninstaller script"""
        print("🗑️  Creating uninstaller...")
        
        uninstaller_script = """#!/usr/bin/env python3
\"\"\"
ViralForge Uninstaller
Removes ViralForge and cleans up system
\"\"\"

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
"""
        
        try:
            uninstaller_file = self.install_dir / "uninstall_viralforge.py"
            with open(uninstaller_file, 'w') as f:
                f.write(uninstaller_script)
            os.chmod(uninstaller_file, 0o755)
            print("✅ Uninstaller created")
            print()
            return True
            
        except Exception as e:
            print(f"❌ Uninstaller creation failed: {e}")
            return False
            
    def run_verification_tests(self) -> bool:
        """Run verification tests to ensure installation works"""
        print("🧪 Running verification tests...")
        
        try:
            # Test core imports
            sys.path.insert(0, str(self.install_dir))
            
            try:
                from viralforge_core import ViralForgeCore
                print("✅ Core module import")
            except ImportError as e:
                print(f"❌ Core module import failed: {e}")
                return False
                
            try:
                from content_engine import ContentEngine
                print("✅ Content engine import")
            except ImportError as e:
                print(f"⚠️  Content engine import failed: {e}")
                
            try:
                from discovery_amplifier import DiscoveryAmplifier
                print("✅ Discovery amplifier import")
            except ImportError as e:
                print(f"⚠️  Discovery amplifier import failed: {e}")
                
            # Test configuration loading
            try:
                forge = ViralForgeCore()
                print("✅ Configuration loading")
            except Exception as e:
                print(f"❌ Configuration loading failed: {e}")
                return False
                
            # Test database initialization
            try:
                if forge.content_db:
                    print("✅ Database initialization")
            except Exception as e:
                print(f"❌ Database initialization failed: {e}")
                return False
                
            print("✅ All verification tests passed")
            print()
            return True
            
        except Exception as e:
            print(f"❌ Verification tests failed: {e}")
            return False
            
    def print_installation_summary(self):
        """Print installation summary and next steps"""
        print("=" * 70)
        print("🎉 VIRALFORGE INSTALLATION COMPLETE!")
        print("=" * 70)
        print()
        print("📁 Installation Directory:", self.install_dir)
        print("🎯 Target: 10,000+ GitHub stars for NeuralSync2")
        print("⚡ Campaign Duration: 30 days")
        print()
        print("🚀 NEXT STEPS:")
        print("1. Quick Test:     python test_viralforge.py")
        print("2. Start Campaign: python launch_viralforge.py") 
        print("3. Shell Launcher: ./viralforge.sh (Unix/Mac)")
        print()
        print("📊 EXPECTED RESULTS:")
        print("• Viral content generation every hour")
        print("• SEO optimization for maximum discoverability")
        print("• Technical showcases demonstrating capabilities")
        print("• Community seeding for organic growth")
        print("• Influencer magnet content for amplification")
        print()
        print("🔍 MONITORING:")
        print("• Check logs/viralforge.log for detailed logs")
        print("• Monitor viral_output/ for generated content")
        print("• Watch GitHub stars: https://github.com/heyfinal/NeuralSync2")
        print()
        print("🛠️  SUPPORT:")
        print("• Documentation: https://neuralsync2.dev/viralforge")
        print("• Issues: https://github.com/heyfinal/NeuralSync2/issues")
        print()
        print("=" * 70)
        
    def install(self) -> bool:
        """Run complete installation process"""
        self.print_header()
        
        # Check system requirements
        if not self.check_system_requirements():
            return False
            
        # Install dependencies
        if not self.install_dependencies():
            return False
            
        # Setup directory structure
        if not self.setup_directory_structure():
            return False
            
        # Create configuration files
        if not self.create_configuration_files():
            return False
            
        # Create launch scripts
        if not self.create_launch_scripts():
            return False
            
        # Create requirements file
        if not self.create_requirements_file():
            return False
            
        # Create uninstaller
        if not self.create_uninstaller():
            return False
            
        # Run verification tests
        if not self.run_verification_tests():
            return False
            
        # Print summary
        self.print_installation_summary()
        
        return True


def main():
    """Main installer entry point"""
    installer = ViralForgeInstaller()
    
    try:
        success = installer.install()
        if success:
            print("✅ Installation completed successfully!")
            sys.exit(0)
        else:
            print("❌ Installation failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Installation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()