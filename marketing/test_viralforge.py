#!/usr/bin/env python3
"""
ViralForge Quick Test
Test installation and generate sample content
"""

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
