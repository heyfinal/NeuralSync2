#!/usr/bin/env python3
"""
ViralForge Launcher
Convenient script to launch viral marketing campaigns
"""

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
        success_metrics={
            "stars": 10000, 
            "viral_coefficient": 2.0, 
            "discovery_rate": 85.0
        }
    )
    
    # Initialize and start campaign
    forge = ViralForgeCore()
    
    try:
        asyncio.run(forge.start_autonomous_campaign(strategy))
    except KeyboardInterrupt:
        print("\n🛑 Campaign stopped by user")
        forge.stop_campaign()
    except Exception as e:
        print(f"\n❌ Campaign error: {e}")
    finally:
        # Show final status
        status = forge.get_campaign_status()
        print(f"\n📊 Final Status: {status}")

if __name__ == "__main__":
    main()
