#!/bin/bash
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
