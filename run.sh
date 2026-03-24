#!/bin/bash
# Novabot - Startup Script for Raspberry Pi

# Navigate to the project directory (directory of the script)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || { echo "❌ Error: Project directory not found"; exit 1; }

# Ensure virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔗 Activating virtual environment..."
source venv/bin/activate

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📥 Installing dependencies (this may take a minute)..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "⚠️ Warning: requirements.txt not found!"
fi

# Run the app
echo "🚀 Starting Novabot..."
python3 main.py
