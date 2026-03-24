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

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️ Warning: .env file NOT found! Please create it with GROQ_API_KEY."
fi

# Pre-flight check for Piper
if [ ! -d "piper" ] || [ ! -x "$(find piper -name piper -type f -executable | head -n 1)" ]; then
    echo "⚠️ Warning: Piper binary not found or not executable. Audio might fail."
fi

# Install requirements (only if venv is new or requested)
if [ -f "requirements.txt" ]; then
    echo "📥 Checking dependencies..."
    pip install -r requirements.txt --quiet
else
    echo "⚠️ Warning: requirements.txt not found!"
fi

# Run the app
echo "🚀 Starting Novabot..."
python3 main.py
