#!/bin/bash
# Novabot - Startup Script for Raspberry Pi

# Navigate to the project directory
PROJECT_DIR="/home/nova/Documents/Novabot"
cd "$PROJECT_DIR" || { echo "❌ Error: Project directory not found at $PROJECT_DIR"; exit 1; }

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "🔗 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️ Warning: venv not found. Attempting to run with system python3..."
fi

# Run the app
echo "🚀 Starting Novabot..."
python3 main.py
