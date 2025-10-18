#!/bin/bash
# Predictive Maintenance - Run Application Script
# macOS/Linux version of run_app.bat

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  🚀 Starting Predictive Maintenance Application          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r backend/requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if backend/app.py exists
if [ ! -f "backend/app.py" ]; then
    echo "❌ backend/app.py not found!"
    exit 1
fi

echo "✅ Environment ready!"
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  📊 Loading ML Models...                                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Run the application
python backend/app.py

# Keep terminal open if there's an error
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Application exited with an error"
    echo "Press any key to exit..."
    read -n 1
fi
