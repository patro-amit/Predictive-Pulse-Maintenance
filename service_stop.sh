#!/bin/bash
# Stop the Predictive Maintenance background service

echo "🛑 Stopping Predictive Maintenance Service..."
echo ""

PLIST_PATH="$HOME/Library/LaunchAgents/com.predictive.maintenance.plist"

if [ -f "$PLIST_PATH" ]; then
    launchctl unload "$PLIST_PATH"
    echo "✅ Service stopped"
else
    echo "⚠️  Service not installed"
fi

# Also kill any running processes
pkill -f "python.*backend/app.py" 2>/dev/null && echo "✅ Killed running processes" || echo "ℹ️  No running processes found"

echo ""
echo "The service is now stopped."
echo "To start it again, run: ./service_install.sh"
echo ""
