#!/bin/bash
# Uninstall the Predictive Maintenance background service

echo "🗑️  Uninstalling Predictive Maintenance Service..."
echo ""

PLIST_PATH="$HOME/Library/LaunchAgents/com.predictive.maintenance.plist"

# Stop the service first
if [ -f "$PLIST_PATH" ]; then
    echo "📍 Stopping service..."
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    
    echo "🗑️  Removing launch agent..."
    rm "$PLIST_PATH"
    
    echo "✅ Service uninstalled"
else
    echo "ℹ️  Service was not installed"
fi

# Kill any running processes
pkill -f "python.*backend/app.py" 2>/dev/null && echo "✅ Killed running processes" || true

echo ""
echo "Service has been completely removed."
echo "To install again, run: ./service_install.sh"
echo ""
