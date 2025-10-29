#!/bin/bash
# Install Predictive Maintenance as a background service

echo "🚀 Installing Predictive Maintenance Background Service..."
echo ""

# Stop current server if running
echo "📍 Stopping any running instances..."
pkill -f "python.*backend/app.py" 2>/dev/null || true
sleep 2

# Copy plist to LaunchAgents
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_SRC="$SCRIPT_DIR/com.predictive.maintenance.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.predictive.maintenance.plist"

echo "📋 Installing launch agent..."
cp "$PLIST_SRC" "$PLIST_DEST"

# Load the service
echo "🔧 Loading service..."
launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"

echo ""
echo "✅ Installation Complete!"
echo ""
echo "📊 Service Details:"
echo "   Name: com.predictive.maintenance"
echo "   URL: http://localhost:8010"
echo "   Status: Running in background"
echo ""
echo "🔍 Useful Commands:"
echo "   Check status: ./services/service_status.sh"
echo "   Stop service: ./services/service_stop.sh"
echo "   View logs: tail -f logs/stdout.log"
echo ""
echo "🌐 Access your app at: http://localhost:8010"
echo ""
