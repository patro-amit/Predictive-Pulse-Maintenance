#!/bin/bash
# Check the status of the Predictive Maintenance service

echo "📊 Predictive Maintenance Service Status"
echo "========================================"
echo ""

# Check if service is loaded
PLIST_PATH="$HOME/Library/LaunchAgents/com.predictive.maintenance.plist"

if [ -f "$PLIST_PATH" ]; then
    echo "✅ Service Installed: YES"
    
    # Check if it's loaded
    if launchctl list | grep -q "com.predictive.maintenance"; then
        echo "✅ Service Loaded: YES"
        
        # Check if process is running
        if pgrep -f "python.*backend/app.py" > /dev/null; then
            PID=$(pgrep -f "python.*backend/app.py")
            echo "✅ Process Running: YES (PID: $PID)"
            
            # Check if port is open
            if lsof -i :8010 > /dev/null 2>&1; then
                echo "✅ Port 8010: LISTENING"
                echo ""
                echo "🌐 App URL: http://localhost:8010"
                echo ""
                echo "📝 Recent Logs:"
                echo "---------------"
                tail -10 /Users/shyampatro/Predictive-Pulse-Maintenance/logs/stdout.log 2>/dev/null || echo "No logs yet"
            else
                echo "❌ Port 8010: NOT LISTENING"
            fi
        else
            echo "❌ Process Running: NO"
            echo ""
            echo "🔍 Check logs for errors:"
            echo "   tail -f logs/stderr.log"
        fi
    else
        echo "❌ Service Loaded: NO"
        echo "   Run: ./service_install.sh"
    fi
else
    echo "❌ Service Installed: NO"
    echo "   Run: ./service_install.sh"
fi

echo ""
