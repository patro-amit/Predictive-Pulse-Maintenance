#!/bin/bash
# Predictive Maintenance Server Startup Script
# This script ensures the server runs with proper environment

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Activate virtual environment
source "$DIR/.venv/bin/activate"

# Set environment variables
export PYTHONUNBUFFERED=1
export PORT=8010

# Log file locations
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"
ACCESS_LOG="$LOG_DIR/access.log"
ERROR_LOG="$LOG_DIR/error.log"

# Start the server
echo "$(date): Starting Predictive Maintenance Server..." >> "$ACCESS_LOG"
python backend/app.py >> "$ACCESS_LOG" 2>> "$ERROR_LOG"
