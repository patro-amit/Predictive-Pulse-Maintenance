#!/bin/bash

# Set up environment for Big Data tools
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"

# Start the FastAPI application with Big Data tools
echo "🚀 Starting Predictive Maintenance System with Big Data Tools"
echo "=================================================="
echo "✅ Java: $(java -version 2>&1 | head -1)"
echo "✅ Apache Spark: 3.5.0"
echo "✅ MongoDB: Available (optional)"
echo "=================================================="
echo ""

cd "$(dirname "$0")/../backend"
../.venv/bin/python app.py
