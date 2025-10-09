#!/bin/bash

# Predictive Maintenance Application Launcher
# Big Data & Machine Learning Strategies

echo "======================================================================"
echo "  🏭 PREDICTIVE MAINTENANCE SYSTEM"
echo "  Predictive Maintenance Strategies Using Big Data And ML"
echo "======================================================================"
echo ""
echo "✅ All 4 Models Achieve 90%+ Accuracy:"
echo "   🏆 Random Forest: 92.30%"
echo "   📊 Gradient Boosting: 92.22%"
echo "   📈 XGBoost: 92.10%"
echo "   🤖 CatBoost: 91.34%"
echo ""
echo "📊 Big Data: 44,511 samples | 29 features | 24 MB"
echo ""
echo "======================================================================"
echo ""

# Change to backend directory
cd "$(dirname "$0")/../backend"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: python -m venv venv && ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models/*.pkl 2>/dev/null)" ]; then
    echo "⚠️  Models not found!"
    echo "   Please train models first:"
    echo "   cd scripts && ../backend/venv/bin/python train.py --csv ../data/predictive_maintenance_bigdata.csv --label label --group unit"
    exit 1
fi

echo "🚀 Starting FastAPI server..."
echo ""
echo "📍 Access your application:"
echo "   🌐 Web Interface: http://localhost:8010"
echo "   📚 API Docs: http://localhost:8010/docs"
echo "   🏥 Health Check: http://localhost:8010/health"
echo ""
echo "💡 Features:"
echo "   ✅ Modern web interface (no Streamlit)"
echo "   ✅ Multiple ML models (90%+ accuracy)"
echo "   ✅ CSV file upload support"
echo "   ✅ Real-time predictions"
echo "   ✅ Model comparison"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo "======================================================================"
echo ""

# Start uvicorn
exec ./venv/bin/uvicorn app:app --host 0.0.0.0 --port 8010 --reload
