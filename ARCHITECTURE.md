# 🏗️ Predictive Maintenance System Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    PREDICTIVE MAINTENANCE SYSTEM                            │
│              Big Data & Machine Learning Architecture                       │
│                                           
╚═══════════════════════════════════════════════════════════════════════════╝

    ┌──────────────────────────────────────────────────────────┐                                  │
└─────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════╗
║                           1. DATA SOURCES LAYER                           ║
    │         🏭 INDUSTRIAL IoT SENSORS & EQUIPMENT            │
    │                                                          │
    │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐       │
    │  │ Temp   │  │Pressure│  │Vibration│ │  RPM   │       │
    │  │Sensors │  │Sensors │  │ Sensors │ │Sensors │       │
    │  │ (s1-s3)│  │(s7,s11)│  │(s15-s17)│ │(s8-s9) │       │
    │  └────────┘  └────────┘  └────────┘  └────────┘       │
    │                                                          │
    │  ┌─────────────────────────────────────────────┐       │
    │  │  21 Sensors + 3 Operational Settings        │       │
    │  │  NASA C-MAPSS Turbofan Engine Dataset       │       │
    │  │  44,511 samples | 18.42% failure rate       │       │
    │  └─────────────────────────────────────────────┘       │
    └──────────────────────────────────────────────────────────┘
                            │
                            │ ① Raw Sensor Data
                            ▼

╔═══════════════════════════════════════════════════════════════════════════╗
║                     2. BIG DATA PROCESSING LAYER                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

    ┌──────────────────────────────────────────────────────────┐
    │           🔥 APACHE SPARK (PySpark 3.5.0)                │
    │               Distributed Data Processing                │
    │                                                          │
    │  ┌─────────────────────────────────────────────┐       │
    │  │  • Large-scale dataset processing           │       │
    │  │  • Parallel feature engineering             │       │
    │  │  • Real-time stream processing              │       │
    │  │  • Distributed computation (10-100x faster) │       │
    │  └─────────────────────────────────────────────┘       │
    └──────────────────────────────────────────────────────────┘
                            │
                            │ ② Processed Data
                            ▼

╔═══════════════════════════════════════════════════════════════════════════╗
║                    3. FEATURE ENGINEERING LAYER                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

    ┌──────────────────────────────────────────────────────────┐
    │        🔧 ADVANCED FEATURE ENGINEERING PIPELINE          │
    │                                                          │
    │  Base Features (24):                                    │
    │  ├─ 21 Sensor readings (s1-s21)                        │
    │  └─ 3 Operational settings (setting1-3)                │
    │                                                          │
    │  Engineered Features (19):                              │
    │  ├─ Averages (4): temp_avg, pressure_avg, etc.        │
    │  ├─ Ratios (2): temp_pressure_ratio, etc.             │
    │  ├─ Interactions (3): temp_vibration, etc.            │
    │  ├─ Polynomials (2): temp_squared, vibration_squared  │
    │  ├─ Rolling Stats (5): rolling mean/std/max           │
    │  └─ Statistics (3): sensor_std, range, mean           │
    │                                                          │
    │  Total Features: 43 (optimized for 94%+ accuracy)      │
    └──────────────────────────────────────────────────────────┘
                            │
                            │ ③ Feature Vectors
                            ▼

╔═══════════════════════════════════════════════════════════════════════════╗
║                      4. MACHINE LEARNING LAYER                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

    ┌──────────────────────────────────────────────────────────┐
    │         🤖 ENSEMBLE ML MODELS (4 MODELS)                 │
    │                                                          │
    │  ┌──────────────┐  ┌──────────────┐                    │
    │  │  CatBoost    │  │   XGBoost    │                    │
    │  │  94.06% Acc  │  │  94.04% Acc  │                    │
    │  │  (Conservative)│  │(Conservative)│                    │
    │  └──────────────┘  └──────────────┘                    │
    │                                                          │
    │  ┌──────────────┐  ┌──────────────┐                    │
    │  │   Gradient   │  │    Random    │                    │
    │  │   Boosting   │  │    Forest    │                    │
    │  │  91.28% Acc  │  │  91.24% Acc  │                    │
    │  │  (DEFAULT ✓) │  │  (Balanced)  │                    │
    │  └──────────────┘  └──────────────┘                    │
    │                                                          │
    │  Preprocessing Pipeline:                                │
    │  └─ RobustScaler → Model → Threshold (0.7)            │
    └──────────────────────────────────────────────────────────┘
                            │
                            │ ④ Predictions
                            ▼

╔═══════════════════════════════════════════════════════════════════════════╗
║                       5. API & BACKEND LAYER                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

    ┌──────────────────────────────────────────────────────────┐
    │          ⚡ FASTAPI REST API (Python 3.13)              │
    │                Port 8010 | Uvicorn ASGI                 │
    │                                                          │
    │  Endpoints:                                             │
    │  ├─ GET  /health         (System status)               │
    │  ├─ GET  /models         (List all models)             │
    │  ├─ POST /predict        (Default model)               │
    │  ├─ POST /predict/{model}(Specific model)              │
    │  ├─ POST /predict/compare(Compare all models)          │
    │  ├─ GET  /bigdata/status (Big Data tools)              │
    │  ├─ GET  /bigdata/spark/insights                       │
    │  └─ POST /bigdata/spark/process                        │
    │                                                          │
    │  ┌─────────────────────────────────────────────┐       │
    │  │  Automatic API Documentation                │       │
    │  │  • Swagger UI: /docs                        │       │
    │  │  • ReDoc: /redoc                            │       │
    │  └─────────────────────────────────────────────┘       │
    └──────────────────────────────────────────────────────────┘
                            │
                            │ ⑤ JSON Response
                            ▼

╔═══════════════════════════════════════════════════════════════════════════╗
║                     6. DATA STORAGE LAYER (OPTIONAL)                      ║
╚═══════════════════════════════════════════════════════════════════════════╝

    ┌──────────────────────────────────────────────────────────┐
    │            🍃 MONGODB (PyMongo 4.6.0)                    │
    │              NoSQL Database for Analytics                │
    │                                                          │
    │  Collections:                                           │
    │  ├─ predictions: Prediction history & results          │
    │  ├─ analytics: Statistical aggregations                │
    │  └─ sensors: Time-series sensor data                   │
    │                                                          │
    │  Document Structure:                                    │
    │  {                                                      │
    │    timestamp: ISODate,                                 │
    │    machine_id: String,                                 │
    │    sensors: {s1...s21, setting1-3},                   │
    │    prediction: "Working" | "Needs_Maintenance",       │
    │    probability: Number,                                │
    │    model_used: String,                                 │
    │    model_accuracy: Number                              │
    │  }                                                      │
    └──────────────────────────────────────────────────────────┘
                            │
                            │ ⑥ Historical Data
                            ▼

╔═══════════════════════════════════════════════════════════════════════════╗
║                       7. FRONTEND & UI LAYER                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

    ┌──────────────────────────────────────────────────────────┐
    │         🌐 WEB APPLICATION (HTML5 + JavaScript)          │
    │              Modern Dark Theme with Glassmorphism        │
    │                                                          │
    │  Features:                                              │
    │  ├─ Model Selection (4 models)                         │
    │  ├─ Scenario-Based Testing:                            │
    │  │  • Normal Operation (healthy machine)               │
    │  │  • High Risk (abnormal readings)                    │
    │  │  • Critical (failure imminent)                      │
    │  ├─ Real-time Predictions (<100ms)                     │
    │  ├─ Confidence Scores & Probabilities                  │
    │  ├─ Multi-model Comparison                             │
    │  └─ Responsive Design (mobile + desktop)               │
    │                                                          │
    │  Access: http://localhost:8010                          │
    └──────────────────────────────────────────────────────────┘
                            │
                            │ ⑦ User Interaction
                            ▼

╔═══════════════════════════════════════════════════════════════════════════╗
║                    8. DEPLOYMENT & SERVICE LAYER                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

    ┌──────────────────────────────────────────────────────────┐
    │         🚀 macOS LaunchAgent Background Service          │
    │                                                          │
    │  Service Configuration:                                 │
    │  ├─ Auto-start on login                                │
    │  ├─ Auto-restart on crash (KeepAlive)                  │
    │  ├─ Working directory: Project root                     │
    │  ├─ Logs: stdout.log, stderr.log, access.log          │
    │  └─ Virtual environment: .venv/                        │
    │                                                          │
    │  Management Scripts:                                    │
    │  ├─ service_install.sh   (Install & start)            │
    │  ├─ service_stop.sh      (Stop service)                │
    │  ├─ service_status.sh    (Check status)                │
    │  └─ service_uninstall.sh (Complete removal)            │
    │                                                          │
    │  Status: Always running at http://localhost:8010        │
    └──────────────────────────────────────────────────────────┘


════════════════════════════════════════════════════════════════════════════
                            DATA FLOW DIAGRAM
════════════════════════════════════════════════════════════════════════════

    Sensor Data → Apache Spark → Feature Engineering → ML Models
         ↓              ↓               ↓                  ↓
    [Raw CSV]    [Distributed]    [43 Features]    [4 Predictions]
                 [Processing]                            ↓
                                                    FastAPI Server
                                                         ↓
                                                   ┌─────┴─────┐
                                                   ↓           ↓
                                             Web UI      MongoDB
                                            [Display]   [Storage]


════════════════════════════════════════════════════════════════════════════
                        TECHNOLOGY STACK SUMMARY
════════════════════════════════════════════════════════════════════════════

┌─────────────────────┬──────────────────────────────────────────────────┐
│ Layer               │ Technologies                                     │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Data Source         │ NASA C-MAPSS Dataset (44,511 samples)          │
│ Big Data Processing │ Apache Spark 3.5.0 (PySpark)                   │
│ Machine Learning    │ XGBoost 2.0.2, CatBoost 1.2.2                  │
│                     │ Scikit-learn 1.3.2, Gradient Boosting          │
│ Feature Engineering │ Pandas 2.1.3, NumPy 1.26.2                     │
│ Backend Framework   │ FastAPI 0.104.1, Uvicorn 0.24.0                │
│ Data Storage        │ MongoDB 4.x (PyMongo 4.6.0) - Optional         │
│ Frontend            │ HTML5, CSS3, Vanilla JavaScript                 │
│ Preprocessing       │ RobustScaler (Scikit-learn)                    │
│ Model Serialization │ Joblib 1.3.2                                    │
│ Service Management  │ macOS LaunchAgent (launchd)                     │
│ Environment         │ Python 3.13, Virtual Environment (.venv)        │
│ Version Control     │ Git + GitHub (Git LFS for models)              │
└─────────────────────┴──────────────────────────────────────────────────┘


════════════════════════════════════════════════════════════════════════════
                         SYSTEM CHARACTERISTICS
════════════════════════════════════════════════════════════════════════════

Performance Metrics:
├─ Prediction Latency: <100ms per sample
├─ Throughput: >1000 requests/second
├─ Model Accuracy: 91.24% - 94.06%
├─ AUROC Score: 0.96 - 0.98
└─ Dataset Size: 44,511 samples (28 MB)

Scalability Features:
├─ Apache Spark: Distributed processing for large datasets
├─ MongoDB: Horizontal scaling for prediction storage
├─ FastAPI: Async I/O for high concurrency
└─ Background Service: Always-on operation

Reliability Features:
├─ Auto-restart on crash (LaunchAgent KeepAlive)
├─ Comprehensive logging (stdout, stderr, access, error)
├─ Multiple model fallback (if default unavailable)
├─ Input validation & error handling
└─ Health check endpoint (/health)


════════════════════════════════════════════════════════════════════════════
                      PREDICTION WORKFLOW DIAGRAM
════════════════════════════════════════════════════════════════════════════

┌────────────┐
│   START    │
└──────┬─────┘
       │
       ▼
┌─────────────────────┐
│  User Input         │
│  (Web UI or API)    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Sensor Data        │
│  (21 sensors +      │
│   3 settings)       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Feature            │
│  Engineering        │
│  (24 → 43 features) │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Data Preprocessing │
│  (RobustScaler)     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐     ┌──────────────┐
│  Model Selection    │────▶│ CatBoost     │──┐
│  (Gradient Boosting │     │ (94.06%)     │  │
│   by default)       │     └──────────────┘  │
└──────┬──────────────┘                       │
       │              ┌──────────────┐        │
       │              │ XGBoost      │────────┤
       │              │ (94.04%)     │        │
       │              └──────────────┘        │
       │                                      │ Compare
       │              ┌──────────────┐        │ (Optional)
       │              │ Gradient     │────────┤
       │              │ Boosting     │        │
       │              │ (91.28%) ✓   │        │
       │              └──────────────┘        │
       │                                      │
       │              ┌──────────────┐        │
       │              │ Random       │────────┘
       └─────────────▶│ Forest       │
                      │ (91.24%)     │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ Prediction   │
                      │ • Probability│
                      │ • Confidence │
                      │ • Label      │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ Threshold    │
                      │ Check (0.7)  │
                      └──────┬───────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
         ┌─────────────┐          ┌─────────────┐
         │  "Working"  │          │  "Needs     │
         │             │          │ Maintenance"│
         └──────┬──────┘          └──────┬──────┘
                │                        │
                └────────────┬───────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ Store Result │
                      │ (MongoDB)    │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ Return JSON  │
                      │ Response     │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  Display     │
                      │  in Web UI   │
                      └──────────────┘
                             │
                             ▼
                       ┌─────────┐
                       │   END   │
                       └─────────┘


════════════════════════════════════════════════════════════════════════════
                      KEY ARCHITECTURAL DECISIONS
════════════════════════════════════════════════════════════════════════════

1. Model Selection Strategy:
   └─ Gradient Boosting (91.28%) as DEFAULT instead of CatBoost (94.06%)
      Reason: Better practical predictions for normal operations
              CatBoost/XGBoost too conservative (high false positive rate)

2. Threshold Adjustment:
   └─ Changed from 0.5 to 0.7 (70% confidence required)
      Reason: Reduce false positives while maintaining high true positive rate

3. Big Data Integration (Optional):
   └─ Apache Spark & MongoDB are OPTIONAL
      Reason: System works without them for datasets <1GB
              Enables scalability for production deployment

4. Feature Engineering:
   └─ 43 features (24 base + 19 engineered)
      Reason: Optimal balance between accuracy and overfitting
              Based on actual failure patterns from NASA dataset

5. Background Service (macOS):
   └─ LaunchAgent with KeepAlive
      Reason: Always-on operation without terminal dependency
              Auto-restart on crash ensures reliability

6. API Design:
   └─ RESTful with automatic OpenAPI documentation
      Reason: Industry standard, easy integration
              Self-documenting for developers


════════════════════════════════════════════════════════════════════════════
                         SECURITY CONSIDERATIONS
════════════════════════════════════════════════════════════════════════════

✓ Input validation on all API endpoints
✓ CORS configuration for cross-origin requests
✓ No sensitive data in predictions or logs
✓ Virtual environment isolation (.venv)
✓ Git LFS for large model files (not in Git history)
✓ .gitignore for cache, logs, and system files

Optional Enhancements:
• Rate limiting for production deployments
• Authentication layer (JWT, OAuth)
• HTTPS/TLS for encrypted communication
• API key management for access control


════════════════════════════════════════════════════════════════════════════
                        DEPLOYMENT ARCHITECTURE
════════════════════════════════════════════════════════════════════════════

Current (Development):
┌─────────────────────────────────────────────┐
│  Local Machine (macOS)                      │
│  ├─ Python 3.13 Virtual Environment         │
│  ├─ FastAPI Server (Port 8010)              │
│  ├─ LaunchAgent Background Service          │
│  ├─ MongoDB (Optional - localhost:27017)    │
│  └─ Apache Spark (Optional - local mode)    │
└─────────────────────────────────────────────┘

Production (Recommended):
┌─────────────────────────────────────────────┐
│  Cloud Infrastructure (AWS/Azure/GCP)       │
│  ├─ Docker Container (FastAPI + Models)     │
│  ├─ Load Balancer (Multi-instance)          │
│  ├─ MongoDB Atlas (Managed Database)        │
│  ├─ Apache Spark Cluster (EMR/Databricks)   │
│  ├─ API Gateway (Rate limiting, auth)       │
│  └─ CDN for Frontend (CloudFront/Cloudflare)│
└─────────────────────────────────────────────┘


════════════════════════════════════════════════════════════════════════════
                          PROJECT REPOSITORY
════════════════════════════════════════════════════════════════════════════

GitHub: https://github.com/patro-amit/Predictive-Pulse-Maintenance
Branch: main
Status: Production Ready ✅

Last Updated: October 21, 2025
Version: 4.0.0
License: Academic & Research Use
