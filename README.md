# 🏭 Predictive Maintenance System
### "Predictive Maintenance Strategies Using Big Data And Machine Learning"

[![Status](https://img.shields.io/badge/status-production--ready-brightgreen)]()
[![Accuracy](https://img.shields.io/badge/accuracy-92.30%25-success)]()
[![Models](https://img.shields.io/badge/models-5-blue)]()
[![Python](https://img.shields.io/badge/python-3.9-blue)]()

> An intelligent system that predicts equipment failures before they happen, using machine learning to analyze sensor data from industrial equipment. Achieves 92.30% accuracy using ensemble methods on 44,511 data samples.

---

## 🎯 Quick Access

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[QUICKSTART.md](docs/QUICKSTART.md)** | Get started in 5 minutes | 5 min |
| **[COLLEGE_PRESENTATION.md](docs/COLLEGE_PRESENTATION.md)** | Complete presentation script & Q&A | 20 min |
| **[HOW_TO_USE.md](docs/HOW_TO_USE.md)** | Detailed usage guide | 15 min |
| **[VISUAL_GUIDE.md](docs/VISUAL_GUIDE.md)** | Step-by-step walkthrough | 10 min |

---

## 🚀 30-Second Setup

```bash
cd backend
/Users/shyampatro/Predictive-Pulse-Maintenance/backend/venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8010 --reload
```

Open: **http://localhost:8010**

---

## ✨ Key Features

- 🤖 **5 ML Models** achieving 90%+ accuracy
- 📊 **44,511 training samples** - Big Data scale
- 🎯 **92.30% accuracy** with Random Forest
- 🌐 **Modern web UI** with glass morphism design
- 🔌 **REST API** with auto-documentation
- ⚡ **<100ms latency** per prediction
- 📱 **Responsive design** for all devices

---

## 📊 Model Performance

| Model | Accuracy | Status |
|-------|----------|---------|
| **Random Forest** | **92.30%** | 🏆 Best |
| Gradient Boosting | 92.22% | ✅ |
| XGBoost | 92.10% | ✅ |
| CatBoost | 91.34% | ✅ |

---

## 📁 Project Structure

```
Predictive-Pulse-Maintenance/
├── README.md                      ← You are here
│
├── backend/                       ← FastAPI server
│   ├── app.py                    ← Main app (5 models)
│   ├── requirements.txt          ← Dependencies
│   ├── models/                   ← Trained ML models (.pkl)
│   └── __pycache__/              ← Python cache
│
├── frontend/                      ← Web UI
│   ├── templates/
│   │   └── index.html           ← Main page
│   └── static/
│       ├── css/
│       │   ├── style_modern.css ← Active UI (dark theme)
│       │   └── archive/         ← Old styles
│       └── js/
│           └── app.js           ← Frontend logic
│
├── data/                          ← Datasets
│   ├── predictive_maintenance_bigdata.csv (44,511 samples)
│   └── cmapss_train_binary.csv  ← NASA C-MAPSS
│
├── scripts/                       ← Python utilities
│   ├── train.py                 ← Model training
│   ├── generate_data.py         ← Data generation
│   ├── graph_*.py               ← 8 graph scripts
│   └── run_all_graphs.py        ← Graph menu
│
├── bin/                           ← Shell scripts
│   ├── graph1.sh - graph8.sh    ← Easy graph runners
│   ├── run_graphs.sh            ← Graph help
│   └── start.sh                 ← Server starter
│
└── docs/                          ← Documentation
    ├── QUICKSTART.md            ← 5-min setup
    ├── COLLEGE_PRESENTATION.md  ← Presentation guide
    ├── HOW_TO_USE.md            ← Usage guide
    ├── VISUAL_GUIDE.md          ← Screenshots
    ├── USER_GUIDE.md            ← Complete manual
    ├── GRAPHS_SUMMARY.md        ← Graph documentation
    ├── READY_TO_USE_COMMANDS.md ← Quick commands
    └── [14 more documentation files]
```

---

## 🎓 For Your Presentation

Read **[COLLEGE_PRESENTATION.md](docs/COLLEGE_PRESENTATION.md)** for:

✅ 20-minute demo script  
✅ What to showcase  
✅ Technical explanations  
✅ Q&A with answers  
✅ Presentation tips  

**Key Points:**
- 92.30% accuracy achieved
- 5 models compared scientifically
- 44,511 samples (Big Data)
- Production-ready system
- Industry relevance

## 📊 Visualizations

Run professional IEEE-style graphs:
```bash
.venv/bin/python scripts/graph_model_accuracy.py      # Graph 1
.venv/bin/python scripts/graph_confusion_matrix.py    # Graph 3
.venv/bin/python scripts/graph_sensor_degradation.py  # Graph 7
```

Or use easy shortcuts:
```bash
./bin/graph1.sh    # Model accuracy comparison
./bin/graph3.sh    # Confusion matrices
./bin/graph7.sh    # Sensor degradation
```

See **[GRAPHS_SUMMARY.md](docs/GRAPHS_SUMMARY.md)** for all 8 graphs!

---

## 🌐 Endpoints

- **GET** `/` - Web UI
- **GET** `/health` - Status check
- **POST** `/predict` - Make prediction
- **GET** `/models` - List models
- **GET** `/docs` - API documentation
- **GET** `/example` - Sample data

---

## 💻 Technology Stack

**Backend:** FastAPI, Python 3.9, Uvicorn  
**ML:** Scikit-learn, XGBoost, CatBoost  
**Frontend:** HTML5, CSS3, Vanilla JS  
**Data:** 44,511 samples, 29 features  

---

## 📚 Documentation Index

### 🎯 Getting Started
- [**QUICKSTART.md**](docs/QUICKSTART.md) - 5-minute setup
- [**HOW_TO_USE.md**](docs/HOW_TO_USE.md) - Detailed usage guide
- [**VISUAL_GUIDE.md**](docs/VISUAL_GUIDE.md) - Step-by-step with screenshots
- [**READY_TO_USE_COMMANDS.md**](docs/READY_TO_USE_COMMANDS.md) - Command reference

### 🎓 Academic & Presentation
- [**COLLEGE_PRESENTATION.md**](docs/COLLEGE_PRESENTATION.md) - Complete presentation script
- [**PRESENTATION_READY.md**](docs/PRESENTATION_READY.md) - Quick presentation guide
- [**SIMPLIFIED_FEATURES.md**](docs/SIMPLIFIED_FEATURES.md) - Feature explanation

### 📊 Graphs & Visualizations
- [**GRAPHS_SUMMARY.md**](docs/GRAPHS_SUMMARY.md) - All 8 IEEE-style graphs
- [**GRAPHS_ERRORS_FIXED.md**](docs/GRAPHS_ERRORS_FIXED.md) - Technical fixes
- [**HOW_TO_RUN_GRAPHS.txt**](docs/HOW_TO_RUN_GRAPHS.txt) - Quick reference

### 🔧 Technical Documentation
- [**USER_GUIDE.md**](docs/USER_GUIDE.md) - Comprehensive manual
- [**EVERYTHING_WORKING.md**](docs/EVERYTHING_WORKING.md) - System status
- [**MODEL_SELECTION_FIXED.md**](docs/MODEL_SELECTION_FIXED.md) - Model selection guide
- [**UI_UPDATE_COMPLETE.md**](docs/UI_UPDATE_COMPLETE.md) - UI documentation

### 📁 Project Documentation
- [**PROJECT_SUMMARY.md**](docs/PROJECT_SUMMARY.md) - Overview
- [**VERIFICATION_REPORT.md**](docs/VERIFICATION_REPORT.md) - Testing results
- [**FIXES_APPLIED.md**](docs/FIXES_APPLIED.md) - Bug fixes log
- [**ORGANIZATION_COMPLETE.md**](docs/ORGANIZATION_COMPLETE.md) - Structure guide

---

## 🎯 Use Cases

✅ Manufacturing equipment monitoring  
✅ Wind turbine maintenance  
✅ Aircraft engine prediction  
✅ Medical equipment reliability  

---

## 💡 Quick Start

1. Start server (see 30-second setup above)
2. Open http://localhost:8010
3. Click "Start Prediction"
4. Click "Use Example"
5. Click "Load Example Data"
6. Click "Predict Maintenance"
7. See results! 🎉

---

## 📞 Support

- **Quick Issues:** Check `QUICKSTART.md`
- **Usage Questions:** See `HOW_TO_USE.md`
- **API Docs:** http://localhost:8010/docs
- **Health Check:** http://localhost:8010/health

---

**Built for predictive maintenance** | **Preventing failures, one prediction at a time** 🔧⚙️

---

**Last Updated:** October 9, 2025 | **Version:** 3.0.0 | **Status:** Production Ready ✅
