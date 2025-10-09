#!/usr/bin/env python3
"""
Cleanup and Organization Script
Removes unnecessary files and organizes the project structure
"""

import os
import shutil
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
print(f"Project root: {ROOT}")

# Files to remove from root
FILES_TO_REMOVE = [
    "test_api.py",
    "start_app.py", 
    "commands.sh",
    "run_backend.sh",
    "run_frontend.sh",
    "setup.sh",
    "SETUP_COMPLETE.md"
]

# Files to remove from backend
BACKEND_FILES_TO_REMOVE = [
    "generate_data.py",
    "train.py"
]

print("\n" + "="*70)
print("  PROJECT CLEANUP AND ORGANIZATION")
print("="*70)

# Remove unnecessary files from root
print("\n📁 Cleaning root directory...")
removed_count = 0
for file in FILES_TO_REMOVE:
    file_path = ROOT / file
    if file_path.exists():
        try:
            file_path.unlink()
            print(f"   ✅ Removed: {file}")
            removed_count += 1
        except Exception as e:
            print(f"   ⚠️  Could not remove {file}: {e}")
    else:
        print(f"   ℹ️  Already removed: {file}")

# Remove duplicate files from backend
print("\n📁 Cleaning backend directory...")
backend_dir = ROOT / "backend"
for file in BACKEND_FILES_TO_REMOVE:
    file_path = backend_dir / file
    if file_path.exists():
        try:
            file_path.unlink()
            print(f"   ✅ Removed: backend/{file}")
            removed_count += 1
        except Exception as e:
            print(f"   ⚠️  Could not remove {file}: {e}")
    else:
        print(f"   ℹ️  Already removed: backend/{file}")

# Remove old Streamlit files
print("\n📁 Cleaning frontend directory...")
frontend_files = [
    "streamlit_app.py",
    "streamlit_app.py.old",
    "requirements.txt"  # Old Streamlit requirements
]
frontend_dir = ROOT / "frontend"
for file in frontend_files:
    file_path = frontend_dir / file
    if file_path.exists():
        try:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
            print(f"   ✅ Removed: frontend/{file}")
            removed_count += 1
        except Exception as e:
            print(f"   ⚠️  Could not remove {file}: {e}")

# Remove frontend venv if exists
frontend_venv = frontend_dir / "venv"
if frontend_venv.exists():
    try:
        shutil.rmtree(frontend_venv)
        print(f"   ✅ Removed: frontend/venv/")
        removed_count += 1
    except Exception as e:
        print(f"   ⚠️  Could not remove frontend/venv: {e}")

# Remove catboost_info from backend
catboost_backend = backend_dir / "catboost_info"
if catboost_backend.exists():
    try:
        shutil.rmtree(catboost_backend)
        print(f"   ✅ Removed: backend/catboost_info/")
        removed_count += 1
    except Exception as e:
        print(f"   ⚠️  Could not remove backend/catboost_info: {e}")

# Remove __pycache__ directories
print("\n📁 Cleaning __pycache__ directories...")
for pycache in ROOT.rglob("__pycache__"):
    if pycache.is_dir():
        try:
            shutil.rmtree(pycache)
            print(f"   ✅ Removed: {pycache.relative_to(ROOT)}")
            removed_count += 1
        except Exception as e:
            print(f"   ⚠️  Could not remove {pycache}: {e}")

# Summary
print("\n" + "="*70)
print(f"✨ CLEANUP COMPLETE!")
print(f"   Removed {removed_count} files/directories")
print("="*70)

# Show final structure
print("\n📊 Final Project Structure:")
print("""
Predictive-Pulse-Maintenance/
├── backend/
│   ├── app.py              ✅ FastAPI server
│   ├── requirements.txt    ✅ Dependencies
│   ├── venv/               ✅ Virtual environment
│   └── models/             ✅ Trained models
├── frontend/
│   ├── templates/
│   │   └── index.html      ✅ Web UI
│   └── static/
│       ├── css/style.css   ✅ Styling
│       └── js/app.js       ✅ JavaScript
├── scripts/
│   ├── generate_data.py    ✅ Data generator
│   ├── train.py            ✅ Model training
│   ├── test_api.py         ✅ API tests
│   ├── start_app.py        ✅ App launcher
│   └── cleanup.py          ✅ This script
├── data/
│   └── predictive_maintenance_bigdata.csv
├── start.sh                ✅ Quick start
├── README.md               ✅ Documentation
├── QUICKSTART.md
├── PROJECT_SUMMARY.md
└── FINAL_SUMMARY.md
""")

print("\n🚀 Project is now clean and organized!")
print("   Run: ./start.sh to launch the application")
