# GitHub Setup Instructions

## 🚀 Quick Setup Guide

Your predictive maintenance project is ready to push to GitHub! Follow these steps:

### Step 1: Create Repository on GitHub

1. Go to: **https://github.com/new**
2. Fill in the details:
   - **Repository name:** `Predictive-Pulse-Maintenance`
   - **Description:** `Intelligent Predictive Maintenance System using Big Data & Machine Learning - 94.06% accuracy with CatBoost on NASA C-MAPSS dataset. Features Apache Spark & MongoDB integration.`
   - **Visibility:** Public (recommended for portfolio) or Private
   - **⚠️ IMPORTANT:** Do NOT check "Add a README file" (we already have a comprehensive one)
   - **⚠️ IMPORTANT:** Do NOT add .gitignore or license (already configured)
3. Click **"Create repository"**

### Step 2: Push Your Code

After creating the repository, run these commands in your terminal:

```bash
cd /Users/shyampatro/Predictive-Pulse-Maintenance

# Add GitHub as remote
git remote add origin https://github.com/shyampatro/Predictive-Pulse-Maintenance.git

# Push your code
git push -u origin main
```

### Step 3: Verify Upload

After pushing, visit:
**https://github.com/shyampatro/Predictive-Pulse-Maintenance**

You should see all your files uploaded!

---

## 📊 Project Summary

### What's Included:
✅ **Source code** - Backend (FastAPI) + Frontend (HTML/CSS/JS)
✅ **ML Models** - 4 trained models (.pkl files)
✅ **Training scripts** - Complete training pipeline
✅ **Datasets** - NASA C-MAPSS data (44,511 samples)
✅ **Configuration** - Requirements, shell scripts, configs
✅ **Documentation** - Comprehensive README.md

### What's Excluded (via .gitignore):
❌ Virtual environment (`.venv/` directory)
❌ Python cache files (`__pycache__/`)
❌ Jupyter checkpoints (`.ipynb_checkpoints/`)
❌ System files (`.DS_Store`, `Thumbs.db`)
❌ Log files (`*.log`, `nohup.out`)
❌ IDE settings (`.vscode/`, `.idea/`)

### Project Structure:
```
Predictive-Pulse-Maintenance/
├── backend/           # FastAPI server
│   ├── app.py        # Main application
│   ├── models/       # Trained ML models (4 .pkl files)
│   └── requirements.txt
├── frontend/         # Web interface
│   ├── static/       # CSS & JavaScript
│   └── templates/    # HTML templates
├── data/            # Training datasets
├── scripts/         # Training & visualization scripts
├── bin/             # Shell scripts
├── README.md        # Main documentation
└── GITHUB_SETUP.md  # This file
```

---

## 🎯 Current System Status

### Models Performance:
- **CatBoost:** 94.06% accuracy (Best Model) 🏆
- **XGBoost:** 94.04% accuracy
- **Gradient Boosting:** 93.54% accuracy
- **Random Forest:** 91.24% accuracy

### Features:
- 43 total features (21 sensors + 3 settings + 19 engineered)
- Feature engineering pipeline with ratios, interactions, rolling stats
- Real-time predictions with <100ms latency
- Scenario-based testing (Normal, High Risk, Critical)
- Apache Spark integration for big data processing
- MongoDB integration for prediction storage

### Recent Updates (October 18, 2025):
- ✅ Fixed normal operation predictions
- ✅ Adjusted sensor value ranges for realistic scenarios
- ✅ Updated frontend cache version (v8.0)
- ✅ Cleaned up documentation files
- ✅ Enhanced README with comprehensive information

---

## 🆘 Troubleshooting

**Error: "remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/shyampatro/Predictive-Pulse-Maintenance.git
```

**Error: "authentication failed"**
- Use Personal Access Token instead of password
- Generate token at: https://github.com/settings/tokens
- Use token as password when prompted

**Need to update remote URL:**
```bash
git remote set-url origin https://github.com/shyampatro/Predictive-Pulse-Maintenance.git
```

---

## 📝 After Upload

### Verify Your Repository
1. Visit: `https://github.com/YOUR_USERNAME/Predictive-Pulse-Maintenance`
2. Check that all files are uploaded
3. README.md should display properly with badges
4. Test clone: `git clone https://github.com/YOUR_USERNAME/Predictive-Pulse-Maintenance.git`

### Repository Features to Enable
- ✅ **Issues** - Track bugs and features
- ✅ **Wiki** - Extended documentation
- ✅ **Discussions** - Community Q&A
- ✅ **Topics** - Add tags: `machine-learning`, `predictive-maintenance`, `fastapi`, `big-data`, `apache-spark`, `mongodb`

### Add Repository Description
Navigate to Settings and add keywords:
- `predictive-maintenance`
- `machine-learning`
- `fastapi`
- `xgboost`
- `catboost`
- `apache-spark`
- `mongodb`
- `nasa-cmapss`
- `industrial-iot`
- `python`

---

## 🎓 For Academic Use

### Citing This Project
```
Predictive Maintenance System using Machine Learning
Dataset: NASA C-MAPSS Turbofan Engine Degradation
Accuracy: 94.06% (CatBoost Ensemble Model)
Features: 43 (21 sensors + 3 settings + 19 engineered)
GitHub: https://github.com/YOUR_USERNAME/Predictive-Pulse-Maintenance
```

### Project Highlights for Portfolio
- Production-ready FastAPI application
- 94.06% accuracy on real-world NASA dataset
- Big Data integration (Apache Spark + MongoDB)
- Modern responsive web interface
- Comprehensive feature engineering pipeline
- RESTful API with Swagger documentation

---

## 🔐 Security Notes

### Sensitive Information
Before pushing, ensure no sensitive data is committed:
- ✅ No API keys or passwords in code
- ✅ No personal data in datasets
- ✅ No production database credentials
- ✅ Virtual environment excluded

### .gitignore is configured to exclude:
```
.venv/
__pycache__/
*.pyc
*.log
.DS_Store
.env
*.key
```

---

## 🚀 Next Steps After Upload

1. **Star Your Own Repo** - Makes it easier to find
2. **Add README Badges** - Already included in README.md
3. **Write a Good Description** - Highlight 94% accuracy and Big Data
4. **Add Topics/Tags** - Helps with discovery
5. **Share** - Add link to resume/portfolio
6. **Keep Updated** - Push improvements regularly

---

## 📞 Support

**Issues with GitHub Setup?**
- Authenticate with Personal Access Token (not password)
- Generate at: https://github.com/settings/tokens
- Use HTTPS (easier) or SSH (more secure)

**Repository Already Exists?**
```bash
# Update existing repository
git remote set-url origin https://github.com/YOUR_USERNAME/Predictive-Pulse-Maintenance.git
git push -u origin main --force  # Use with caution!
```

---

**Happy Coding & Good Luck with Your Project!** 🚀⚙️

**Last Updated:** October 18, 2025
