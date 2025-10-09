# GitHub Setup Instructions

## 🚀 Quick Setup Guide

Your project is ready to push to GitHub! Follow these steps:

### Step 1: Create Repository on GitHub

1. Go to: **https://github.com/new**
2. Fill in the details:
   - **Repository name:** `Predictive-Pulse-Maintenance`
   - **Description:** `Predictive Maintenance using Big Data & Machine Learning - 92%+ accuracy with Random Forest, XGBoost, Gradient Boosting, and CatBoost models`
   - **Visibility:** Public (recommended) or Private
   - **⚠️ IMPORTANT:** Do NOT check "Add a README file" (we already have one)
   - **⚠️ IMPORTANT:** Do NOT add .gitignore or license (we already have them)
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

### What's Uploaded (935 MB total):
✅ Source code (Backend + Frontend)
✅ Training scripts (29 files)
✅ Datasets (44,511 samples, 28 MB)
✅ Configuration files
✅ README.md with project documentation

### What's Excluded:
❌ Virtual environment (`.venv/` - 533 MB)
❌ Model files (`*.pkl` - 373 MB)
❌ Cache files (`__pycache__/`)

### Why Models are Excluded:
GitHub has a 100 MB file size limit. Your trained models (373 MB) exceed this.

**Solutions:**
1. **Users retrain models** - Run `python scripts/train.py --csv data/predictive_maintenance_bigdata.csv --label label --group unit`
2. **Use Git LFS** - Upload large files separately
3. **Host elsewhere** - Google Drive, Dropbox, GitHub Releases

---

## 🔄 Alternative: Use Git LFS for Models

If you want to include the trained models:

```bash
# Install Git LFS
brew install git-lfs
git lfs install

# Track model files
git lfs track "backend/models/*.pkl"
git add .gitattributes
git add backend/models/*.pkl
git commit -m "Add trained models via Git LFS"
git push
```

---

## ✅ Commit Summary

**Initial commit includes:**
- 59 files
- 63,723 lines of code
- FastAPI backend with 5 ML models
- Modern web UI
- Complete training pipeline
- 13 visualization scripts

**Models Included:**
- Random Forest: 92.26% accuracy
- XGBoost: 92.10% accuracy
- Gradient Boosting: 92.16% accuracy
- CatBoost: 91.34% accuracy

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

Add these badges to your README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![Accuracy](https://img.shields.io/badge/accuracy-92.26%25-success.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
```

Happy coding! 🚀
