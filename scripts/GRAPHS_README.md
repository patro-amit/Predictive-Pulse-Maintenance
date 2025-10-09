# 📊 IEEE Conference Style Graphs

Professional visualization scripts for the Predictive Maintenance project, styled for IEEE conference papers.

## 🎯 Available Graphs (8 Total)

### 1. **Model Accuracy Comparison** 📈
- **File**: `graph_model_accuracy.py`
- **Description**: Bar chart comparing accuracy and F1-scores across all models
- **Run**: `python scripts/graph_model_accuracy.py`

### 2. **ROC & Precision-Recall Analysis** 📊
- **File**: `graph_roc_precision.py`
- **Description**: Dual visualization showing AUROC scores and Precision-Recall trade-offs
- **Run**: `python scripts/graph_roc_precision.py`

### 3. **Confusion Matrix Heatmaps** 🔥
- **File**: `graph_confusion_matrix.py`
- **Description**: 2x2 grid showing confusion matrices for all 4 models
- **Run**: `python scripts/graph_confusion_matrix.py`

### 4. **Training vs Testing Performance** 📉
- **File**: `graph_train_test_performance.py`
- **Description**: Dataset split analysis and positive rate comparison
- **Run**: `python scripts/graph_train_test_performance.py`

### 5. **Sensor Correlation Heatmap** 🌡️
- **File**: `graph_sensor_correlation.py`
- **Description**: Correlation matrix of sensor features
- **Run**: `python scripts/graph_sensor_correlation.py`

### 6. **Feature Importance Ranking** ⭐
- **File**: `graph_feature_importance.py`
- **Description**: Horizontal bar chart showing most important features
- **Run**: `python scripts/graph_feature_importance.py`

### 7. **Sensor Degradation Time Series** 📉
- **File**: `graph_sensor_degradation.py`
- **Description**: Time series showing sensor degradation patterns over engine life
- **Run**: `python scripts/graph_sensor_degradation.py`

### 8. **Model Comparison Radar Chart** 🎯
- **File**: `graph_radar_comparison.py`
- **Description**: Radar chart comparing all models across 5 metrics
- **Run**: `python scripts/graph_radar_comparison.py`

---

## 🚀 Quick Start

### Install Dependencies
```bash
pip install matplotlib seaborn
```

### Run Individual Graph
```bash
# Example: Run accuracy comparison
python scripts/graph_model_accuracy.py
```

### Run All Graphs (Interactive Menu)
```bash
python scripts/run_all_graphs.py
```

---

## 📋 Features

✅ **Popup Windows**: All graphs open in separate popup windows  
✅ **IEEE Style**: Professional formatting for conference papers  
✅ **High DPI**: 100 DPI for crisp, publication-quality images  
✅ **Times New Roman**: IEEE standard font  
✅ **Interactive**: Zoom, pan, save from popup window  
✅ **Close on Stop**: Windows close when you close them or press Ctrl+C  

---

## 🎨 Graph Characteristics

All graphs feature:
- **Professional styling**: IEEE conference paper standards
- **Clear labels**: Bold, readable text with proper sizing
- **Grid lines**: Subtle gridlines for easier reading
- **Legends**: Positioned for clarity with shadows
- **Color schemes**: Carefully selected for print and screen
- **Annotations**: Value labels where appropriate
- **High contrast**: Readable in both color and grayscale

---

## 💡 Usage Tips

### Save Graph as Image
1. Run the graph script
2. Click the **Save** icon in the popup window
3. Choose format: PNG (recommended), PDF, SVG, or EPS
4. For papers: Use 300+ DPI for print quality

### Zoom and Pan
- **Zoom**: Click the magnifying glass icon, then click and drag
- **Pan**: Click the cross-arrow icon, then drag
- **Reset**: Click the home icon

### Close Graph
- Click the X button on the window
- Or press `Ctrl+C` in the terminal

### Modify Graph
Each script is self-contained and easy to customize:
- Colors: Line 13-20 (color definitions)
- Font size: Line 13 (`plt.rcParams['font.size']`)
- Figure size: In each function (`figsize=(width, height)`)

---

## 📁 Directory Structure

```
scripts/
├── run_all_graphs.py              # Master script with menu
├── graph_model_accuracy.py        # Graph 1
├── graph_roc_precision.py         # Graph 2
├── graph_confusion_matrix.py      # Graph 3
├── graph_train_test_performance.py # Graph 4
├── graph_sensor_correlation.py    # Graph 5
├── graph_feature_importance.py    # Graph 6
├── graph_sensor_degradation.py    # Graph 7
├── graph_radar_comparison.py      # Graph 8
└── GRAPHS_README.md              # This file
```

---

## 🔧 Troubleshooting

### Error: "No module named 'matplotlib'"
```bash
pip install matplotlib seaborn
```

### Error: "Unable to find file"
Make sure you're running from the project root:
```bash
cd /Users/shyampatro/Predictive-Pulse-Maintenance
python scripts/graph_model_accuracy.py
```

### Graph doesn't appear
- Check if another graph window is open
- Try closing all Python windows
- On macOS: Grant terminal access to control computer

### Poor quality when saving
- Use PNG or PDF format
- Increase DPI: Modify `plt.rcParams['figure.dpi'] = 300`
- Use vector formats (PDF, SVG, EPS) for papers

---

## 📄 For IEEE Conference Papers

### Recommended Settings for Publication:
```python
plt.rcParams['figure.dpi'] = 300  # High resolution
plt.savefig('figure.pdf', bbox_inches='tight')  # PDF for papers
```

### Caption Examples:
- "Fig. 1. Comparative accuracy and F1-score analysis across four machine learning models for predictive maintenance."
- "Fig. 2. Confusion matrix heatmaps demonstrating classification performance of Random Forest, Gradient Boosting, XGBoost, and CatBoost models."
- "Fig. 3. Time series analysis showing sensor degradation patterns over operational cycles (NASA C-MAPSS dataset)."

---

## 🎓 For College Presentation

### Tips:
1. **Run graphs during presentation**: Shows live data processing
2. **Explain each metric**: Define accuracy, precision, recall, F1-score
3. **Highlight best model**: Point out Random Forest's 92.30% accuracy
4. **Show degradation**: Use time series to explain predictive maintenance concept
5. **Compare models**: Use radar chart to show trade-offs

### Presentation Order:
1. Start with accuracy comparison (easy to understand)
2. Show confusion matrices (explain TP, TN, FP, FN)
3. Display sensor degradation (real-world context)
4. End with radar chart (comprehensive comparison)

---

## ✨ Summary

**8 Professional Graphs** | **IEEE Style** | **Popup Windows** | **Easy to Run**

Created for: Predictive Maintenance Big Data Project  
Date: October 2025  
Format: IEEE Conference Standard

Run any graph with: `python scripts/graph_<name>.py`  
Run all graphs: `python scripts/run_all_graphs.py`

---

**Happy Visualizing! 📊✨**
