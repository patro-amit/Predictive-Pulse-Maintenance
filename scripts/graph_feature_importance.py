"""
═══════════════════════════════════════════════════════════════════════════
GRAPH 6: Feature Importance Ranking
═══════════════════════════════════════════════════════════════════════════
IEEE Conference Style Visualization
Displays: Single horizontal bar chart
Purpose: Rank features by their contribution to model predictions
Run: .venv/bin/python scripts/graph_feature_importance.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
import json

# ═══════════════════════════════════════════════════════════════════════════
# IEEE STYLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'figure.dpi': 100
})

def create_feature_importance_chart():
    """
    Creates horizontal bar chart ranking feature importance
    
    Graph Type: Single plot (horizontal bars)
    Purpose: Identify most influential features for prediction
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA PREPARATION
    # ═══════════════════════════════════════════════════════════════════════════
    # Simulated feature importance (you can load from your model)
    features = ['temp_avg', 'pressure_avg', 'vibration_avg', 'rpm_avg', 
                'cycle', 'setting1', 'setting2', 'setting3',
                's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    
    # Simulated importance scores (normalized)
    importance = [0.152, 0.145, 0.138, 0.125, 0.095, 0.082, 0.075, 0.068,
                 0.035, 0.028, 0.022, 0.018, 0.015, 0.012, 0.008, 0.002]
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = [importance[i] for i in sorted_idx]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE CREATION - CLEAN MINIMAL STYLE (LIKE PDF)
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.canvas.manager.set_window_title('Fig. 2: Column Data RandomForest')  # type: ignore
    
    # Create horizontal bars - SIMPLE BLUE COLOR LIKE PDF
    bars = ax.barh(sorted_features, sorted_importance, color='steelblue')
    
    # Customize - MINIMAL LIKE PDF
    ax.set_xlabel('Importance')
    ax.set_title('Top 20 Feature Importances', pad=15)
    ax.set_xlim(0, max(importance) * 1.05)
    ax.grid(False)  # No grid like PDF
    
    plt.tight_layout()
    
    # Add IEEE-style figure caption at bottom (like PDF)
    fig.text(0.5, 0.02, 'Fig. 2.  Column Data RandomForest',
             ha='center', va='bottom', fontsize=11,
             style='italic', weight='bold')
    
    plt.subplots_adjust(bottom=0.10)
    plt.show()

if __name__ == '__main__':
    print("=" * 60)
    print("GRAPH 6: Feature Importance Ranking")
    print("=" * 60)
    create_feature_importance_chart()
