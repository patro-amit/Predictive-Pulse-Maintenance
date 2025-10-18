"""
═══════════════════════════════════════════════════════════════════════════
GRAPH 1: Model Accuracy Comparison
═══════════════════════════════════════════════════════════════════════════
IEEE Conference Style Visualization
Displays: Single grouped bar chart comparing Accuracy and F1-Score
Run: .venv/bin/python scripts/graph_model_accuracy.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# IEEE STYLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 100
})

def create_accuracy_comparison():
    """
    Creates a grouped bar chart comparing accuracy and F1-score across ML models
    
    Graph Type: Single plot with grouped bars
    Purpose: Visual comparison of model performance metrics
    Models: Random Forest, Gradient Boosting, XGBoost, CatBoost
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    data_path = Path(__file__).parent.parent / 'backend' / 'models' / 'model_comparison.csv'
    df = pd.read_csv(data_path)
    
    # Prepare data
    models = df['model_name'].str.replace('_', ' ').str.title()
    
    # Parse accuracy - handle both percentage strings and decimals
    if df['accuracy'].dtype == 'object':
        accuracy = df['accuracy'].str.rstrip('%').astype(float)
    else:
        accuracy = df['accuracy'] * 100
    
    # Parse F1 score
    f1_score = df['f1'] * 100
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE CREATION
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set proper window title (fixes "Figure 1" issue)
    fig.canvas.manager.set_window_title('GRAPH 1: Model Accuracy Comparison')  # type: ignore
    
    # Bar positioning
    x = np.arange(len(models))
    width = 0.35
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BAR CHART PLOTTING
    # ═══════════════════════════════════════════════════════════════════════════
    bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', 
                   color='#2E86AB', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, f1_score, width, label='F1-Score (%)', 
                   color='#A23B72', edgecolor='black', linewidth=1.2)
    
    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STYLING & ANNOTATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    ax.set_xlabel('Machine Learning Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Metrics (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparative Analysis of ML Model Performance\nfor Predictive Maintenance', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    
    # Add 90% baseline reference line
    ax.axhline(y=90, color='red', linestyle='--', linewidth=1, alpha=0.5, label='90% Baseline')
    
    plt.tight_layout()
    
    # Add IEEE-style figure caption
    fig.text(0.5, 0.01, 'Fig. 5.  Model Performance Comparison',
             ha='center', va='bottom', fontsize=12,
             style='italic', weight='bold')
    
    plt.subplots_adjust(bottom=0.12)  # Make room for caption
    plt.show()

if __name__ == '__main__':
    print("=" * 70)
    print("GRAPH 1: Model Accuracy Comparison")
    print("=" * 70)
    create_accuracy_comparison()
