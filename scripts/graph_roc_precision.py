"""
═══════════════════════════════════════════════════════════════════════════
GRAPH 2: ROC & Precision-Recall Analysis
═══════════════════════════════════════════════════════════════════════════
IEEE Conference Style Visualization
Displays: 2 SUBPLOTS (Intentional Dual-View Design)
  - Left: AUROC horizontal bar chart
  - Right: Precision vs Recall grouped bars
Purpose: Compare ROC performance and precision-recall trade-offs
Run: .venv/bin/python scripts/graph_roc_precision.py
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
    'figure.dpi': 100
})

def create_roc_precision_chart():
    """
    Creates dual-view comparison of ROC and Precision-Recall metrics
    
    Graph Type: 2 subplots (1x2 layout) - THIS IS INTENTIONAL
    Left Plot: AUROC horizontal bars for all models
    Right Plot: Precision vs Recall grouped bar comparison
    Note: Two plots allow simultaneous comparison of different metric aspects
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    data_path = Path(__file__).parent.parent / 'backend' / 'models' / 'model_comparison.csv'
    df = pd.read_csv(data_path)
    
    # Prepare data
    models = df['model_name'].str.replace('_', ' ').str.title()
    auroc = df['auroc'] * 100
    precision = df['precision'] * 100
    recall = df['recall'] * 100
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE CREATION (2 Subplots)
    # ═══════════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set proper window title (fixes "Figure 1" issue)
    fig.canvas.manager.set_window_title('GRAPH 2: ROC & Precision-Recall Analysis')  # type: ignore
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEFT SUBPLOT: AUROC Horizontal Bars
    # ═══════════════════════════════════════════════════════════════════════════
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars1 = ax1.barh(models, auroc, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}%',
                ha='left', va='center', fontsize=9, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('AUROC Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Area Under ROC Curve', fontsize=12, fontweight='bold')
    ax1.set_xlim([95, 100])
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.axvline(x=97, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RIGHT SUBPLOT: Precision vs Recall Grouped Bars
    # ═══════════════════════════════════════════════════════════════════════════
    x = np.arange(len(models))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, precision, width, label='Precision', 
                    color='#6C5CE7', edgecolor='black', linewidth=1.2)
    bars3 = ax2.bar(x + width/2, recall, width, label='Recall', 
                    color='#00B894', edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Machine Learning Models', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.legend(loc='lower right', frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 100)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUPER TITLE & FINALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    plt.suptitle('ROC and Precision-Recall Analysis for Predictive Maintenance Models',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("=" * 70)
    print("GRAPH 2: ROC & Precision-Recall Analysis (2 Subplots)")
    print("=" * 70)
    create_roc_precision_chart()
