"""
═══════════════════════════════════════════════════════════════════════════
GRAPH 3: Confusion Matrix Analysis
═══════════════════════════════════════════════════════════════════════════
IEEE Conference Style Visualization
Displays: 4 SUBPLOTS (Intentional Multi-Model Comparison)
  - 2x2 grid showing confusion matrix for each model
Purpose: Side-by-side comparison of classification performance
Run: .venv/bin/python scripts/graph_confusion_matrix.py
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

def create_confusion_matrices():
    """
    Creates 2x2 grid of confusion matrices for all ML models
    
    Graph Type: 4 subplots (2x2 grid) - THIS IS INTENTIONAL
    Purpose: Allows direct comparison of TP, TN, FP, FN across all models
    Note: Multi-subplot design is standard for comparative confusion matrix analysis
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    data_path = Path(__file__).parent.parent / 'backend' / 'models' / 'model_comparison.csv'
    df = pd.read_csv(data_path)
    
    # Parse accuracy - handle percentage strings
    if df['accuracy'].dtype == 'object':
        df['accuracy'] = df['accuracy'].str.rstrip('%').astype(float) / 100
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE CREATION (4 Subplots in 2x2 Grid)
    # ═══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Set proper window title (fixes "Figure 1" issue)
    fig.canvas.manager.set_window_title('GRAPH 3: Confusion Matrix Comparison')  # type: ignore
    
    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        
        # Create confusion matrix
        cm = np.array([
            [row['true_negatives'], row['false_positives']],
            [row['false_negatives'], row['true_positives']]
        ])
        
        # Calculate percentages
        cm_percent = cm / cm.sum() * 100
        
        # Create heatmap
        im = ax.imshow(cm, cmap='Blues', alpha=0.8, vmin=0, vmax=7000)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > 3500 else 'black'
                text = ax.text(j, i, f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)',
                             ha="center", va="center", color=color,
                             fontsize=12, fontweight='bold')
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted: No\nMaintenance', 'Predicted:\nMaintenance'], fontsize=9)
        ax.set_yticklabels(['Actual: No\nMaintenance', 'Actual:\nMaintenance'], fontsize=9)
        
        # Title with accuracy
        model_name = row['model_name'].replace('_', ' ').title()
        accuracy = row['accuracy'] * 100
        ax.set_title(f'{model_name}\nAccuracy: {accuracy:.2f}%', 
                    fontsize=11, fontweight='bold', pad=10)
        
        # Grid
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Count', fontsize=9)
    
    plt.suptitle('Confusion Matrix Analysis for Predictive Maintenance Models',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.99))  # type: ignore
    plt.show()

if __name__ == '__main__':
    print("=" * 60)
    print("GRAPH 3: Confusion Matrix Heatmaps")
    print("=" * 60)
    create_confusion_matrices()
