"""
═══════════════════════════════════════════════════════════════════════════
GRAPH 10: Enhanced Classification Report Heatmap
═══════════════════════════════════════════════════════════════════════════
IEEE Conference Style Visualization
Displays: Single heatmap showing precision, recall, f1-score for all models
Purpose: Comprehensive classification metrics comparison
Run: .venv/bin/python scripts/graph_classification_heatmap.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# ═══════════════════════════════════════════════════════════════════════════
# IEEE STYLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'figure.figsize': (14, 9),
    'figure.dpi': 100
})

def create_classification_heatmap():
    """
    Creates enhanced classification report heatmap for all models
    
    Graph Type: Single heatmap (NOT multiple plots - one comprehensive view)
    Purpose: Show precision, recall, f1-score across all models
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    data_path = Path(__file__).parent.parent / 'backend' / 'models' / 'model_comparison.csv'
    df = pd.read_csv(data_path)
    
    # Parse accuracy - handle percentage strings
    if df['accuracy'].dtype == 'object':
        df['accuracy'] = df['accuracy'].str.rstrip('%').astype(float) / 100
    
    # Prepare data for heatmap
    models = df['model_name'].str.replace('_', ' ').str.title().tolist()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']
    
    # Create matrix
    data_matrix = []
    for _, row in df.iterrows():
        data_matrix.append([
            row['accuracy'] * 100,
            row['precision'] * 100,
            row['recall'] * 100,
            row['f1'] * 100,
            row['auroc'] * 100
        ])
    
    data_matrix = np.array(data_matrix)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE CREATION
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.canvas.manager.set_window_title('GRAPH 10: Classification Report Heatmap')  # type: ignore
    
    # Create heatmap with cleaner, simpler colors - Blues colormap
    im = ax.imshow(data_matrix, cmap='Blues', aspect='auto', vmin=90, vmax=100)
    
    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_yticklabels(models, fontsize=12, fontweight='bold')
    
    # Keep x labels horizontal (NO rotation)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add text annotations with values
    for i in range(len(models)):
        for j in range(len(metrics)):
            value = data_matrix[i, j]
            # Color logic for Blues colormap: light blue = low values, dark blue = high values
            if value < 93:
                text_color = 'black'  # Light blue background
            elif value < 96:
                text_color = 'darkblue'  # Medium blue background
            else:
                text_color = 'white'  # Dark blue background
            
            text = ax.text(j, i, f'{value:.1f}%',
                         ha="center", va="center", color=text_color,
                         fontsize=13, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Performance Score (%)', rotation=270, labelpad=20, 
                   fontsize=12, fontweight='bold')
    
    # Title - shorter and cleaner
    ax.set_title('Classification Heatmap',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.set_xticks(np.arange(len(metrics))-0.5, minor=True)
    ax.set_yticks(np.arange(len(models))-0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=3)
    
    # NO red border or summary box - removed for clean appearance
    
    plt.tight_layout()
    
    # Add IEEE-style figure caption - FULLY BOLD
    fig.text(0.5, 0.01, 'Fig. 8. Enhanced Classification Heatmap',
             ha='center', va='bottom', fontsize=12,
             style='italic', weight='bold')
    
    plt.subplots_adjust(bottom=0.06)  # Make room for caption
    plt.show()

if __name__ == '__main__':
    print("=" * 70)
    print("GRAPH 10: Enhanced Classification Report Heatmap")
    print("=" * 70)
    create_classification_heatmap()
