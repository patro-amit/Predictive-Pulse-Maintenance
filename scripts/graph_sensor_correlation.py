"""
═══════════════════════════════════════════════════════════════════════════
GRAPH 5: Sensor Feature Correlation Heatmap
═══════════════════════════════════════════════════════════════════════════
IEEE Conference Style Visualization
Displays: Single heatmap showing correlation matrix
Purpose: Identify feature relationships and multicollinearity
Run: .venv/bin/python scripts/graph_sensor_correlation.py
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
    'font.size': 9,
    'figure.dpi': 100
})

def create_correlation_heatmap():
    """
    Creates correlation heatmap of sensor features
    
    Graph Type: Single plot (correlation matrix heatmap)
    Purpose: Visualize relationships between 16 sensor features
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    data_path = Path(__file__).parent.parent / 'data' / 'cmapss_train_binary.csv'
    df = pd.read_csv(data_path)
    
    # Select important sensor features from NASA C-MAPSS dataset
    feature_cols = ['cycle', 'setting1', 'setting2', 'setting3', 
                   's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
                   's9', 's10', 's11', 's12']
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE CREATION
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set proper window title (fixes "Figure 1" issue)
    fig.canvas.manager.set_window_title('GRAPH 5: Sensor Correlation Heatmap')  # type: ignore
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_yticks(np.arange(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right')
    ax.set_yticklabels(feature_cols)
    
    # Add text annotations
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            value = corr_matrix.iloc[i, j]
            # Convert to float for comparison
            val_float = float(str(value)) if value is not None else 0.0
            color = 'white' if abs(val_float) > 0.5 else 'black'
            text = ax.text(j, i, f'{val_float:.2f}',
                         ha="center", va="center", color=color,
                         fontsize=7, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    # Title
    ax.set_title('Sensor Feature Correlation Matrix\nfor Predictive Maintenance Dataset',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.set_xticks(np.arange(len(feature_cols))-0.5, minor=True)
    ax.set_yticks(np.arange(len(feature_cols))-0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("=" * 60)
    print("GRAPH 5: Sensor Feature Correlation Heatmap")
    print("=" * 60)
    print("Loading sensor data...")
    create_correlation_heatmap()
