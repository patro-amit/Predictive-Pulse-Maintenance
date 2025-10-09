"""
═══════════════════════════════════════════════════════════════════════════
GRAPH 8: Multi-Metric Model Comparison (Radar Chart)
═══════════════════════════════════════════════════════════════════════════
IEEE Conference Style Visualization
Displays: Single polar/radar plot
Purpose: Holistic comparison of all models across 5 metrics simultaneously
Run: .venv/bin/python scripts/graph_radar_comparison.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from math import pi

# ═══════════════════════════════════════════════════════════════════════════
# IEEE STYLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'figure.dpi': 100
})

def create_radar_comparison():
    """
    Creates radar/spider chart comparing all models across 5 metrics
    
    Graph Type: Single plot (polar projection)
    Purpose: Simultaneous visualization of multiple performance dimensions
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    data_path = Path(__file__).parent.parent / 'backend' / 'models' / 'model_comparison.csv'
    df = pd.read_csv(data_path)
    
    # Metrics to compare (normalized to 0-1 scale)
    metrics = ['accuracy', 'f1', 'auroc', 'precision', 'recall']
    metric_labels = ['Accuracy', 'F1-Score', 'AUROC', 'Precision', 'Recall']
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE CREATION (Polar Projection)
    # ═══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Set proper window title (fixes "Figure 1" issue)
    fig.canvas.manager.set_window_title('GRAPH 8: Multi-Metric Model Comparison')  # type: ignore
    
    # Colors for each model
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    
    # Plot each model
    for i, (idx, row) in enumerate(df.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        model_name = row['model_name'].replace('_', ' ').title()
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
               color=colors[i], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    # Fix axis to go in the right order and start at 12 o'clock
    if hasattr(ax, 'set_theta_offset'):
        ax.set_theta_offset(pi / 2)  # type: ignore
    if hasattr(ax, 'set_theta_direction'):
        ax.set_theta_direction(-1)  # type: ignore
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
              frameon=True, shadow=True, fontsize=11)
    
    # Add title
    plt.title('Multi-Metric Performance Comparison\nPredictive Maintenance Models',
             fontsize=14, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("=" * 60)
    print("GRAPH 8: Model Comparison Radar Chart")
    print("=" * 60)
    create_radar_comparison()
