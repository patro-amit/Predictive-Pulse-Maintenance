#!/usr/bin/env python3
"""
GRAPH 2A: AUROC Comparison (Single Plot)
Area Under ROC Curve comparison across all models
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# IEEE Conference Style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 100

def create_auroc_chart():
    """Single plot showing AUROC comparison"""
    print("="*70)
    print("GRAPH 2A: AUROC Comparison - Horizontal Bar Chart")
    print("="*70)
    
    # Load data
    data_path = Path(__file__).parent.parent / 'backend' / 'models' / 'model_comparison.csv'
    df = pd.read_csv(data_path)
    
    # Prepare data
    models = df['model_name'].str.replace('_', ' ').str.title()
    auroc = df['auroc'] * 100
    
    # Create figure - SINGLE PLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('GRAPH 2A: AUROC Comparison')  # type: ignore
    
    # Colors for each model
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # Create horizontal bar chart
    bars = ax.barh(models, auroc, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, auroc)):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{value:.2f}%',
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Add reference line at 95%
    ax.axvline(x=95, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='95% Threshold')
    
    # Styling
    ax.set_xlabel('AUROC Score (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Machine Learning Models', fontweight='bold', fontsize=12)
    ax.set_title('Area Under ROC Curve - Model Comparison', fontweight='bold', fontsize=14, pad=15)
    
    # Set x-axis limits
    ax.set_xlim(95, 100)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, shadow=True)
    
    plt.tight_layout()
    
    print(f"✓ AUROC comparison chart generated")
    print(f"  Best Model: {models[auroc.argmax()]} ({auroc.max():.2f}%)")
    
    plt.show()

if __name__ == '__main__':
    create_auroc_chart()
