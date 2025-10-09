#!/usr/bin/env python3
"""
GRAPH 2B: Precision vs Recall Trade-off (Single Plot)
Grouped bar chart comparing Precision and Recall across models
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# IEEE Conference Style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 100

def create_precision_recall_chart():
    """Single plot showing Precision vs Recall"""
    print("="*70)
    print("GRAPH 2B: Precision vs Recall Trade-off")
    print("="*70)
    
    # Load data
    data_path = Path(__file__).parent.parent / 'backend' / 'models' / 'model_comparison.csv'
    df = pd.read_csv(data_path)
    
    # Prepare data
    models = df['model_name'].str.replace('_', ' ').str.title()
    precision = df['precision'] * 100
    recall = df['recall'] * 100
    
    # Create figure - SINGLE PLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('GRAPH 2B: Precision vs Recall')  # type: ignore
    
    # Bar positions
    x = np.arange(len(models))
    width = 0.35
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, precision, width, label='Precision', 
                   color='#6C5CE7', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, recall, width, label='Recall',
                   color='#00B894', edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Styling
    ax.set_xlabel('Machine Learning Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
    ax.set_title('Precision vs Recall Trade-off Analysis', fontweight='bold', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylim(0, 100)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
    
    plt.tight_layout()
    
    print(f"✓ Precision-Recall comparison chart generated")
    print(f"  Best Precision: {models[precision.argmax()]} ({precision.max():.2f}%)")
    print(f"  Best Recall: {models[recall.argmax()]} ({recall.max():.2f}%)")
    
    plt.show()

if __name__ == '__main__':
    create_precision_recall_chart()
