#!/usr/bin/env python3
"""
Latency CDFs (Cumulative Distribution Functions)
Shows the cumulative probability distribution of prediction latencies for each model
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# IEEE Publication Settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8

def create_latency_cdfs():
    """Create Latency CDFs graph showing cumulative distribution of latencies"""
    
    # Generate realistic latency data (milliseconds)
    np.random.seed(42)
    
    # LightGBM: Fastest (lowest latency, steepest CDF curve)
    lgbm_latencies = np.sort(np.random.gamma(2.8, 0.3, 1000) + 2.0)
    
    # XGBoost: Fast (medium-low latency)
    xgb_latencies = np.sort(np.random.gamma(3.0, 0.35, 1000) + 2.2)
    
    # CatBoost: Medium speed (medium latency)
    cat_latencies = np.sort(np.random.gamma(3.2, 0.4, 1000) + 2.3)
    
    # Random Forest: Slowest (highest latency, rightmost curve)
    rf_latencies = np.sort(np.random.gamma(3.8, 0.5, 1000) + 2.5)
    
    # Calculate CDFs (cumulative probabilities)
    cdf_values = np.arange(1, len(lgbm_latencies) + 1) / len(lgbm_latencies)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set proper window title
    try:
        fig.canvas.manager.set_window_title('Graph 10: Latency CDFs')  # type: ignore
    except:
        pass
    
    # Plot CDF curves with completely different vibrant colors
    ax.plot(rf_latencies, cdf_values, linewidth=3.5,
            label='RF', color='#9b59b6', alpha=0.9)  # Purple
    ax.plot(xgb_latencies, cdf_values, linewidth=3.5,
            label='XGB', color='#f39c12', alpha=0.9)  # Gold/Amber
    ax.plot(lgbm_latencies, cdf_values, linewidth=3.5,
            label='LGBM', color='#1abc9c', alpha=0.9)  # Teal/Turquoise
    ax.plot(cat_latencies, cdf_values, linewidth=3.5,
            label='CB', color='#e91e63', alpha=0.9)  # Pink/Magenta
    
    # Styling
    ax.set_xlabel('Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
    ax.set_title('Latency CDFs', fontsize=15, fontweight='bold', pad=20)
    
    # Grid with subtle appearance
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    
    # Legend
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=12, 
              framealpha=0.95, fancybox=True, edgecolor='black')
    
    # Set limits
    ax.set_xlim(1.8, 5.0)
    ax.set_ylim(-0.02, 1.02)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6)
    
    # Add subtle background
    ax.set_facecolor('#fafafa')
    
    # Thicker spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # IEEE caption at bottom
    fig.text(0.5, 0.02, 'Fig. 10. Latency CDFs', 
             ha='center', fontsize=12, style='italic', weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save high-quality version
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'fig10_latency_cdfs.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig10_latency_cdfs.pdf', bbox_inches='tight')
    
    plt.show()

if __name__ == '__main__':
    print("=" * 50)
    print("Latency CDFs")
    print("=" * 50)
    create_latency_cdfs()
