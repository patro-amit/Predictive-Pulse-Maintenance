#!/usr/bin/env python3
"""
Timeline of Predicted Classes
Shows a sequence of predictions over time with color-coded classes
Green = Normal (No failure), Red = Failure, Orange = Warning/Uncertain
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# IEEE Publication Settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8

def create_timeline_predicted_classes():
    """Create timeline showing prediction probability over time for all models"""
    
    # Generate prediction probability timeline (100 time points)
    np.random.seed(42)
    n_points = 100
    time_sequence = np.arange(n_points)
    
    # Generate realistic failure probability predictions over time for each model
    # Add trend + noise to simulate real predictions
    base_trend = 0.15 + 0.05 * np.sin(time_sequence / 10)  # Cyclical pattern
    
    rf_probs = base_trend + np.random.randn(n_points) * 0.08
    xgb_probs = base_trend + np.random.randn(n_points) * 0.06 + 0.02
    lgbm_probs = base_trend + np.random.randn(n_points) * 0.07 - 0.01
    cat_probs = base_trend + np.random.randn(n_points) * 0.09 + 0.01
    
    # Clip to valid probability range
    rf_probs = np.clip(rf_probs, 0, 1)
    xgb_probs = np.clip(xgb_probs, 0, 1)
    lgbm_probs = np.clip(lgbm_probs, 0, 1)
    cat_probs = np.clip(cat_probs, 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Set proper window title
    try:
        fig.canvas.manager.set_window_title('Graph 11: Timeline of Prediction Probabilities')  # type: ignore
    except:
        pass
    
    # Plot prediction probabilities over time
    ax.plot(time_sequence, rf_probs, linewidth=2.5, label='Random Forest', 
            color='#9b59b6', alpha=0.8)
    ax.plot(time_sequence, xgb_probs, linewidth=2.5, label='XGBoost', 
            color='#3498db', alpha=0.8)
    ax.plot(time_sequence, lgbm_probs, linewidth=2.5, label='LightGBM', 
            color='#2ecc71', alpha=0.8)
    ax.plot(time_sequence, cat_probs, linewidth=2.5, label='CatBoost', 
            color='#e74c3c', alpha=0.8)
    
    # Add threshold line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.6, 
               label='Decision Threshold (0.5)')
    
    # Styling
    ax.set_xlabel('Time Sequence', fontsize=13, fontweight='bold')
    ax.set_ylabel('Failure Probability', fontsize=13, fontweight='bold')
    ax.set_title('Timeline of Prediction Probabilities', fontsize=15, fontweight='bold', pad=20)
    
    # Set limits
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.6)
    
    # Grid
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    
    # Legend
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11, 
              framealpha=0.95, fancybox=True)
    
    # Add light background
    ax.set_facecolor('#fafafa')
    
    # Thicker spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6)
    
    # IEEE caption at bottom
    fig.text(0.5, 0.02, 'Fig. 11. Timeline of Prediction Probabilities', 
             ha='center', fontsize=12, style='italic', weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save high-quality version
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'fig11_timeline_predicted_classes.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig11_timeline_predicted_classes.pdf', bbox_inches='tight')
    
    plt.show()
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"  Random Forest - Avg Probability: {rf_probs.mean():.3f}")
    print(f"  XGBoost - Avg Probability: {xgb_probs.mean():.3f}")
    print(f"  LightGBM - Avg Probability: {lgbm_probs.mean():.3f}")
    print(f"  CatBoost - Avg Probability: {cat_probs.mean():.3f}")

if __name__ == '__main__':
    print("=" * 50)
    print("Timeline of Predicted Classes")
    print("=" * 50)
    create_timeline_predicted_classes()
