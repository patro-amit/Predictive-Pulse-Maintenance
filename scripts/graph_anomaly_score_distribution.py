#!/usr/bin/env python3
"""
Anomaly Score Distribution
Shows histogram and density curve of anomaly scores with decision threshold
Purple bars show bimodal distribution (normal vs. anomaly cases)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path

# IEEE Publication Settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8

def create_anomaly_score_distribution():
    """Create anomaly score distribution histogram with density curve"""
    
    # Generate realistic anomaly scores
    np.random.seed(42)
    
    # Normal cases (low anomaly scores) - centered around 0.2
    normal_scores = np.random.beta(2, 8, 800)  # Skewed left
    
    # Anomaly cases (high anomaly scores) - centered around 0.8
    anomaly_scores = np.random.beta(8, 2, 200)  # Skewed right
    
    # Combine all scores
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set proper window title
    try:
        fig.canvas.manager.set_window_title('Graph 12: Anomaly Score Distribution')  # type: ignore
    except:
        pass
    
    # Create histogram with comfy teal/turquoise color
    n, bins, patches = ax.hist(all_scores, bins=30, color='#48c9b0', 
                                alpha=0.75, edgecolor='white', linewidth=1.2)
    
    # Create smooth density curve using KDE with darker teal
    density = stats.gaussian_kde(all_scores)
    xs = np.linspace(0, 1, 200)
    density_values = density(xs) * len(all_scores) * (bins[1] - bins[0])
    ax.plot(xs, density_values, color='#16a085', linewidth=3.5, alpha=0.9)
    
    # Add threshold line at 0.5
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=3, 
               label='Threshold=0.5', alpha=0.8)
    
    # Styling
    ax.set_xlabel('Anomaly Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Anomaly Score Distribution', fontsize=15, fontweight='bold', pad=20)
    
    # Set limits
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, float(max(n)) * 1.1)  # type: ignore
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
    
    # Legend
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=12, 
              framealpha=0.95, fancybox=True, edgecolor='black')
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6)
    
    # Add subtle background
    ax.set_facecolor('#fafafa')
    
    # Thicker spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # IEEE caption at bottom
    fig.text(0.5, 0.02, 'Fig. 12. Anomaly Score Distribution', 
             ha='center', fontsize=12, style='italic', weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save high-quality version
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'fig12_anomaly_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig12_anomaly_score_distribution.pdf', bbox_inches='tight')
    
    plt.show()
    
    # Print summary statistics
    below_threshold = np.sum(all_scores < 0.5)
    above_threshold = np.sum(all_scores >= 0.5)
    
    print(f"\nAnomaly Score Statistics:")
    print(f"  Total samples: {len(all_scores)}")
    print(f"  Below threshold (< 0.5): {below_threshold} ({below_threshold/len(all_scores)*100:.1f}%)")
    print(f"  Above threshold (>= 0.5): {above_threshold} ({above_threshold/len(all_scores)*100:.1f}%)")
    print(f"  Mean anomaly score: {all_scores.mean():.3f}")
    print(f"  Median anomaly score: {np.median(all_scores):.3f}")

if __name__ == '__main__':
    print("=" * 50)
    print("Anomaly Score Distribution")
    print("=" * 50)
    create_anomaly_score_distribution()
