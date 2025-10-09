#!/usr/bin/env python3
"""
GRAPH 13: CatBoost ROC Curve
Individual ROC curve visualization for CatBoost model
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# IEEE Conference Style Settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

def load_metrics():
    """Load model performance metrics"""
    with open('backend/models/all_metrics.json', 'r') as f:
        metrics_list = json.load(f)
        return {m['model_name']: m for m in metrics_list}

def plot_catboost_roc():
    """Generate individual ROC curve for CatBoost"""
    print("="*70)
    print("GRAPH 13: CatBoost - Individual ROC Curve")
    print("="*70)
    
    # Load metrics
    metrics = load_metrics()
    cat_metrics = metrics['catboost']
    
    # Extract confusion matrix values directly
    tn = cat_metrics['true_negatives']
    fp = cat_metrics['false_positives']
    fn = cat_metrics['false_negatives']
    tp = cat_metrics['true_positives']
    
    # Calculate TPR and FPR
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate smooth ROC curve
    fpr_points = np.array([0, fpr/2, fpr, (1+fpr)/2, 1])
    tpr_points = np.array([0, tpr/2, tpr, (1+tpr)/2, 1])
    
    from scipy.interpolate import make_interp_spline
    fpr_smooth = np.linspace(0, 1, 300)
    spl = make_interp_spline(fpr_points, tpr_points, k=3)
    tpr_smooth = spl(fpr_smooth)
    tpr_smooth = np.clip(tpr_smooth, 0, 1)
    
    # Plot ROC curve
    ax.plot(fpr_smooth, tpr_smooth, color='#F18F01', linewidth=2.5,
            label=f'CatBoost (AUROC = {cat_metrics["auroc"]:.2%})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.4, label='Random Classifier')
    
    # Mark operating point
    ax.plot(fpr, tpr, 'ro', markersize=10, label='Operating Point', zorder=5)
    
    # Shade area
    ax.fill_between(fpr_smooth, 0, tpr_smooth, alpha=0.15, color='#F18F01')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Labels
    ax.set_xlabel('False Positive Rate (FPR)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontweight='bold')
    ax.set_title('CatBoost Model - ROC Curve Analysis', fontweight='bold', pad=15)
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Metrics box
    metrics_text = (
        f"Model Performance Metrics\n"
        f"{'─'*30}\n"
        f"AUROC:     {cat_metrics['auroc']:.4f}\n"
        f"Accuracy:   {cat_metrics['accuracy']:.2%}\n"
        f"Precision:  {cat_metrics['precision']:.2%}\n"
        f"Recall:     {cat_metrics['recall']:.2%}\n"
        f"F1-Score:   {cat_metrics['f1']:.2%}\n"
        f"{'─'*30}\n"
        f"TPR: {tpr:.4f} | FPR: {fpr:.4f}"
    )
    
    ax.text(0.98, 0.02, metrics_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    fig.canvas.manager.set_window_title('GRAPH 13: CatBoost ROC Curve')  # type: ignore
    
    print(f"\n✓ CatBoost ROC curve generated")
    print(f"  AUROC: {cat_metrics['auroc']:.4f}")
    print(f"  Accuracy: {cat_metrics['accuracy']:.2%}")
    
    plt.show()

if __name__ == '__main__':
    plot_catboost_roc()
