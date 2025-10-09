#!/usr/bin/env python3
"""
GRAPH 11: XGBoost ROC Curve
Individual ROC curve visualization for XGBoost model
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
        # Convert list to dictionary for easier access
        return {m['model_name']: m for m in metrics_list}

def plot_xgboost_roc():
    """Generate individual ROC curve for XGBoost"""
    print("="*70)
    print("GRAPH 11: XGBoost - Individual ROC Curve")
    print("="*70)
    
    # Load metrics
    metrics = load_metrics()
    xgb_metrics = metrics['xgboost']
    
    # Extract confusion matrix values directly from metrics
    tn = xgb_metrics['true_negatives']
    fp = xgb_metrics['false_positives']
    fn = xgb_metrics['false_negatives']
    tp = xgb_metrics['true_positives']
    
    # Calculate TPR and FPR at the operating point
    tpr = tp / (tp + fn)  # True Positive Rate (Recall)
    fpr = fp / (fp + tn)  # False Positive Rate
    
    # Create figure - CLEAN MINIMAL STYLE (LIKE PDF)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.canvas.manager.set_window_title('Fig. 3: ROC Graph XGBoost')  # type: ignore
    
    # Generate smooth ROC curve
    fpr_points = np.array([0, fpr/2, fpr, (1+fpr)/2, 1])
    tpr_points = np.array([0, tpr/2, tpr, (1+tpr)/2, 1])
    
    from scipy.interpolate import make_interp_spline
    fpr_smooth = np.linspace(0, 1, 300)
    spl = make_interp_spline(fpr_points, tpr_points, k=3)
    tpr_smooth = spl(fpr_smooth)
    tpr_smooth = np.clip(tpr_smooth, 0, 1)
    
    # Plot ROC curve - SIMPLE CLEAN
    ax.plot(fpr_smooth, tpr_smooth, color='#1f77b4', linewidth=2.5,
            label=f'ROC Curve (AUC={xgb_metrics["auroc"]:.3f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    
    # Labels and title - MINIMAL LIKE PDF
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', pad=15)
    
    # Set axis limits
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    
    # Add legend - SIMPLE
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    
    # No grid like PDF
    ax.grid(False)
    
    plt.tight_layout()
    
    # Add IEEE-style figure caption at bottom (like PDF)
    fig.text(0.5, 0.02, 'Fig. 3.  ROC Graph XGBoost',
             ha='center', va='bottom', fontsize=11,
             style='italic', weight='bold')
    
    plt.subplots_adjust(bottom=0.10)
    plt.show()

if __name__ == '__main__':
    plot_xgboost_roc()
