#!/usr/bin/env python3
"""
Fig. 4: Column Data XGBoost (Feature Importance)
Clean minimal style matching PDF
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# IEEE Style Configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

def create_xgboost_feature_importance():
    """Feature importance for XGBoost - clean style"""
    
    print("="*70)
    print("Fig. 4: Column Data XGBoost")
    print("="*70)
    
    # Sample feature importance data for XGBoost (top 20)
    # These would typically come from model.feature_importances_
    features = [
        'Dependents_Yes', 'LoanTerm', 'Unemployed', 'NumCreditLines',
        'EmploymentType_Full-time', 'Purpose_Other', 'Purpose_Education',
        'CoSigner_No', 'Purpose_Business', 'Education_Master\'s',
        'Status_Divorced', 'Status_Married', 'Education_High School',
        'Status_Married', 'Education_PhD', 'CoSigner_Yes',
        'Mortgage_Yes', 'Purpose_Home', 'Dependents_No', 'Mortgage_No'
    ]
    
    importance = [0.078, 0.065, 0.061, 0.058, 0.047, 0.041, 0.037, 0.035,
                  0.034, 0.033, 0.032, 0.031, 0.030, 0.029, 0.028, 0.027,
                  0.026, 0.025, 0.023, 0.020]
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = [importance[i] for i in sorted_idx]
    
    # Create figure - CLEAN MINIMAL STYLE (LIKE PDF)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.canvas.manager.set_window_title('Fig. 4: Column Data XGBoost')  # type: ignore
    
    # Create horizontal bars - SIMPLE BLUE COLOR LIKE PDF
    bars = ax.barh(sorted_features, sorted_importance, color='steelblue')
    
    # Customize - MINIMAL LIKE PDF
    ax.set_xlabel('Importance')
    ax.set_title('Top 20 Feature Importances', pad=15)
    ax.set_xlim(0, max(importance) * 1.05)
    ax.grid(False)  # No grid like PDF
    
    plt.tight_layout()
    
    # Add IEEE-style figure caption at bottom (like PDF)
    fig.text(0.5, 0.02, 'Fig. 4.  Column Data XGBoost',
             ha='center', va='bottom', fontsize=11,
             style='italic', weight='bold')
    
    plt.subplots_adjust(bottom=0.10)
    plt.show()


if __name__ == '__main__':
    create_xgboost_feature_importance()
