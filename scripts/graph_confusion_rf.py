#!/usr/bin/env python3
"""
GRAPH 3A: Random Forest Confusion Matrix (Single Plot)
Detailed confusion matrix for Random Forest model only
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# IEEE Conference Style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 100

def create_rf_confusion_matrix():
    """Single confusion matrix for Random Forest"""
    print("="*70)
    print("GRAPH 3A: Random Forest Confusion Matrix")
    print("="*70)
    
    # Load data
    data_path = Path(__file__).parent.parent / 'backend' / 'models' / 'model_comparison.csv'
    df = pd.read_csv(data_path)
    
    # Get Random Forest data
    rf_data = df[df['model_name'] == 'random_forest'].iloc[0]
    
    # Create confusion matrix
    cm = [[rf_data['true_negatives'], rf_data['false_positives']],
          [rf_data['false_negatives'], rf_data['true_positives']]]
    
    # Calculate percentages
    total = sum(sum(row) for row in cm)
    cm_pct = [[f"{val}\n({val/total*100:.1f}%)" for val in row] for row in cm]
    
    # Create figure - SINGLE PLOT
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.canvas.manager.set_window_title('GRAPH 3A: Random Forest Confusion Matrix')  # type: ignore
    
    # Create heatmap
    sns.heatmap(cm, annot=cm_pct, fmt='', cmap='Blues', cbar=True,
                square=True, linewidths=2, linecolor='black',
                cbar_kws={'label': 'Count'}, ax=ax, vmin=0, vmax=7000)
    
    # Labels
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
    ax.set_ylabel('Actual Label', fontweight='bold', fontsize=12)
    ax.set_title(f'Random Forest\nAccuracy: {rf_data["accuracy"]*100:.2f}%',
                fontweight='bold', fontsize=14, pad=15)
    
    # Set tick labels
    ax.set_xticklabels(['Predicted: No\nMaintenance', 'Predicted:\nMaintenance'], rotation=0)
    ax.set_yticklabels(['Actual: No\nMaintenance', 'Actual:\nMaintenance'], rotation=0)
    
    plt.tight_layout()
    
    print(f"✓ Random Forest confusion matrix generated")
    print(f"  Accuracy: {rf_data['accuracy']*100:.2f}%")
    print(f"  True Positives: {rf_data['true_positives']}")
    print(f"  True Negatives: {rf_data['true_negatives']}")
    
    plt.show()

if __name__ == '__main__':
    create_rf_confusion_matrix()
