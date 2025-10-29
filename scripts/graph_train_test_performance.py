"""
═══════════════════════════════════════════════════════════════════════════
GRAPH 4: Training vs Testing Dataset Analysis
═══════════════════════════════════════════════════════════════════════════
IEEE Conference Style Visualization
Displays: 2 SUBPLOTS (Intentional Complementary Views)
  - Left: Pie chart showing train/test split ratio
  - Right: Bar chart showing positive class rates
Purpose: Dataset characteristics and class balance visualization
Run: .venv/bin/python scripts/graph_train_test_performance.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# IEEE STYLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'figure.dpi': 100
})

def create_train_test_analysis():
    """
    Creates dual-view analysis of confusion matrix results
    
    Graph Type: 2 subplots (1x2 layout) - THIS IS INTENTIONAL
    Left Plot: Pie chart showing overall dataset classification results
    Right Plot: Bar chart showing true positives vs false negatives
    Note: Two complementary views provide complete model performance understanding
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    data_path = Path(__file__).parent.parent / 'backend' / 'models' / 'model_comparison.csv'
    df = pd.read_csv(data_path)
    
    # Parse accuracy if it's a percentage string
    if df['accuracy'].dtype == 'object':
        df['accuracy'] = df['accuracy'].str.rstrip('%').astype(float) / 100
    
    # Calculate dataset info from confusion matrix values
    # Use the first model (catboost) as representative
    tp = df['true_positives'].iloc[0]
    fp = df['false_positives'].iloc[0]
    tn = df['true_negatives'].iloc[0]
    fn = df['false_negatives'].iloc[0]
    
    # Total samples in test set
    n_test = tp + fp + tn + fn
    # Training set is typically 80% (assume 80-20 split)
    n_train = int(n_test * 4)  # 80-20 split approximation
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE CREATION (2 Subplots)
    # ═══════════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set proper window title (fixes "Figure 1" issue)
    fig.canvas.manager.set_window_title('GRAPH 4: Train vs Test Dataset Analysis')  # type: ignore
    
    # Left: Dataset sizes
    sizes = [n_train, n_test]
    labels = ['Training Set', 'Testing Set']
    colors = ['#3498DB', '#E74C3C']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels,
                                        colors=colors, autopct='%1.1f%%',
                                        shadow=True, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    # Add sample counts
    for i, autotext in enumerate(autotexts):
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # Add legend with counts
    ax1.legend([f'{labels[i]}\n({sizes[i]:,} samples)' for i in range(len(labels))],
              loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    
    ax1.set_title('Dataset Distribution\nTraining vs Testing Split (80-20)',
                 fontsize=12, fontweight='bold', pad=10)
    
    # Right: Positive class detection rate for each model
    models = df['model_name'].str.replace('_', ' ').str.title()
    
    # Calculate positive detection rate (recall)
    pos_rate = df['recall'] * 100
    
    x = np.arange(len(models))
    width = 0.7
    
    # Create a single bar chart showing recall rate for each model
    bars = ax2.bar(x, pos_rate, width, label='Recall (True Positive Rate)',
                   color='#3498DB', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Models', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Recall Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Model Recall: True Positive Detection Rate\n(Ability to Catch Failures)',
                 fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 100)
    
    # Add horizontal line for 80% recall reference (good performance)
    ax2.axhline(y=80, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='80% Target')
    
    plt.suptitle('Training and Testing Dataset Characteristics',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("=" * 60)
    print("GRAPH 4: Training vs Testing Analysis")
    print("=" * 60)
    create_train_test_analysis()
