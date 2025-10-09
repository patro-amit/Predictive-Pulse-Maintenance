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
    Creates dual-view analysis of dataset characteristics
    
    Graph Type: 2 subplots (1x2 layout) - THIS IS INTENTIONAL
    Left Plot: Pie chart showing dataset split ratio
    Right Plot: Bar chart showing positive rate comparison
    Note: Two complementary views provide complete dataset understanding
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    data_path = Path(__file__).parent.parent / 'backend' / 'models' / 'model_comparison.csv'
    df = pd.read_csv(data_path)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE CREATION (2 Subplots)
    # ═══════════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set proper window title (fixes "Figure 1" issue)
    fig.canvas.manager.set_window_title('GRAPH 4: Train vs Test Dataset Analysis')  # type: ignore
    
    # Left: Dataset sizes
    models = df['model_name'].str.replace('_', ' ').str.title()
    n_train = df['n_train'].iloc[0]  # Same for all models
    n_test = df['n_test'].iloc[0]
    
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
    
    ax1.set_title('Dataset Distribution\nTraining vs Testing Split',
                 fontsize=12, fontweight='bold', pad=10)
    
    # Right: Positive rate comparison
    train_pos = df['positive_rate_train'] * 100
    test_pos = df['positive_rate_test'] * 100
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, train_pos, width, label='Train Positive Rate',
                   color='#27AE60', edgecolor='black', linewidth=1.2, alpha=0.8)
    bars2 = ax2.bar(x + width/2, test_pos, width, label='Test Positive Rate',
                   color='#E67E22', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Models', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Positive Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Class Balance: Maintenance Required Rate',
                 fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 25)
    
    # Add horizontal line for balance reference
    ax2.axhline(y=20, color='red', linestyle='--', linewidth=1, alpha=0.5, label='20% Reference')
    
    plt.suptitle('Training and Testing Dataset Characteristics',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("=" * 60)
    print("GRAPH 4: Training vs Testing Analysis")
    print("=" * 60)
    create_train_test_analysis()
