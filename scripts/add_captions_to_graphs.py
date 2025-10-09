#!/usr/bin/env python3
"""
Add IEEE-style "Fig. X." captions to all existing graphs
This script modifies graph files to add publication-style captions
"""

import re
from pathlib import Path

# Graph files and their figure numbers/captions
GRAPHS_TO_UPDATE = [
    {
        'file': 'graph_roc_random_forest.py',
        'fig_num': 1,
        'caption': 'ROC Graph RandomForest',
        'search': "    plt.tight_layout()\n    plt.show()",
        'replace': """    plt.tight_layout()
    
    # Add IEEE-style figure caption
    fig.text(0.5, 0.02, 'Fig. 1.  ROC Graph RandomForest',
             ha='center', va='bottom', fontsize=11, 
             style='italic', weight='bold')
    
    plt.subplots_adjust(bottom=0.08)  # Make room for caption
    plt.show()"""
    },
    {
        'file': 'graph_roc_xgboost.py',
        'fig_num': 3,
        'caption': 'ROC Graph XGBoost',
        'search': "    # Set window title\n    fig.canvas.manager.set_window_title('GRAPH 11: XGBoost ROC Curve')  # type: ignore",
        'replace': """    # Set window title
    fig.canvas.manager.set_window_title('GRAPH 11: XGBoost ROC Curve')  # type: ignore
    
    # Add IEEE-style figure caption
    fig.text(0.5, 0.02, 'Fig. 3.  ROC Graph XGBoost',
             ha='center', va='bottom', fontsize=11,
             style='italic', weight='bold')
    
    plt.subplots_adjust(bottom=0.10)  # Make room for caption"""
    },
    {
        'file': 'graph_model_accuracy.py',
        'fig_num': 5,
        'caption': 'Model Performance Comparison',
        'search_pattern': r'plt\.tight_layout\(\)\s*plt\.show\(\)',
        'add_before_show': """
    # Add IEEE-style figure caption
    fig.text(0.5, 0.02, 'Fig. 5.  Model Performance Comparison',
             ha='center', va='bottom', fontsize=11,
             style='italic', weight='bold')
    
    plt.subplots_adjust(bottom=0.12)  # Make room for caption
"""
    },
]

def add_caption_to_file(file_path, fig_num, caption):
    """Add IEEE-style caption to a graph file"""
    
    print(f"\n{'='*70}")
    print(f"Processing: {file_path.name}")
    print(f"Adding: Fig. {fig_num}.  {caption}")
    print(f"{'='*70}")
    
    # Read the file
    content = file_path.read_text()
    
    # Check if caption already exists
    if f'Fig. {fig_num}.' in content:
        print(f"  ⚠️  Caption already exists, skipping...")
        return False
    
    # Add caption before plt.show()
    # Look for plt.tight_layout() followed by plt.show()
    pattern = r'(\s+)(plt\.tight_layout\(\))\s*(plt\.show\(\))'
    
    caption_code = f'''\\1\\2
\\1
\\1# Add IEEE-style figure caption
\\1fig.text(0.5, 0.02, 'Fig. {fig_num}.  {caption}',
\\1         ha='center', va='bottom', fontsize=11,
\\1         style='italic', weight='bold')
\\1
\\1plt.subplots_adjust(bottom=0.10)  # Make room for caption
\\1\\3'''
    
    new_content = re.sub(pattern, caption_code, content)
    
    if new_content != content:
        file_path.write_text(new_content)
        print(f"  ✅ Caption added successfully!")
        return True
    else:
        print(f"  ⚠️  Could not find insertion point (plt.tight_layout + plt.show)")
        return False


def main():
    """Add captions to all graph files"""
    
    print("="*70)
    print("ADDING IEEE-STYLE CAPTIONS TO GRAPHS")
    print("="*70)
    
    scripts_dir = Path(__file__).parent
    
    # Define all graphs with their captions
    graphs = [
        ('graph_roc_random_forest.py', 1, 'ROC Graph RandomForest'),
        ('graph_feature_importance.py', 2, 'Column Data RandomForest'),
        ('graph_roc_xgboost.py', 3, 'ROC Graph XGBoost'),
        ('graph_model_accuracy.py', 5, 'Model Performance Comparison'),
        ('graph_roc_precision.py', 6, 'ROC Curve'),
        ('graph_precision_recall.py', 7, 'Precision-Recall Curve'),
        ('graph_classification_heatmap.py', 8, 'Enhanced Classification Heatmap'),
        ('graph_confusion_matrix.py', 15, 'Enhanced Classification Report Heatmap'),
    ]
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for filename, fig_num, caption in graphs:
        file_path = scripts_dir / filename
        
        if not file_path.exists():
            print(f"\n⚠️  File not found: {filename}")
            fail_count += 1
            continue
        
        result = add_caption_to_file(file_path, fig_num, caption)
        
        if result:
            success_count += 1
        elif result is False:
            skip_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Successfully added captions: {success_count}")
    print(f"⚠️  Already had captions (skipped): {skip_count}")
    print(f"❌ Failed or not found: {fail_count}")
    print("="*70)
    
    if success_count > 0:
        print("\n🎉 Captions added! Re-run your graphs to see the changes:")
        print("\n.venv/bin/python scripts/graph_roc_random_forest.py")
        print(".venv/bin/python scripts/graph_roc_xgboost.py")
        print(".venv/bin/python scripts/graph_model_accuracy.py")
        print("# ... etc")


if __name__ == '__main__':
    main()
