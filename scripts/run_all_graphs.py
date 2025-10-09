"""
MASTER SCRIPT: Run All Graphs
IEEE Conference Style Visualizations for Predictive Maintenance

This script provides easy commands to run each graph individually.

REQUIREMENTS:
  pip install matplotlib seaborn

INDIVIDUAL GRAPH COMMANDS:
  1. Model Accuracy Comparison:       python scripts/graph_model_accuracy.py
  2. ROC & Precision-Recall:          python scripts/graph_roc_precision.py
  3. Confusion Matrix Heatmaps:       python scripts/graph_confusion_matrix.py
  4. Training vs Testing Analysis:    python scripts/graph_train_test_performance.py
  5. Sensor Correlation Heatmap:      python scripts/graph_sensor_correlation.py
  6. Feature Importance Ranking:      python scripts/graph_feature_importance.py
  7. Sensor Degradation Time Series:  python scripts/graph_sensor_degradation.py
  8. Model Comparison Radar Chart:    python scripts/graph_radar_comparison.py

RUN ALL GRAPHS:
  python scripts/run_all_graphs.py
"""

import subprocess
import sys
from pathlib import Path

# Graph scripts
GRAPHS = [
    ("Graph 1: Model Accuracy Comparison", "graph_model_accuracy.py"),
    ("Graph 2: ROC & Precision-Recall", "graph_roc_precision.py"),
    ("Graph 3: Confusion Matrix Heatmaps", "graph_confusion_matrix.py"),
    ("Graph 4: Training vs Testing Analysis", "graph_train_test_performance.py"),
    ("Graph 5: Sensor Correlation Heatmap", "graph_sensor_correlation.py"),
    ("Graph 6: Feature Importance Ranking", "graph_feature_importance.py"),
    ("Graph 7: Sensor Degradation Time Series", "graph_sensor_degradation.py"),
    ("Graph 8: Model Comparison Radar Chart", "graph_radar_comparison.py"),
]

def run_graph(script_name):
    """Run a single graph script"""
    script_path = Path(__file__).parent / script_name
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except Exception as e:
        print(f"Error running {script_name}: {e}")

def show_menu():
    """Display interactive menu"""
    print("\n" + "=" * 70)
    print("PREDICTIVE MAINTENANCE - IEEE CONFERENCE GRAPHS")
    print("=" * 70)
    print("\nAvailable Graphs:")
    print("-" * 70)
    
    for i, (title, _) in enumerate(GRAPHS, 1):
        print(f"  {i}. {title}")
    
    print(f"  9. Run ALL graphs sequentially")
    print(f"  0. Exit")
    print("-" * 70)

def main():
    """Main interactive menu"""
    
    # Check if matplotlib is installed
    try:
        import matplotlib
        print("✓ matplotlib is installed")
    except ImportError:
        print("\n⚠️  matplotlib is not installed!")
        print("Install with: pip install matplotlib seaborn")
        response = input("\nWould you like to install it now? (y/n): ")
        if response.lower() == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
        else:
            return
    
    while True:
        show_menu()
        choice = input("\nEnter your choice (0-9): ").strip()
        
        if choice == '0':
            print("\nExiting. Thank you!")
            break
        elif choice == '9':
            print("\n" + "=" * 70)
            print("RUNNING ALL GRAPHS")
            print("=" * 70)
            for title, script in GRAPHS:
                print(f"\n▶ {title}")
                print("-" * 70)
                run_graph(script)
                input("\nPress Enter to continue to next graph...")
        elif choice.isdigit() and 1 <= int(choice) <= len(GRAPHS):
            idx = int(choice) - 1
            title, script = GRAPHS[idx]
            print(f"\n▶ {title}")
            print("-" * 70)
            run_graph(script)
            input("\nPress Enter to return to menu...")
        else:
            print("\n⚠️  Invalid choice. Please try again.")

if __name__ == '__main__':
    print(__doc__)
    response = input("\nWould you like to see the interactive menu? (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("\nTo run a specific graph, use the commands listed above.")
        print("Example: python scripts/graph_model_accuracy.py")
