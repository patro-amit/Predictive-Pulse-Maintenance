#!/usr/bin/env python3
"""
Flow State Diagram
Shows the workflow of the predictive maintenance system
Different style and colors from the reference
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# IEEE Publication Settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

def create_flow_state_diagram():
    """Create flow state diagram showing system workflow with sub-flows"""
    
    # Create larger figure to accommodate sub-flows
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Set proper window title
    try:
        fig.canvas.manager.set_window_title('Graph 13: Flow State Diagram')  # type: ignore
    except:
        pass
    
    # Define main stages with sub-flows
    stages = [
        'Sensor\nData',
        'Feature\nEngineering',
        'ML Model\nTraining',
        'Failure\nPrediction',
        'Maintenance\nScheduling'
    ]
    
    # Sub-flow details for each stage
    sub_flows = [
        ['Temperature', 'Vibration', 'Pressure'],
        ['Extraction', 'Selection', 'Scaling'],
        ['RandomForest', 'XGBoost', 'LightGBM'],
        ['Classify', 'Threshold', 'Alert'],
        ['Priority', 'Resource', 'Execute']
    ]
    
    # Distinct, vibrant, visible colors
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Stage positions - MUCH LARGER
    n_stages = len(stages)
    stage_width = 2.5
    stage_height = 1.8
    spacing = 1.0
    
    start_x = 1.0
    y_center = 5.0
    
    # Draw main boxes, sub-flows, and arrows
    for i, (stage, color, sub_flow_list) in enumerate(zip(stages, colors, sub_flows)):
        x_pos = start_x + i * (stage_width + spacing)
        
        # Create main stage box
        box = FancyBboxPatch(
            (x_pos, y_center - stage_height/2),
            stage_width, stage_height,
            boxstyle="round,pad=0.15",
            edgecolor='#2c3e50',
            facecolor=color,
            linewidth=3,
            alpha=0.85
        )
        ax.add_patch(box)
        
        # Add main stage text - MUCH LARGER
        ax.text(x_pos + stage_width/2, y_center,
                stage,
                ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='white')
        
        # Add sub-flow items below main box - PROPERLY SPACED
        sub_flow_y = y_center - stage_height/2 - 1.5
        sub_box_width = 0.7  # Fixed width for each sub-box
        sub_box_height = 0.6
        total_sub_width = len(sub_flow_list) * sub_box_width + (len(sub_flow_list) - 1) * 0.1
        sub_start_x = x_pos + (stage_width - total_sub_width) / 2  # Center the sub-boxes
        
        for j, sub_item in enumerate(sub_flow_list):
            sub_x = sub_start_x + j * (sub_box_width + 0.1)
            
            # Clean sub-flow box with proper spacing
            sub_box = FancyBboxPatch(
                (sub_x, sub_flow_y - sub_box_height/2),
                sub_box_width, sub_box_height,
                boxstyle="round,pad=0.08",
                edgecolor=color,
                facecolor='white',
                linewidth=2.5,
                alpha=0.95
            )
            ax.add_patch(sub_box)
            
            # Clear sub-flow text - properly centered
            ax.text(sub_x + sub_box_width/2, sub_flow_y,
                    sub_item,
                    ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')
            
            # Thicker connecting line from main box to sub-flow
            ax.plot([x_pos + stage_width/2, sub_x + sub_box_width/2],
                   [y_center - stage_height/2, sub_flow_y + sub_box_height/2],
                   color=color, linewidth=2, alpha=0.5, linestyle=':')
        
        # Add arrow to next stage (except for last stage)
        if i < n_stages - 1:
            arrow_start_x = x_pos + stage_width
            arrow_end_x = x_pos + stage_width + spacing
            
            arrow = FancyArrowPatch(
                (arrow_start_x, y_center),
                (arrow_end_x, y_center),
                arrowstyle='->,head_width=0.5,head_length=0.5',
                color='#34495e',
                linewidth=4,
                alpha=0.9
            )
            ax.add_patch(arrow)
    
    # Title - LARGER
    ax.text(start_x + (n_stages * (stage_width + spacing) - spacing) / 2, 
            y_center + 2.0,
            'Flow State Diagram',
            ha='center', fontsize=20, fontweight='bold',
            color='#2c3e50')
    
    # Set limits and remove axes - adjusted for larger elements and sub-flows
    ax.set_xlim(0, start_x + n_stages * (stage_width + spacing) + 1.0)
    ax.set_ylim(0.5, 7.5)
    ax.axis('off')
    
    # Clean white background
    ax.set_facecolor('white')
    
    # IEEE caption at bottom
    fig.text(0.5, 0.05, 'Fig. 13. Flow State Diagram of Predictive Maintenance System', 
             ha='center', fontsize=12, style='italic', weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save high-quality version
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'fig13_flow_state_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    plt.savefig(output_dir / 'fig13_flow_state_diagram.pdf', bbox_inches='tight', 
                facecolor='white')
    
    plt.show()
    
    print("\nFlow State Stages:")
    for i, stage in enumerate(stages, 1):
        print(f"  {i}. {stage.replace(chr(10), ' ')}")

if __name__ == '__main__':
    print("=" * 50)
    print("Flow State Diagram")
    print("=" * 50)
    create_flow_state_diagram()
