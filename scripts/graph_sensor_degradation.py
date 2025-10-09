"""
═══════════════════════════════════════════════════════════════════════════
GRAPH 7: Sensor Degradation Patterns Over Time
═══════════════════════════════════════════════════════════════════════════
IEEE Conference Style Visualization
Displays: 4 SUBPLOTS (Intentional Multi-Sensor Time Series)
  - 2x2 grid showing temperature, pressure, vibration, RPM over cycles
Purpose: Visualize sensor degradation patterns throughout engine life
Run: .venv/bin/python scripts/graph_sensor_degradation.py
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

def create_degradation_pattern():
    """
    Creates 2x2 grid of time series showing sensor degradation
    
    Graph Type: 4 subplots (2x2 grid) - THIS IS INTENTIONAL
    Purpose: Compare degradation patterns across different sensor types
    Note: Multi-subplot allows simultaneous viewing of all key sensor trends
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    data_path = Path(__file__).parent.parent / 'data' / 'cmapss_train_binary.csv'
    df = pd.read_csv(data_path)
    
    # Select one unit for visualization
    unit_data = df[df['unit'] == 1].sort_values('cycle')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE CREATION (4 Subplots in 2x2 Grid)
    # ═══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Set proper window title (fixes "Figure 1" issue)
    fig.canvas.manager.set_window_title('GRAPH 7: Sensor Degradation Patterns')  # type: ignore
    
    # Define sensors to plot (NASA C-MAPSS sensor readings)
    sensors = [
        ('s2', 'Sensor 2 - LPC Outlet Temp', '#E74C3C'),
        ('s3', 'Sensor 3 - HPC Outlet Temp', '#3498DB'),
        ('s4', 'Sensor 4 - LPT Outlet Temp', '#2ECC71'),
        ('s7', 'Sensor 7 - Total Pressure', '#F39C12')
    ]
    
    for idx, (sensor, label, color) in enumerate(sensors):
        ax = axes[idx // 2, idx % 2]
        
        # Normalize sensor values for visualization
        normalized = (unit_data[sensor] - unit_data[sensor].mean()) / unit_data[sensor].std()
        
        # Plot
        ax.plot(unit_data['cycle'], normalized, color=color, linewidth=2, label=label)
        ax.fill_between(unit_data['cycle'], normalized, alpha=0.3, color=color)
        
        # Add trend line
        z = np.polyfit(unit_data['cycle'], normalized, 2)
        p = np.poly1d(z)
        ax.plot(unit_data['cycle'], p(unit_data['cycle']), 
               linestyle='--', color='red', linewidth=2, alpha=0.7, label='Trend')
        
        # Mark failure point (if exists)
        failure_cycle = unit_data[unit_data['label'] == 1]['cycle'].min()
        if not pd.isna(failure_cycle):
            ax.axvline(x=failure_cycle, color='red', linestyle=':', 
                      linewidth=2, alpha=0.8, label='Failure Point')
        
        # Customize
        ax.set_xlabel('Operational Cycle', fontsize=11, fontweight='bold')
        ax.set_ylabel('Normalized Value', fontsize=11, fontweight='bold')
        ax.set_title(f'{label} Degradation Pattern', fontsize=12, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Sensor Degradation Patterns Over Engine Life Cycle\n(Unit #1 - NASA C-MAPSS Dataset)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.99))  # type: ignore
    plt.show()

if __name__ == '__main__':
    print("=" * 60)
    print("GRAPH 7: Sensor Degradation Time Series")
    print("=" * 60)
    print("Loading time series data...")
    create_degradation_pattern()
