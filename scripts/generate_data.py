"""
Big Data Generator for Predictive Maintenance
Aligned with: "Predictive Maintenance Strategies Using Big Data And Machine Learning"

Generates large-scale synthetic sensor data from industrial equipment with:
- Multiple failure modes
- Realistic degradation patterns
- Big Data characteristics (volume, velocity, variety)
- Complex feature interactions
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Set random seed for reproducibility
rng = np.random.default_rng(42)

def generate_big_data_maintenance(n_units=200, output_dir="../data"):
    """
    Generate large-scale predictive maintenance data for Big Data ML strategies.
    
    Args:
        n_units: Number of industrial machines/units (default: 200 for Big Data)
        output_dir: Directory to save the generated CSV
        
    Features:
    - 200+ units for Big Data volume
    - 24 sensor readings per cycle
    - Multiple failure modes (wear, overheating, vibration)
    - Realistic operating conditions
    - Complex degradation patterns
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    print(f"🏭 Generating Big Data for Predictive Maintenance...")
    print(f"   Units: {n_units}")
    print(f"   Target: Large-scale industrial sensor data\n")
    
    for unit in range(1, n_units + 1):
        # Vary lifecycle: 150-300 cycles (more realistic for industrial equipment)
        cycles = rng.integers(150, 300)
        
        # Unit-specific characteristics (equipment variability)
        base_temp = rng.normal(100, 10)  # Base temperature
        base_pressure = rng.normal(50, 5)  # Base pressure
        base_rpm = rng.normal(3000, 300)  # Base RPM
        
        # Failure mode (affects degradation pattern)
        failure_mode = rng.choice(['wear', 'thermal', 'vibration'], p=[0.4, 0.35, 0.25])
        
        for c in range(1, cycles + 1):
            # Timestamp for Big Data context
            timestamp = datetime.now() + timedelta(hours=c)
            
            # Operating conditions (3 settings with realistic ranges)
            setting1 = rng.normal(0.3, 0.05)      # Operating mode
            setting2 = rng.normal(-0.03, 0.02)    # Environmental factor
            setting3 = rng.normal(0.002, 0.001)   # Load condition
            
            # Degradation factor (non-linear)
            progress = c / cycles
            degrade = progress ** 1.5  # Non-linear degradation
            
            # 21 sensor readings with realistic physics-based degradation
            sensors = []
            
            # Temperature sensors (s1-s4): Increase with degradation
            temp_drift = 5 * degrade if failure_mode == 'thermal' else 2 * degrade
            sensors.extend([
                base_temp + rng.normal(0, 2) + temp_drift,
                base_temp + 5 + rng.normal(0, 1.5) + temp_drift * 1.2,
                base_temp - 3 + rng.normal(0, 1.8) + temp_drift * 0.8,
                base_temp + 2 + rng.normal(0, 2.2) + temp_drift * 1.1
            ])
            
            # Pressure sensors (s5-s8): Decrease with wear
            pressure_drift = -3 * degrade if failure_mode == 'wear' else -1 * degrade
            sensors.extend([
                base_pressure + rng.normal(0, 1) + pressure_drift,
                base_pressure + 2 + rng.normal(0, 0.8) + pressure_drift * 1.1,
                base_pressure - 1 + rng.normal(0, 1.2) + pressure_drift * 0.9,
                base_pressure + 3 + rng.normal(0, 1.5) + pressure_drift * 1.2
            ])
            
            # Vibration sensors (s9-s12): Increase with mechanical issues
            vib_drift = 4 * degrade if failure_mode == 'vibration' else 1.5 * degrade
            sensors.extend([
                rng.normal(0, 0.5) + vib_drift,
                rng.normal(0, 0.4) + vib_drift * 1.3,
                rng.normal(0, 0.6) + vib_drift * 1.1,
                rng.normal(0, 0.45) + vib_drift * 1.4
            ])
            
            # RPM/Flow sensors (s13-s16): Decrease with degradation
            rpm_drift = -50 * degrade
            sensors.extend([
                base_rpm + rng.normal(0, 20) + rpm_drift,
                base_rpm + 100 + rng.normal(0, 15) + rpm_drift * 1.1,
                base_rpm - 50 + rng.normal(0, 25) + rpm_drift * 0.9,
                base_rpm + 200 + rng.normal(0, 30) + rpm_drift * 1.2
            ])
            
            # Additional sensors (s17-s21): Mixed patterns
            sensors.extend([
                rng.normal(50, 5) + 2 * degrade,      # Flow rate
                rng.normal(30, 3) - 1.5 * degrade,    # Efficiency
                rng.normal(10, 1) + 3 * degrade,      # Energy consumption
                rng.normal(0, 0.3) + 0.8 * degrade,   # Acoustic emission
                rng.normal(100, 10) - 5 * degrade     # Output quality
            ])
            
            # Add operational noise (Big Data characteristic - noisy data)
            sensors = [s + rng.normal(0, 0.05 * abs(s)) for s in sensors]
            
            # Feature engineering: Derived features for better ML performance
            temp_avg = np.mean(sensors[0:4])
            pressure_avg = np.mean(sensors[4:8])
            vibration_avg = np.mean(sensors[8:12])
            rpm_avg = np.mean(sensors[12:16])
            
            # Calculate Remaining Useful Life (RUL)
            RUL = cycles - c
            
            # Multi-threshold labeling for better accuracy
            # Early warning: RUL <= 40 (more conservative)
            label = int(RUL <= 40)
            
            # Append row with all features
            rows.append([
                unit, c, timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                setting1, setting2, setting3,
                *sensors,
                temp_avg, pressure_avg, vibration_avg, rpm_avg,  # Engineered features
                failure_mode,
                RUL, label
            ])
        
        if unit % 20 == 0:
            print(f"  ✓ Generated data for {unit}/{n_units} units")
    
    # Create DataFrame with comprehensive column names
    cols = (
        ["unit", "cycle", "timestamp"] +
        [f"setting{i}" for i in range(1, 4)] +
        [f"s{i}" for i in range(1, 22)] +
        ["temp_avg", "pressure_avg", "vibration_avg", "rpm_avg"] +
        ["failure_mode", "RUL", "label"]
    )
    
    df = pd.DataFrame(rows, columns=cols)
    
    # Save to CSV
    out_csv = out_dir / "predictive_maintenance_bigdata.csv"
    df.to_csv(out_csv, index=False)
    
    print(f"\n✅ Big Data Generated Successfully!")
    print(f"   File: {out_csv.resolve()}")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Total Data Points: {df.shape[0] * df.shape[1]:,}")
    print(f"   Size: {os.path.getsize(out_csv) / (1024*1024):.2f} MB")
    print(f"\n📊 Label Distribution:")
    print(f"   Needs Maintenance (1): {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
    print(f"   Normal Operation (0): {(1-df['label']).sum():,} ({(1-df['label']).mean()*100:.1f}%)")
    print(f"\n🔧 Failure Modes:")
    print(df['failure_mode'].value_counts())
    
    return df

if __name__ == "__main__":
    print("="*70)
    print("  PREDICTIVE MAINTENANCE BIG DATA GENERATOR")
    print("  Topic: Predictive Maintenance Strategies Using Big Data And ML")
    print("="*70)
    print()
    
    # Generate large-scale data (200 units = ~40,000-50,000 data points)
    df = generate_big_data_maintenance(n_units=200, output_dir="../data")
    
    print("\n📈 Statistical Summary:")
    print(df[['temp_avg', 'pressure_avg', 'vibration_avg', 'rpm_avg', 'RUL']].describe())
    
    print("\n✨ Data generation complete! Ready for Big Data ML training.")
