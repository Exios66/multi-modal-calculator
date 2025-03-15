#!/usr/bin/env python
"""
Test script for the cognitive workload pipeline.
This script demonstrates how to use the functions from generate_report.py.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from generate_report import cognitive_workload_pipeline, generate_clinical_report, create_timestamped_output_dir

# Create a sample dataset for testing
def create_sample_data(n_samples=500):
    """Create a sample dataset with physiological data."""
    # Start time
    start_time = datetime.now() - timedelta(hours=2)
    
    # Create timestamps at 10-second intervals
    timestamps = [start_time + timedelta(seconds=10*i) for i in range(n_samples)]
    
    # Generate random SpO2 values (normally around 97-99%)
    spo2_base = np.random.normal(98, 1, n_samples)
    
    # Add some dips in SpO2 to simulate hypoxemia events
    hypoxemia_1_start = 55
    hypoxemia_1_end = 63
    hypoxemia_2_start = 73
    hypoxemia_2_end = 76
    
    # Create dips in SpO2
    for i in range(hypoxemia_1_start, hypoxemia_1_end):
        if i < n_samples:
            spo2_base[i] = max(90, spo2_base[i] - 5)
    
    for i in range(hypoxemia_2_start, hypoxemia_2_end):
        if i < n_samples:
            spo2_base[i] = max(92, spo2_base[i] - 3)
    
    # Generate random pulse rate values (normally around 70-80 bpm)
    pulse_base = np.random.normal(75, 5, n_samples)
    
    # Add some elevated pulse rate periods
    stress_start = 82
    stress_end = 113
    
    # Create elevated pulse rate
    for i in range(stress_start, stress_end):
        if i < n_samples:
            pulse_base[i] = min(100, pulse_base[i] + 15)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'SpO2': np.round(spo2_base, 1),
        'pulse_rate': np.round(pulse_base, 1)
    })
    
    # Add some random outliers
    outlier_indices = np.random.choice(n_samples, 5, replace=False)
    for idx in outlier_indices:
        if np.random.random() > 0.5:
            df.loc[idx, 'SpO2'] = max(70, min(100, df.loc[idx, 'SpO2'] + np.random.choice([-15, 15])))
        else:
            df.loc[idx, 'pulse_rate'] = max(40, min(140, df.loc[idx, 'pulse_rate'] + np.random.choice([-30, 30])))
    
    return df

# Create sample data
print("Creating sample physiological data...")
sample_data = create_sample_data(n_samples=500)

# Create a timestamped output directory
output_dir = create_timestamped_output_dir()

# Save sample data to CSV
sample_data_path = os.path.join(output_dir, 'sample_physiological_data.csv')
sample_data.to_csv(sample_data_path, index=False)
print(f"Sample data saved to {sample_data_path}")

# Run the cognitive workload pipeline
print("\nRunning cognitive workload pipeline...")
processed_df, report = cognitive_workload_pipeline(sample_data_path, output_dir=output_dir)

# Generate a clinical report
report_path = os.path.join(output_dir, 'clinical_report.md')
print(f"\nGenerating clinical report to {report_path}...")
generate_clinical_report(report, processed_df, report_path)

print("\nDone! Check the output directory for results.")
print(f"Output directory: {output_dir}") 