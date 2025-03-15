# Pilot Data Documentation

## Overview
This directory contains synthetic clinical datasets and experimental data used for developing and testing the Multi-Modal Calculator's analytical capabilities.

## Data Structure
```
pilot-data/
│
├── synthetic/               # Algorithmically generated test cases
│   ├── vital_signs/         # Simulated patient monitoring data
│   └── clinical_events/     # Synthetic medical event sequences
│
└── experimental/            # Real-world test recordings
    ├── sensor_capture/      # Raw device input samples
    └── calculation_logs/    # Historical operation records
```

## Generation Parameters
Synthetic datasets are created using `ClinicalDataGenerator` with configurable:
- Temporal parameters: `base_time`, `duration_hours`, `resolution_min`
- Physiological ranges:
  ```json
  {
    "SpO2": [88, 100],
    "pulse_rate": [60, 140],
    "blood_pressure_sys": [90, 180],
    "resp_rate": [12, 30],
    "temperature": [36.0, 38.5]
  }
  ```
- Event definitions for clinical scenarios:
  - Hypoxemia (SpO2 < 90%)
  - Tachycardia (pulse_rate > 110bpm)
  - Fever (temperature > 37.8°C)

## Usage Example
```python
from src.utility.synthetic import ClinicalDataGenerator

# Initialize generator with custom configuration
generator = ClinicalDataGenerator({
    'duration_hours': 6,
    'resolution_min': 5,
    'signal_ranges': {
        'SpO2': (85, 99),


