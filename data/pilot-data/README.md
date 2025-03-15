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
        'temperature': (36.5, 39.0)
    }
})

# Generate dataset with 72 data points (6hr * 12pts/hr)
dataset = generator.generate_base_signals(72)

# Save as CSV with timestamp
filename = f"vital_signs_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
pd.DataFrame(dataset).to_csv(filename, index=False)
```

## Key Features

- 🧪 Realistic cross-signal correlations
- ⚠️ Configurable clinical event simulation
- 🔒 Built-in parameter validation
- 📈 Signal smoothing with Savitzky-Golay filter

## Notes

1. All data is synthetic - **NOT** real patient data
2. Default configuration generates 3 hours of data at 1-minute resolution
3. Event simulations introduce coordinated vital sign changes
4. Custom configurations must pass `validate_config` checks
