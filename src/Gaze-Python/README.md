# Gaze Calculator

A comprehensive eye-tracking data processing and analysis tool for research and applications in visual attention, human-computer interaction, and cognitive science.

## Features

- Load and process raw eye-tracking data
- Detect fixations and saccades using velocity-based or DBSCAN clustering methods
- Calculate gaze metrics and statistics
- Perform Continuous Wavelet Transform (CWT) analysis
- Generate visualizations including:
  - Gaze trajectories
  - Fixation and saccade plots
  - Velocity profiles
  - Heatmaps
  - CWT scalograms
  - Distribution plots

## Data Organization

The processed data is organized in a structured folder hierarchy:

```
processed_data/
├── MM#DD#YYYY/           # Date-formatted folders (e.g., 03#11#2025/)
│   ├── csv/              # CSV data files
│   │   ├── processed_gaze_data_YYYYMMDD_HHMMSS.csv
│   │   ├── fixations_YYYYMMDD_HHMMSS.csv
│   │   ├── gaze_metrics_YYYYMMDD_HHMMSS.csv
│   │   └── ...
│   ├── graphs/           # Visualization files
│   │   ├── gaze_trajectory_YYYYMMDD_HHMMSS.png
│   │   ├── velocity_profile_YYYYMMDD_HHMMSS.png
│   │   ├── gaze_heatmap_YYYYMMDD_HHMMSS.png
│   │   └── ...
│   └── data/             # Other data files
│       ├── wavelet_data_YYYYMMDD_HHMMSS.npz
│       └── ...
└── ...
```

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/gaze-calculator.git
   cd gaze-calculator
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv gaze_env
   source gaze_env/bin/activate  # On Windows: gaze_env\Scripts\activate
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```python
from gaze_data_processor import GazeDataProcessor

# Initialize the processor with the path to your data file
processor = GazeDataProcessor("data/your_gaze_data.csv")

# Run the complete processing pipeline
results = processor.run_pipeline()
```

### Advanced Usage

```python
# Initialize the processor
processor = GazeDataProcessor("data/your_gaze_data.csv")

# Load and preprocess the data
processor.load_data()

# Detect fixations and saccades
processor.detect_fixations_and_saccades(
    velocity_threshold=100,  # pixels/sec
    min_fixation_duration=0.1,  # seconds
    min_saccade_velocity=300  # pixels/sec
)

# Or use DBSCAN clustering for fixation detection
processor.detect_fixations_dbscan(
    eps=30,  # maximum distance between points in a cluster
    min_samples=5,  # minimum points to form a core point
    min_duration=0.1  # minimum fixation duration in seconds
)

# Compute CWT metrics
processor.compute_cwt_metrics()

# Create visualizations
processor.visualize_results()

# Save processed data
processor.save_processed_data()
```

## Data Migration

If you have existing data that needs to be migrated to the new folder structure, you can use the provided migration script:

```
python migrate_data.py
```

This script will:

1. Scan the processed_data directory for existing files
2. Extract dates from filenames
3. Create appropriate date-formatted folders (MM#DD#YYYY)
4. Move files to their corresponding subdirectories based on file type
5. Optionally clean up old files after migration (with confirmation)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyWavelets](https://pywavelets.readthedocs.io/) for wavelet transform functionality
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualization
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data processing
- [scikit-learn](https://scikit-learn.org/) for DBSCAN clustering
