# Cognitive Workload Analysis Pipeline

This directory contains scripts for analyzing physiological data to assess cognitive workload.

## Files

- `generate_report.py`: Contains the main functions for the cognitive workload analysis pipeline.
- `test_report.py`: A test script that demonstrates how to use the functions in `generate_report.py`.

## Requirements

The following Python packages are required:

```
numpy
pandas
scipy
matplotlib
seaborn
feature-engine
statsmodels
scikit-learn
```

You can install these packages using pip:

```bash
pip install numpy pandas scipy matplotlib seaborn feature-engine statsmodels scikit-learn
```

## Usage

### Using the test script

The easiest way to see how the pipeline works is to run the test script:

```bash
python test_report.py
```

This will:

1. Generate sample physiological data
2. Create a timestamped directory in the `processed_data` folder (e.g., `processed_data/analysis_2023-03-15_123456`)
3. Save the sample data to the timestamped directory
4. Run the cognitive workload pipeline on this data
5. Generate a clinical report and various plots in the timestamped directory

### Using the pipeline with your own data

To use the pipeline with your own data, you can import the functions from `generate_report.py`:

```python
from generate_report import cognitive_workload_pipeline, generate_clinical_report, create_timestamped_output_dir

# Create a timestamped output directory
output_dir = create_timestamped_output_dir()

# Run the pipeline on your data
processed_df, report = cognitive_workload_pipeline('path/to/your/data.csv', output_dir=output_dir)

# Generate a clinical report
generate_clinical_report(report, processed_df, os.path.join(output_dir, 'clinical_report.md'))
```

Alternatively, you can let the pipeline create the timestamped directory automatically:

```python
from generate_report import cognitive_workload_pipeline, generate_clinical_report

# Run the pipeline on your data (output_dir=None will create a timestamped directory)
processed_df, report = cognitive_workload_pipeline('path/to/your/data.csv', output_dir=None)

# The report will be saved to the automatically created directory
```

Your input data should be a CSV file with at least the following columns:

- `timestamp`: A datetime column
- `SpO2`: Oxygen saturation percentage
- `pulse_rate`: Pulse rate in beats per minute

## Output

The pipeline generates the following outputs in a timestamped directory within the `processed_data` folder:

1. A processed DataFrame with engineered features and cluster labels
2. A report dictionary containing analysis results and metrics
3. Various plots saved to the output directory:
   - Main time-series analysis
   - Feature correlations
   - Workload distribution
   - Physiological clusters
   - Critical period details
4. A clinical report in Markdown format

## Customization

You can customize the pipeline by modifying the parameters in the `cognitive_workload_pipeline` function:

- Adjust the window sizes for rolling statistics
- Change the thresholds for workload levels
- Modify the feature engineering steps
- Adjust the clustering parameters

For more advanced customization, you can modify the functions in `generate_report.py` directly.
