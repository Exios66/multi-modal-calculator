# Working Memory Tasks

## Instructions

1. Data Acquisition & Import:

* The load_reverse_digit_span method reads a text file, groups lines into trials (assuming blank lines as separators), and extracts the final line per trial.
* The load_nback_data method reads a CSV file containing trial-level data for the n‑back task.

2. Data Preprocessing:

* The preprocess_digit_span and preprocess_nback methods convert columns to the appropriate data types and drop incomplete records.
  
3. Metric Computation:

* The digit span metrics (best score, reaction time mean/variance, and error counts) are computed in compute_metrics_digit_span.
* The n‑back metrics (hits, false alarms, hit rate, false alarm rate, d‑prime, and reaction time summaries) are computed in compute_metrics_nback. A helper function adjusts extreme values for the d‑prime calculation.

4. Statistical Analysis:

* Descriptive statistics are computed with descriptive_statistics.
* A paired t‑test comparing reaction times between two conditions is run in inferential_tests.
* A sample regression model can be generated via regression_analysis if the appropriate predictor (e.g., age) is present.

5. Visualization and Reporting:

* The visualize_results method produces histograms, box plots, scatter plots, and prints summary tables.

6. Running the Pipeline:

* The run_pipeline method ties everything together. It will use provided file paths if available or generate sample data if not.

This script can be run as-is and will generate both printed outputs and interactive plots. You can further adapt each section to match your specific data formats and research questions.
