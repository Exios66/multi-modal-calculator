import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_rel, chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

class WorkingMemoryPipeline:
    def __init__(self, digit_span_file=None, nback_file=None):
        """
        Initialize the pipeline with optional file paths.
        If file paths are not provided, sample data will be generated.
        """
        self.digit_span_file = digit_span_file
        self.nback_file = nback_file
        self.digit_df = None
        self.nback_df = None
        self.metrics_digit = None
        self.metrics_nback = None

    def load_reverse_digit_span(self, file_path):
        """
        Load and parse a plain-text file from the reverse digit span task.
        Assumes each trial is separated by a blank line and the final line
        of each trial contains comma-separated values: trial_id, best_score,
        reaction_time, and (optionally) error_details.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        trials = []
        current_trial = []
        for line in lines:
            if line.strip() == "":
                if current_trial:
                    trials.append(current_trial)
                    current_trial = []
            else:
                current_trial.append(line.strip())
        if current_trial:
            trials.append(current_trial)

        data = []
        for trial in trials:
            last_line = trial[-1]
            parts = last_line.split(',')
            # Expecting at least trial_id and best_score; reaction_time and error_details are optional.
            trial_id = parts[0].strip() if len(parts) > 0 else None
            try:
                best_score = float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() != "" else np.nan
            except ValueError:
                best_score = np.nan
            try:
                reaction_time = float(parts[2].strip()) if len(parts) > 2 and parts[2].strip() != "" else np.nan
            except ValueError:
                reaction_time = np.nan
            error_details = parts[3].strip() if len(parts) > 3 else ""
            data.append({
                'trial_id': trial_id,
                'best_score': best_score,
                'reaction_time': reaction_time,
                'error_details': error_details
            })

        df = pd.DataFrame(data)
        return df

    def load_nback_data(self, file_path):
        """
        Load the n-back task data from a CSV file.
        Expected columns: trial, condition, stimulus, response, correct,
        reaction_time, and target (1 if a target is present, 0 otherwise).
        """
        df = pd.read_csv(file_path)
        return df

    def preprocess_digit_span(self, df):
        """
        Clean and convert columns for the reverse digit span DataFrame.
        """
        df['best_score'] = pd.to_numeric(df['best_score'], errors='coerce')
        df['reaction_time'] = pd.to_numeric(df['reaction_time'], errors='coerce')
        df = df.dropna(subset=['best_score'])
        return df

    def preprocess_nback(self, df):
        """
        Clean the n-back DataFrame by ensuring proper data types
        and removing incomplete records.
        """
        df['trial'] = pd.to_numeric(df['trial'], errors='coerce')
        df['reaction_time'] = pd.to_numeric(df['reaction_time'], errors='coerce')
        # Ensure correct is treated as integer (0 or 1)
        df['correct'] = df['correct'].astype(int)
        # Drop rows with missing essential values
        df = df.dropna(subset=['trial', 'reaction_time', 'correct'])
        # Ensure 'response' is a string for later comparisons
        df['response'] = df['response'].astype(str)
        return df

    def compute_metrics_digit_span(self, df):
        """
        Compute key metrics for the reverse digit span task:
          - Overall best score (maximum across trials)
          - Mean and variance of reaction times
          - Frequency counts for error types (if provided)
        """
        best_score = df['best_score'].max()
        rt_mean = df['reaction_time'].mean()
        rt_var = df['reaction_time'].var()
        error_counts = df['error_details'].value_counts()
        metrics = {
            'best_score': best_score,
            'reaction_time_mean': rt_mean,
            'reaction_time_variance': rt_var,
            'error_counts': error_counts.to_dict()
        }
        return metrics

    def compute_metrics_nback(self, df):
        """
        Compute key performance indices for the n-back task:
          - Hits: Correct "yes" responses when target is present.
          - False Alarms: "Yes" responses when no target is present.
          - Hit rate and false alarm rate.
          - d-prime calculation (using adjusted rates for extreme values).
          - Reaction time mean and standard deviation for correct target detections.
          - Load effects: Summary of accuracy and reaction time by condition.
        """
        # Ensure the target column exists (if not, assume a fallback where target equals correct)
        if 'target' not in df.columns:
            df['target'] = df['correct']

        # Count hits: target present and response is 'yes' (case insensitive)
        hits = df[(df['target'] == 1) & (df['response'].str.lower() == 'yes')].shape[0]
        # Count false alarms: target absent but response is 'yes'
        false_alarms = df[(df['target'] == 0) & (df['response'].str.lower() == 'yes')].shape[0]

        total_targets = df[df['target'] == 1].shape[0]
        total_non_targets = df[df['target'] == 0].shape[0]

        hit_rate = hits / total_targets if total_targets > 0 else np.nan
        false_alarm_rate = false_alarms / total_non_targets if total_non_targets > 0 else np.nan

        # Function to adjust extreme rates (0 or 1)
        def adjust_rate(rate, n):
            if rate >= 1:
                return 1 - 0.5 / n
            elif rate <= 0:
                return 0.5 / n
            return rate

        hit_rate_adj = adjust_rate(hit_rate, total_targets) if total_targets > 0 else np.nan
        false_alarm_rate_adj = adjust_rate(false_alarm_rate, total_non_targets) if total_non_targets > 0 else np.nan
        d_prime = norm.ppf(hit_rate_adj) - norm.ppf(false_alarm_rate_adj) if (not np.isnan(hit_rate_adj) and not np.isnan(false_alarm_rate_adj)) else np.nan

        # Reaction time for correctly detected targets
        rt_correct = df[(df['target'] == 1) & (df['correct'] == 1)]['reaction_time']
        rt_mean = rt_correct.mean()
        rt_std = rt_correct.std()

        # Load effects: compare performance by condition (e.g., 1-back vs 2-back)
        load_effects = df.groupby('condition').agg({'correct': 'mean', 'reaction_time': 'mean'}).reset_index()

        metrics = {
            'hits': hits,
            'false_alarms': false_alarms,
            'hit_rate': hit_rate,
            'false_alarm_rate': false_alarm_rate,
            'd_prime': d_prime,
            'rt_mean': rt_mean,
            'rt_std': rt_std,
            'load_effects': load_effects
        }
        return metrics

    def descriptive_statistics(self, df, task='digit_span'):
        """
        Compute descriptive statistics (count, mean, std, min, max, etc.)
        for the provided DataFrame.
        """
        return df.describe()

    def inferential_tests(self, df_nback):
        """
        Perform an inferential test comparing reaction times between
        two n-back conditions (lowest vs. highest load).
        Uses a paired t-test.
        """
        conditions = df_nback['condition'].unique()
        if len(conditions) < 2:
            return None

        # Sort conditions assuming they are named like '1-back', '2-back', etc.
        try:
            cond_sorted = sorted(conditions, key=lambda x: int(x.split('-')[0]))
        except Exception:
            cond_sorted = sorted(conditions)
        cond_low = cond_sorted[0]
        cond_high = cond_sorted[-1]

        rt_low = df_nback[df_nback['condition'] == cond_low]['reaction_time']
        rt_high = df_nback[df_nback['condition'] == cond_high]['reaction_time']

        # Ensure the lengths are the same (paired comparison)
        min_len = min(len(rt_low), len(rt_high))
        rt_low = rt_low.iloc[:min_len]
        rt_high = rt_high.iloc[:min_len]

        t_stat, p_value = ttest_rel(rt_low, rt_high)
        return {'t_stat': t_stat, 'p_value': p_value, 'conditions': (cond_low, cond_high)}

    def regression_analysis(self, df, dependent, independent):
        """
        Build and return a regression model (using OLS) predicting the
        dependent variable from the independent variable.
        For example, predicting best_score from age.
        """
        df = df.dropna(subset=[dependent, independent])
        X = sm.add_constant(df[independent])
        y = df[dependent]
        model = sm.OLS(y, X).fit()
        return model

    def visualize_results(self, digit_df, nback_df, metrics_digit, metrics_nback):
        """
        Generate visualizations including histograms for digit span scores and reaction times,
        box plots for n-back reaction times by condition, and a scatter plot for digit span vs.
        reaction time. Also print summary tables of computed metrics.
        """
        # --- Reverse Digit Span Visualizations ---
        plt.figure()
        plt.hist(digit_df['best_score'].dropna(), bins=10, edgecolor='black')
        plt.title('Histogram of Best Digit Span Scores')
        plt.xlabel('Digit Span Score')
        plt.ylabel('Frequency')
        plt.show()

        plt.figure()
        plt.hist(digit_df['reaction_time'].dropna(), bins=10, edgecolor='black')
        plt.title('Histogram of Reaction Times (Digit Span)')
        plt.xlabel('Reaction Time (ms)')
        plt.ylabel('Frequency')
        plt.show()

        # --- N-Back Visualizations ---
        # Box plot of reaction times by n-back condition
        conditions = nback_df['condition'].unique()
        data_to_plot = [nback_df[nback_df['condition'] == cond]['reaction_time'].dropna() for cond in conditions]
        plt.figure()
        plt.boxplot(data_to_plot, labels=conditions)
        plt.title('Reaction Time by N-Back Condition')
        plt.xlabel('Condition')
        plt.ylabel('Reaction Time (ms)')
        plt.show()

        # Scatter plot: Digit span score vs. reaction time
        plt.figure()
        plt.scatter(digit_df['best_score'], digit_df['reaction_time'])
        plt.title('Digit Span Score vs Reaction Time')
        plt.xlabel('Digit Span Score')
        plt.ylabel('Reaction Time (ms)')
        plt.show()

        # --- Print Summary Tables ---
        print("=== Digit Span Metrics ===")
        for key, value in metrics_digit.items():
            print(f"{key}: {value}")
        print("\n=== N-Back Metrics ===")
        for key, value in metrics_nback.items():
            if key != 'load_effects':
                print(f"{key}: {value}")
            else:
                print("Load Effects by Condition:")
                print(value)

    def run_pipeline(self):
        """
        Execute the entire pipeline:
          1. Data Acquisition: Load digit span and n-back data.
          2. Preprocessing: Clean and structure the data.
          3. Metric Computation: Calculate performance indices.
          4. Statistical Analysis: Descriptive stats, inferential tests, and regression.
          5. Visualization: Generate plots and summary tables.
        """
        # --- Data Acquisition ---
        if self.digit_span_file and os.path.exists(self.digit_span_file):
            self.digit_df = self.load_reverse_digit_span(self.digit_span_file)
        else:
            # Generate sample reverse digit span data
            sample_data = (
                "Trial1,5,600,order_error\n"
                "Trial1,5,610,order_error\n"
                "\n"
                "Trial2,6,580,\n"
                "Trial2,6,590,\n"
                "\n"
                "Trial3,4,700,other_error\n"
                "Trial3,4,710,other_error\n"
            )
            sample_digit_file = 'sample_digit_span.txt'
            with open(sample_digit_file, 'w') as f:
                f.write(sample_data)
            self.digit_df = self.load_reverse_digit_span(sample_digit_file)

        self.digit_df = self.preprocess_digit_span(self.digit_df)

        if self.nback_file and os.path.exists(self.nback_file):
            self.nback_df = self.load_nback_data(self.nback_file)
        else:
            # Generate sample n-back CSV data
            sample_nback = (
                "trial,condition,stimulus,response,correct,reaction_time,target\n"
                "1,1-back,A,yes,1,500,1\n"
                "2,1-back,B,no,0,550,0\n"
                "3,2-back,C,yes,1,600,1\n"
                "4,2-back,D,yes,0,650,0\n"
                "5,1-back,E,no,1,520,0\n"
                "6,2-back,F,yes,1,630,1\n"
                "7,2-back,G,no,0,640,0\n"
                "8,1-back,H,yes,1,510,1\n"
            )
            sample_nback_file = 'sample_nback.csv'
            with open(sample_nback_file, 'w') as f:
                f.write(sample_nback)
            self.nback_df = self.load_nback_data(sample_nback_file)

        self.nback_df = self.preprocess_nback(self.nback_df)

        # --- Metric Computation ---
        self.metrics_digit = self.compute_metrics_digit_span(self.digit_df)
        self.metrics_nback = self.compute_metrics_nback(self.nback_df)

        # --- Descriptive Statistics ---
        print("=== Descriptive Statistics: Reverse Digit Span ===")
        print(self.descriptive_statistics(self.digit_df, task='digit_span'))
        print("\n=== Descriptive Statistics: N-Back Task ===")
        print(self.descriptive_statistics(self.nback_df, task='nback'))

        # --- Inferential Testing ---
        inferential_results = self.inferential_tests(self.nback_df)
        print("\n=== Inferential Test Results (Paired t-test on Reaction Times) ===")
        print(inferential_results)

        # --- Regression Analysis ---
        # Example: If an 'age' column were available in digit_df, you could run:
        if 'age' in self.digit_df.columns:
            model = self.regression_analysis(self.digit_df, dependent='best_score', independent='age')
            print("\n=== Regression Analysis Summary (Digit Span Score ~ Age) ===")
            print(model.summary())
        else:
            print("\nNo 'age' column in digit span data; skipping regression analysis.")

        # --- Visualization and Reporting ---
        self.visualize_results(self.digit_df, self.nback_df, self.metrics_digit, self.metrics_nback)


if __name__ == "__main__":
    pipeline = WorkingMemoryPipeline()
    pipeline.run_pipeline()
