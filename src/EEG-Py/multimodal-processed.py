# Full Integrated Data Pipeline for EEG, Heart Rate, and Gaze Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans

# Function to load and preprocess EEG data
def load_eeg_data(alpha_path, theta_path):
    """Loads, cleans, and processes Alpha and Theta EEG data."""
    alpha_df = pd.read_csv(alpha_path)
    theta_df = pd.read_csv(theta_path)
    
    # Select only numerical data
    alpha_numeric = alpha_df.select_dtypes(include=[np.number])
    theta_numeric = theta_df.select_dtypes(include=[np.number])

    # Convert to NumPy format and add a channel axis
    alpha_data = alpha_numeric.to_numpy()[:, np.newaxis, :]
    theta_data = theta_numeric.to_numpy()[:, np.newaxis, :]

    return alpha_data, theta_data

# Function to segment EEG data using rolling windows
def segment_eeg_data(data, window_size=1000, stride=500):
    """Applies rolling window segmentation to EEG data."""
    num_epochs, num_channels, num_samples = data.shape
    windows = []

    for start in range(0, num_epochs - window_size + 1, stride):
        window = data[start:start + window_size, :, :]
        windows.append(window)

    return np.array(windows)

# Function to extract EEG features (Mean, Variance, Skewness)
def extract_eeg_features(data):
    """Computes summary statistics for EEG data."""
    features = []
    for i in range(data.shape[0]):
        segment = data[i]
        features.append([
            np.mean(segment),  
            np.var(segment),   
            skew(segment.flatten())  
        ])
    
    return pd.DataFrame(features, columns=["Mean", "Variance", "Skewness"])

# Function to cluster EEG data
def cluster_eeg_data(df, num_clusters=4):
    """Applies K-Means clustering to EEG summary statistics."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(df)
    df["Cluster"] = kmeans.labels_
    return df

# Function to load and preprocess heart rate data
def load_heart_rate_data(hr_path):
    """Loads and processes heart rate data."""
    hr_df = pd.read_csv(hr_path)
    return hr_df

# Function to segment heart rate data using rolling windows
def segment_heart_rate_data(data, window_size=100, stride=50):
    """Applies rolling window segmentation to heart rate data."""
    windows = [data[start:start + window_size] for start in range(0, len(data) - window_size + 1, stride)]
    return np.array(windows)

# Function to extract heart rate features (Mean, Variance, Skewness)
def extract_heart_rate_features(data):
    """Computes statistical features for heart rate data."""
    return pd.DataFrame({
        "Mean": np.mean(data, axis=1),
        "Variance": np.var(data, axis=1),
        "Skewness": skew(data, axis=1)
    })

# Function to load and preprocess gaze tracking data
def load_gaze_data(gaze_path):
    """Loads and processes gaze tracking data."""
    gaze_df = pd.read_csv(gaze_path)
    # Compute rolling means for gaze-related features
    return gaze_df[["left_pupil_size", "right_pupil_size", "gaze_x", "gaze_y"]].rolling(100).mean().dropna()

# Function to integrate all features into a dataset
def integrate_features(eeg_features, heart_rate_features, gaze_features):
    """Combines EEG, heart rate, and gaze features."""
    min_length = min(len(eeg_features), len(heart_rate_features), len(gaze_features))
    return pd.concat([
        eeg_features.iloc[:min_length].reset_index(drop=True),
        heart_rate_features.iloc[:min_length].reset_index(drop=True),
        gaze_features.iloc[:min_length].reset_index(drop=True)
    ], axis=1)

# Function to train a nonlinear model to predict workload intensity
def train_nonlinear_model(data, target):
    """Trains a Random Forest model to predict EEG workload clusters."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)

# Function to execute the full pipeline
def run_pipeline(alpha_path, theta_path, hr_path, gaze_path):
    """Executes the full EEG, Heart Rate, and Gaze Analysis Pipeline."""
    
    # Load data
    alpha_data, theta_data = load_eeg_data(alpha_path, theta_path)
    heart_rate_df = load_heart_rate_data(hr_path)
    gaze_features = load_gaze_data(gaze_path)

    # Segment EEG data
    alpha_windows = segment_eeg_data(alpha_data)
    theta_windows = segment_eeg_data(theta_data)

    # Extract EEG features
    alpha_features = extract_eeg_features(alpha_windows)
    theta_features = extract_eeg_features(theta_windows)

    # Cluster EEG data
    alpha_clustered = cluster_eeg_data(alpha_features)
    theta_clustered = cluster_eeg_data(theta_features)

    # Segment heart rate data
    pulse_windows = segment_heart_rate_data(heart_rate_df["pulse_rate"].values)
    spo2_windows = segment_heart_rate_data(heart_rate_df["SpO2"].values)

    # Extract heart rate features
    pulse_features = extract_heart_rate_features(pulse_windows)
    spo2_features = extract_heart_rate_features(spo2_windows)

    # Integrate all features
    combined_data = integrate_features(
        pd.concat([alpha_clustered, theta_clustered], axis=1),
        pd.concat([spo2_features, pulse_features], axis=1),
        gaze_features
    )

    # Train nonlinear model
    mae, r2 = train_nonlinear_model(combined_data, alpha_clustered["Cluster"])

    # Print performance
    print(f"Nonlinear Model Performance: MAE={mae:.3f}, R²={r2:.3f}")
    
    return combined_data

# Run the full pipeline with placeholder file paths (replace with actual paths)
full_results = run_pipeline(
    alpha_path="/mnt/data/Alpha_Real_Movements.csv",
    theta_path="/mnt/data/Theta_Real_Movements.csv",
    hr_path="/mnt/data/Dataset_201.csv",
    gaze_path="/mnt/data/Generated_Gaze_Tracking_Data.csv"
)

# Display final integrated dataset
import ace_tools as tools
tools.display_dataframe_to_user(name="Final Integrated Multimodal Dataset", dataframe=full_results)