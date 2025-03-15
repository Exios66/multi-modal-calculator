#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaze Data Processing Pipeline

This script processes eye-tracking data to identify fixations, saccades, 
and compute metrics useful for Continuous Wavelet Transform (CWT) analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
import pywt
import os
from sklearn.cluster import DBSCAN
from datetime import datetime

class GazeDataProcessor:
    """
    A class for processing eye-tracking gaze data.
    """
    
    def __init__(self, file_path, output_dir='processed_data'):
        """
        Initialize the GazeDataProcessor with the path to the gaze data file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing gaze data
        output_dir : str
            Directory to save processed data and visualizations
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.raw_data = None
        self.processed_data = None
        self.fixations = None
        self.saccades = None
        self.metrics = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_data(self):
        """
        Load gaze data from CSV file and perform initial preprocessing.
        """
        print(f"Loading data from {self.file_path}...")
        self.raw_data = pd.read_csv(self.file_path)
        
        # Basic data validation
        required_columns = ['timestamp', 'x', 'y']
        for col in required_columns:
            if col not in self.raw_data.columns:
                raise ValueError(f"Required column '{col}' not found in the data")
        
        # Convert timestamp to seconds for easier analysis
        self.raw_data['timestamp_sec'] = (self.raw_data['timestamp'] - self.raw_data['timestamp'].iloc[0]) / 1000.0
        
        # Calculate time differences between consecutive samples
        self.raw_data['dt'] = self.raw_data['timestamp_sec'].diff().fillna(0)
        
        # Calculate velocities (pixels per second)
        self.raw_data['dx'] = self.raw_data['x'].diff().fillna(0)
        self.raw_data['dy'] = self.raw_data['y'].diff().fillna(0)
        
        # Calculate Euclidean distance between consecutive points
        self.raw_data['distance'] = np.sqrt(self.raw_data['dx']**2 + self.raw_data['dy']**2)
        
        # Calculate velocity (pixels per second)
        self.raw_data['velocity'] = self.raw_data['distance'] / self.raw_data['dt']
        self.raw_data['velocity'] = self.raw_data['velocity'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate acceleration (pixels per second^2)
        self.raw_data['acceleration'] = self.raw_data['velocity'].diff().fillna(0) / self.raw_data['dt']
        self.raw_data['acceleration'] = self.raw_data['acceleration'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate angular velocity (change in direction)
        self.raw_data['angle'] = np.arctan2(self.raw_data['dy'], self.raw_data['dx'])
        self.raw_data['angle_change'] = self.raw_data['angle'].diff().fillna(0)
        
        # Adjust angle changes to be between -pi and pi
        self.raw_data['angle_change'] = ((self.raw_data['angle_change'] + np.pi) % (2 * np.pi)) - np.pi
        
        # Calculate angular velocity (radians per second)
        self.raw_data['angular_velocity'] = self.raw_data['angle_change'] / self.raw_data['dt']
        self.raw_data['angular_velocity'] = self.raw_data['angular_velocity'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        self.processed_data = self.raw_data.copy()
        print(f"Loaded {len(self.raw_data)} data points.")
        
        return self.processed_data
    
    def detect_fixations_and_saccades(self, velocity_threshold=100, min_fixation_duration=0.1, min_saccade_velocity=300):
        """
        Detect fixations and saccades based on velocity thresholds.
        
        Parameters:
        -----------
        velocity_threshold : float
            Velocity threshold for distinguishing between fixations and saccades (pixels/sec)
        min_fixation_duration : float
            Minimum duration for a fixation to be considered valid (seconds)
        min_saccade_velocity : float
            Minimum velocity for a movement to be considered a saccade (pixels/sec)
        """
        if self.processed_data is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        print("Detecting fixations and saccades...")
        
        # Identify potential fixations (low velocity) and saccades (high velocity)
        self.processed_data['is_fixation'] = self.processed_data['velocity'] < velocity_threshold
        
        # Group consecutive fixation points
        self.processed_data['fixation_group'] = (self.processed_data['is_fixation'].diff() != 0).cumsum()
        
        # Extract fixation groups
        fixation_groups = []
        for group_id, group_data in self.processed_data[self.processed_data['is_fixation']].groupby('fixation_group'):
            duration = group_data['timestamp_sec'].iloc[-1] - group_data['timestamp_sec'].iloc[0]
            if duration >= min_fixation_duration:
                fixation_groups.append({
                    'start_time': group_data['timestamp_sec'].iloc[0],
                    'end_time': group_data['timestamp_sec'].iloc[-1],
                    'duration': duration,
                    'mean_x': group_data['x'].mean(),
                    'mean_y': group_data['y'].mean(),
                    'std_x': group_data['x'].std(),
                    'std_y': group_data['y'].std(),
                    'points': len(group_data)
                })
        
        self.fixations = pd.DataFrame(fixation_groups)
        
        # Extract saccades (movements between fixations)
        saccade_groups = []
        if len(fixation_groups) > 1:
            for i in range(len(fixation_groups) - 1):
                start_time = fixation_groups[i]['end_time']
                end_time = fixation_groups[i + 1]['start_time']
                
                # Get data points during this saccade
                saccade_data = self.processed_data[
                    (self.processed_data['timestamp_sec'] >= start_time) & 
                    (self.processed_data['timestamp_sec'] <= end_time)
                ]
                
                if len(saccade_data) > 0 and saccade_data['velocity'].max() >= min_saccade_velocity:
                    saccade_groups.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time,
                        'start_x': fixation_groups[i]['mean_x'],
                        'start_y': fixation_groups[i]['mean_y'],
                        'end_x': fixation_groups[i + 1]['mean_x'],
                        'end_y': fixation_groups[i + 1]['mean_y'],
                        'amplitude': euclidean(
                            [fixation_groups[i]['mean_x'], fixation_groups[i]['mean_y']],
                            [fixation_groups[i + 1]['mean_x'], fixation_groups[i + 1]['mean_y']]
                        ),
                        'max_velocity': saccade_data['velocity'].max(),
                        'mean_velocity': saccade_data['velocity'].mean(),
                        'points': len(saccade_data)
                    })
        
        self.saccades = pd.DataFrame(saccade_groups)
        
        print(f"Detected {len(self.fixations)} fixations and {len(self.saccades)} saccades.")
        
        return self.fixations, self.saccades
    
    def detect_fixations_dbscan(self, eps=30, min_samples=5, min_duration=0.1):
        """
        Detect fixations using DBSCAN clustering algorithm.
        
        Parameters:
        -----------
        eps : float
            The maximum distance between two samples for them to be considered in the same cluster
        min_samples : int
            The minimum number of samples in a neighborhood for a point to be considered a core point
        min_duration : float
            Minimum duration for a fixation to be considered valid (seconds)
        """
        if self.processed_data is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        print("Detecting fixations using DBSCAN clustering...")
        
        # Prepare data for clustering
        X = self.processed_data[['x', 'y']].values
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        
        # Add cluster labels to the data
        self.processed_data['cluster'] = clustering.labels_
        
        # Extract fixation clusters (ignoring noise points labeled as -1)
        fixation_clusters = []
        for cluster_id in sorted(set(clustering.labels_)):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_data = self.processed_data[self.processed_data['cluster'] == cluster_id]
            duration = cluster_data['timestamp_sec'].iloc[-1] - cluster_data['timestamp_sec'].iloc[0]
            
            if duration >= min_duration:
                fixation_clusters.append({
                    'cluster_id': cluster_id,
                    'start_time': cluster_data['timestamp_sec'].iloc[0],
                    'end_time': cluster_data['timestamp_sec'].iloc[-1],
                    'duration': duration,
                    'mean_x': cluster_data['x'].mean(),
                    'mean_y': cluster_data['y'].mean(),
                    'std_x': cluster_data['x'].std(),
                    'std_y': cluster_data['y'].std(),
                    'points': len(cluster_data)
                })
        
        self.fixations_dbscan = pd.DataFrame(fixation_clusters)
        
        print(f"Detected {len(self.fixations_dbscan)} fixation clusters using DBSCAN.")
        
        return self.fixations_dbscan
    
    def compute_cwt_metrics(self):
        """
        Compute metrics useful for Continuous Wavelet Transform (CWT) analysis.
        """
        if self.processed_data is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        print("Computing metrics for CWT analysis...")
        
        # 1. Compute wavelet coefficients for x and y coordinates
        scales = np.arange(1, 128)
        
        # X coordinate wavelet transform
        x_signal = self.processed_data['x'].values
        x_coeffs, x_freqs = pywt.cwt(x_signal, scales, 'morl')
        
        # Y coordinate wavelet transform
        y_signal = self.processed_data['y'].values
        y_coeffs, y_freqs = pywt.cwt(y_signal, scales, 'morl')
        
        # Store wavelet coefficients
        self.metrics['x_wavelet_coeffs'] = x_coeffs
        self.metrics['y_wavelet_coeffs'] = y_coeffs
        self.metrics['wavelet_freqs'] = x_freqs
        self.metrics['wavelet_scales'] = scales
        
        # 2. Compute power spectrum
        x_power = np.abs(x_coeffs)**2
        y_power = np.abs(y_coeffs)**2
        
        self.metrics['x_power_spectrum'] = x_power
        self.metrics['y_power_spectrum'] = y_power
        
        # 3. Compute wavelet entropy
        x_entropy = np.array([entropy(np.abs(x_coeffs[i, :])) for i in range(len(scales))])
        y_entropy = np.array([entropy(np.abs(y_coeffs[i, :])) for i in range(len(scales))])
        
        self.metrics['x_wavelet_entropy'] = x_entropy
        self.metrics['y_wavelet_entropy'] = y_entropy
        
        # 4. Compute gaze path complexity metrics
        # Spatial complexity (total path length)
        self.metrics['path_length'] = self.processed_data['distance'].sum()
        
        # Temporal complexity (variability in velocity)
        self.metrics['velocity_std'] = self.processed_data['velocity'].std()
        
        # Directional complexity (variability in direction changes)
        self.metrics['direction_changes'] = np.sum(np.abs(self.processed_data['angle_change']) > 0.1)
        
        # 5. Compute fixation and saccade metrics
        if self.fixations is not None and not self.fixations.empty:
            self.metrics['fixation_count'] = len(self.fixations)
            self.metrics['mean_fixation_duration'] = self.fixations['duration'].mean()
            self.metrics['total_fixation_time'] = self.fixations['duration'].sum()
        else:
            self.metrics['fixation_count'] = 0
            self.metrics['mean_fixation_duration'] = 0
            self.metrics['total_fixation_time'] = 0
            
        if self.saccades is not None and not self.saccades.empty:
            self.metrics['saccade_count'] = len(self.saccades)
            self.metrics['mean_saccade_amplitude'] = self.saccades['amplitude'].mean()
            self.metrics['mean_saccade_velocity'] = self.saccades['mean_velocity'].mean()
            self.metrics['max_saccade_velocity'] = self.saccades['max_velocity'].max()
        else:
            self.metrics['saccade_count'] = 0
            self.metrics['mean_saccade_amplitude'] = 0
            self.metrics['mean_saccade_velocity'] = 0
            self.metrics['max_saccade_velocity'] = 0
        
        # 6. Compute scanpath metrics
        if self.fixations is not None and len(self.fixations) > 1:
            # Scanpath length (sum of distances between consecutive fixations)
            scanpath_length = 0
            for i in range(len(self.fixations) - 1):
                scanpath_length += euclidean(
                    [self.fixations['mean_x'].iloc[i], self.fixations['mean_y'].iloc[i]],
                    [self.fixations['mean_x'].iloc[i+1], self.fixations['mean_y'].iloc[i+1]]
                )
            self.metrics['scanpath_length'] = scanpath_length
            
            # Scanpath area (convex hull area of fixation points)
            from scipy.spatial import ConvexHull
            try:
                points = self.fixations[['mean_x', 'mean_y']].values
                hull = ConvexHull(points)
                self.metrics['scanpath_area'] = hull.volume
            except:
                self.metrics['scanpath_area'] = 0
        else:
            self.metrics['scanpath_length'] = 0
            self.metrics['scanpath_area'] = 0
        
        print("Computed CWT and gaze metrics successfully.")
        
        return self.metrics
    
    def visualize_results(self):
        """
        Create visualizations of the gaze data, fixations, saccades, and CWT analysis.
        Saves graphs to date-formatted folders (MM#DD#YYYY).
        """
        if self.processed_data is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        print("Creating visualizations...")
        
        # Create timestamp for output files
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
        
        # Create date-formatted directory structure (MM#DD#YYYY)
        date_folder = current_datetime.strftime("%m#%d#%Y")
        date_folder_path = os.path.join(self.output_dir, date_folder)
        
        # Create necessary directories
        os.makedirs(date_folder_path, exist_ok=True)
        
        # Create graphs subdirectory
        graphs_folder = os.path.join(date_folder_path, 'graphs')
        os.makedirs(graphs_folder, exist_ok=True)
        
        # 1. Plot raw gaze trajectory
        plt.figure(figsize=(10, 8))
        plt.plot(self.processed_data['x'], self.processed_data['y'], 'b-', alpha=0.5, linewidth=0.5)
        plt.scatter(self.processed_data['x'], self.processed_data['y'], c=self.processed_data['timestamp_sec'], 
                   cmap='viridis', s=5, alpha=0.7)
        plt.colorbar(label='Time (seconds)')
        plt.title('Raw Gaze Trajectory')
        plt.xlabel('X coordinate (pixels)')
        plt.ylabel('Y coordinate (pixels)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_folder, f'gaze_trajectory_{timestamp}.png'), dpi=300)
        
        # 2. Plot fixations and saccades
        if self.fixations is not None and not self.fixations.empty and self.saccades is not None and not self.saccades.empty:
            plt.figure(figsize=(12, 10))
            
            # Plot gaze trajectory
            plt.plot(self.processed_data['x'], self.processed_data['y'], 'b-', alpha=0.2, linewidth=0.5)
            
            # Plot fixations
            for i, fix in self.fixations.iterrows():
                circle = plt.Circle((fix['mean_x'], fix['mean_y']), 
                                   radius=max(10, fix['duration'] * 50),  # Scale circle by duration
                                   alpha=0.5, color='red')
                plt.gca().add_patch(circle)
                plt.text(fix['mean_x'], fix['mean_y'], str(i+1), 
                        fontsize=8, ha='center', va='center', color='white')
            
            # Plot saccades
            for i, sac in self.saccades.iterrows():
                plt.arrow(sac['start_x'], sac['start_y'], 
                         sac['end_x'] - sac['start_x'], sac['end_y'] - sac['start_y'],
                         head_width=15, head_length=15, fc='green', ec='green', alpha=0.7)
            
            plt.title('Fixations and Saccades')
            plt.xlabel('X coordinate (pixels)')
            plt.ylabel('Y coordinate (pixels)')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_folder, f'fixations_saccades_{timestamp}.png'), dpi=300)
        
        # 3. Plot velocity profile
        plt.figure(figsize=(12, 6))
        plt.plot(self.processed_data['timestamp_sec'], self.processed_data['velocity'], 'b-')
        plt.axhline(y=100, color='r', linestyle='--', label='Fixation threshold')
        plt.title('Gaze Velocity Profile')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Velocity (pixels/second)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_folder, f'velocity_profile_{timestamp}.png'), dpi=300)
        
        # 4. Plot CWT scalogram for x and y coordinates
        if 'x_wavelet_coeffs' in self.metrics and 'y_wavelet_coeffs' in self.metrics:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # X coordinate scalogram
            im1 = ax1.imshow(np.abs(self.metrics['x_wavelet_coeffs']), 
                           aspect='auto', cmap='jet',
                           extent=[0, len(self.processed_data), 
                                  self.metrics['wavelet_scales'][-1], 
                                  self.metrics['wavelet_scales'][0]])
            ax1.set_title('CWT Scalogram - X Coordinate')
            ax1.set_ylabel('Scale')
            plt.colorbar(im1, ax=ax1, label='Magnitude')
            
            # Y coordinate scalogram
            im2 = ax2.imshow(np.abs(self.metrics['y_wavelet_coeffs']), 
                           aspect='auto', cmap='jet',
                           extent=[0, len(self.processed_data), 
                                  self.metrics['wavelet_scales'][-1], 
                                  self.metrics['wavelet_scales'][0]])
            ax2.set_title('CWT Scalogram - Y Coordinate')
            ax2.set_xlabel('Time (samples)')
            ax2.set_ylabel('Scale')
            plt.colorbar(im2, ax=ax2, label='Magnitude')
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_folder, f'cwt_scalogram_{timestamp}.png'), dpi=300)
        
        # 5. Plot fixation duration distribution
        if self.fixations is not None and not self.fixations.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.fixations['duration'], bins=20, kde=True)
            plt.title('Fixation Duration Distribution')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_folder, f'fixation_duration_dist_{timestamp}.png'), dpi=300)
        
        # 6. Plot saccade amplitude distribution
        if self.saccades is not None and not self.saccades.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.saccades['amplitude'], bins=20, kde=True)
            plt.title('Saccade Amplitude Distribution')
            plt.xlabel('Amplitude (pixels)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_folder, f'saccade_amplitude_dist_{timestamp}.png'), dpi=300)
        
        # 7. Plot heatmap of gaze density
        plt.figure(figsize=(12, 10))
        sns.kdeplot(x=self.processed_data['x'], y=self.processed_data['y'], 
                   cmap='hot', fill=True, thresh=0.05)
        plt.title('Gaze Density Heatmap')
        plt.xlabel('X coordinate (pixels)')
        plt.ylabel('Y coordinate (pixels)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_folder, f'gaze_heatmap_{timestamp}.png'), dpi=300)
        
        print(f"Visualizations saved to {graphs_folder} directory.")
    
    def save_processed_data(self):
        """
        Save processed data, fixations, saccades, and metrics to CSV files.
        Organizes data into folders by date in the format "MM#DD#YYYY".
        """
        if self.processed_data is None:
            raise ValueError("Data must be loaded and processed first")
        
        print("Saving processed data...")
        
        # Create timestamp for output files
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
        
        # Create date-formatted directory structure (MM#DD#YYYY)
        date_folder = current_datetime.strftime("%m#%d#%Y")
        date_folder_path = os.path.join(self.output_dir, date_folder)
        
        # Create necessary directories
        os.makedirs(date_folder_path, exist_ok=True)
        
        # Create subdirectories for different types of data
        csv_folder = os.path.join(date_folder_path, 'csv')
        graphs_folder = os.path.join(date_folder_path, 'graphs')
        data_folder = os.path.join(date_folder_path, 'data')
        
        os.makedirs(csv_folder, exist_ok=True)
        os.makedirs(graphs_folder, exist_ok=True)
        os.makedirs(data_folder, exist_ok=True)
        
        # Save processed data
        self.processed_data.to_csv(os.path.join(csv_folder, f'processed_gaze_data_{timestamp}.csv'), index=False)
        
        # Save fixations
        if self.fixations is not None and not self.fixations.empty:
            self.fixations.to_csv(os.path.join(csv_folder, f'fixations_{timestamp}.csv'), index=False)
        
        # Save saccades
        if self.saccades is not None and not self.saccades.empty:
            self.saccades.to_csv(os.path.join(csv_folder, f'saccades_{timestamp}.csv'), index=False)
        
        # Save metrics
        if self.metrics:
            # Save scalar metrics
            scalar_metrics = {k: v for k, v in self.metrics.items() if np.isscalar(v)}
            pd.DataFrame([scalar_metrics]).to_csv(os.path.join(csv_folder, f'gaze_metrics_{timestamp}.csv'), index=False)
            
            # Save wavelet coefficients and other array data
            np.savez(os.path.join(data_folder, f'wavelet_data_{timestamp}.npz'),
                    x_wavelet_coeffs=self.metrics.get('x_wavelet_coeffs', None),
                    y_wavelet_coeffs=self.metrics.get('y_wavelet_coeffs', None),
                    wavelet_freqs=self.metrics.get('wavelet_freqs', None),
                    wavelet_scales=self.metrics.get('wavelet_scales', None),
                    x_power_spectrum=self.metrics.get('x_power_spectrum', None),
                    y_power_spectrum=self.metrics.get('y_power_spectrum', None),
                    x_wavelet_entropy=self.metrics.get('x_wavelet_entropy', None),
                    y_wavelet_entropy=self.metrics.get('y_wavelet_entropy', None))
        
        print(f"Data saved to {date_folder_path} directory.")
    
    def run_pipeline(self, velocity_threshold=100, min_fixation_duration=0.1, 
                    min_saccade_velocity=300, use_dbscan=False, eps=30, min_samples=5):
        """
        Run the complete gaze data processing pipeline.
        
        Parameters:
        -----------
        velocity_threshold : float
            Velocity threshold for distinguishing between fixations and saccades (pixels/sec)
        min_fixation_duration : float
            Minimum duration for a fixation to be considered valid (seconds)
        min_saccade_velocity : float
            Minimum velocity for a movement to be considered a saccade (pixels/sec)
        use_dbscan : bool
            Whether to use DBSCAN clustering for fixation detection
        eps : float
            The maximum distance between two samples for DBSCAN clustering
        min_samples : int
            The minimum number of samples in a neighborhood for DBSCAN clustering
        """
        print("Starting gaze data processing pipeline...")
        
        # Step 1: Load and preprocess data
        self.load_data()
        
        # Step 2: Detect fixations and saccades
        if use_dbscan:
            self.detect_fixations_dbscan(eps=eps, min_samples=min_samples, min_duration=min_fixation_duration)
            self.fixations = self.fixations_dbscan
        else:
            self.detect_fixations_and_saccades(
                velocity_threshold=velocity_threshold,
                min_fixation_duration=min_fixation_duration,
                min_saccade_velocity=min_saccade_velocity
            )
        
        # Step 3: Compute metrics for CWT analysis
        self.compute_cwt_metrics()
        
        # Step 4: Visualize results
        self.visualize_results()
        
        # Step 5: Save processed data
        self.save_processed_data()
        
        print("Gaze data processing pipeline completed successfully!")
        
        return {
            'processed_data': self.processed_data,
            'fixations': self.fixations,
            'saccades': self.saccades,
            'metrics': self.metrics
        }


if __name__ == "__main__":
    # Example usage
    file_path = "data/gaze_data_fallback_2025-03-11T16_07_19.760Z.csv"
    processor = GazeDataProcessor(file_path)
    
    # Run the complete pipeline
    results = processor.run_pipeline(
        velocity_threshold=100,  # pixels/sec
        min_fixation_duration=0.1,  # seconds
        min_saccade_velocity=300,  # pixels/sec
        use_dbscan=False  # Set to True to use DBSCAN clustering instead of velocity-based detection
    )
    
    # Print summary of results
    print("\nSummary of Results:")
    print(f"Total data points: {len(results['processed_data'])}")
    
    if results['fixations'] is not None and not results['fixations'].empty:
        print(f"Number of fixations: {len(results['fixations'])}")
        print(f"Mean fixation duration: {results['fixations']['duration'].mean():.3f} seconds")
    else:
        print("No fixations detected.")
    
    if results['saccades'] is not None and not results['saccades'].empty:
        print(f"Number of saccades: {len(results['saccades'])}")
        print(f"Mean saccade amplitude: {results['saccades']['amplitude'].mean():.2f} pixels")
        print(f"Mean saccade velocity: {results['saccades']['mean_velocity'].mean():.2f} pixels/sec")
    else:
        print("No saccades detected.")
    
    print(f"Total scanpath length: {results['metrics'].get('scanpath_length', 'N/A')}")
    print(f"Path complexity (direction changes): {results['metrics'].get('direction_changes', 'N/A')}")
    
    print("\nProcessed data and visualizations saved to 'processed_data' directory.") 
