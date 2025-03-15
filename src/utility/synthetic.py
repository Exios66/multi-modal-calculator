import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Union, Dict, List
import json
from scipy.signal import savgol_filter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalDataGenerator:
    """Production-grade synthetic clinical data generator with enhanced features"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'base_time': datetime(2025, 2, 26, 16, 0),
            'duration_hours': 3,
            'resolution_min': 1,
            'signal_ranges': {
                'SpO2': (88, 100),
                'pulse_rate': (60, 140),
                'blood_pressure_sys': (90, 180),
                'resp_rate': (12, 30),
                'temperature': (36.0, 38.5)
            },
            'event_definitions': {
                'hypoxemia': {
                    'SpO2': {'min': 70, 'max': 90},
                    'pulse_rate': {'delta': +15},
                    'resp_rate': {'delta': +5}
                },
                'tachycardia': {
                    'pulse_rate': {'min': 110, 'max': 140},
                    'blood_pressure_sys': {'delta': +20}
                },
                'fever': {
                    'temperature': {'min': 37.8, 'max': 39.5},
                    'pulse_rate': {'delta': +10}
                }
            }
        }
        self.validate_config()
        
    def validate_config(self):
        """Ensure configuration parameters are valid"""
        required_keys = ['base_time', 'duration_hours', 'resolution_min',
                        'signal_ranges', 'event_definitions']
        if not all(key in self.config for key in required_keys):
            raise ValueError("Invalid configuration: missing required parameters")
            
        for param, (low, high) in self.config['signal_ranges'].items():
            if low >= high:
                raise ValueError(f"Invalid range for {param}: {low}-{high}")

    def generate_base_signals(self, num_points: int) -> Dict[str, np.ndarray]:
        """Generate baseline physiological signals with cross-correlation"""
        signals = {}
        sr = self.config['signal_ranges']
        
        # Base vital signs with realistic correlations
        signals['SpO2'] = np.clip(np.random.normal(97.5, 1.0, num_points), *sr['SpO2'])
        signals['pulse_rate'] = np.random.normal(80, 3, num_points)
        signals['blood_pressure_sys'] = 120 + 0.5*(signals['pulse_rate'] - 80) + np.random.normal(0, 3, num_points)
        signals['resp_rate'] = 18 + 0.2*(signals['pulse_rate'] - 80) + np.random.normal(0, 1, num_points)
        signals['temperature'] = np.random.normal(36.8, 0.2, num_points)
        
        return signals
    
    def apply_clinical_event(self, signals: Dict[str, np.ndarray], event_type: str,
                           start_idx: int, duration: int, intensity: float = 1.0):
        """Apply realistic clinical event patterns to signals"""
        event_config = self.config['event_definitions'].get(event_type)
        if not event_config:
            raise ValueError(f"Unknown event type: {event_type}")
            
        end_idx = min(start_idx + duration, len(signals['SpO2']))
        actual_duration = end_idx - start_idx
        x = np.linspace(0, 1, actual_duration)
        
        for param, rules in event_config.items():
            if param not in signals:
                continue
                
            current_values = signals[param][start_idx:end_idx]
            base_value = np.mean(current_values)
            
            # Apply different effect patterns based on parameter type
            if 'delta' in rules:
                effect = rules['delta'] * intensity * np.sin(np.pi * x)  # Smooth sinusoidal effect
            elif 'min' in rules and 'max' in rules:
                effect = np.linspace(base_value, (rules['min'] + rules['max'])/2, actual_duration)
                effect += np.random.normal(0, (rules['max'] - rules['min'])/10, actual_duration)
            
            # Apply smoothing and clipping
            signals[param][start_idx:end_idx] = np.clip(
                savgol_filter(current_values + effect, 5, 2),
                *self.config['signal_ranges'].get(param, (-np.inf, np.inf))
            )
            
        return signals
    
    def add_sensor_noise(self, signals: Dict[str, np.ndarray], noise_level: float = 0.1):
        """Add realistic sensor noise patterns"""
        for param, values in signals.items():
            noise = np.random.normal(0, noise_level * np.std(values), len(values))
            signals[param] = np.clip(values + noise, *self.config['signal_ranges'][param])
        return signals
    
    def generate_dataset(self, events: List[Dict] = None) -> pd.DataFrame:
        """Generate complete dataset with configurable clinical events"""
        num_points = self.config['duration_hours'] * 60 // self.config['resolution_min']
        timestamps = [self.config['base_time'] + timedelta(minutes=i*self.config['resolution_min'])
                     for i in range(num_points)]
        
        signals = self.generate_base_signals(num_points)
        
        # Apply predefined clinical events
        default_events = [
            {'type': 'hypoxemia', 'start': 30, 'duration': 10, 'intensity': 0.8},
            {'type': 'tachycardia', 'start': 65, 'duration': 8, 'intensity': 1.0},
            {'type': 'fever', 'start': 120, 'duration': 15, 'intensity': 0.7}
        ]
        
        for event in events or default_events:
            try:
                signals = self.apply_clinical_event(
                    signals, event['type'],
                    event['start'], event['duration'],
                    event.get('intensity', 1.0)
                )
                logger.info(f"Applied {event['type']} event at {event['start']}min")
            except Exception as e:
                logger.error(f"Failed to apply event {event}: {str(e)}")
        
        signals = self.add_sensor_noise(signals)
        
        # Create DataFrame with proper formatting
        df = pd.DataFrame({
            'timestamp': timestamps,
            **{k: np.round(v, 1) if k != 'pulse_rate' else v.astype(int)
              for k, v in signals.items()}
        })
        
        # Add metadata
        df.attrs['generation_config'] = json.dumps(self.config, default=str)
        df.attrs['event_log'] = json.dumps(events or default_events)
        
        return df

def generate_clinical_data(output_format: str = 'dataframe', **kwargs) -> Union[pd.DataFrame, str, bytes]:
    """
    Generate synthetic clinical data with multiple output format options
    
    Args:
        output_format: One of 'dataframe', 'csv', 'parquet', 'dict'
        **kwargs: Configuration overrides
        
    Returns:
        Requested format containing synthetic clinical data
    """
    generator = ClinicalDataGenerator(kwargs.get('config'))
    
    try:
        df = generator.generate_dataset(kwargs.get('events'))
        
        if output_format == 'dataframe':
            return df
        elif output_format == 'csv':
            return df.to_csv(index=False)
        elif output_format == 'parquet':
            return df.to_parquet(index=False)
        elif output_format == 'dict':
            return df.to_dict(orient='records')
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
    except Exception as e:
        logger.error(f"Data generation failed: {str(e)}")
        raise

# Example production usage
if __name__ == "__main__":
    try:
        # Generate data with custom events
        clinical_data = generate_clinical_data(
            output_format='dataframe',
            events=[
                {'type': 'hypoxemia', 'start': 45, 'duration': 15, 'intensity': 0.9},
                {'type': 'tachycardia', 'start': 90, 'duration': 20, 'intensity': 1.2}
            ]
        )
        
        # Save outputs
        clinical_data.to_csv('clinical_data.csv', index=False)
        clinical_data.to_parquet('clinical_data.parquet', index=False)
        
        logger.info("Successfully generated clinical datasets")
        
    except Exception as e:
        logger.critical(f"Critical failure in data generation: {str(e)}")
        raise
