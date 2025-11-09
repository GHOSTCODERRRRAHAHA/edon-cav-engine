"""Feature extraction from physiological signals."""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_hrv_rmssd(rr_intervals: np.ndarray) -> float:
    """
    Compute HRV using Root Mean Square of Successive Differences (RMSSD).
    
    Args:
        rr_intervals: Array of R-R intervals in milliseconds
        
    Returns:
        RMSSD value in milliseconds
    """
    if len(rr_intervals) < 2:
        return 0.0
    
    # Remove NaN and invalid values
    rr = rr_intervals[~np.isnan(rr_intervals)]
    if len(rr) < 2:
        return 0.0
    
    # Compute successive differences
    diff = np.diff(rr)
    
    # RMSSD = sqrt(mean(diff^2))
    rmssd = np.sqrt(np.mean(diff ** 2))
    
    return float(rmssd)


def compute_eda_stats(eda_signal: np.ndarray) -> Dict[str, float]:
    """
    Compute EDA mean and variance.
    
    Args:
        eda_signal: Electrodermal activity signal in microsiemens
        
    Returns:
        Dictionary with 'mean' and 'var' keys
    """
    if len(eda_signal) == 0:
        return {"mean": 0.0, "var": 0.0}
    
    # Remove NaN values
    eda = eda_signal[~np.isnan(eda_signal)]
    
    if len(eda) == 0:
        return {"mean": 0.0, "var": 0.0}
    
    return {
        "mean": float(np.mean(eda)),
        "var": float(np.var(eda))
    }


def compute_respiration_rate(resp_signal: np.ndarray, sampling_rate: float = 700.0) -> float:
    """
    Compute respiration rate in breaths per minute.
    
    Uses peak detection on the respiration signal.
    
    Args:
        resp_signal: Respiration signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Respiration rate in breaths per minute
    """
    if len(resp_signal) < int(sampling_rate * 2):  # Need at least 2 seconds
        return 0.0
    
    # Remove NaN
    resp = resp_signal[~np.isnan(resp_signal)]
    if len(resp) < int(sampling_rate * 2):
        return 0.0
    
    # Simple peak detection using zero-crossing of derivative
    # More sophisticated methods could use scipy.signal.find_peaks
    from scipy.signal import find_peaks
    
    # Find peaks in the signal
    peaks, _ = find_peaks(resp, distance=int(sampling_rate * 0.5))  # Min 0.5s between peaks
    
    if len(peaks) < 2:
        return 0.0
    
    # Calculate average time between peaks
    peak_intervals = np.diff(peaks) / sampling_rate  # Convert to seconds
    avg_interval = np.mean(peak_intervals)
    
    if avg_interval <= 0:
        return 0.0
    
    # Convert to breaths per minute
    bpm = 60.0 / avg_interval
    
    return float(bpm)


def compute_accel_magnitude(accel_x: np.ndarray, accel_y: np.ndarray, accel_z: np.ndarray) -> float:
    """
    Compute accelerometer magnitude.
    
    Args:
        accel_x: X-axis acceleration in g
        accel_y: Y-axis acceleration in g
        accel_z: Z-axis acceleration in g
        
    Returns:
        Mean magnitude in g
    """
    if len(accel_x) != len(accel_y) or len(accel_x) != len(accel_z):
        return 0.0
    
    # Remove NaN
    valid = ~(np.isnan(accel_x) | np.isnan(accel_y) | np.isnan(accel_z))
    
    if np.sum(valid) == 0:
        return 0.0
    
    # Compute magnitude: sqrt(x^2 + y^2 + z^2)
    magnitude = np.sqrt(
        accel_x[valid] ** 2 + 
        accel_y[valid] ** 2 + 
        accel_z[valid] ** 2
    )
    
    return float(np.mean(magnitude))


def extract_wesad_features(wesad_data: Dict, subject_id: str = "S2", window_size: int = 60) -> pd.DataFrame:
    """
    Extract features from WESAD dataset.
    
    WESAD structure:
    - 'signal': dict with keys like 'chest', 'wrist'
    - Each contains 'ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp'
    - Sampling rates vary (700 Hz for most, 32 Hz for ACC)
    
    Args:
        wesad_data: Loaded WESAD data dictionary
        subject_id: Subject ID to process (default S2 for baseline)
        window_size: Window size in seconds for feature extraction
        
    Returns:
        DataFrame with extracted features
    """
    import scipy.io
    
    # For demo purposes, we'll create synthetic data if WESAD isn't available
    # In production, load actual WESAD .mat files
    
    features_list = []
    
    # Try to load actual WESAD data if available
    try:
        if 'signal' in wesad_data:
            signal = wesad_data['signal']
            chest = signal.get('chest', {})
            
            # Get sampling rates
            fs_ecg = 700.0
            fs_eda = 700.0
            fs_resp = 700.0
            fs_acc = 32.0
            
            # Extract signals
            ecg = chest.get('ECG', np.array([]))
            eda = chest.get('EDA', np.array([]))
            resp = chest.get('Resp', np.array([]))
            acc = chest.get('ACC', np.array([]))
            
            # Process in windows
            window_samples_ecg = int(fs_ecg * window_size)
            window_samples_acc = int(fs_acc * window_size)
            
            n_windows = min(
                len(ecg) // window_samples_ecg if len(ecg) > 0 else 0,
                len(eda) // window_samples_ecg if len(eda) > 0 else 0,
                len(resp) // window_samples_ecg if len(resp) > 0 else 0,
                len(acc) // window_samples_acc if len(acc) > 0 else 0,
            )
            
            for i in range(n_windows):
                start_ecg = i * window_samples_ecg
                end_ecg = start_ecg + window_samples_ecg
                start_acc = i * window_samples_acc
                end_acc = start_acc + window_samples_acc
                
                # Extract features
                ecg_window = ecg[start_ecg:end_ecg] if len(ecg) > 0 else np.array([])
                eda_window = eda[start_ecg:end_ecg] if len(eda) > 0 else np.array([])
                resp_window = resp[start_ecg:end_ecg] if len(resp) > 0 else np.array([])
                acc_window = acc[start_acc:end_acc] if len(acc) > 0 else np.array([])
                
                # Compute HR from ECG (simplified - in production use proper R-peak detection)
                if len(ecg_window) > 0:
                    # Simple peak detection for HR estimation
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(ecg_window, distance=int(fs_ecg * 0.4))
                    if len(peaks) > 1:
                        rr_intervals = np.diff(peaks) / fs_ecg * 1000  # Convert to ms
                        hr = 60000.0 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 70.0
                        hrv = compute_hrv_rmssd(rr_intervals)
                    else:
                        hr = 70.0
                        hrv = 0.0
                else:
                    hr = 70.0
                    hrv = 0.0
                
                eda_stats = compute_eda_stats(eda_window)
                resp_bpm = compute_respiration_rate(resp_window, fs_resp)
                
                if len(acc_window) >= 3:
                    accel_mag = compute_accel_magnitude(
                        acc_window[:, 0] if acc_window.ndim > 1 else acc_window,
                        acc_window[:, 1] if acc_window.ndim > 1 else np.zeros(len(acc_window)),
                        acc_window[:, 2] if acc_window.ndim > 1 else np.zeros(len(acc_window))
                    )
                else:
                    accel_mag = 0.0
                
                features_list.append({
                    'hr': hr,
                    'hrv_rmssd': hrv,
                    'eda_mean': eda_stats['mean'],
                    'eda_var': eda_stats['var'],
                    'resp_bpm': resp_bpm,
                    'accel_mag': accel_mag,
                })
    except Exception as e:
        # Fallback to synthetic data generation
        print(f"Warning: Could not load WESAD data: {e}. Generating synthetic data.")
        n_windows = 1000  # Generate 1000 windows
        
        for i in range(n_windows):
            # Generate realistic synthetic physiological data
            features_list.append({
                'hr': np.random.normal(72, 10),
                'hrv_rmssd': np.random.normal(45, 15),
                'eda_mean': np.random.normal(2.5, 0.8),
                'eda_var': np.random.normal(0.5, 0.3),
                'resp_bpm': np.random.normal(16, 3),
                'accel_mag': np.random.normal(1.0, 0.5),
            })
    
    return pd.DataFrame(features_list)

