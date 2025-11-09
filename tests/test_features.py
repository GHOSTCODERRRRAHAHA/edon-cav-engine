"""Tests for feature extraction."""

import numpy as np
import pytest
from src.features import (
    compute_hrv_rmssd,
    compute_eda_stats,
    compute_respiration_rate,
    compute_accel_magnitude
)


def test_hrv_rmssd():
    """Test HRV RMSSD computation."""
    # Create synthetic RR intervals
    rr_intervals = np.array([800, 820, 810, 830, 815, 825])  # ms
    
    rmssd = compute_hrv_rmssd(rr_intervals)
    
    assert rmssd > 0
    assert isinstance(rmssd, float)
    
    # Test with insufficient data
    assert compute_hrv_rmssd(np.array([800])) == 0.0
    assert compute_hrv_rmssd(np.array([])) == 0.0


def test_eda_stats():
    """Test EDA statistics computation."""
    eda_signal = np.array([2.0, 2.5, 2.3, 2.7, 2.1])
    
    stats = compute_eda_stats(eda_signal)
    
    assert "mean" in stats
    assert "var" in stats
    assert stats["mean"] > 0
    assert stats["var"] >= 0
    
    # Test with empty array
    stats_empty = compute_eda_stats(np.array([]))
    assert stats_empty["mean"] == 0.0
    assert stats_empty["var"] == 0.0


def test_respiration_rate():
    """Test respiration rate computation."""
    # Create synthetic respiration signal (sinusoidal)
    sampling_rate = 700.0
    duration = 10.0  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration))
    resp_rate = 16.0  # breaths per minute
    resp_signal = np.sin(2 * np.pi * (resp_rate / 60) * t)
    
    computed_rate = compute_respiration_rate(resp_signal, sampling_rate)
    
    # Should be close to 16 BPM (allow some tolerance)
    assert 10 <= computed_rate <= 25
    
    # Test with insufficient data
    assert compute_respiration_rate(np.array([1, 2]), 700.0) == 0.0


def test_accel_magnitude():
    """Test accelerometer magnitude computation."""
    accel_x = np.array([0.1, 0.2, 0.15])
    accel_y = np.array([0.2, 0.1, 0.25])
    accel_z = np.array([0.9, 0.95, 0.92])
    
    mag = compute_accel_magnitude(accel_x, accel_y, accel_z)
    
    assert mag > 0
    assert isinstance(mag, float)
    
    # Test with mismatched lengths
    assert compute_accel_magnitude(
        np.array([1]), np.array([1, 2]), np.array([1])
    ) == 0.0

