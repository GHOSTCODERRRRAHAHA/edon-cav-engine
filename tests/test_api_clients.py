"""Tests for API clients."""

import pytest
from src.api_clients import (
    get_weather_data,
    get_air_quality,
    get_circadian_data,
    get_activity_label
)


def test_get_activity_label():
    """Test activity classification."""
    # Running
    assert get_activity_label(2.5, 110) == "running"
    
    # Walking
    assert get_activity_label(1.8, 90) == "walking"
    
    # Sitting
    assert get_activity_label(0.3, 65) == "sitting"
    
    # Standing (default)
    assert get_activity_label(1.0, 75) == "standing"


def test_get_weather_data_no_key():
    """Test weather data with no API key (returns defaults)."""
    result = get_weather_data(40.7128, -74.0060, api_key=None)
    
    assert "temp_c" in result
    assert "humidity" in result
    assert "cloud" in result
    assert isinstance(result["temp_c"], (int, float))
    assert 0 <= result["humidity"] <= 100
    assert 0 <= result["cloud"] <= 100


def test_get_air_quality_no_key():
    """Test air quality with no API key (returns defaults)."""
    result = get_air_quality(40.7128, -74.0060, api_key=None)
    
    assert "aqi" in result
    assert "pm25" in result
    assert "ozone" in result
    assert isinstance(result["aqi"], int)
    assert result["pm25"] > 0
    assert result["ozone"] > 0


def test_get_circadian_data():
    """Test circadian data retrieval."""
    result = get_circadian_data(40.7128, -74.0060)
    
    assert "hour" in result
    assert "is_daylight" in result
    assert 0 <= result["hour"] <= 23
    assert result["is_daylight"] in [0, 1]

