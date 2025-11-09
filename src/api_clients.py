"""API clients for environmental and circadian data."""

import os
import requests
from typing import Dict, Optional
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()


def get_weather_data(lat: float, lon: float, api_key: Optional[str] = None) -> Dict:
    """
    Fetch weather data from OpenWeatherMap API.
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: OpenWeatherMap API key (or from env)
        
    Returns:
        Dictionary with temp_c, humidity, cloud
    """
    api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        # Return synthetic data if API key not available
        return {
            "temp_c": 22.0,
            "humidity": 60,
            "cloud": 30
        }
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric"
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        return {
            "temp_c": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "cloud": data["clouds"]["all"]
        }
    except Exception as e:
        print(f"Warning: Could not fetch weather data: {e}. Using defaults.")
        return {
            "temp_c": 22.0,
            "humidity": 60,
            "cloud": 30
        }


def get_air_quality(lat: float, lon: float, api_key: Optional[str] = None) -> Dict:
    """
    Fetch air quality data from AirNow API.
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: AirNow API key (or from env)
        
    Returns:
        Dictionary with aqi, pm25, ozone
    """
    api_key = api_key or os.getenv("AIRNOW_API_KEY")
    
    if not api_key:
        # Return synthetic data if API key not available
        return {
            "aqi": 50,
            "pm25": 12.0,
            "ozone": 0.05
        }
    
    try:
        # AirNow API endpoint
        url = "https://www.airnowapi.org/aq/observation/latLong/current/"
        params = {
            "latitude": lat,
            "longitude": lon,
            "format": "application/json",
            "API_KEY": api_key
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Parse response (may contain multiple pollutants)
        aqi = 50  # Default
        pm25 = 12.0
        ozone = 0.05
        
        for obs in data:
            param = obs.get("ParameterName", "")
            aqi_val = obs.get("AQI", 50)
            
            if param == "PM2.5":
                pm25 = obs.get("Value", 12.0)
                aqi = max(aqi, aqi_val)
            elif param == "O3":
                ozone = obs.get("Value", 0.05) / 1000.0  # Convert to ppm
                aqi = max(aqi, aqi_val)
        
        return {
            "aqi": aqi,
            "pm25": pm25,
            "ozone": ozone
        }
    except Exception as e:
        print(f"Warning: Could not fetch air quality data: {e}. Using defaults.")
        return {
            "aqi": 50,
            "pm25": 12.0,
            "ozone": 0.05
        }


def get_circadian_data(lat: float, lon: float, timezone: Optional[str] = None) -> Dict:
    """
    Fetch circadian data (hour, daylight status) from WorldTimeAPI.
    
    WorldTimeAPI is free and doesn't require an API key.
    Uses the /timezone/{area}/{location} endpoint format.
    
    Args:
        lat: Latitude (not directly used, but kept for API consistency)
        lon: Longitude (not directly used, but kept for API consistency)
        timezone: Timezone in format "Area/Location" (e.g., "America/New_York")
        
    Returns:
        Dictionary with hour, is_daylight
    """
    timezone = timezone or os.getenv("DEFAULT_TIMEZONE", "America/New_York")
    
    try:
        # WorldTimeAPI - free public API, no key required
        # Format: http://worldtimeapi.org/api/timezone/{area}/{location}
        url = f"http://worldtimeapi.org/api/timezone/{timezone}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Parse datetime from ISO8601 format
        dt_str = data.get("datetime", "")
        if dt_str:
            # Handle timezone offset in ISO8601 format
            dt_str = dt_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(dt_str)
            hour = dt.hour
            
            # Use DST flag if available, otherwise estimate (6 AM to 8 PM)
            dst = data.get("dst", False)
            if dst:
                # During daylight saving, extend daylight hours
                is_daylight = 1 if 5 <= hour < 21 else 0
            else:
                is_daylight = 1 if 6 <= hour < 20 else 0
            
            return {
                "hour": hour,
                "is_daylight": is_daylight
            }
        else:
            raise ValueError("No datetime in response")
    except Exception as e:
        print(f"Warning: Could not fetch circadian data: {e}. Using local time.")
        # Fallback to local time
        now = datetime.now()
        hour = now.hour
        is_daylight = 1 if 6 <= hour < 20 else 0
        
        return {
            "hour": hour,
            "is_daylight": is_daylight
        }


def get_activity_label(accel_mag: float, hr: float) -> str:
    """
    Simple activity classification based on accelerometer and heart rate.
    
    Args:
        accel_mag: Accelerometer magnitude
        hr: Heart rate
        
    Returns:
        Activity label
    """
    # Simple heuristic-based classification
    if accel_mag > 2.0 or hr > 100:
        return "running"
    elif accel_mag > 1.5 or hr > 85:
        return "walking"
    elif accel_mag < 0.5 and hr < 70:
        return "sitting"
    else:
        return "standing"

