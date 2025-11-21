#!/usr/bin/env python3
"""
EDON Client SDK

Simple Python client for interacting with the EDON CAV API.
Provides easy-to-use methods for OEM partners to integrate EDON.
"""

import requests
import json
from typing import Dict, Optional, List
from pathlib import Path
import os


class EDONClient:
    """
    EDON API Client for OEM partners.
    
    Provides methods to interact with the EDON CAV API including:
    - Computing CAV scores from sensor windows
    - Retrieving memory statistics
    - Managing memory
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize EDON client.
        
        Args:
            base_url: Base URL of the EDON API server
        """
        env_base = os.getenv("EDON_API_BASE")
        self.base_url = (env_base or base_url).rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        # Optional auth
        if os.getenv("EDON_AUTH_ENABLED", "false").lower() in ("1", "true", "yes"):
            token = os.getenv("EDON_API_TOKEN")
            if token:
                self.session.headers["Authorization"] = f"Bearer {token}"
    
    def post_cav(
        self,
        eda: List[float],
        temp: List[float],
        bvp: List[float],
        acc_x: List[float],
        acc_y: List[float],
        acc_z: List[float],
        temp_c: float,
        humidity: float,
        aqi: int,
        local_hour: int = 12
    ) -> Dict:
        """
        Compute CAV score from sensor window.
        
        Args:
            eda: EDA signal (240 samples)
            temp: Temperature signal (240 samples)
            bvp: Blood volume pulse signal (240 samples)
            acc_x: Accelerometer X-axis (240 samples)
            acc_y: Accelerometer Y-axis (240 samples)
            acc_z: Accelerometer Z-axis (240 samples)
            temp_c: Environmental temperature in Celsius
            humidity: Humidity percentage
            aqi: Air Quality Index
            local_hour: Local hour [0-23]
        
        Returns:
            Dictionary with CAV results including:
            - cav_raw: Raw CAV score [0-10000]
            - cav_smooth: Smoothed CAV score [0-10000]
            - state: State string (overload, balanced, focus, restorative)
            - parts: Component scores (bio, env, circadian, p_stress)
            - adaptive: Adaptive adjustments (z_cav, sensitivity, env_weight_adj)
        
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        payload = {
            "EDA": eda,
            "TEMP": temp,
            "BVP": bvp,
            "ACC_x": acc_x,
            "ACC_y": acc_y,
            "ACC_z": acc_z,
            "temp_c": temp_c,
            "humidity": humidity,
            "aqi": aqi,
            "local_hour": local_hour
        }
        
        response = self.session.post(
            f"{self.base_url}/cav",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def post_cav_from_dict(self, payload: Dict) -> Dict:
        """
        Compute CAV score from payload dictionary.
        
        Args:
            payload: Dictionary with sensor data and environmental parameters
        
        Returns:
            Dictionary with CAV results
        """
        response = self.session.post(
            f"{self.base_url}/cav",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def get_memory_summary(self) -> Dict:
        """
        Get 24-hour memory statistics summary.
        
        Returns:
            Dictionary with:
            - total_records: Number of records in last 24h
            - window_hours: Rolling window size
            - hourly_stats: Per-hour statistics
            - overall_stats: Overall statistics
        
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        response = self.session.get(
            f"{self.base_url}/memory/summary",
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    
    def clear_memory(self) -> Dict:
        """
        Clear all memory (buffer and database).
        
        WARNING: This will delete all stored CAV history.
        Use for testing or resetting the adaptive engine.
        
        Returns:
            Dictionary with status and message
        
        Raises:
            requests.exceptions.RequestException: If API call fails
        """
        response = self.session.post(
            f"{self.base_url}/memory/clear",
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict:
        """
        Check API health status.
        
        Returns:
            Dictionary with health status
        """
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=2
        )
        response.raise_for_status()
        return response.json()
    
    def get_telemetry(self) -> Dict:
        """
        Get server telemetry statistics.
        
        Returns:
            Dictionary with request count, average latency, uptime
        """
        response = self.session.get(
            f"{self.base_url}/telemetry",
            timeout=2
        )
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of EDON client."""
    # Initialize client
    client = EDONClient()
    
    # Load sample payload
    sample_path = Path(__file__).parent / "sample_payload.json"
    if not sample_path.exists():
        print(f"Error: {sample_path} not found")
        print("Please create sample_payload.json with valid sensor data")
        return
    
    with open(sample_path, 'r') as f:
        sample_payload = json.load(f)
    
    # Check health
    print("Checking API health...")
    try:
        health = client.health_check()
        print(f"API Status: {'OK' if health.get('ok') else 'ERROR'}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Compute CAV
    print("\nComputing CAV score...")
    try:
        result = client.post_cav_from_dict(sample_payload)
        print(f"CAV Score: {result['cav_smooth']}")
        print(f"State: {result['state']}")
        print(f"Parts: {result['parts']}")
        if 'adaptive' in result:
            adaptive = result['adaptive']
            print(f"Adaptive:")
            print(f"  Z-score: {adaptive['z_cav']:.2f}")
            print(f"  Sensitivity: {adaptive['sensitivity']:.2f}")
            print(f"  Env weight adj: {adaptive['env_weight_adj']:.2f}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Get memory summary
    print("\nGetting memory summary...")
    try:
        summary = client.get_memory_summary()
        print(f"Total records: {summary['total_records']}")
        if summary.get('overall_stats'):
            overall = summary['overall_stats']
            print(f"CAV Mean: {overall['cav_mean']:.1f}")
            print(f"CAV Std: {overall['cav_std']:.1f}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

