#!/usr/bin/env python3
"""Example: Using EDON SDK with REST transport (default)."""

import sys
from pathlib import Path

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sdk" / "python"))

from edon_sdk.client import EdonClient, TransportType

def main():
    # Create client with REST transport (default)
    client = EdonClient(
        base_url="http://127.0.0.1:8000",
        transport=TransportType.REST,
    )
    
    # Example sensor window
    window = {
        "EDA": [0.1] * 240,
        "TEMP": [36.5] * 240,
        "BVP": [0.5] * 240,
        "ACC_x": [0.0] * 240,
        "ACC_y": [0.0] * 240,
        "ACC_z": [1.0] * 240,
        "temp_c": 22.0,
        "humidity": 50.0,
        "aqi": 35,
        "local_hour": 14,
    }
    
    # Compute CAV
    print("Computing CAV via REST...")
    result = client.cav(window)
    print(f"State: {result['state']}")
    print(f"CAV Smooth: {result['cav_smooth']}")
    print(f"P-Stress: {result['parts']['p_stress']:.3f}")
    
    # Classify
    state = client.classify(window)
    print(f"\nClassified state: {state}")

if __name__ == "__main__":
    main()

