#!/usr/bin/env python3
"""Simple example: Using EDON SDK with REST transport."""

from edon import EdonClient

def main():
    # Initialize client (uses EDON_BASE_URL env var or default)
    client = EdonClient()
    
    # Create sensor window (240 samples per signal)
    window = {
        "EDA": [0.1] * 240,          # Electrodermal activity
        "TEMP": [36.5] * 240,        # Temperature
        "BVP": [0.5] * 240,          # Blood volume pulse
        "ACC_x": [0.0] * 240,        # Accelerometer X
        "ACC_y": [0.0] * 240,        # Accelerometer Y
        "ACC_z": [1.0] * 240,        # Accelerometer Z
        "temp_c": 22.0,              # Environmental temperature (°C)
        "humidity": 50.0,            # Humidity (%)
        "aqi": 35,                   # Air Quality Index
        "local_hour": 14,            # Local hour [0-23]
    }
    
    # Compute CAV
    print("Computing CAV...")
    result = client.cav(window)
    
    print(f"State: {result['state']}")
    print(f"CAV Raw: {result['cav_raw']}")
    print(f"CAV Smooth: {result['cav_smooth']}")
    print(f"P-Stress: {result['parts']['p_stress']:.3f}")
    
    # Classify state
    state = client.classify(window)
    print(f"\nClassified state: {state}")
    
    # Health check
    health = client.health()
    print(f"\nService health: {health.get('ok', False)}")

if __name__ == "__main__":
    main()

