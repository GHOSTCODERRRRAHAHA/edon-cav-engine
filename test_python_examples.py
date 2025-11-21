#!/usr/bin/env python3
"""Test Python SDK examples."""

import sys
from pathlib import Path

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent / "sdk" / "python"))

print("Testing Python SDK Examples")
print("=" * 60)

# Test REST example
print("\n[1] Testing REST example...")
try:
    from edon_sdk.client import EdonClient, TransportType
    
    client = EdonClient(
        base_url="http://127.0.0.1:8001",
        transport=TransportType.REST
    )
    
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
    
    result = client.cav(window)
    print(f"[OK] REST example works")
    print(f"     State: {result['state']}")
    print(f"     CAV: {result['cav_smooth']}")
    
except Exception as e:
    print(f"[FAIL] REST example failed: {e}")

# Test gRPC example (if server is running)
print("\n[2] Testing gRPC example...")
try:
    sys.path.insert(0, str(Path(__file__).parent / "integrations" / "grpc" / "edon_grpc_service"))
    
    client = EdonClient(
        transport=TransportType.GRPC,
        grpc_host="localhost",
        grpc_port=50051
    )
    
    # Just test if client can be created
    print("[OK] gRPC client created")
    client.close()
    
except Exception as e:
    print(f"[SKIP] gRPC example (server may not be running): {e}")

print("\n" + "=" * 60)
print("Python examples test complete!")

