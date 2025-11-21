#!/usr/bin/env python3
"""Test gRPC client connection."""

import sys
import time
from pathlib import Path

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent / "sdk" / "python"))
sys.path.insert(0, str(Path(__file__).parent / "integrations" / "grpc" / "edon_grpc_service"))

from edon_sdk.client import EdonClient, TransportType

def main():
    print("Testing gRPC Client...")
    print("=" * 60)
    
    # Wait a moment for server to start
    print("Waiting for gRPC server to be ready...")
    time.sleep(2)
    
    try:
        # Create gRPC client
        client = EdonClient(
            transport=TransportType.GRPC,
            grpc_host="localhost",
            grpc_port=50051
        )
        
        # Test health
        print("\n[1] Testing health check...")
        health = client.health()
        print(f"Health: {health}")
        
        # Create test window
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
        
        # Test single request
        print("\n[2] Testing single CAV computation...")
        result = client.cav(window)
        print(f"[OK] State: {result['state']}")
        print(f"     CAV Smooth: {result['cav_smooth']}")
        print(f"     P-Stress: {result['parts']['p_stress']:.3f}")
        if 'controls' in result:
            print(f"     Controls - Speed: {result['controls']['speed']:.2f}, "
                  f"Torque: {result['controls']['torque']:.2f}, "
                  f"Safety: {result['controls']['safety']:.2f}")
        
        # Test classify
        print("\n[3] Testing classify method...")
        state = client.classify(window)
        print(f"[OK] Classified state: {state}")
        
        # Test streaming (just one update)
        print("\n[4] Testing streaming (single update)...")
        count = 0
        for update in client.stream(window):
            count += 1
            print(f"[OK] Stream update #{count}: {update['state']}, CAV: {update['cav_smooth']}")
            if count >= 1:  # Just get one update for testing
                break
        
        client.close()
        print("\n[OK] All gRPC tests passed!")
        
    except Exception as e:
        print(f"\n[FAIL] gRPC test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

