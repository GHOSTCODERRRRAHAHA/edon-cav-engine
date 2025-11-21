#!/usr/bin/env python3
"""Test script for EDON integrations."""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("EDON Integration Tests")
print("=" * 60)

# Test 1: Python SDK REST transport
print("\n[1] Testing Python SDK REST transport...")
try:
    from sdk.python.edon_sdk.client import EdonClient, TransportType
    
    # Check if REST API is running
    client = EdonClient(base_url="http://127.0.0.1:8001", transport=TransportType.REST)
    health = client.health()
    print(f"[OK] REST API health check: {health.get('ok', False)}")
    
    # Test CAV computation
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
    print(f"[OK] CAV computation successful")
    print(f"  State: {result['state']}")
    print(f"  CAV Smooth: {result['cav_smooth']}")
    print(f"  P-Stress: {result['parts']['p_stress']:.3f}")
    
    # Test classify
    state = client.classify(window)
    print(f"[OK] Classify method: {state}")
    
except Exception as e:
    print(f"[FAIL] REST transport test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Protobuf generation
print("\n[2] Testing protobuf generation...")
try:
    import subprocess
    grpc_dir = Path(__file__).parent / "integrations" / "grpc" / "edon_grpc_service"
    os.chdir(grpc_dir)
    
    result = subprocess.run(
        ["python", "-m", "grpc_tools.protoc", "-I.", "--python_out=.", "--grpc_python_out=.", "edon.proto"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        print("[OK] Protobuf files generated successfully")
        if (grpc_dir / "edon_pb2.py").exists():
            print("  [OK] edon_pb2.py exists")
        if (grpc_dir / "edon_pb2_grpc.py").exists():
            print("  [OK] edon_pb2_grpc.py exists")
    else:
        print(f"[FAIL] Protobuf generation failed")
        print(f"  Return code: {result.returncode}")
        if result.stderr:
            print(f"  Error: {result.stderr[:500]}")
        if result.stdout:
            print(f"  Output: {result.stdout[:500]}")
            
except Exception as e:
    print(f"[FAIL] Protobuf generation test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Python SDK gRPC transport (if protobuf generated)
print("\n[3] Testing Python SDK gRPC transport...")
try:
    grpc_dir = Path(__file__).parent / "integrations" / "grpc" / "edon_grpc_service"
    if (grpc_dir / "edon_pb2.py").exists():
        sys.path.insert(0, str(grpc_dir))
        from sdk.python.edon_sdk.client import EdonClient, TransportType
        
        client = EdonClient(
            transport=TransportType.GRPC,
            grpc_host="localhost",
            grpc_port=50051
        )
        print("[OK] gRPC client created (server may not be running)")
        print("  Note: Start gRPC server to test full functionality")
        client.close()
    else:
        print("[SKIP] Skipping gRPC test - protobuf files not generated")
        
except Exception as e:
    print(f"[FAIL] gRPC transport test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Engine direct import
print("\n[4] Testing engine direct import...")
try:
    from app.engine import CAVEngine, classify_state
    
    engine = CAVEngine()
    print("[OK] CAVEngine imported and initialized")
    
    window = {
        "EDA": [0.1] * 240,
        "TEMP": [36.5] * 240,
        "BVP": [0.5] * 240,
        "ACC_x": [0.0] * 240,
        "ACC_y": [0.0] * 240,
        "ACC_z": [1.0] * 240,
    }
    
    cav_raw, cav_smooth, state, parts = engine.cav_from_window(
        window=window,
        temp_c=22.0,
        humidity=50.0,
        aqi=35,
        local_hour=14
    )
    
    print(f"[OK] Direct engine computation successful")
    print(f"  State: {state}")
    print(f"  CAV Smooth: {cav_smooth}")
    print(f"  P-Stress: {parts['p_stress']:.3f}")
    
except Exception as e:
    print(f"[FAIL] Engine test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Tests complete!")
print("=" * 60)

