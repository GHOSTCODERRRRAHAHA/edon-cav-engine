#!/usr/bin/env python3
"""Simple gRPC server test."""

import sys
import time
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "integrations" / "grpc" / "edon_grpc_service"))

print("Starting gRPC server test...")
print("=" * 60)

# Start server in background
import subprocess
server = subprocess.Popen(
    [sys.executable, "integrations/grpc/edon_grpc_service/edon_grpc_server.py", "--port", "50051"],
    cwd=str(project_root),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print(f"Server started (PID: {server.pid})")
print("Waiting 3 seconds for server to initialize...")
time.sleep(3)

# Check if server is still running
if server.poll() is not None:
    stdout, stderr = server.communicate()
    print(f"[FAIL] Server exited with code {server.returncode}")
    print(f"STDOUT: {stdout.decode()[:500]}")
    print(f"STDERR: {stderr.decode()[:500]}")
    sys.exit(1)

print("[OK] Server is running")

# Test client
try:
    from sdk.python.edon_sdk.client import EdonClient, TransportType
    
    client = EdonClient(
        transport=TransportType.GRPC,
        grpc_host="localhost",
        grpc_port=50051
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
    
    print("\nTesting gRPC call...")
    result = client.cav(window)
    print(f"[OK] gRPC call successful!")
    print(f"     State: {result['state']}")
    print(f"     CAV: {result['cav_smooth']}")
    print(f"     P-Stress: {result['parts']['p_stress']:.3f}")
    
    client.close()
    
except Exception as e:
    print(f"[FAIL] Client test failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\nStopping server...")
    server.terminate()
    server.wait(timeout=5)
    print("Done!")

