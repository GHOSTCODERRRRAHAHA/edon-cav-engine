#!/usr/bin/env python3
"""Quick test summary for v2 functionality."""

import sys
import requests
import json

print("="*70)
print("EDON v2 Test Summary")
print("="*70)

# Test 1: REST Health
print("\n1. Testing REST Health...")
try:
    r = requests.get("http://127.0.0.1:8002/health", timeout=2)
    if r.status_code == 200:
        health = r.json()
        print(f"   ✓ REST server running (mode: {health.get('mode')})")
        print(f"   ✓ Neural loaded: {health.get('neural_loaded')}")
        print(f"   ✓ PCA loaded: {health.get('pca_loaded')}")
    else:
        print(f"   ✗ REST server returned {r.status_code}")
except Exception as e:
    print(f"   ⚠ REST server not running: {e}")

# Test 2: REST Batch
print("\n2. Testing REST Batch...")
try:
    window = {
        "physio": {"EDA": [0.25]*240, "BVP": [0.5]*240},
        "motion": {"ACC_x": [0.0]*240, "ACC_y": [0.0]*240, "ACC_z": [1.0]*240},
        "env": {"temp_c": 22.0, "humidity": 45.0, "aqi": 20},
        "task": {"id": "test", "complexity": 0.5}
    }
    r = requests.post("http://127.0.0.1:8002/v2/oem/cav/batch", json={"windows": [window]}, timeout=5)
    if r.status_code == 200:
        result = r.json()["results"][0]
        print(f"   ✓ State: {result['state_class']}")
        print(f"   ✓ p_stress: {result['p_stress']:.3f}")
        print(f"   ✓ p_chaos: {result['p_chaos']:.3f}")
        print(f"   ✓ CAV vector length: {len(result['cav_vector'])}")
    else:
        print(f"   ✗ Batch returned {r.status_code}")
except Exception as e:
    print(f"   ⚠ Batch test failed: {e}")

# Test 3: SDK REST
print("\n3. Testing Python SDK (REST)...")
try:
    sys.path.insert(0, "sdk/python")
    from edon import EdonClient
    
    client = EdonClient(base_url="http://127.0.0.1:8002")
    window = {
        "physio": {"EDA": [0.25]*240, "BVP": [0.5]*240},
        "motion": {"ACC_x": [0.0]*240, "ACC_y": [0.0]*240, "ACC_z": [1.0]*240},
        "env": {"temp_c": 22.0, "humidity": 45.0, "aqi": 20},
        "task": {"id": "test", "complexity": 0.5}
    }
    resp = client.cav_batch_v2(windows=[window])
    result = resp["results"][0]
    print(f"   ✓ SDK REST working")
    print(f"   ✓ State: {result['state_class']}")
except Exception as e:
    print(f"   ⚠ SDK REST test failed: {e}")

# Test 4: gRPC Server Check
print("\n4. Testing gRPC Server...")
try:
    import grpc
    sys.path.insert(0, "integrations/grpc/edon_v2_service")
    from integrations.grpc.edon_v2_service import edon_v2_pb2_grpc
    
    channel = grpc.insecure_channel("127.0.0.1:50052")
    stub = edon_v2_pb2_grpc.EdonV2ServiceStub(channel)
    
    # Try health check
    from integrations.grpc.edon_v2_service import edon_v2_pb2
    req = edon_v2_pb2.HealthRequest()
    resp = stub.Health(req, timeout=2)
    print(f"   ✓ gRPC server running (mode: {resp.mode})")
    print(f"   ✓ Neural loaded: {resp.neural_loaded}")
    print(f"   ✓ PCA loaded: {resp.pca_loaded}")
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.UNAVAILABLE:
        print("   ⚠ gRPC server not running on port 50052")
    else:
        print(f"   ✗ gRPC error: {e}")
except Exception as e:
    print(f"   ⚠ gRPC test failed: {e}")

# Test 5: SDK gRPC
print("\n5. Testing Python SDK (gRPC)...")
try:
    sys.path.insert(0, "sdk/python")
    from edon import EdonClient, TransportType
    
    client = EdonClient(
        transport=TransportType.GRPC,
        grpc_port=50052,
        grpc_version="v2"
    )
    health = client.health()
    if health.get("ok"):
        print(f"   ✓ SDK gRPC working (mode: {health.get('mode')})")
    else:
        print(f"   ⚠ SDK gRPC health check failed")
except Exception as e:
    print(f"   ⚠ SDK gRPC test failed: {e}")

# Test 6: WebSocket
print("\n6. Testing WebSocket...")
try:
    import asyncio
    import websockets
    
    async def test_ws():
        uri = "ws://127.0.0.1:8002/v2/stream/cav"
        async with websockets.connect(uri, timeout=2) as ws:
            window = {
                "physio": {"EDA": [0.25]*240, "BVP": [0.5]*240},
                "motion": {"ACC_x": [0.0]*240, "ACC_y": [0.0]*240, "ACC_z": [1.0]*240},
                "env": {"temp_c": 22.0, "humidity": 45.0, "aqi": 20},
                "task": {"id": "test", "complexity": 0.5}
            }
            await ws.send(json.dumps(window))
            result = json.loads(await ws.recv())
            if result.get("ok"):
                print(f"   ✓ WebSocket working")
                print(f"   ✓ State: {result['state_class']}")
            else:
                print(f"   ✗ WebSocket error: {result.get('error')}")
    
    asyncio.run(test_ws())
except ImportError:
    print("   ⚠ websockets library not installed")
except Exception as e:
    print(f"   ⚠ WebSocket test failed: {e}")

print("\n" + "="*70)
print("Test Summary Complete")
print("="*70)
print("\nNote: Some tests may be skipped if servers are not running.")
print("To start servers:")
print("  REST: python -m uvicorn app.main:app --port 8002 (EDON_MODE=v2)")
print("  gRPC: python -m integrations.grpc.edon_v2_service.server --port 50052")

