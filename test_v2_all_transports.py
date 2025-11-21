#!/usr/bin/env python3
"""Comprehensive test for v2 REST, gRPC, and WebSocket streaming."""

import sys
import json
import time
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import requests

# Try to import SDK
try:
    sys.path.insert(0, "sdk/python")
    from edon import EdonClient, TransportType
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("[WARNING] SDK not available, skipping SDK tests")

# Try to import websockets
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("[WARNING] websockets not installed, skipping WebSocket tests")

REST_URL = "http://127.0.0.1:8001"
GRPC_PORT = 50052


def create_test_window(eda_base=0.25, motion_level=0.1, task_id="test"):
    """Create a test v2 window."""
    return {
        "physio": {
            "EDA": [eda_base] * 240,
            "BVP": [0.5] * 240
        },
        "motion": {
            "ACC_x": [0.0] * 240,
            "ACC_y": [0.0] * 240,
            "ACC_z": [1.0] * 240
        },
        "env": {
            "temp_c": 22.0,
            "humidity": 45.0,
            "aqi": 20
        },
        "task": {
            "id": task_id,
            "complexity": 0.5
        }
    }


def test_rest():
    """Test REST API."""
    print("\n" + "=" * 70)
    print("TEST 1: REST API")
    print("=" * 70)
    
    try:
        # Health check
        print("\n1.1 Health Check...")
        health = requests.get(f"{REST_URL}/health", timeout=2).json()
        print(f"   Mode: {health.get('mode')}")
        print(f"   Engine: {health.get('engine')}")
        print(f"   Neural Loaded: {health.get('neural_loaded')}")
        print(f"   PCA Loaded: {health.get('pca_loaded')}")
        
        if health.get('mode') != 'v2':
            print("   [ERROR] Server not in v2 mode!")
            return False
        
        # Batch request
        print("\n1.2 Batch Request...")
        window = create_test_window()
        start_time = time.time()
        response = requests.post(
            f"{REST_URL}/v2/oem/cav/batch",
            json={"windows": [window]},
            timeout=5
        )
        response.raise_for_status()
        result = response.json()
        latency = (time.time() - start_time) * 1000
        
        res = result["results"][0]
        print(f"   [OK] Success!")
        print(f"   State: {res['state_class']}")
        print(f"   Stress: {res['p_stress']:.3f}")
        print(f"   Chaos: {res['p_chaos']:.3f}")
        print(f"   CAV Vector: {len(res['cav_vector'])} dimensions")
        print(f"   Latency: {latency:.2f} ms")
        
        return {
            "ok": True,
            "state": res['state_class'],
            "stress": res['p_stress'],
            "chaos": res['p_chaos'],
            "latency": latency
        }
        
    except requests.exceptions.ConnectionError:
        print("   [SKIP] REST server not running")
        return None
    except Exception as e:
        print(f"   [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grpc():
    """Test gRPC API."""
    print("\n" + "=" * 70)
    print("TEST 2: gRPC API")
    print("=" * 70)
    
    if not SDK_AVAILABLE:
        print("\n[SKIP] SDK not available")
        return None
    
    try:
        # Create client
        print("\n2.1 Creating gRPC client...")
        client = EdonClient(
            transport=TransportType.GRPC,
            grpc_host="127.0.0.1",
            grpc_port=GRPC_PORT,
            grpc_version="v2"
        )
        print("   [OK] Client created")
        
        # Health check
        print("\n2.2 Health Check...")
        health = client.health()
        if not health.get("ok"):
            print(f"   [ERROR] Health check failed: {health.get('error')}")
            return False
        
        print(f"   Mode: {health.get('mode')}")
        print(f"   Engine: {health.get('engine')}")
        print(f"   Neural Loaded: {health.get('neural_loaded')}")
        print(f"   PCA Loaded: {health.get('pca_loaded')}")
        
        # Batch request
        print("\n2.3 Batch Request...")
        window = create_test_window()
        start_time = time.time()
        response = client.cav_batch_v2_grpc(windows=[window])
        latency = (time.time() - start_time) * 1000
        
        result = response["results"][0]
        if not result["ok"]:
            print(f"   [ERROR] {result.get('error')}")
            return False
        
        print(f"   [OK] Success!")
        print(f"   State: {result['state_class']}")
        print(f"   Stress: {result['p_stress']:.3f}")
        print(f"   Chaos: {result['p_chaos']:.3f}")
        print(f"   CAV Vector: {len(result['cav_vector'])} dimensions")
        print(f"   Latency: {latency:.2f} ms")
        
        return {
            "ok": True,
            "state": result['state_class'],
            "stress": result['p_stress'],
            "chaos": result['p_chaos'],
            "latency": latency
        }
        
    except Exception as e:
        error_str = str(e)
        if "Connection refused" in error_str or "Failed to connect" in error_str:
            print(f"   [SKIP] gRPC server not running on port {GRPC_PORT}")
            return None
        print(f"   [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grpc_streaming():
    """Test gRPC bidirectional streaming."""
    print("\n" + "=" * 70)
    print("TEST 3: gRPC Bidirectional Streaming")
    print("=" * 70)
    
    if not SDK_AVAILABLE:
        print("\n[SKIP] SDK not available")
        return None
    
    try:
        print("\n3.1 Creating gRPC client...")
        client = EdonClient(
            transport=TransportType.GRPC,
            grpc_host="127.0.0.1",
            grpc_port=GRPC_PORT,
            grpc_version="v2"
        )
        
        # Create multiple windows
        print("\n3.2 Streaming 3 windows...")
        windows = [
            create_test_window(eda_base=0.1, motion_level=0.05, task_id="rest"),
            create_test_window(eda_base=0.3, motion_level=0.15, task_id="work"),
            create_test_window(eda_base=0.5, motion_level=0.25, task_id="stress")
        ]
        
        start_time = time.time()
        results = list(client.stream_v2_grpc(windows))
        total_latency = (time.time() - start_time) * 1000
        
        if len(results) != len(windows):
            print(f"   [ERROR] Expected {len(windows)} results, got {len(results)}")
            return False
        
        print(f"   [OK] Processed {len(results)} windows in {total_latency:.2f} ms")
        for i, result in enumerate(results):
            if result["ok"]:
                print(f"   Window {i+1}: State={result['state_class']}, "
                      f"Stress={result['p_stress']:.3f}, Chaos={result['p_chaos']:.3f}")
            else:
                print(f"   Window {i+1}: [ERROR] {result.get('error')}")
                return False
        
        return True
        
    except Exception as e:
        error_str = str(e)
        if "Connection refused" in error_str or "Failed to connect" in error_str:
            print(f"   [SKIP] gRPC server not running on port {GRPC_PORT}")
            return None
        print(f"   [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_websocket():
    """Test WebSocket streaming."""
    print("\n" + "=" * 70)
    print("TEST 4: WebSocket Streaming")
    print("=" * 70)
    
    if not WEBSOCKETS_AVAILABLE:
        print("\n[SKIP] websockets library not installed")
        return None
    
    try:
        uri = f"ws://127.0.0.1:8001/v2/stream/cav"
        print(f"\n4.1 Connecting to {uri}...")
        
        async with websockets.connect(uri, open_timeout=5) as websocket:
            print("   [OK] Connected")
            
            # Send 3 windows
            print("\n4.2 Streaming 3 windows...")
            windows = [
                create_test_window(eda_base=0.1, task_id="rest"),
                create_test_window(eda_base=0.3, task_id="work"),
                create_test_window(eda_base=0.5, task_id="stress")
            ]
            
            start_time = time.time()
            for i, window in enumerate(windows):
                await websocket.send(json.dumps(window))
                response = await websocket.recv()
                result = json.loads(response)
                
                if result.get("ok"):
                    print(f"   Window {i+1}: State={result['state_class']}, "
                          f"Stress={result['p_stress']:.3f}, Chaos={result['p_chaos']:.3f}")
                else:
                    print(f"   Window {i+1}: [ERROR] {result.get('error')}")
                    return False
            
            total_latency = (time.time() - start_time) * 1000
            print(f"   [OK] Processed 3 windows in {total_latency:.2f} ms")
            return True
            
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "Not Found" in error_str:
            print("   [SKIP] WebSocket endpoint not available (server not in v2 mode?)")
            return None
        if "Connection refused" in error_str or "Failed to connect" in error_str:
            print(f"   [SKIP] REST server not running on port 8001")
            return None
        # Check for InvalidStatusCode in a different way
        if hasattr(e, 'status_code') and e.status_code == 404:
            print("   [SKIP] WebSocket endpoint not available (server not in v2 mode?)")
            return None
        print(f"   [ERROR] Connection error: {e}")
        return False
    except (ConnectionRefusedError, OSError) as e:
        print(f"   [SKIP] REST server not running on port 8001")
        return None
    except Exception as e:
        print(f"   [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_rest_vs_grpc(rest_result, grpc_result):
    """Compare REST and gRPC results for consistency."""
    print("\n" + "=" * 70)
    print("TEST 5: REST vs gRPC Consistency")
    print("=" * 70)
    
    if rest_result is None or grpc_result is None:
        print("\n[SKIP] Both REST and gRPC must be available for comparison")
        return None
    
    print("\n5.1 Comparing results...")
    
    state_match = rest_result["state"] == grpc_result["state"]
    stress_diff = abs(rest_result["stress"] - grpc_result["stress"])
    chaos_diff = abs(rest_result["chaos"] - grpc_result["chaos"])
    
    print(f"   REST:  State={rest_result['state']}, Stress={rest_result['stress']:.3f}, "
          f"Chaos={rest_result['chaos']:.3f}")
    print(f"   gRPC:  State={grpc_result['state']}, Stress={grpc_result['stress']:.3f}, "
          f"Chaos={grpc_result['chaos']:.3f}")
    
    if state_match and stress_diff < 0.01 and chaos_diff < 0.01:
        print(f"   [OK] Results match!")
        print(f"   State match: {state_match}")
        print(f"   Stress diff: {stress_diff:.6f} (< 0.01)")
        print(f"   Chaos diff: {chaos_diff:.6f} (< 0.01)")
        return True
    else:
        print(f"   [ERROR] Results don't match!")
        print(f"   State match: {state_match}")
        print(f"   Stress diff: {stress_diff:.6f}")
        print(f"   Chaos diff: {chaos_diff:.6f}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("EDON v2 Comprehensive Transport Test")
    print("=" * 70)
    print()
    print("Prerequisites:")
    print("  1. REST server: python -m uvicorn app.main:app --port 8001 (EDON_MODE=v2)")
    print("  2. gRPC server: python -m integrations.grpc.edon_v2_service.server --port 50052")
    print()
    
    results = {}
    
    # Test 1: REST
    results["rest"] = test_rest()
    
    # Test 2: gRPC
    results["grpc"] = test_grpc()
    
    # Test 3: gRPC Streaming
    results["grpc_stream"] = test_grpc_streaming()
    
    # Test 4: WebSocket
    if WEBSOCKETS_AVAILABLE:
        results["websocket"] = asyncio.run(test_websocket())
    else:
        results["websocket"] = None
    
    # Test 5: Compare REST vs gRPC
    results["comparison"] = compare_rest_vs_grpc(results["rest"], results["grpc"])
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    failed = 0
    skipped = 0
    
    test_names = {
        "rest": "REST API",
        "grpc": "gRPC API",
        "grpc_stream": "gRPC Streaming",
        "websocket": "WebSocket Streaming",
        "comparison": "REST vs gRPC Consistency"
    }
    
    for key, result in results.items():
        name = test_names.get(key, key)
        if result is True or (isinstance(result, dict) and result.get("ok")):
            status = "[OK] PASS"
            passed += 1
        elif result is False:
            status = "[ERROR] FAIL"
            failed += 1
        else:
            status = "[SKIP]"
            skipped += 1
        print(f"  {status}: {name}")
    
    print()
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print()
    
    if failed > 0:
        print("[ERROR] Some tests failed")
        sys.exit(1)
    elif passed == 0:
        print("[WARNING] All tests skipped (servers not running?)")
        print("  Start servers and run again")
        sys.exit(0)
    else:
        print("[OK] All available tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

