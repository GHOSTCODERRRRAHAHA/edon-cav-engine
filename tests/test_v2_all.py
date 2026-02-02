"""Comprehensive test suite for v2 REST, gRPC, and WebSocket."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import time
import math
import random
import json
import asyncio
import subprocess
import signal
from typing import Optional
import pytest

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("WARNING: websockets not installed, skipping WebSocket tests")

try:
    from edon import EdonClient, TransportType
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("WARNING: EDON SDK not installed, skipping SDK tests")


def create_test_window(eda_base=0.25, motion_level=0.1):
    """Create a test v2 window."""
    return {
        "physio": {
            "EDA": [eda_base] * 240,
            "BVP": [0.5 + 0.1 * math.sin(i / 20) for i in range(240)]
        },
        "motion": {
            "ACC_x": [random.gauss(0, motion_level) for _ in range(240)],
            "ACC_y": [random.gauss(0, motion_level) for _ in range(240)],
            "ACC_z": [1.0 + random.gauss(0, motion_level * 0.5) for _ in range(240)]
        },
        "env": {
            "temp_c": 22.0,
            "humidity": 45.0,
            "aqi": 20
        },
        "task": {
            "id": "test",
            "complexity": 0.5
        }
    }


def test_v2_rest_batch():
    """Test v2 REST batch endpoint."""
    print("\n" + "=" * 70)
    print("Test 1: v2 REST Batch")
    print("=" * 70)
    
    if not SDK_AVAILABLE:
        pytest.skip("SDK not available")
    
    try:
        import requests
        
        window = create_test_window()
        url = "http://127.0.0.1:8002/v2/oem/cav/batch"
        
        response = requests.post(url, json={"windows": [window]}, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        assert "results" in data, "Response should have 'results'"
        assert len(data["results"]) == 1, "Should have one result"
        
        result = data["results"][0]
        assert result["ok"] == True, "Result should be ok"
        assert result["state_class"] in ["restorative", "focus", "balanced", "alert", "overload", "emergency"]
        assert len(result["cav_vector"]) == 128, "CAV vector should be 128-dim"
        
        print(f"  ✓ State: {result['state_class']}")
        print(f"  ✓ p_stress: {result['p_stress']:.3f}")
        print(f"  ✓ p_chaos: {result['p_chaos']:.3f}")
        print(f"  ✓ Latency: {data.get('latency_ms', 0):.2f}ms")
        
        return None
        
    except requests.exceptions.ConnectionError:
        pytest.skip("REST server not running on port 8002")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"REST v2 batch error: {e}"


def test_v2_grpc_batch():
    """Test v2 gRPC batch endpoint."""
    print("\n" + "=" * 70)
    print("Test 2: v2 gRPC Batch")
    print("=" * 70)
    
    if not SDK_AVAILABLE:
        pytest.skip("SDK not available")
    
    try:
        client = EdonClient(
            transport=TransportType.GRPC,
            grpc_host="127.0.0.1",
            grpc_port=50052,
            grpc_version="v2"
        )
        
        # Test health
        health = client.health()
        print(f"  Health: {health.get('mode', 'unknown')} mode")
        
        if not health.get("ok"):
            pytest.skip("gRPC server not healthy")
        
        # Test batch
        window = create_test_window()
        response = client.cav_batch_v2_grpc(windows=[window], device_profile="humanoid_full")
        
        assert "results" in response, "Response should have 'results'"
        assert len(response["results"]) == 1, "Should have one result"
        
        result = response["results"][0]
        assert result["ok"] == True, "Result should be ok"
        assert result["state_class"] in ["restorative", "focus", "balanced", "alert", "overload", "emergency"]
        assert len(result["cav_vector"]) == 128, "CAV vector should be 128-dim"
        
        print(f"  ✓ State: {result['state_class']}")
        print(f"  ✓ p_stress: {result['p_stress']:.3f}")
        print(f"  ✓ p_chaos: {result['p_chaos']:.3f}")
        print(f"  ✓ Latency: {response.get('latency_ms', 0):.2f}ms")
        
        return None
        
    except Exception as e:
        error_str = str(e)
        if "Connection refused" in error_str or "Failed to connect" in error_str:
            pytest.skip("gRPC server not running on port 50052")
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"gRPC v2 batch error: {e}"


def test_v2_rest_vs_grpc():
    """Test that REST and gRPC return same results."""
    print("\n" + "=" * 70)
    print("Test 3: REST vs gRPC Consistency")
    print("=" * 70)
    
    if not SDK_AVAILABLE:
        pytest.skip("SDK not available")
    
    try:
        import requests
        
        window = create_test_window()
        
        # REST
        try:
            rest_url = "http://127.0.0.1:8002/v2/oem/cav/batch"
            rest_response = requests.post(rest_url, json={"windows": [window]}, timeout=5)
            rest_response.raise_for_status()
            rest_data = rest_response.json()
            rest_result = rest_data["results"][0]
        except requests.exceptions.ConnectionError:
            pytest.skip("REST server not running")
        
        # gRPC
        try:
            grpc_client = EdonClient(
                transport=TransportType.GRPC,
                grpc_host="127.0.0.1",
                grpc_port=50052,
                grpc_version="v2"
            )
            grpc_response = grpc_client.cav_batch_v2_grpc(windows=[window])
            grpc_result = grpc_response["results"][0]
        except Exception as e:
            if "Connection refused" in str(e) or "Failed to connect" in str(e):
                pytest.skip("gRPC server not running")
            raise
        
        # Compare
        assert rest_result["state_class"] == grpc_result["state_class"], \
            f"State mismatch: REST={rest_result['state_class']}, gRPC={grpc_result['state_class']}"
        
        assert abs(rest_result["p_stress"] - grpc_result["p_stress"]) < 0.01, \
            f"p_stress mismatch: REST={rest_result['p_stress']}, gRPC={grpc_result['p_stress']}"
        
        assert abs(rest_result["p_chaos"] - grpc_result["p_chaos"]) < 0.01, \
            f"p_chaos mismatch: REST={rest_result['p_chaos']}, gRPC={grpc_result['p_chaos']}"
        
        print(f"  ✓ State matches: {rest_result['state_class']}")
        print(f"  ✓ p_stress matches: {rest_result['p_stress']:.3f}")
        print(f"  ✓ p_chaos matches: {rest_result['p_chaos']:.3f}")
        
        return None
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"REST vs gRPC error: {e}"


@pytest.mark.asyncio
async def test_v2_websocket_stream():
    """Test v2 WebSocket streaming."""
    print("\n" + "=" * 70)
    print("Test 4: v2 WebSocket Streaming")
    print("=" * 70)
    
    if not WEBSOCKETS_AVAILABLE:
        pytest.skip("websockets library not installed")
    
    try:
        uri = "ws://127.0.0.1:8002/v2/stream/cav"
        
        async with websockets.connect(uri, timeout=5) as websocket:
            print("  ✓ Connected to WebSocket")
            
            # Send a few windows
            for i in range(3):
                window = create_test_window(eda_base=0.2 + i * 0.1)
                window["task"]["id"] = f"test_{i}"
                
                await websocket.send(json.dumps(window))
                response = await websocket.recv()
                result = json.loads(response)
                
                if result.get("ok"):
                    print(f"  ✓ Window {i+1}: State={result['state_class']}, Stress={result['p_stress']:.3f}")
                else:
                    print(f"  ✗ Window {i+1} error: {result.get('error')}")
                    assert False, f"Window {i+1} error: {result.get('error')}"
                
                await asyncio.sleep(0.1)
            
            return None
            
    except websockets.exceptions.InvalidStatusCode as e:
        if e.status_code == 404:
            pytest.skip("WebSocket endpoint not available (server not in v2 mode?)")
        assert False, f"WebSocket connection error: {e}"
    except (ConnectionRefusedError, OSError) as e:
        pytest.skip("REST server not running on port 8002")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"WebSocket error: {e}"


def test_v2_grpc_streaming():
    """Test v2 gRPC bidirectional streaming."""
    print("\n" + "=" * 70)
    print("Test 5: v2 gRPC Bidirectional Streaming")
    print("=" * 70)
    
    if not SDK_AVAILABLE:
        pytest.skip("SDK not available")
    
    try:
        client = EdonClient(
            transport=TransportType.GRPC,
            grpc_host="127.0.0.1",
            grpc_port=50052,
            grpc_version="v2"
        )
        
        # Create multiple windows
        windows = [
            create_test_window(eda_base=0.1, motion_level=0.05),
            create_test_window(eda_base=0.3, motion_level=0.15),
            create_test_window(eda_base=0.5, motion_level=0.25)
        ]
        
        # Stream processing
        results = []
        for result in client.stream_v2_grpc(windows):
            results.append(result)
            if result["ok"]:
                print(f"  ✓ State: {result['state_class']}, Stress: {result['p_stress']:.3f}")
            else:
                print(f"  ✗ Error: {result.get('error')}")
                assert False, f"Streaming error: {result.get('error')}"
        
        assert len(results) == len(windows), f"Expected {len(windows)} results, got {len(results)}"
        print(f"  ✓ Processed {len(results)} windows")
        
        return None
        
    except Exception as e:
        error_str = str(e)
        if "Connection refused" in error_str or "Failed to connect" in error_str:
            pytest.skip("gRPC server not running on port 50052")
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"gRPC streaming error: {e}"


def main():
    """Run all tests."""
    print("=" * 70)
    print("EDON v2 Comprehensive Test Suite")
    print("=" * 70)
    print()
    print("Prerequisites:")
    print("  1. REST server: python -m uvicorn app.main:app --port 8002 (EDON_MODE=v2)")
    print("  2. gRPC server: python -m integrations.grpc.edon_v2_service.server --port 50052")
    print()
    
    results = []
    
    # Test 1: REST batch
    results.append(("v2 REST Batch", test_v2_rest_batch()))
    
    # Test 2: gRPC batch
    results.append(("v2 gRPC Batch", test_v2_grpc_batch()))
    
    # Test 3: REST vs gRPC consistency
    results.append(("REST vs gRPC", test_v2_rest_vs_grpc()))
    
    # Test 4: WebSocket streaming
    if WEBSOCKETS_AVAILABLE:
        results.append(("v2 WebSocket Stream", asyncio.run(test_v2_websocket_stream())))
    else:
        results.append(("v2 WebSocket Stream", None))
    
    # Test 5: gRPC streaming
    results.append(("v2 gRPC Stream", test_v2_grpc_streaming()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results:
        if result is True:
            status = "✓ PASS"
            passed += 1
        elif result is False:
            status = "✗ FAIL"
            failed += 1
        else:
            status = "⚠ SKIP"
            skipped += 1
        print(f"  {status}: {name}")
    
    print()
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print()
    
    if failed > 0:
        print("✗ Some tests failed")
        sys.exit(1)
    elif passed == 0:
        print("⚠ All tests skipped (servers not running?)")
        print("  Start servers and run again")
        sys.exit(0)
    else:
        print("✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

