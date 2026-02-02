"""Integration tests for v2 gRPC API."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import time
import math
import random
from edon import EdonClient, TransportType
import pytest


def create_test_window():
    """Create a test v2 window."""
    return {
        "physio": {
            "EDA": [0.25] * 240,
            "BVP": [0.5 + 0.1 * math.sin(i / 20) for i in range(240)]
        },
        "motion": {
            "ACC_x": [random.gauss(0, 0.1) for _ in range(240)],
            "ACC_y": [random.gauss(0, 0.1) for _ in range(240)],
            "ACC_z": [1.0 + random.gauss(0, 0.05) for _ in range(240)]
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


def test_v2_grpc_batch():
    """Test v2 gRPC batch computation."""
    print("Testing v2 gRPC batch...")
    
    try:
        # Create v2 gRPC client
        client = EdonClient(
            transport=TransportType.GRPC,
            grpc_host="127.0.0.1",
            grpc_port=50052,
            grpc_version="v2"
        )
        
        # Test health
        health = client.health()
        print(f"  Health: {health}")
        assert health.get("ok") == True, "Health check failed"
        assert health.get("mode") == "v2", "Should be v2 mode"
        
        # Test batch
        window = create_test_window()
        response = client.cav_batch_v2_grpc(windows=[window], device_profile="humanoid_full")
        
        print(f"  Response keys: {list(response.keys())}")
        assert "results" in response, "Response should have 'results'"
        assert len(response["results"]) == 1, "Should have one result"
        
        result = response["results"][0]
        assert result["ok"] == True, "Result should be ok"
        assert result["state_class"] in ["restorative", "focus", "balanced", "alert", "overload", "emergency"]
        assert len(result["cav_vector"]) == 128, "CAV vector should be 128-dim"
        
        print(f"  ✓ State: {result['state_class']}")
        print(f"  ✓ p_stress: {result['p_stress']:.3f}")
        print(f"  ✓ p_chaos: {result['p_chaos']:.3f}")
        
        return None
        
    except Exception as e:
        assert False, f"v2 gRPC batch error: {e}"


def test_v2_rest_vs_grpc():
    """Test that REST and gRPC return same results."""
    print("\nTesting REST vs gRPC consistency...")
    
    try:
        window = create_test_window()
        
        # REST client
        rest_client = EdonClient(base_url="http://127.0.0.1:8002")
        rest_response = rest_client.cav_batch_v2(windows=[window], device_profile="humanoid_full")
        rest_result = rest_response["results"][0]
        
        # gRPC client
        grpc_client = EdonClient(
            transport=TransportType.GRPC,
            grpc_host="127.0.0.1",
            grpc_port=50052,
            grpc_version="v2"
        )
        grpc_response = grpc_client.cav_batch_v2_grpc(windows=[window], device_profile="humanoid_full")
        grpc_result = grpc_response["results"][0]
        
        # Compare
        assert rest_result["state_class"] == grpc_result["state_class"], \
            f"State mismatch: REST={rest_result['state_class']}, gRPC={grpc_result['state_class']}"
        
        # Allow small floating point differences
        assert abs(rest_result["p_stress"] - grpc_result["p_stress"]) < 0.01, \
            f"p_stress mismatch: REST={rest_result['p_stress']}, gRPC={grpc_result['p_stress']}"
        
        assert abs(rest_result["p_chaos"] - grpc_result["p_chaos"]) < 0.01, \
            f"p_chaos mismatch: REST={rest_result['p_chaos']}, gRPC={grpc_result['p_chaos']}"
        
        print(f"  ✓ State matches: {rest_result['state_class']}")
        print(f"  ✓ p_stress matches: {rest_result['p_stress']:.3f}")
        print(f"  ✓ p_chaos matches: {rest_result['p_chaos']:.3f}")
        
        return None
        
    except Exception as e:
        assert False, f"REST vs gRPC error: {e}"


if __name__ == "__main__":
    print("=" * 70)
    print("EDON v2 gRPC Integration Tests")
    print("=" * 70)
    print()
    print("Prerequisites:")
    print("  1. Start REST server: python -m uvicorn app.main:app --port 8002")
    print("  2. Set EDON_MODE=v2")
    print("  3. Start gRPC server: python -m integrations.grpc.edon_v2_service.server --port 50052")
    print()
    
    results = []
    
    # Test 1: v2 gRPC batch
    results.append(("v2 gRPC batch", test_v2_grpc_batch()))
    
    # Test 2: REST vs gRPC consistency
    results.append(("REST vs gRPC", test_v2_rest_vs_grpc()))
    
    # Summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)

