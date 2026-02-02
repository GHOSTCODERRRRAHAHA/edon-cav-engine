#!/usr/bin/env python3
"""Test EDON v2 multimodal API endpoint."""

import requests
import json
import time
import os
import numpy as np
from typing import Dict, Any
import pytest

API_URL = os.getenv("EDON_API_URL", "http://127.0.0.1:8002")


def create_test_window(device_profile: str = None) -> Dict[str, Any]:
    """Create a test window with multimodal inputs."""
    window = {}
    
    if device_profile != "drone_nav":
        # Add physio data
        window["physio"] = {
            "EDA": [0.1 + 0.01 * i for i in range(240)],
            "TEMP": [36.5] * 240,
            "BVP": [0.5 + 0.1 * np.sin(i / 10.0) for i in range(240)],
            "ACC_x": [0.0] * 240,
            "ACC_y": [0.0] * 240,
            "ACC_z": [1.0] * 240
        }
    
    # Add motion data
    window["motion"] = {
        "velocity_magnitude": 0.5,
        "torque_mean": 25.0,
        "force_mean": 10.0
    }
    
    # Add env data
    window["env"] = {
        "temp_c": 22.0,
        "humidity": 50.0,
        "aqi": 35,
        "local_hour": 14
    }
    
    if device_profile != "wearable_limited":
        # Add vision data
        window["vision"] = {
            "embedding": [0.1] * 128,  # 128-dim embedding
            "objects": ["person", "desk", "chair"],
            "scene_type": "indoor",
            "activity_context": "sitting"
        }
        
        # Add audio data (not for drone_nav)
        if device_profile != "drone_nav":
            window["audio"] = {
                "embedding": [0.05] * 64,  # 64-dim embedding
                "keywords": ["calm", "quiet"],
                "speech_activity": 0.2,
                "emotion": "calm"
            }
    
    # Add task data
    if device_profile != "wearable_limited":
        window["task"] = {
            "goal": "operate robot",
            "confidence": 0.8,
            "priority": 5,
            "complexity": 0.4
        }
    
    # Add system data
    if device_profile == "drone_nav" or device_profile is None:
        window["system"] = {
            "cpu_usage": 0.3,
            "memory_usage": 0.4,
            "network_latency": 15.0,
            "error_rate": 0.01,
            "battery_level": 0.85
        }
    
    if device_profile:
        window["device_profile"] = device_profile
    
    return window


def test_v2_endpoint():
    """Test v2 batch endpoint."""
    print("=" * 70)
    print("Testing EDON v2 Multimodal API")
    print("=" * 70)
    
    # Test 1: Health check
    print("\n[Test 1] Health Check")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        assert response.status_code == 200
        data = response.json()
        print(f"   [OK] Service: {data.get('service')}")
        print(f"   [OK] Version: {data.get('version')}")
        
        # Check v2 endpoint is listed
        endpoints = data.get('endpoints', {})
        if 'v2' in endpoints:
            print(f"   [OK] v2 endpoint found: {endpoints['v2']}")
        else:
            print(f"   [WARN] v2 endpoint not in root response")
    except Exception as e:
        pytest.skip(f"Health check failed: {e}")
    
    # Test 2: Basic v2 request (no device profile)
    print("\n[Test 2] Basic v2 Request (All Modalities)")
    try:
        window = create_test_window()
        payload = {"windows": [window]}
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/v2/oem/cav/batch",
            json=payload,
            timeout=10
        )
        latency = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert len(data["results"]) == 1
        result = data["results"][0]
        
        assert result["ok"] is True
        assert "cav_vector" in result
        assert len(result["cav_vector"]) == 128, f"Expected 128-dim vector, got {len(result['cav_vector'])}"
        assert "state_class" in result
        assert result["state_class"] in ["restorative", "focus", "balanced", "overload", "alert", "emergency"]
        assert "influences" in result
        assert "speed_scale" in result["influences"]
        assert "torque_scale" in result["influences"]
        
        print(f"   [OK] Status: {response.status_code}")
        print(f"   [OK] Latency: {latency:.2f}ms")
        print(f"   [OK] CAV Vector: {len(result['cav_vector'])}-dim")
        print(f"   [OK] State: {result['state_class']}")
        print(f"   [OK] P-Stress: {result.get('p_stress', 'N/A'):.3f}")
        print(f"   [OK] Speed Scale: {result['influences']['speed_scale']:.2f}")
        print(f"   [OK] Confidence: {result.get('confidence', 'N/A'):.3f}")
        
        if "metadata" in result:
            meta = result["metadata"]
            print(f"   [OK] Modalities: {meta.get('modalities_present', [])}")
            print(f"   [OK] PCA Fitted: {meta.get('pca_fitted', False)}")
    except Exception as e:
        assert False, f"Basic request failed: {e}"
    
    # Test 3: Device Profile - humanoid_full
    print("\n[Test 3] Device Profile: humanoid_full")
    try:
        window = create_test_window(device_profile="humanoid_full")
        payload = {"windows": [window]}
        
        response = requests.post(
            f"{API_URL}/v2/oem/cav/batch",
            json=payload,
            timeout=10
        )
        
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]
        
        assert result["ok"] is True
        if "metadata" in result and result["metadata"].get("device_profile"):
            print(f"   [OK] Profile: {result['metadata']['device_profile']}")
        print(f"   [OK] State: {result['state_class']}")
        print(f"   [OK] CAV Vector: {len(result['cav_vector'])}-dim")
    except Exception as e:
        print(f"   [FAIL] humanoid_full test failed: {e}")
        assert False
    
    # Test 4: Device Profile - wearable_limited
    print("\n[Test 4] Device Profile: wearable_limited")
    try:
        window = create_test_window(device_profile="wearable_limited")
        payload = {"windows": [window]}
        
        response = requests.post(
            f"{API_URL}/v2/oem/cav/batch",
            json=payload,
            timeout=10
        )
        
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]
        
        assert result["ok"] is True
        print(f"   [OK] State: {result['state_class']}")
        print(f"   [OK] CAV Vector: {len(result['cav_vector'])}-dim")
    except Exception as e:
        print(f"   [FAIL] wearable_limited test failed: {e}")
        assert False
    
    # Test 5: Device Profile - drone_nav
    print("\n[Test 5] Device Profile: drone_nav")
    try:
        window = create_test_window(device_profile="drone_nav")
        payload = {"windows": [window]}
        
        response = requests.post(
            f"{API_URL}/v2/oem/cav/batch",
            json=payload,
            timeout=10
        )
        
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]
        
        assert result["ok"] is True
        print(f"   [OK] State: {result['state_class']}")
        print(f"   [OK] CAV Vector: {len(result['cav_vector'])}-dim")
    except Exception as e:
        print(f"   [FAIL] drone_nav test failed: {e}")
        assert False
    
    # Test 6: Batch request (multiple windows)
    print("\n[Test 6] Batch Request (3 windows)")
    try:
        windows = [
            create_test_window(),
            create_test_window(device_profile="wearable_limited"),
            create_test_window(device_profile="drone_nav")
        ]
        payload = {"windows": windows}
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/v2/oem/cav/batch",
            json=payload,
            timeout=15
        )
        latency = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) == 3
        all_ok = all(r["ok"] for r in data["results"])
        
        assert all_ok
        print(f"   [OK] All {len(data['results'])} windows processed")
        print(f"   [OK] Total latency: {latency:.2f}ms")
        print(f"   [OK] Avg latency: {latency/3:.2f}ms per window")
        
        for i, result in enumerate(data["results"]):
            print(f"   [OK] Window {i+1}: {result['state_class']} (CAV: {len(result['cav_vector'])}-dim)")
    except Exception as e:
        print(f"   [FAIL] Batch test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False
    
    # Test 7: Minimal request (only env)
    print("\n[Test 7] Minimal Request (env only)")
    try:
        window = {
            "env": {
                "temp_c": 22.0,
                "humidity": 50.0,
                "aqi": 35,
                "local_hour": 14
            }
        }
        # Don't set device_profile for minimal test
        payload = {"windows": [window]}
        
        response = requests.post(
            f"{API_URL}/v2/oem/cav/batch",
            json=payload,
            timeout=10
        )
        
        assert response.status_code == 200
        data = response.json()
        result = data["results"][0]
        
        assert result["ok"] is True
        print(f"   [OK] Minimal request processed")
        print(f"   [OK] State: {result['state_class']}")
    except Exception as e:
        print(f"   [FAIL] Minimal request failed: {e}")
        import traceback
        traceback.print_exc()
        assert False
    
    print("\n" + "=" * 70)
    print("All v2 tests passed! [SUCCESS]")
    print("=" * 70)
    return None


if __name__ == "__main__":
    import sys
    
    try:
        success = test_v2_endpoint()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

