#!/usr/bin/env python3
"""Quick test for v2 API endpoint."""

import requests
import json

BASE_URL = "http://127.0.0.1:8001"

print("=" * 70)
print("EDON v2 API Quick Test")
print("=" * 70)
print()

# 1. Check health
print("1. Checking health...")
try:
    health = requests.get(f"{BASE_URL}/health", timeout=2).json()
    print(f"   Mode: {health.get('mode')}")
    print(f"   Engine: {health.get('engine')}")
    print(f"   Neural Loaded: {health.get('neural_loaded')}")
    print(f"   PCA Loaded: {health.get('pca_loaded')}")
    
    if health.get('mode') != 'v2':
        print("\n[WARNING] Server is not in v2 mode!")
        print("   Restart server with: EDON_MODE=v2")
        exit(1)
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    exit(1)

# 2. Test v2 batch endpoint
print("\n2. Testing v2 batch endpoint...")
print(f"   POST {BASE_URL}/v2/oem/cav/batch")

window = {
    "physio": {
        "EDA": [0.25] * 240,
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
        "id": "test",
        "complexity": 0.5
    }
}

body = {
    "windows": [window]
}

try:
    response = requests.post(
        f"{BASE_URL}/v2/oem/cav/batch",
        json=body,
        timeout=5
    )
    response.raise_for_status()
    result = response.json()
    
    res = result["results"][0]
    print("   [OK] Success!")
    print(f"   State: {res['state_class']}")
    print(f"   Stress: {res['p_stress']:.3f}")
    print(f"   Chaos: {res['p_chaos']:.3f}")
    print(f"   CAV Vector: {len(res['cav_vector'])} dimensions")
    print(f"   Latency: {result.get('latency_ms', 0):.2f} ms")
    print()
    print("   Influences:")
    print(f"     Speed Scale: {res['influences']['speed_scale']:.3f}")
    print(f"     Safety Scale: {res['influences']['safety_scale']:.3f}")
    print(f"     Caution Flag: {res['influences']['caution_flag']}")
    
except requests.exceptions.HTTPError as e:
    print(f"   [ERROR] HTTP Error: {e.response.status_code}")
    try:
        error_detail = e.response.json()
        print(f"   Detail: {error_detail}")
    except:
        print(f"   Response: {e.response.text}")
    exit(1)
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    exit(1)

print()
print("=" * 70)
print("[OK] All tests passed!")
print("=" * 70)

