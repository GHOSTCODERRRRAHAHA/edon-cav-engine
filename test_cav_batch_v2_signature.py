#!/usr/bin/env python3
"""Test script to verify cav_batch_v2 signature works correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sdk', 'python'))

from edon import EdonClient
import math
import random

# Test the signature
client = EdonClient("http://127.0.0.1:8002")

window = {
    "physio": {
        "EDA": [0.25 for _ in range(240)],
        "BVP": [math.sin(i/6) for i in range(240)],
    },
    "motion": {
        "ACC_x": [random.gauss(0, 1) for _ in range(240)],
        "ACC_y": [random.gauss(0, 1) for _ in range(240)],
        "ACC_z": [random.gauss(0, 1) for _ in range(240)],
    },
    "env": {
        "temp_c": 22.0,
        "humidity": 45.0,
        "aqi": 15,
    },
    "task": {
        "id": "walking",
        "difficulty": 0.5,
    },
}

print("Testing cav_batch_v2 signature...")
print()

# Test 1: New ergonomic signature
print("Test 1: New ergonomic signature")
try:
    res1 = client.cav_batch_v2(windows=[window], device_profile="humanoid_full")
    print("✓ Test 1 passed: New signature works")
    print(f"  Response keys: {list(res1.keys())}")
except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Backwards compatible raw payload
print("Test 2: Backwards compatible raw payload")
try:
    res2 = client.cav_batch_v2({"windows": [window]})
    print("✓ Test 2 passed: Backwards compatible signature works")
    print(f"  Response keys: {list(res2.keys())}")
except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("All tests completed!")

