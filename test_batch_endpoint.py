"""Quick test of batch endpoint."""
import requests
import json
import math

def make_window(seed=0):
    return {
        "EDA": [k * 0.01 for k in range(240)],
        "TEMP": [36.5 for _ in range(240)],
        "BVP": [math.sin((k + seed) / 12.0) for k in range(240)],
        "ACC_x": [0.0] * 240,
        "ACC_y": [0.0] * 240,
        "ACC_z": [1.0] * 240,
        "temp_c": 22.0,
        "humidity": 50.0,
        "aqi": 35,
        "local_hour": 14
    }

payload = {
    "windows": [make_window(i) for i in range(3)]
}

try:
    print("Testing /oem/cav/batch endpoint...")
    r = requests.post("http://127.0.0.1:8000/oem/cav/batch", json=payload, timeout=10)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Results: {len(data.get('results', []))} windows")
        print(f"Latency: {data.get('latency_ms', 0):.2f}ms")
        if data.get('results'):
            first = data['results'][0]
            print(f"First result - OK: {first.get('ok')}, State: {first.get('state')}, CAV: {first.get('cav_smooth')}")
    else:
        print(f"Error: {r.text[:500]}")
except Exception as e:
    print(f"Exception: {e}")

