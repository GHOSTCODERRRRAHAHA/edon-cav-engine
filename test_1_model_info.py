"""Test 1: Model Info"""
import requests
import json
import os

print("\n=== TEST 1: Model Info ===\n")

try:
    r = requests.get("http://127.0.0.1:8001/models/info", timeout=5)
    if r.status_code == 200:
        info = r.json()
        print(f"Model Name: {info.get('name')}")
        print(f"Model Hash: {info.get('sha256', 'N/A')}")
        print(f"Features: {info.get('features')}")
        print(f"Window: {info.get('window')}")
        print(f"PCA Dims: {info.get('pca_dim')}")
        
        # Save to reports
        os.makedirs("reports", exist_ok=True)
        with open("reports/model_info.json", "w") as f:
            json.dump(info, f, indent=2)
        print("\n[OK] Model info saved to reports/model_info.json")
        exit(0)
    else:
        print(f"[FAIL] Status {r.status_code}: {r.text[:200]}")
        exit(1)
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

