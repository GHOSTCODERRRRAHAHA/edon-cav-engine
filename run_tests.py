"""Run validation tests programmatically."""
import subprocess
import sys
import time
import requests
import json
import os

def check_server():
    """Check if server is running."""
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if r.status_code == 200:
            print("[OK] Server is running")
            return True
    except:
        pass
    print("[FAIL] Server is not running")
    return False

def test_model_info():
    """Test 1: Model Info"""
    print("\n=== TEST 1: Model Info ===")
    try:
        r = requests.get("http://127.0.0.1:8000/models/info", timeout=5)
        if r.status_code == 200:
            info = r.json()
            print(f"Model Name: {info.get('name')}")
            print(f"Model Hash: {info.get('sha256', 'N/A')[:16]}...")
            print(f"Features: {info.get('features')}")
            print(f"Window: {info.get('window')}")
            print(f"PCA Dims: {info.get('pca_dim')}")
            
            os.makedirs("reports", exist_ok=True)
            with open("reports/model_info.json", "w") as f:
                json.dump(info, f, indent=2)
            print("[OK] Model info saved")
            return True
        else:
            print(f"[FAIL] Status {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

def test_load_test():
    """Test 3: Load Test"""
    print("\n=== TEST 3: Load Test ===")
    try:
        result = subprocess.run(
            [sys.executable, "tools/load_test.py", 
             "--url", "http://127.0.0.1:8000/oem/cav/batch",
             "--requests", "50",
             "--windows", "3",
             "--concurrent", "5"],
            capture_output=True,
            text=True,
            timeout=60
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

if __name__ == "__main__":
    print("\n=== EDON Pilot Validation Tests ===\n")
    
    if not check_server():
        print("Please start the server first:")
        print("  .\\venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)
    
    results = []
    results.append(("Model Info", test_model_info()))
    results.append(("Load Test", test_load_test()))
    
    print("\n=== Validation Tests Complete ===\n")
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")
    
    if all(passed for _, passed in results):
        sys.exit(0)
    else:
        sys.exit(1)

