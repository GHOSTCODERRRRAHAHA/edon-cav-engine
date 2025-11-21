"""Run validation tests one by one with user confirmation."""
import subprocess
import sys
import requests
import time

def check_server():
    """Check if server is running."""
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if r.status_code == 200:
            print("[OK] Server is running\n")
            return True
    except:
        pass
    print("[FAIL] Server is not running")
    print("Please start the server first:")
    print("  .\\venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000\n")
    return False

def run_test(test_name, script_name):
    """Run a single test."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"{'='*60}\n")
    
    result = subprocess.run([sys.executable, script_name])
    return result.returncode == 0

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EDON Pilot Validation Tests - One by One")
    print("="*60)
    
    if not check_server():
        sys.exit(1)
    
    tests = [
        ("Test 1: Model Info", "test_1_model_info.py"),
        ("Test 2: Evaluation (WESAD)", "test_2_evaluation.py"),
        ("Test 3: Load Test", "test_3_load_test.py"),
    ]
    
    results = []
    
    for test_name, script_name in tests:
        input(f"\nPress Enter to run {test_name}...")
        passed = run_test(test_name, script_name)
        results.append((test_name, passed))
        
        if passed:
            print(f"\n✓ {test_name} PASSED")
        else:
            print(f"\n✗ {test_name} FAILED")
            response = input("Continue to next test? (y/n): ")
            if response.lower() != 'y':
                break
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)

