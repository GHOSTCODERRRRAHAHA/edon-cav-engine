"""Test 3: Load Test"""
import subprocess
import sys

print("\n=== TEST 3: Load Test (50 requests, 3 windows each) ===\n")

result = subprocess.run(
    [sys.executable, "tools/load_test.py",
     "--url", "http://127.0.0.1:8001/oem/cav/batch",
     "--requests", "50",
     "--windows", "3",
     "--concurrent", "5"],
    capture_output=False,
    text=True
)

if result.returncode == 0:
    print("\n[OK] Load test passed (p95 less than 120ms and success rate >= 95%)")
    exit(0)
else:
    print(f"\n[FAIL] Load test failed (exit code {result.returncode})")
    exit(1)

