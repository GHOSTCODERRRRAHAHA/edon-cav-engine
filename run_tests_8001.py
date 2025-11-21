"""Run all validation tests on port 8001"""
import subprocess
import sys
import time

print("\n" + "="*60)
print("EDON Validation Tests - Port 8001")
print("="*60 + "\n")

# Test 1: Model Info
print("\n[TEST 1] Model Info")
print("-" * 60)
try:
    result = subprocess.run([sys.executable, "test_1_model_info.py"], 
                          capture_output=False, text=True)
    if result.returncode == 0:
        print("\n✓ TEST 1 PASSED")
    else:
        print("\n✗ TEST 1 FAILED")
except Exception as e:
    print(f"Error running test 1: {e}")

# Test 2: Evaluation
print("\n\n[TEST 2] Evaluation (WESAD Ground Truth)")
print("-" * 60)
try:
    result = subprocess.run([sys.executable, "test_2_evaluation.py"],
                          capture_output=False, text=True)
    if result.returncode == 0:
        print("\n✓ TEST 2 PASSED")
    else:
        print("\n✗ TEST 2 FAILED or SKIPPED")
except Exception as e:
    print(f"Error running test 2: {e}")

# Test 3: Load Test
print("\n\n[TEST 3] Load Test")
print("-" * 60)
try:
    result = subprocess.run([sys.executable, "test_3_load_test.py"],
                          capture_output=False, text=True)
    if result.returncode == 0:
        print("\n✓ TEST 3 PASSED")
    else:
        print("\n✗ TEST 3 FAILED")
except Exception as e:
    print(f"Error running test 3: {e}")

print("\n" + "="*60)
print("All Tests Complete")
print("="*60 + "\n")

