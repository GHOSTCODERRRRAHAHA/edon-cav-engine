#!/usr/bin/env python3
"""Simple test runner for v2 functionality."""

import sys
import subprocess
from pathlib import Path

def run_test(test_file):
    """Run a test file and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print('='*70)
    
    result = subprocess.run(
        [sys.executable, str(test_file)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    tests = [
        Path("tests/test_v2_grpc.py"),
        Path("tests/test_v2_stream_websocket.py"),
    ]
    
    results = []
    for test in tests:
        if test.exists():
            results.append((test.name, run_test(test)))
        else:
            print(f"\n⚠ Test file not found: {test}")
            results.append((test.name, None))
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    for name, result in results:
        if result is True:
            print(f"✓ {name}")
        elif result is False:
            print(f"✗ {name}")
        else:
            print(f"⚠ {name} (skipped)")
    
    all_passed = all(r[1] for r in results if r[1] is not None)
    sys.exit(0 if all_passed else 1)

