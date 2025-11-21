"""Load testing script for batch endpoint."""

import time
import math
import random
import requests
import statistics
from typing import List
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

WINDOW_LEN = 240


def make_window(seed: int = None) -> dict:
    """Create a raw CAVRequest window matching OpenAPI spec."""
    if seed is None:
        seed = random.randint(0, 10000)
    return {
        "EDA": [k * 0.01 for k in range(WINDOW_LEN)],
        "TEMP": [36.5 for _ in range(WINDOW_LEN)],
        "BVP": [math.sin((k + seed) / 12.0) for k in range(WINDOW_LEN)],
        "ACC_x": [0.0] * WINDOW_LEN,
        "ACC_y": [0.0] * WINDOW_LEN,
        "ACC_z": [1.0] * WINDOW_LEN,
        "temp_c": 22.0,
        "humidity": 50.0,
        "aqi": 35,
        "local_hour": 14
    }


def create_test_payload(n_windows: int = 1) -> dict:
    """Create a test payload with n raw windows matching OpenAPI."""
    return {
        "windows": [make_window(seed=i) for i in range(n_windows)]
    }


def make_request(url: str, payload: dict) -> tuple:
    """Make a single request and return (latency_ms, status_code, success)."""
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=30.0)
        latency_ms = (time.time() - start) * 1000.0
        success = response.status_code == 200
        if not success:
            # Log first error for debugging
            if not hasattr(make_request, '_error_logged'):
                print(f"\n[DEBUG] Request failed with status {response.status_code}: {response.text[:200]}")
                make_request._error_logged = True
        return (latency_ms, response.status_code, success)
    except Exception as e:
        latency_ms = (time.time() - start) * 1000.0
        if not hasattr(make_request, '_error_logged'):
            print(f"\n[DEBUG] Request exception: {e}")
            make_request._error_logged = True
        return (latency_ms, 0, False)


def run_load_test(
    url: str,
    n_requests: int,
    n_windows: int,
    n_concurrent: int
) -> dict:
    """Run load test and return statistics."""
    payload = create_test_payload(n_windows)
    latencies: List[float] = []
    successes = 0
    errors = 0
    status_codes = {}
    first_error = None
    
    print(f"Running load test: {n_requests} requests, {n_windows} windows/req, {n_concurrent} concurrent...")
    
    with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        futures = [executor.submit(make_request, url, payload) for _ in range(n_requests)]
        
        for future in as_completed(futures):
            latency_ms, status_code, success = future.result()
            latencies.append(latency_ms)
            status_codes[status_code] = status_codes.get(status_code, 0) + 1
            if success:
                successes += 1
            else:
                errors += 1
                if first_error is None and status_code != 0:
                    # Try to get error details from a test request
                    try:
                        test_response = requests.post(url, json=payload, timeout=5.0)
                        first_error = f"Status {test_response.status_code}: {test_response.text[:300]}"
                    except:
                        pass
    
    if not latencies:
        return {"error": "No successful requests"}
    
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    
    result = {
        "n_requests": n_requests,
        "n_windows_per_request": n_windows,
        "n_concurrent": n_concurrent,
        "successes": successes,
        "errors": errors,
        "success_rate": successes / n_requests,
        "status_codes": status_codes,
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "median": p50,
            "p95": p95,
            "p99": p99,
            "min": min(latencies),
            "max": max(latencies)
        }
    }
    if first_error:
        result["first_error"] = first_error
    return result


def main():
    parser = argparse.ArgumentParser(description="Load test batch endpoint")
    parser.add_argument("--url", type=str, default="http://localhost:8000/oem/cav/batch", help="API URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--windows", type=int, default=1, help="Number of windows per request")
    parser.add_argument("--concurrent", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--p95-threshold", type=float, default=120.0, help="P95 latency threshold (ms)")
    
    args = parser.parse_args()
    
    results = run_load_test(args.url, args.requests, args.windows, args.concurrent)
    
    if "error" in results:
        print(f"✗ Load test failed: {results['error']}")
        return 1
    
    print("\n" + "="*60)
    print("LOAD TEST RESULTS")
    print("="*60)
    print(f"Requests: {results['n_requests']}")
    print(f"Windows per request: {results['n_windows_per_request']}")
    print(f"Concurrent: {results['n_concurrent']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    if results.get('status_codes'):
        print(f"Status codes: {results['status_codes']}")
    if results.get('first_error'):
        print(f"\nFirst error: {results['first_error']}")
    print(f"\nLatency (ms):")
    print(f"  Mean:   {results['latency_ms']['mean']:.2f}")
    print(f"  Median: {results['latency_ms']['median']:.2f}")
    print(f"  P95:    {results['latency_ms']['p95']:.2f}")
    print(f"  P99:    {results['latency_ms']['p99']:.2f}")
    print(f"  Min:    {results['latency_ms']['min']:.2f}")
    print(f"  Max:    {results['latency_ms']['max']:.2f}")
    
    # Check threshold
    p95 = results['latency_ms']['p95']
    success_rate = results['success_rate']
    
    # Both latency and success rate must pass
    latency_ok = p95 <= args.p95_threshold
    success_ok = success_rate >= 0.95  # 95% success rate required
    
    if latency_ok and success_ok:
        print(f"\n✓ P95 latency ({p95:.2f}ms) <= threshold ({args.p95_threshold}ms)")
        print(f"✓ Success rate ({success_rate:.2%}) >= 95%")
        return 0
    else:
        if not latency_ok:
            print(f"\n✗ P95 latency ({p95:.2f}ms) > threshold ({args.p95_threshold}ms)")
        if not success_ok:
            print(f"✗ Success rate ({success_rate:.2%}) < 95%")
        return 1


if __name__ == "__main__":
    exit(main())

