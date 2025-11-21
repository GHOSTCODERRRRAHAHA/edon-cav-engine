#!/usr/bin/env python3
"""
Load Test for EDON gRPC API

Sends N windows via gRPC and measures latency statistics.
"""

import sys
import time
import statistics
from pathlib import Path
from typing import List

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent / "integrations" / "grpc" / "edon_grpc_service"))

from edon import EdonClient, TransportType


def make_test_window() -> dict:
    """Create a test sensor window."""
    return {
        "EDA": [0.1] * 240,
        "TEMP": [36.5] * 240,
        "BVP": [0.5] * 240,
        "ACC_x": [0.0] * 240,
        "ACC_y": [0.0] * 240,
        "ACC_z": [1.0] * 240,
        "temp_c": 22.0,
        "humidity": 50.0,
        "aqi": 35,
        "local_hour": 14,
    }


def benchmark_grpc(n: int = 1000, host: str = "localhost", port: int = 50051) -> dict:
    """Benchmark gRPC API latency."""
    client = EdonClient(
        transport=TransportType.GRPC,
        grpc_host=host,
        grpc_port=port
    )
    
    window = make_test_window()
    latencies: List[float] = []
    errors = 0
    
    print(f"Benchmarking gRPC API with {n} requests...")
    print(f"Target: {host}:{port}\n")
    
    for i in range(n):
        start = time.time()
        try:
            result = client.cav(window)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n} requests...")
        except Exception as e:
            errors += 1
            print(f"  Error on request {i + 1}: {e}")
    
    client.close()
    
    if not latencies:
        return {"error": "No successful requests"}
    
    latencies.sort()
    
    return {
        "total_requests": n,
        "successful": len(latencies),
        "errors": errors,
        "avg_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p90_ms": latencies[int(len(latencies) * 0.90)],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "throughput_per_sec": len(latencies) / (sum(latencies) / 1000.0) if latencies else 0.0,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark EDON gRPC API latency")
    parser.add_argument("--n", type=int, default=1000, help="Number of requests (default: 1000)")
    parser.add_argument("--host", type=str, default="localhost", help="gRPC host")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EDON gRPC API Latency Benchmark")
    print("=" * 70)
    print()
    
    results = benchmark_grpc(n=args.n, host=args.host, port=args.port)
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return 1
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Total Requests:     {results['total_requests']}")
    print(f"Successful:         {results['successful']}")
    print(f"Errors:             {results['errors']}")
    print(f"\nLatency Statistics:")
    print(f"  Average:          {results['avg_latency_ms']:.2f} ms")
    print(f"  Median (p50):     {results['median_latency_ms']:.2f} ms")
    print(f"  p90:              {results['p90_ms']:.2f} ms")
    print(f"  p95:              {results['p95_ms']:.2f} ms")
    print(f"  p99:              {results['p99_ms']:.2f} ms")
    print(f"  Min:              {results['min_ms']:.2f} ms")
    print(f"  Max:              {results['max_ms']:.2f} ms")
    print(f"  Std Dev:          {results['std_dev_ms']:.2f} ms")
    print(f"\nThroughput:         {results['throughput_per_sec']:.2f} windows/sec")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

