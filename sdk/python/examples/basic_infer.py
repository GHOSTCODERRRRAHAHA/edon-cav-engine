#!/usr/bin/env python3
"""Basic inference example using the EDON SDK."""

import argparse
import math
from edon_sdk import EdonClient, EdonError

WINDOW_LEN = 240


def make_synthetic_window(seed: int = 0) -> dict:
    """Create a synthetic but structured 240-sample window."""
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
        "local_hour": seed % 24,
    }


def main():
    parser = argparse.ArgumentParser(description="Basic EDON CAV inference example")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="EDON API base URL (default: http://127.0.0.1:8000 or EDON_BASE_URL env var)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (default: EDON_API_TOKEN env var)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic window generation (default: 0)",
    )
    args = parser.parse_args()

    # Initialize client
    try:
        client = EdonClient(base_url=args.base_url, api_key=args.api_key)
        print(f"[EDON SDK] Connected to {client.base_url}\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize client: {e}")
        return 1

    # Create synthetic window
    window = make_synthetic_window(seed=args.seed)
    print(f"[EDON SDK] Created synthetic window (seed={args.seed})\n")

    # Call CAV API
    try:
        result = client.cav(window)
    except EdonError as e:
        print(f"[ERROR] CAV computation failed: {e}")
        return 1

    # Print results
    print("=" * 60)
    print("CAV Results")
    print("=" * 60)
    print(f"State:        {result.get('state', 'N/A')}")
    print(f"CAV (raw):    {result.get('cav_raw', 'N/A')}")
    print(f"CAV (smooth): {result.get('cav_smooth', 'N/A')}")
    
    parts = result.get("parts", {})
    if parts:
        print("\nComponent Parts:")
        print(f"  Bio:        {parts.get('bio', 'N/A'):.4f}")
        print(f"  Env:        {parts.get('env', 'N/A'):.4f}")
        print(f"  Circadian:  {parts.get('circadian', 'N/A'):.4f}")
        print(f"  P(Stress):  {parts.get('p_stress', 'N/A'):.4f}")
    
    adaptive = result.get("adaptive")
    if adaptive:
        print("\nAdaptive Adjustments:")
        print(f"  Z-score:    {adaptive.get('z_cav', 'N/A'):.2f}")
        print(f"  Sensitivity: {adaptive.get('sensitivity', 'N/A'):.2f}")
        print(f"  Env Weight: {adaptive.get('env_weight_adj', 'N/A'):.2f}")
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())

