#!/usr/bin/env python3
"""Test the Adaptive Memory Engine directly (without API server)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.adaptive_memory import AdaptiveMemoryEngine
import json

def test_memory_engine_direct():
    """Test the memory engine directly."""
    print("=" * 60)
    print("Testing Adaptive Memory Engine (Direct)")
    print("=" * 60)
    
    # Initialize memory engine
    print("\n1. Initializing memory engine...")
    memory = AdaptiveMemoryEngine(db_path="data/test_memory.db")
    print("   [OK] Memory engine initialized")
    
    # Clear memory
    print("\n2. Clearing memory...")
    memory.clear()
    print("   [OK] Memory cleared")
    
    # Load a test window from JSONL
    print("\n3. Loading test window from dataset...")
    with open("outputs/oem_sample_windows.jsonl", "r") as f:
        first_window = json.loads(f.readline())
    
    print(f"   Window ID: {first_window['window_id']}")
    print(f"   Has raw signals: {'EDA' in first_window}")
    
    # Record some test data
    print("\n4. Recording test data...")
    test_records = [
        {
            "cav_raw": 9996,
            "cav_smooth": 9732,
            "state": "restorative",
            "parts": {"bio": 0.999, "env": 1.0, "circadian": 1.0, "p_stress": 0.001},
            "temp_c": 24.0,
            "humidity": 50.0,
            "aqi": 42,
            "local_hour": 12
        },
        {
            "cav_raw": 9997,
            "cav_smooth": 9759,
            "state": "restorative",
            "parts": {"bio": 0.999, "env": 1.0, "circadian": 1.0, "p_stress": 0.001},
            "temp_c": 24.0,
            "humidity": 50.0,
            "aqi": 42,
            "local_hour": 12
        },
        {
            "cav_raw": 9996,
            "cav_smooth": 9771,
            "state": "restorative",
            "parts": {"bio": 0.999, "env": 1.0, "circadian": 1.0, "p_stress": 0.001},
            "temp_c": 24.0,
            "humidity": 50.0,
            "aqi": 42,
            "local_hour": 12
        },
        {
            "cav_raw": 9996,
            "cav_smooth": 9776,
            "state": "restorative",
            "parts": {"bio": 0.999, "env": 1.0, "circadian": 1.0, "p_stress": 0.001},
            "temp_c": 24.0,
            "humidity": 50.0,
            "aqi": 42,
            "local_hour": 12
        },
        {
            "cav_raw": 9996,
            "cav_smooth": 9780,
            "state": "restorative",
            "parts": {"bio": 0.999, "env": 1.0, "circadian": 1.0, "p_stress": 0.001},
            "temp_c": 24.0,
            "humidity": 50.0,
            "aqi": 42,
            "local_hour": 12
        }
    ]
    
    for i, record in enumerate(test_records):
        memory.record(**record)
        print(f"   Recorded record {i+1}/5")
    
    # Force update hourly stats
    print("\n5. Updating hourly statistics...")
    memory._update_hourly_stats()
    print("   [OK] Hourly statistics updated")
    
    # Test adaptive computation
    print("\n6. Testing adaptive computation...")
    adaptive = memory.compute_adaptive(
        cav_smooth=9780,
        state="restorative",
        aqi=42,
        local_hour=12
    )
    
    print(f"   Z-score: {adaptive['z_cav']:.2f}")
    print(f"   Sensitivity: {adaptive['sensitivity']:.2f}")
    print(f"   Env weight adj: {adaptive['env_weight_adj']:.2f}")
    
    # Get summary
    print("\n7. Getting memory summary...")
    summary = memory.get_summary()
    
    print(f"   Total records: {summary['total_records']}")
    print(f"   Window hours: {summary['window_hours']}")
    
    if summary['overall_stats']:
        overall = summary['overall_stats']
        print(f"   Overall CAV mean: {overall['cav_mean']:.1f}")
        print(f"   Overall CAV std: {overall['cav_std']:.1f}")
        print(f"   State distribution:")
        for state, prob in overall['state_distribution'].items():
            print(f"      {state}: {prob:.3f}")
    
    if 12 in summary['hourly_stats']:
        hour12 = summary['hourly_stats'][12]
        print(f"\n   Hour 12 statistics:")
        print(f"      CAV mean: {hour12['cav_mean']:.1f}")
        print(f"      CAV std: {hour12['cav_std']:.1f}")
        print(f"      State probabilities:")
        for state, prob in hour12['state_probs'].items():
            print(f"         {state}: {prob:.3f}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Memory engine test complete!")
    print("=" * 60)
    
    # Cleanup
    print("\n8. Cleaning up test database...")
    memory.clear()
    print("   [OK] Test database cleared")

if __name__ == "__main__":
    test_memory_engine_direct()

