"""Comprehensive test for v2 state classification with demo windows."""

import requests
import json
import math
import random
import pytest

# Server URL
url = "http://127.0.0.1:8002/v2/oem/cav/batch"


def create_restorative_demo():
    """Create a restorative state demo window."""
    return {
        "physio": {
            "EDA": [0.01 * i for i in range(240)],  # Very low EDA
            "BVP": [0.5 + 0.1 * math.sin(i / 20) for i in range(240)]  # Stable BVP
        },
        "motion": {
            "ACC_x": [0.0] * 240,
            "ACC_y": [0.0] * 240,
            "ACC_z": [0.98] * 240  # Minimal motion
        },
        "env": {
            "temp_c": 22.0,  # Comfortable
            "humidity": 45.0,  # Comfortable
            "aqi": 15  # Clean air
        },
        "task": {
            "id": "rest",
            "complexity": 0.0  # No task
        }
    }


def create_focus_demo():
    """Create a focus state demo window."""
    return {
        "physio": {
            "EDA": [0.03 * i for i in range(240)],  # Moderate EDA
            "BVP": [0.5 + 0.2 * math.sin(i / 8) for i in range(240)]  # Rhythmic BVP
        },
        "motion": {
            "ACC_x": [0.05] * 240,  # Small controlled movements
            "ACC_y": [0.02] * 240,
            "ACC_z": [1.0] * 240
        },
        "env": {
            "temp_c": 20.0,  # Cool, optimal
            "humidity": 35.0,  # Low humidity
            "aqi": 20  # Clean air
        },
        "task": {
            "id": "work",
            "complexity": 0.4  # Moderate complexity
        }
    }


def create_balanced_demo():
    """Create a balanced state demo window."""
    return {
        "physio": {
            "EDA": [0.05 * i for i in range(240)],  # Moderate-high EDA
            "BVP": [0.5 + 0.25 * math.sin(i / 10) for i in range(240)]  # Variable BVP
        },
        "motion": {
            "ACC_x": [0.1] * 240,
            "ACC_y": [0.05] * 240,
            "ACC_z": [1.1] * 240
        },
        "env": {
            "temp_c": 25.0,  # Moderate temp
            "humidity": 60.0,  # Moderate humidity
            "aqi": 50  # Moderate air quality
        },
        "task": {
            "id": "routine",
            "complexity": 0.5  # Moderate complexity
        }
    }


def create_emergency_demo():
    """Create an emergency/overload state demo window."""
    return {
        "physio": {
            "EDA": [0.8 + 0.1 * random.uniform(-1, 1) for _ in range(240)],  # Very high EDA
            "BVP": [random.uniform(-1.5, 1.5) for _ in range(240)]  # Erratic BVP
        },
        "motion": {
            "ACC_x": [random.gauss(0, 4) for _ in range(240)],  # Chaotic motion
            "ACC_y": [random.gauss(0, 4) for _ in range(240)],
            "ACC_z": [random.gauss(1.0, 3) for _ in range(240)]
        },
        "env": {
            "temp_c": 38.0,  # Extreme heat
            "humidity": 95.0,  # Extreme humidity
            "aqi": 250  # Very poor air quality
        },
        "task": {
            "id": "critical_failure",
            "complexity": 1.0,  # Maximum complexity
            "deadline_proximity": 1.0  # Urgent deadline
        },
        "system": {
            "battery_level": 0.05,  # Critical battery
            "cpu_temp": 95.0,  # Critical CPU temp
            "system_load": 0.99  # Critical system load
        }
    }


def run_window(name, window, expected_state=None, min_stress=None, max_stress=None,
               min_chaos=None, max_chaos=None):
    """Run a single window and verify results."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    try:
        response = requests.post(url, json={"windows": [window]}, timeout=5)
        response.raise_for_status()
        result = response.json()
        
        if "results" in result and len(result["results"]) > 0:
            item = result["results"][0]
            
            assert item.get("ok", False), f"Request failed: {item.get('error', 'Unknown error')}"
            
            state = item.get("state_class")
            p_stress = item.get("p_stress")
            p_chaos = item.get("p_chaos")
            influences = item.get("influences", {})
            metadata = item.get("metadata", {})
            
            print(f"State Class: {state}")
            print(f"p_stress: {p_stress:.3f}")
            print(f"p_chaos: {p_chaos:.3f}")
            print(f"Influences:")
            print(f"  speed_scale: {influences.get('speed_scale', 'N/A'):.3f}")
            print(f"  torque_scale: {influences.get('torque_scale', 'N/A'):.3f}")
            print(f"  caution_flag: {influences.get('caution_flag', 'N/A')}")
            print(f"  emergency_flag: {influences.get('emergency_flag', 'N/A')}")
            print(f"Metadata:")
            print(f"  modalities: {metadata.get('modalities_present', [])}")
            print(f"  device_profile: {metadata.get('device_profile', 'None')}")
            print(f"  pca_fitted: {metadata.get('pca_fitted', False)}")
            
            # Verify expectations
            if expected_state and state != expected_state:
                assert False, f"Expected state '{expected_state}', got '{state}'"
            if min_stress is not None and p_stress < min_stress:
                assert False, f"p_stress {p_stress:.3f} < {min_stress}"
            if max_stress is not None and p_stress > max_stress:
                assert False, f"p_stress {p_stress:.3f} > {max_stress}"
            if min_chaos is not None and p_chaos < min_chaos:
                assert False, f"p_chaos {p_chaos:.3f} < {min_chaos}"
            if max_chaos is not None and p_chaos > max_chaos:
                assert False, f"p_chaos {p_chaos:.3f} > {max_chaos}"
            
            print(f"[PASS] All checks passed")
        else:
            print("Unexpected response format:")
            print(json.dumps(result, indent=2))
            assert False, "Unexpected response format"
            
    except requests.exceptions.ConnectionError:
        pytest.skip("Could not connect to v2 server on port 8002")
    except Exception as e:
        assert False, f"Demo state error: {e}"


@pytest.mark.parametrize(
    "name,window,expected_state,min_stress,max_stress,min_chaos,max_chaos",
    [
        ("Restorative Demo", create_restorative_demo(), "restorative", None, 0.20, None, 0.15),
        ("Focus Demo", create_focus_demo(), "focus", 0.20, 0.45, None, 0.20),
        ("Balanced Demo", create_balanced_demo(), "balanced", None, 0.70, None, 0.40),
        ("Emergency Demo", create_emergency_demo(), "emergency", 0.90, None, 0.75, None),
    ],
)
def test_v2_demo_state(
    name, window, expected_state, min_stress, max_stress, min_chaos, max_chaos
):
    run_window(
        name,
        window,
        expected_state=expected_state,
        min_stress=min_stress,
        max_stress=max_stress,
        min_chaos=min_chaos,
        max_chaos=max_chaos,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("EDON v2 State Classification Test")
    print("=" * 70)
    print("\nTesting all v2 states with demo windows")
    print("Expected states:")
    print("  - restorative: p_stress < 0.20, p_chaos < 0.15")
    print("  - focus: 0.20 <= p_stress <= 0.45, p_chaos < 0.20")
    print("  - balanced: p_stress < 0.70, p_chaos < 0.40 (not focus)")
    print("  - alert: p_stress >= 0.70, p_chaos < 0.60")
    print("  - overload: p_stress >= 0.80, p_chaos >= 0.60")
    print("  - emergency: p_stress >= 0.90, p_chaos >= 0.75")
    print("=" * 70)
    
    # Set random seed for reproducible emergency demo
    random.seed(42)
    
    results = []
    
    # Test all states
    results.append(("Restorative", run_window(
        "Restorative Demo",
        create_restorative_demo(),
        expected_state="restorative",
        max_stress=0.20,
        max_chaos=0.15
    )))
    
    results.append(("Focus", run_window(
        "Focus Demo",
        create_focus_demo(),
        expected_state="focus",
        min_stress=0.20,
        max_stress=0.45,
        max_chaos=0.20
    )))
    
    results.append(("Balanced", run_window(
        "Balanced Demo",
        create_balanced_demo(),
        expected_state="balanced"
    )))
    
    results.append(("Emergency/Overload", run_window(
        "Emergency Demo",
        create_emergency_demo(),
        expected_state=None,  # Can be overload or emergency
        min_stress=0.80,  # Should be high
        min_chaos=0.60  # Should be high
    )))
    
    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
    
    all_passed = all(p for _, p in results)
    print(f"\n{'='*70}")
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print(f"{'='*70}")

