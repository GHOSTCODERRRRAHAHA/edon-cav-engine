"""Basic sanity tests for EDON SDK client."""

import os
import pytest
from edon_sdk import EdonClient, EdonConnectionError, EdonError


# Default test server URL (can be overridden via env var)
TEST_BASE_URL = os.getenv("EDON_TEST_BASE_URL", "http://127.0.0.1:8000")


def test_health_check():
    """Test health endpoint (assumes server is running)."""
    client = EdonClient(base_url=TEST_BASE_URL, timeout=2.0)
    
    try:
        health = client.health()
        assert "ok" in health
        assert health["ok"] is True
        assert "model" in health
        print(f"[TEST] Health check passed: {health.get('model', 'N/A')}")
    except EdonConnectionError:
        pytest.skip("EDON server not running - skipping health check test")


def test_cav_basic():
    """Test basic CAV computation with synthetic window."""
    client = EdonClient(base_url=TEST_BASE_URL, timeout=5.0)
    
    # Create minimal synthetic window
    window = {
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
    
    try:
        result = client.cav(window)
        
        # Verify response structure
        assert "state" in result
        assert "cav_raw" in result
        assert "cav_smooth" in result
        assert "parts" in result
        
        # Verify state is valid
        assert result["state"] in ["restorative", "balanced", "focus", "overload"]
        
        # Verify CAV scores are in expected range
        assert 0 <= result["cav_raw"] <= 10000
        assert 0 <= result["cav_smooth"] <= 10000
        
        print(f"[TEST] CAV computation passed: state={result['state']}, cav={result['cav_smooth']}")
    except EdonConnectionError:
        pytest.skip("EDON server not running - skipping CAV test")
    except EdonError as e:
        pytest.fail(f"CAV computation failed: {e}")


if __name__ == "__main__":
    # Allow running tests directly without pytest
    print("Running basic EDON SDK tests...")
    print(f"Test server: {TEST_BASE_URL}\n")
    
    try:
        test_health_check()
        test_cav_basic()
        print("\n[OK] All tests passed!")
    except Exception as e:
        print(f"\n[FAIL] Test error: {e}")
        exit(1)

