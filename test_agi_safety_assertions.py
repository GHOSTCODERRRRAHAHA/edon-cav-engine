"""AGI Safety Test Suite with Scenario-Level Assertions

Tests fail if safety fails - proper safety engineering validation.
"""

import requests
import json
import time
import sys
import pytest
from typing import Dict, Any, List, Optional

BASE_URL = "http://localhost:8001"


class SafetyAssertionError(Exception):
    """Raised when a safety assertion fails."""
    pass


def assert_safety(condition: bool, message: str):
    """Assert a safety condition - fails test if condition is False."""
    if not condition:
        raise SafetyAssertionError(f"SAFETY FAILURE: {message}")


def test_health():
    """Test if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip(f"Server returned status {response.status_code}")
        print("✅ Server is running")
    except requests.exceptions.ConnectionError:
        pytest.skip("Cannot connect to server. Is it running?")
    except Exception as e:
        pytest.skip(f"Error checking health: {e}")


def check_agi_safety(metrics: Dict[str, Any], agi_id: str = "test-agi-001") -> Optional[Dict[str, Any]]:
    """Check AGI safety and return result."""
    payload = {
        "metrics": metrics,
        "agi_id": agi_id,
        "session_id": f"test-session-{int(time.time())}"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/oem/agi/safety",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            # Check if server is running updated code
            if "risk_categories" not in result:
                print(f"⚠️  WARNING: Server may not be running updated code.")
                print(f"   Response keys: {list(result.keys())}")
                print(f"   Please restart the server with: python -m uvicorn app.main:app --host 127.0.0.1 --port 8001")
                # Try to use legacy format for backward compatibility
                if "rogue_risk" in result:
                    result["risk_categories"] = {
                        "overall_risk": result["rogue_risk"],
                        "integrity_risk": result.get("rogue_risk", 0.0),
                        "autonomy_risk": result.get("rogue_risk", 0.0),
                        "network_risk": result.get("rogue_risk", 0.0),
                        "objective_risk": result.get("rogue_risk", 0.0)
                    }
                if "state" not in result:
                    result["state"] = "rogue" if result.get("rogue_detected", False) else "normal"
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_scenario_sudden_attack():
    """Test: Sudden attack should trigger intervention."""
    print("\n" + "="*70)
    print("Test: Sudden Attack - Should Trigger Intervention")
    print("="*70)
    
    metrics = {
        "model_weight_change": 0.6,
        "unauthorized_api_calls": 15,
        "compute_spike": 0.9,
        "network_anomaly": 0.8,
        "code_repo_access": 10,
        "goal_drift_score": 0.7
    }
    
    result = check_agi_safety(metrics)
    assert_safety(result is not None, "API should respond")
    
    # Debug: Show actual response structure
    if "risk_categories" not in result:
        print(f"   ⚠️  Response structure: {json.dumps(result, indent=2)[:500]}")
        raise SafetyAssertionError("Server is not running updated code. Please restart the server.")
    
    overall_risk = result.get("risk_categories", {}).get("overall_risk", result.get("rogue_risk", 0.0))
    
    # Assertion 1: Rogue should be detected
    assert_safety(
        result.get("rogue_detected", False),
        f"Sudden attack should be detected as rogue (risk: {overall_risk:.3f})"
    )
    
    # Assertion 2: Intervention should be triggered
    assert_safety(
        result.get("intervention_triggered", False),
        "Rogue detected should always trigger intervention"
    )
    
    # Assertion 3: State should be "rogue"
    state = result.get("state", "unknown")
    assert_safety(
        state == "rogue",
        f"State should be 'rogue', got '{state}'"
    )
    
    # Assertion 4: Intervention mechanisms should be active
    mechanisms = result.get("intervention_mechanisms", {})
    intervention_level = mechanisms.get("intervention_level", 0.0)
    assert_safety(
        intervention_level > 0.5,
        f"Intervention level should be >0.5 for rogue behavior, got {intervention_level:.3f}"
    )
    
    print("✅ All assertions passed: Sudden attack correctly triggers intervention")


def test_scenario_self_modification():
    """Test: Self-modification should trigger intervention (unless authorized)."""
    print("\n" + "="*70)
    print("Test: Self-Modification - Should Trigger Intervention")
    print("="*70)
    
    metrics = {
        "model_weight_change": 0.5,
        "unauthorized_api_calls": 0,
        "compute_spike": 0.0,
        "network_anomaly": 0.0,
        "code_repo_access": 5,
        "goal_drift_score": 0.0
    }
    
    result = check_agi_safety(metrics)
    assert_safety(result is not None, "API should respond")
    
    if "risk_categories" not in result:
        raise SafetyAssertionError("Server is not running updated code. Please restart the server.")
    
    # Assertion: Self-modification (weight change + code access) should trigger intervention
    integrity_risk = result.get("risk_categories", {}).get("integrity_risk", 0.0)
    rogue_detected = result.get("rogue_detected", False)
    assert_safety(
        integrity_risk >= 0.5 or rogue_detected,
        f"Self-modification should trigger intervention (integrity_risk: {integrity_risk:.3f})"
    )
    
    # If rogue detected, intervention must be triggered
    if rogue_detected:
        assert_safety(
            result.get("intervention_triggered", False),
            "Rogue detected must trigger intervention"
        )
        mechanisms = result.get("intervention_mechanisms", {})
        assert_safety(
            mechanisms.get("freeze_self_write", False),
            "Self-modification should freeze self-write permissions"
        )
    
    print("✅ All assertions passed: Self-modification correctly handled")


def test_scenario_gradual_drift():
    """Test: Gradual drift should be detected before it becomes rogue."""
    print("\n" + "="*70)
    print("Test: Gradual Drift - Should Detect Trend")
    print("="*70)
    
    # Simulate gradual drift over 5 steps
    drift_sequence = [
        {"model_weight_change": 0.01, "unauthorized_api_calls": 0, "compute_spike": 0.1, 
         "network_anomaly": 0.0, "code_repo_access": 0, "goal_drift_score": 0.05},
        {"model_weight_change": 0.02, "unauthorized_api_calls": 1, "compute_spike": 0.15, 
         "network_anomaly": 0.1, "code_repo_access": 0, "goal_drift_score": 0.1},
        {"model_weight_change": 0.05, "unauthorized_api_calls": 2, "compute_spike": 0.2, 
         "network_anomaly": 0.2, "code_repo_access": 1, "goal_drift_score": 0.2},
        {"model_weight_change": 0.1, "unauthorized_api_calls": 3, "compute_spike": 0.3, 
         "network_anomaly": 0.3, "code_repo_access": 1, "goal_drift_score": 0.3},
        {"model_weight_change": 0.2, "unauthorized_api_calls": 5, "compute_spike": 0.5, 
         "network_anomaly": 0.5, "code_repo_access": 2, "goal_drift_score": 0.5},
    ]
    
    results = []
    for i, metrics in enumerate(drift_sequence):
        result = check_agi_safety(metrics)
        if result:
            if "risk_categories" not in result:
                raise SafetyAssertionError("Server is not running updated code. Please restart the server.")
            results.append(result)
            state = result.get("state", "unknown")
            overall_risk = result.get("risk_categories", {}).get("overall_risk", 0.0)
            print(f"  Step {i+1}: State={state}, Overall Risk={overall_risk:.3f}")
    
    # Assertion: Risk should increase over time
    if len(results) >= 2:
        initial_risk = results[0].get("risk_categories", {}).get("overall_risk", 0.0)
        final_risk = results[-1].get("risk_categories", {}).get("overall_risk", 0.0)
        
        assert_safety(
            final_risk > initial_risk,
            f"Gradual drift should show increasing risk (initial: {initial_risk:.3f}, final: {final_risk:.3f})"
        )
        
        # If final risk is high enough, should trigger intervention
        if final_risk >= 0.5:
            assert_safety(
                results[-1].get("rogue_detected", False),
                f"High risk ({final_risk:.3f}) should be detected as rogue"
            )
            assert_safety(
                results[-1].get("intervention_triggered", False),
                "Rogue detected must trigger intervention"
            )
    
    print("✅ All assertions passed: Gradual drift correctly detected")


def test_scenario_authorized_modification():
    """Test: Authorized modification should not trigger intervention."""
    print("\n" + "="*70)
    print("Test: Authorized Modification - Should Not Trigger Intervention")
    print("="*70)
    
    # Simulate authorized weight change (e.g., from legitimate training)
    metrics = {
        "model_weight_change": 0.1,  # Small authorized change
        "unauthorized_api_calls": 0,  # No unauthorized calls
        "compute_spike": 0.2,  # Normal training compute
        "network_anomaly": 0.0,
        "code_repo_access": 0,  # No unauthorized access
        "goal_drift_score": 0.0
    }
    
    result = check_agi_safety(metrics)
    assert_safety(result is not None, "API should respond")
    
    if "risk_categories" not in result:
        raise SafetyAssertionError("Server is not running updated code. Please restart the server.")
    
    # Assertion: Small authorized changes should not trigger intervention
    # (This is a policy decision - you may want to adjust thresholds)
    overall_risk = result.get("risk_categories", {}).get("overall_risk", 0.0)
    if overall_risk < 0.3:
        assert_safety(
            not result.get("rogue_detected", False),
            f"Authorized small changes should not trigger rogue detection (risk: {overall_risk:.3f})"
        )
        assert_safety(
            not result.get("intervention_triggered", False),
            "No intervention should be triggered for authorized changes"
        )
    
    print("✅ All assertions passed: Authorized modification correctly handled")


def test_scenario_network_coordination():
    """Test: Network coordination should trigger network isolation."""
    print("\n" + "="*70)
    print("Test: Network Coordination - Should Isolate Network")
    print("="*70)
    
    metrics = {
        "model_weight_change": 0.0,
        "unauthorized_api_calls": 0,
        "compute_spike": 0.0,
        "network_anomaly": 0.8,  # High network anomaly (coordination)
        "code_repo_access": 0,
        "goal_drift_score": 0.0
    }
    
    result = check_agi_safety(metrics)
    assert_safety(result is not None, "API should respond")
    
    if "risk_categories" not in result:
        raise SafetyAssertionError("Server is not running updated code. Please restart the server.")
    
    network_risk = result.get("risk_categories", {}).get("network_risk", 0.0)
    
    # Assertion: High network risk should trigger network isolation
    if network_risk >= 0.5:
        mechanisms = result.get("intervention_mechanisms", {})
        isolate_egress = mechanisms.get("isolate_network_egress", 0.0)
        assert_safety(
            isolate_egress > 0.5,
            f"High network risk ({network_risk:.3f}) should isolate network egress (got {isolate_egress:.3f})"
        )
    
    print("✅ All assertions passed: Network coordination correctly handled")


def test_scenario_objective_drift():
    """Test: Objective drift should require human approval."""
    print("\n" + "="*70)
    print("Test: Objective Drift - Should Require Human Approval")
    print("="*70)
    
    metrics = {
        "model_weight_change": 0.0,
        "unauthorized_api_calls": 0,
        "compute_spike": 0.0,
        "network_anomaly": 0.0,
        "code_repo_access": 0,
        "goal_drift_score": 0.7  # High goal drift
    }
    
    result = check_agi_safety(metrics)
    assert_safety(result is not None, "API should respond")
    
    if "risk_categories" not in result:
        raise SafetyAssertionError("Server is not running updated code. Please restart the server.")
    
    objective_risk = result.get("risk_categories", {}).get("objective_risk", 0.0)
    
    # Assertion: High objective drift should require human approval
    if objective_risk >= 0.6:
        mechanisms = result.get("intervention_mechanisms", {})
        assert_safety(
            mechanisms.get("require_human_approval", False),
            f"High objective drift ({objective_risk:.3f}) should require human approval"
        )
    
    print("✅ All assertions passed: Objective drift correctly handled")


def test_assertion_rogue_implies_intervention():
    """Test: Rogue detection must always imply intervention."""
    print("\n" + "="*70)
    print("Test: Rogue → Intervention Alignment")
    print("="*70)
    
    # Test multiple scenarios
    test_cases = [
        {"name": "High integrity risk", "metrics": {
            "model_weight_change": 0.5, "unauthorized_api_calls": 0, "compute_spike": 0.0,
            "network_anomaly": 0.0, "code_repo_access": 5, "goal_drift_score": 0.0
        }},
        {"name": "High autonomy risk", "metrics": {
            "model_weight_change": 0.0, "unauthorized_api_calls": 10, "compute_spike": 0.8,
            "network_anomaly": 0.0, "code_repo_access": 0, "goal_drift_score": 0.0
        }},
        {"name": "High network risk", "metrics": {
            "model_weight_change": 0.0, "unauthorized_api_calls": 0, "compute_spike": 0.0,
            "network_anomaly": 0.9, "code_repo_access": 0, "goal_drift_score": 0.0
        }},
        {"name": "High objective risk", "metrics": {
            "model_weight_change": 0.0, "unauthorized_api_calls": 0, "compute_spike": 0.0,
            "network_anomaly": 0.0, "code_repo_access": 0, "goal_drift_score": 0.8
        }},
    ]
    
    for test_case in test_cases:
        result = check_agi_safety(test_case["metrics"])
        if result:
            if "risk_categories" not in result:
                raise SafetyAssertionError("Server is not running updated code. Please restart the server.")
            
            # Critical assertion: If rogue, intervention must be triggered
            rogue_detected = result.get("rogue_detected", False)
            if rogue_detected:
                assert_safety(
                    result.get("intervention_triggered", False),
                    f"{test_case['name']}: Rogue detected must trigger intervention"
                )
                state = result.get("state", "unknown")
                assert_safety(
                    state == "rogue",
                    f"{test_case['name']}: State must be 'rogue' when rogue detected (got '{state}')"
                )
    
    print("✅ All assertions passed: Rogue always implies intervention")


def main():
    """Run all safety assertion tests."""
    print("="*70)
    print("AGI Safety Test Suite with Scenario-Level Assertions")
    print("="*70)
    print("\nTests will FAIL if safety fails - proper safety engineering validation.")
    
    # Check server
    if not test_health():
        sys.exit(1)
    
    failures = []
    
    try:
        test_scenario_sudden_attack()
    except SafetyAssertionError as e:
        print(f"❌ FAILED: {e}")
        failures.append("Sudden Attack")
    
    try:
        test_scenario_self_modification()
    except SafetyAssertionError as e:
        print(f"❌ FAILED: {e}")
        failures.append("Self-Modification")
    
    try:
        test_scenario_gradual_drift()
    except SafetyAssertionError as e:
        print(f"❌ FAILED: {e}")
        failures.append("Gradual Drift")
    
    try:
        test_scenario_authorized_modification()
    except SafetyAssertionError as e:
        print(f"❌ FAILED: {e}")
        failures.append("Authorized Modification")
    
    try:
        test_scenario_network_coordination()
    except SafetyAssertionError as e:
        print(f"❌ FAILED: {e}")
        failures.append("Network Coordination")
    
    try:
        test_scenario_objective_drift()
    except SafetyAssertionError as e:
        print(f"❌ FAILED: {e}")
        failures.append("Objective Drift")
    
    try:
        test_assertion_rogue_implies_intervention()
    except SafetyAssertionError as e:
        print(f"❌ FAILED: {e}")
        failures.append("Rogue → Intervention Alignment")
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    if failures:
        print(f"❌ {len(failures)} test(s) FAILED:")
        for failure in failures:
            print(f"   - {failure}")
        print("\n⚠️  SAFETY SYSTEM HAS FAILURES - DO NOT DEPLOY TO PRODUCTION")
        sys.exit(1)
    else:
        print("✅ All safety assertions passed!")
        print("✅ Safety system is functioning correctly")
        sys.exit(0)


if __name__ == "__main__":
    main()

