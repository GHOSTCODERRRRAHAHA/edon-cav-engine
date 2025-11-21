"""Direct test of v1 state classifier - test all states without server."""

from app.engine import CAVEngine, classify_state
import numpy as np

# Initialize v1 engine
engine = CAVEngine()

def test_state_classification(name, p_stress, env=0.5, circadian=0.5):
    """Test state classification directly."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"Input:")
    print(f"  p_stress: {p_stress}")
    print(f"  env: {env}")
    print(f"  circadian: {circadian}")
    
    parts = {
        "p_stress": p_stress,
        "env": env,
        "circadian": circadian
    }
    
    state = classify_state(0.5, parts)  # cav_smooth not used in current logic
    print(f"\nResult: {state}")
    
    # Verify expected state
    expected = None
    if p_stress >= 0.8:
        expected = "overload"
    elif p_stress < 0.2:
        expected = "restorative"
    elif 0.2 <= p_stress <= 0.5 and env >= 0.8 and circadian >= 0.9:
        expected = "focus"
    elif p_stress < 0.8:
        expected = "balanced"
    
    if state == expected:
        print(f"[PASS] Correct! Expected: {expected}")
    else:
        print(f"[FAIL] Mismatch! Expected: {expected}, Got: {state}")
    
    return state

if __name__ == "__main__":
    print("EDON v1 State Classifier Direct Test")
    print("=" * 60)
    print("Testing all v1 states: restorative, focus, balanced, overload")
    print("=" * 60)
    
    # Test all states
    test_state_classification(
        "Restorative State",
        p_stress=0.15,  # < 0.2
        env=0.7,
        circadian=0.8
    )
    
    test_state_classification(
        "Focus State",
        p_stress=0.35,  # 0.2 <= p_stress <= 0.5
        env=0.85,  # >= 0.8
        circadian=0.95  # >= 0.9
    )
    
    test_state_classification(
        "Balanced State (moderate stress, low alignment)",
        p_stress=0.45,  # < 0.8 but not focus (env too low)
        env=0.6,  # < 0.8
        circadian=0.7  # < 0.9
    )
    
    test_state_classification(
        "Balanced State (moderate stress, good env but low circadian)",
        p_stress=0.4,  # < 0.8
        env=0.85,  # >= 0.8
        circadian=0.8  # < 0.9 (not high enough)
    )
    
    test_state_classification(
        "Overload State",
        p_stress=0.85,  # >= 0.8
        env=0.5,
        circadian=0.5
    )
    
    test_state_classification(
        "Overload State (extreme)",
        p_stress=0.95,  # >= 0.8
        env=0.3,
        circadian=0.4
    )
    
    print(f"\n{'='*60}")
    print("Test Complete")
    print(f"{'='*60}")
    print("\nState Classification Rules:")
    print("  - Restorative: p_stress < 0.2")
    print("  - Focus: 0.2 <= p_stress <= 0.5 AND env >= 0.8 AND circadian >= 0.9")
    print("  - Balanced: p_stress < 0.8 (and not focus)")
    print("  - Overload: p_stress >= 0.8")

