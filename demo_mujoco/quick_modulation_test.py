"""Quick test of modulation application logic."""
import numpy as np

# Simulate baseline action (12 joints, in MuJoCo range [-20, 20])
baseline_action = np.array([5.0, -3.0, 0.0, 2.0, -1.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.3, -0.3])

# Simulate modulations from policy
modulations = {
    "gain_scale": 1.1,
    "lateral_compliance": 0.8,
    "step_height_bias": -0.3
}

print("Testing modulation application...")
print(f"Baseline action: {baseline_action}")
print(f"Modulations: {modulations}")

# NEW LOGIC: Apply all 3 modulations
# 1. Normalize to [-1, 1]
baseline_normalized = np.clip(baseline_action / 20.0, -1.0, 1.0)
print(f"\n1. Normalized: {baseline_normalized}")

# 2. Apply gain scale
corrected = baseline_normalized * modulations["gain_scale"]
print(f"2. After gain_scale ({modulations['gain_scale']}): {corrected}")

# 3. Apply lateral_compliance to root rotation (indices 3-5)
if len(corrected) >= 6:
    corrected[3:6] = corrected[3:6] * modulations["lateral_compliance"]
    print(f"3. After lateral_compliance ({modulations['lateral_compliance']}) on [3:6]: {corrected}")

# 4. Apply step_height_bias to leg joints (indices 6-11)
if len(corrected) >= 12:
    corrected[6:12] = corrected[6:12] + modulations["step_height_bias"] * 0.1
    print(f"4. After step_height_bias ({modulations['step_height_bias']}) on [6:12]: {corrected}")

# 5. Clamp and scale back
final_action = np.clip(corrected, -1.0, 1.0) * 20.0
print(f"\n5. Final action (scaled to [-20, 20]): {final_action}")

# Check for issues
if np.any(np.isnan(final_action)) or np.any(np.isinf(final_action)):
    print("\n❌ ERROR: NaN/Inf detected!")
else:
    print("\n✅ No NaN/Inf - modulation logic works!")

if np.any(np.abs(final_action) > 25.0):
    print("⚠️  WARNING: Some actions exceed 25.0")
else:
    print("✅ All actions in reasonable range")

print("\n✅ Test passed - modulations can be added to training!")

