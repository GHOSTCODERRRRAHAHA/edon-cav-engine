# Reward Alignment Summary

## Analysis Results

### Initial State
- **Reward-EDON Score correlation**: 0.251 (target: >0.7)
- **Reward-Length correlation**: -0.412 (should be positive)
- **Reward-Stability correlation**: -0.631 (good)

### Final State (After Adjustments)
- **Reward-EDON Score correlation**: 0.446 (improved, but still below target)
- **Reward-Length correlation**: +0.029 (now positive!)
- **Reward-Stability correlation**: -0.789 (excellent)

## Reward Function Adjustments

### Key Changes
1. **Increased alive bonus**: 0.2 → 0.6 per step
   - Matches length_bonus in EDON Score (min(20, length/50))
   - Makes longer episodes have less negative (or more positive) reward
   - Creates positive correlation with length

2. **Adjusted intervention penalty**: -20.0 → -25.0
   - Stronger penalty for interventions
   - Matches intervention_score in EDON Score (100 - interventions * 2)

3. **Adjusted tilt penalty**: -8.0 → -8.0 (base) + extra for dangerous tilt
   - Non-linear penalty that saturates
   - Matches stability_score in EDON Score (100 * (1 - min(1, stability * 10)))

4. **Adjusted velocity penalty**: -3.0 → -3.0
   - Maintains stability focus

5. **Adjusted action delta penalty**: -0.3 → -0.3
   - Maintains smoothness constraint

## Reward Statistics
- **Mean**: -470.95 (target: -200 to -800) ✅
- **Std**: 73.21
- **Range**: [-607.04, -352.51]

## PPO Hyperparameters (Bolder)
- **Learning Rate**: 3e-4 → **5e-4** (67% increase)
- **Update Epochs**: 10 → **12** (20% increase)
- **Gamma**: 0.995 (unchanged)

## Training Plan
- **Episodes**: 300 (increased from 200)
- **Model name**: `edon_v7_ep300_aligned`
- **Expected improvements**:
  - Better reward signal alignment
  - More aggressive policy updates
  - Longer training for better convergence

## Next Steps
1. Monitor training progress (KL, entropy, rewards)
2. Evaluate trained model vs baseline
3. If correlation still low, consider:
   - Further increasing alive bonus
   - Adjusting penalty scaling
   - Adding explicit length-based rewards


