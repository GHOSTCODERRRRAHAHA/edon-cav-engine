# v8 Policy Learning Bug - FIXED

## Problem Identified

**Issue**: Policy loss = 0.0000, policy not learning during training

**Root Cause**: Advantages computed by GAE were all **zero**

### Why Advantages Were Zero

1. **GAE with returns as values**: When using returns as value estimates in GAE:
   ```
   delta = reward + gamma * next_value - value
   ```
   If `value = return`, then:
   ```
   delta = reward + gamma * return - return
        = reward + (gamma - 1) * return
   ```
   For gamma = 0.995, this is very small.

2. **Normalization kills signal**: After GAE, advantages were normalized:
   ```python
   advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
   ```
   If all advantages are similar (or zero), normalization makes them exactly zero.

3. **Result**: Policy loss = 0, gradients = 0, policy never updates

## Fix Applied

**Changed from**: GAE with returns as values (causes zero advantages)

**Changed to**: Simple advantage = return - baseline (mean return)

```python
# OLD (BROKEN):
advantages, returns_gae = ppo.compute_gae(
    rewards=trajectory["rewards"],
    values=returns,  # This causes zero advantages!
    dones=trajectory["dones"],
    next_value=0.0
)

# NEW (FIXED):
mean_return = np.mean(returns)
advantages = [r - mean_return for r in returns]

# Normalize advantages (important for stable learning)
if len(advantages) > 1:
    adv_mean = np.mean(advantages)
    adv_std = np.std(advantages)
    if adv_std > 1e-8:
        advantages = [(a - adv_mean) / adv_std for a in advantages]
```

## Expected Improvement

With this fix:
- Advantages will have non-zero variance
- Policy loss will be non-zero
- Gradients will flow
- Policy will actually learn

## Next Steps

1. **Retrain v8 with fix**: Run training again with fixed advantage computation
2. **Verify learning**: Check that policy loss > 0 and policy is updating
3. **Compare results**: Should see improvement over previous v8

## Files Modified

- `training/train_edon_v8_strategy.py`: Fixed advantage computation (line ~277)

