# v8 Strategy Policy Model Size

## File Size Estimate

**Trained model file size: ~0.11-0.13 MB** (110-130 KB)

### Breakdown:

1. **Model parameters**: 28,551 parameters
   - Input size: 25 features
   - Hidden layers: [128, 128, 64]
   - Outputs: 4 strategies + 3 modulations
   - Size: 28,551 × 4 bytes (float32) = **0.11 MB**

2. **Checkpoint metadata**:
   - `policy_state_dict`: 0.11 MB
   - `input_size`: < 1 KB
   - `episodes`: < 1 KB
   - `final_avg_score`: < 1 KB
   - PyTorch overhead: ~10-20%

3. **Total file size**: **~0.13 MB** (130 KB)

## Comparison with Other Models

- `edon_fail_risk_v1.pt`: 0.017 MB (17 KB)
- `edon_v6.pt`: 0.110 MB (110 KB)
- `edon_v6_1.pt`: 0.111 MB (111 KB)
- `edon_v7.pt`: 0.220 MB (220 KB)
- **`edon_v8_strategy_v1.pt`**: **~0.11-0.13 MB** (estimated)

## Storage Impact

**Training 300 episodes on C drive:**
- Model file: **0.13 MB**
- Training logs: ~0-1 MB (if saved)
- **Total: < 2 MB**

**Very small!** C drive has plenty of space for this.

## Notes

- The model is very lightweight (smaller than v7)
- No need to worry about disk space
- Can easily train on C drive or D drive
- Model size doesn't change with number of training episodes (only weights change)

