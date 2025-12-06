# V4.1 Incremental Testing Plan

## Objective

Test different parameter combinations incrementally and save the configuration that achieves the **highest percentage improvement consistently** (not just best single run).

## Testing Strategy

### 1. Test Configurations

6 configurations to test:

| Test | Name | BASE_GAIN | PREFALL_MIN | PREFALL_MAX | Notes |
|------|------|-----------|-------------|-------------|-------|
| 1 | V4.1_baseline | 0.50 | 0.15 | 0.60 | Baseline (from +7.0% result) |
| 2 | V4.1_base_055 | 0.55 | 0.15 | 0.60 | Higher base gain |
| 3 | V4.1_prefall_wide | 0.50 | 0.10 | 0.65 | Wider PREFALL range |
| 4 | V4.1_prefall_narrow | 0.50 | 0.18 | 0.55 | Narrower PREFALL range |
| 5 | V4.1_prefall_high | 0.50 | 0.15 | 0.70 | Higher PREFALL max |
| 6 | V4.1_base_045_wide | 0.45 | 0.12 | 0.65 | Lower base + wider PREFALL |

### 2. Testing Process

For each configuration:
1. **Apply parameters** to `run_eval.py` code
2. **Run baseline** (3 seeds × 30 episodes each)
3. **Run EDON** (3 seeds × 30 episodes each)
4. **Compute improvements** (interventions, stability, average)
5. **Calculate statistics** (mean, std, min, max)

### 3. Selection Criteria

**Best configuration** = Highest **mean improvement** with:
- **Consistent performance** (low std deviation preferred)
- **Positive results** across all seeds (or at least 2/3)
- **Reasonable range** (min not too negative)

### 4. Output

The script will:
- Print rankings of all configurations
- **Save best configuration** to `results/v41_incremental/best_config_v41.json`
- **Apply best configuration** to `run_eval.py` automatically
- Save all individual results for analysis

## Running the Tests

```bash
python test_v41_incremental.py
```

**Expected runtime**: ~2-4 hours (depending on simulation speed)
- 6 configurations
- 3 seeds per configuration
- 2 modes (baseline + EDON)
- 30 episodes per run

## Expected Results

Based on historical data:
- **V4.1_baseline**: Expected +6-7% (baseline from +7.0% result)
- **V4.1_base_055**: May improve stability further
- **V4.1_prefall_wide**: May help in high-risk situations
- **V4.1_prefall_narrow**: May reduce overcorrection
- **V4.1_prefall_high**: May provide stronger corrections
- **V4.1_base_045_wide**: May balance base gain and PREFALL

## Success Criteria

**Target**: Find configuration that achieves:
- **Mean improvement**: +5% to +8% (consistent)
- **Std deviation**: < 2% (consistent across seeds)
- **Min improvement**: > 0% (all seeds positive or at least 2/3)

## Files Created

- `results/v41_incremental/best_config_v41.json` - Best configuration
- `results/v41_incremental/*_baseline_seed*.json` - Baseline results
- `results/v41_incremental/*_edon_seed*.json` - EDON results

## Next Steps After Testing

1. **Review results** - Check which configuration performed best
2. **Validate best config** - Run 5-10 seeds to confirm consistency
3. **Lock configuration** - Save as V4.1_FINAL
4. **Test across profiles** - Verify on medium_stress, hell_stress
5. **Document findings** - Update V4.1 documentation

## Notes

- Tests run on `high_stress` profile (most challenging)
- EDON gain fixed at 0.75 (optimal from previous tests)
- Each configuration tested with 3 seeds for consistency check
- Script automatically applies best configuration to code





