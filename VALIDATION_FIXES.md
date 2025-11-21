# Validation Test Fixes Applied

## Changes Made to `app/routes/batch.py`

1. **Fixed raw window tracking**: Added `is_raw_list` to track which windows are raw, preventing double-checking `looks_raw()`.

2. **Handler now properly**:
   - Detects raw vs feature-map windows
   - Featurizes raw windows or normalizes feature maps
   - Runs feature guard only when all windows are feature maps (gated by `EDON_STRICT_FEATURES` env flag)
   - Processes windows sequentially with thread-safe locking
   - Returns proper `BatchResponse` format

## Testing

The handler should now work correctly with:
- Raw windows (EDA, TEMP, BVP, ACC_x, ACC_y, ACC_z arrays of length 240)
- Feature maps (eda_mean, eda_std, bvp_mean, bvp_std, acc_mean, acc_std)

## To Run Validation Tests

1. Start the server:
   ```powershell
   cd C:\Users\cjbig\Desktop\EDON\edon-cav-engine
   .\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

2. In another terminal, run:
   ```powershell
   cd C:\Users\cjbig\Desktop\EDON\edon-cav-engine
   .\run_validation_tests.ps1
   ```

Or use the Python test script:
```powershell
.\venv\Scripts\python.exe run_tests.py
```

## Expected Results

- **Test 1 (Model Info)**: Should return model metadata
- **Test 2 (Evaluation)**: May skip if WESAD data not found
- **Test 3 (Load Test)**: Should show >=95% success rate and p95 < 120ms

## Known Limitations

- Feature-map-only inference is not yet supported (engine requires raw windows)
- The handler will return an error for pure feature-map inputs until `ENGINE.cav_from_features_batch()` is implemented

