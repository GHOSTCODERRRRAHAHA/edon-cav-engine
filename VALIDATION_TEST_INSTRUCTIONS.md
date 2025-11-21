# Validation Test Instructions

I've created a PowerShell script to run all 3 validation tests. Here's how to use it:

## Quick Start

```powershell
cd C:\Users\cjbig\Desktop\EDON\edon-cav-engine
.\run_validation_tests.ps1
```

## What the Script Does

### Test 1: Model Info Verification
- Checks which model is loaded (should be v3.2 LGBM)
- Saves model metadata to `reports/model_info.json`
- Verifies model hash and configuration

### Test 2: Ground Truth Evaluation
- Runs `eval_wesad.py` on WESAD dataset
- Computes accuracy, AUROC, confusion matrix
- Saves results to `reports/last_eval.json`
- Limited to 50 windows for speed (can be increased)

### Test 3: Load Test
- Tests `/oem/cav/batch` endpoint
- 50 requests with 3 windows each
- 5 concurrent requests
- Verifies p95 latency < 120ms

## Manual Execution (If Script Fails)

### 1. Start Server (if not running)
```powershell
cd C:\Users\cjbig\Desktop\EDON\edon-cav-engine
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2. Test 1: Model Info
```powershell
curl http://127.0.0.1:8000/models/info
```

### 3. Test 2: Evaluation
```powershell
.\venv\Scripts\python.exe tools\eval_wesad.py --data ..\data\raw\wesad\wesad_wrist_4hz.csv --api http://127.0.0.1:8000 --output reports\last_eval.json --limit 50
```

### 4. Test 3: Load Test
```powershell
.\venv\Scripts\python.exe tools\load_test.py --url http://127.0.0.1:8000/oem/cav/batch --requests 50 --windows 3 --concurrent 5
```

## Expected Results

### Model Info
- **Model Name**: Should be `cav_state_v3_2` (not `cav_engine_v4_0`)
- **Algorithm**: LGBM (LightGBM)
- **Features**: 6

### Evaluation
- **Accuracy**: Should be > 0.65 (65%)
- **AUROC**: Should be > 0.70 (70%)
- **Confusion Matrix**: Should show reasonable class distribution

### Load Test
- **P95 Latency**: Should be < 120ms
- **Success Rate**: Should be 100%
- **Mean Latency**: Typically 20-50ms

## Troubleshooting

### Server Not Starting
- Check if port 8000 is already in use
- Verify virtual environment is activated
- Check for Python errors in the console

### Evaluation Fails
- Verify WESAD data exists at `..\data\raw\wesad\wesad_wrist_4hz.csv`
- Check API is responding: `curl http://127.0.0.1:8000/health`
- Reduce `--limit` if dataset is too large

### Load Test Fails
- Verify server is running
- Check rate limiting isn't blocking requests
- Increase `--concurrent` gradually if needed

## Results Location

All results are saved to `reports/`:
- `reports/model_info.json` - Model metadata
- `reports/last_eval.json` - Evaluation metrics

## Next Steps After Validation

1. **If all tests pass**: EDON is ready for pilot! ✅
2. **If model is wrong**: Update model discovery or set `EDON_MODEL_DIR`
3. **If accuracy is low**: Review model training or use more data
4. **If latency is high**: Optimize batch processing or reduce window size

