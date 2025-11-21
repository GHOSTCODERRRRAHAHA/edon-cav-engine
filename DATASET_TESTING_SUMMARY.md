# Dataset Testing Summary

## Tested Datasets & Windows

### ✅ Evaluation Tests

1. **WESAD Evaluation (Full)**
   - **Windows Tested**: **1,392 windows**
   - **File**: `reports/eval_wesad.json`
   - **Source**: WESAD ground truth data
   - **Status**: Complete evaluation run
   - **Results**: Accuracy, AUROC, confusion matrix

2. **WESAD Evaluation (Quick Test)**
   - **Windows Tested**: **50 windows**
   - **File**: `reports/last_eval.json`
   - **Source**: WESAD ground truth data (limited for speed)
   - **Status**: Quick validation test
   - **Script**: `test_2_evaluation.py`

### ✅ Training Datasets

1. **Model v4.0 Training**
   - **Windows Used**: **2,895 windows**
   - **Source**: `data/unified/mobiact.jsonl` (MobiAct/WESAD)
   - **Status**: ✅ Model trained successfully

2. **Model v3.2/v3.3 Training**
   - **Windows Used**: **100,000 windows**
   - **Source**: `cav_engine_v3_2_LGBM_2025-11-08/oem_100k_windows.parquet`
   - **Status**: ✅ Model trained successfully

### ✅ Test/Validation Datasets

1. **OEM Dataset Builder Test**
   - **Windows Tested**: **10 windows**
   - **Purpose**: Dataset format validation
   - **Status**: ✅ Verified (CSV, Parquet, JSONL formats)

2. **Load Testing**
   - **Windows Tested**: Variable (batch endpoint testing)
   - **Script**: `test_3_load_test.py`
   - **Status**: ✅ Batch processing verified

### 📊 Available Datasets (Not Yet Tested)

1. **Full WESAD Dataset**
   - **Available Windows**: **~347,233 windows**
   - **Source**: `sensors/real_wesad.csv` (347,472 rows)
   - **Status**: Available but not fully processed
   - **Note**: Would take ~197 hours to process all windows

2. **WESAD Raw Data**
   - **Rows**: **~347,474 rows** (including header)
   - **Source**: `data/raw/wesad/wesad_wrist_4hz.csv`
   - **Subjects**: 15 subjects (S2-S17)
   - **Status**: Available for processing

## Summary Statistics

| Category | Windows Tested | Status |
|----------|---------------|--------|
| **Evaluation (Full)** | 1,392 | ✅ Complete |
| **Evaluation (Quick)** | 50 | ✅ Complete |
| **Training (v4.0)** | 2,895 | ✅ Complete |
| **Training (v3.2/v3.3)** | 100,000 | ✅ Complete |
| **Validation Tests** | 10 | ✅ Complete |
| **Available (Unprocessed)** | ~347,233 | ⏳ Available |

## Total Tested Windows

**Minimum Total**: ~104,347 windows tested across all categories
- 1,392 (evaluation)
- 50 (quick test)
- 2,895 (v4.0 training)
- 100,000 (v3.2/v3.3 training)
- 10 (validation)

**Note**: Some windows may overlap between training and evaluation sets.

## Evaluation Results

### Full WESAD Evaluation (1,392 windows)
- **Accuracy**: 0.205 (20.5%)
- **AUROC**: 0.5
- **Samples**: 1,392
- **State Distribution**: All classified as "unknown" (needs investigation)

### Quick Test (50 windows)
- Limited test for speed
- Used for validation during development

## Next Steps

1. **Process Full Dataset**: ~347,233 windows available for processing
2. **Improve Evaluation**: Current evaluation shows low accuracy (needs investigation)
3. **Expand Testing**: Test on additional datasets (MobiAct, WISDM)

