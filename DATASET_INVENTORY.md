# Dataset Inventory

## Processed Datasets (Ready for Training)

### 1. MobiAct/WESAD Dataset
- **Location**: `edon-cav-engine/data/unified/mobiact.jsonl`
- **Windows**: **2,895 windows**
- **Source**: WESAD dataset (`data/raw/wesad/wesad_wrist_4hz.csv`)
- **Format**: JSONL with 240-sample windows
- **Features**: acc_x, acc_y, acc_z, eda, temp, bvp (synthetic for missing)
- **Status**: ✅ Ready for training

### 2. WISDM Dataset
- **Location**: `edon-cav-engine/data/unified/wisdm.jsonl`
- **Windows**: **0 windows** (empty - no source data available)
- **Status**: ⚠️ No data

### 3. Merged Dataset (v1.0)
- **Location**: `edon-cav-engine/data/unified/all_v10.jsonl`
- **Windows**: **2,895 windows** (from MobiAct/WESAD only)
- **Status**: ✅ Ready for training (used to train v4.0 model)

## Raw Source Datasets

### 4. WESAD Raw Data
- **Location**: `data/raw/wesad/wesad_wrist_4hz.csv`
- **Rows**: ~347,474 rows (including header)
- **Format**: CSV with columns: subject, t_sec, EDA, TEMP, BVP, ACC_x, ACC_y, ACC_z, ACC_mag, label
- **Subjects**: 15 subjects (S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17)
- **Status**: ✅ Available (used to generate MobiAct JSONL)

### 5. WESAD Full Dataset
- **Location**: `data/raw/wesad/WESAD/`
- **Format**: Individual CSV files per subject (ACC.csv, BVP.csv, EDA.csv, TEMP.csv, etc.)
- **Subjects**: 15 subjects with full sensor data
- **Status**: ✅ Available

### 6. Large Processed Dataset
- **Location**: `cav_engine_v3_2_LGBM_2025-11-08/oem_100k_windows.parquet`
- **Windows**: **100,000 windows** (already processed)
- **Format**: Parquet
- **Status**: ✅ Available (can be converted to JSONL if needed)

## Sample/Test Datasets

### 7. Sample Data
- **Location**: `edon-cav-engine/data/sample.jsonl`
- **Purpose**: Test/sample data
- **Status**: Available

### 8. Overload Ramp
- **Location**: `edon-cav-engine/data/overload_ramp.jsonl`
- **Purpose**: Test scenario data
- **Status**: Available

### 9. Real WESAD CSV
- **Location**: `edon-cav-engine/sensors/real_wesad.csv`
- **Purpose**: Sensor data for testing
- **Status**: Available

## Summary

| Dataset | Location | Windows/Rows | Status |
|---------|----------|--------------|--------|
| **MobiAct/WESAD** | `data/unified/mobiact.jsonl` | **2,895** | ✅ Ready |
| **WISDM** | `data/unified/wisdm.jsonl` | **0** | ⚠️ Empty |
| **All v1.0** | `data/unified/all_v10.jsonl` | **2,895** | ✅ Ready |
| **WESAD Raw** | `data/raw/wesad/wesad_wrist_4hz.csv` | **~347k rows** | ✅ Source |
| **100k Parquet** | `cav_engine_v3_2_LGBM_2025-11-08/oem_100k_windows.parquet` | **100,000** | ✅ Available |

## Model Training Status

- **Model v4.0**: Trained on **2,895 windows** from MobiAct/WESAD dataset
- **Training successful**: ✅ Model saved to `models/cav_engine_v4_0/cav_engine_v4_0.pkl`

## Notes

- The build script successfully parsed WESAD data and created 2,895 windows
- WISDM parsing produced 0 windows (no source data available)
- The merged `all_v10.jsonl` contains only MobiAct/WESAD data (2,895 windows)
- The 100k parquet file is available but wasn't used in the v1.0 build

