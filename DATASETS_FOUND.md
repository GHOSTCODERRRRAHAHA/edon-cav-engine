# Datasets Found in EDON

## Available Datasets

### 1. WESAD Dataset (Ready to Use)
**Location:** `data/raw/wesad/wesad_wrist_4hz.csv` (from parent EDON folder)
**Format:** CSV with columns: `subject,t_sec,EDA,TEMP,BVP,ACC_x,ACC_y,ACC_z,ACC_mag,label`
**Size:** ~347k rows
**Status:** ✅ Can be used with MobiAct parser (has acc_x, acc_y, acc_z columns)

### 2. Processed Datasets (Empty)
**Location:** `edon-cav-engine/data/unified/`
- `wisdm.jsonl` - Empty file
- `mobiact.jsonl` - Empty file

### 3. Large Processed Dataset
**Location:** `cav_engine_v3_2_LGBM_2025-11-08/oem_100k_windows.parquet`
**Status:** ✅ 100k windows already processed

### 4. Sample Data
**Location:** `edon-cav-engine/data/`
- `sample.jsonl` - Sample windows
- `overload_ramp.jsonl` - Test data

## Missing Datasets

### WISDM Raw Data
**Expected:** `data/external/wisdm/` (directory with .txt files)
**Status:** ❌ Not found

### MobiAct Raw Data  
**Expected:** `data/external/mobiact/` (directory with .csv files)
**Status:** ❌ Not found

## Recommendations

1. **Use WESAD data** - The `wesad_wrist_4hz.csv` can be converted to MobiAct format
2. **Use existing parquet** - The 100k windows parquet can be converted to JSONL
3. **Create synthetic data** - Generate minimal test data for immediate testing

## Quick Fix Options

### Option A: Use WESAD as MobiAct source
```powershell
# Copy WESAD CSV to external/mobiact
New-Item -ItemType Directory -Force -Path data\external\mobiact
Copy-Item ..\..\data\raw\wesad\wesad_wrist_4hz.csv data\external\mobiact\mobiact.csv
```

### Option B: Use existing parquet
```python
import pandas as pd
df = pd.read_parquet('cav_engine_v3_2_LGBM_2025-11-08/oem_100k_windows.parquet')
# Convert to JSONL format
```

### Option C: Create minimal synthetic data
```powershell
# Create minimal test dataset
@'
{"acc_x":[0,0,0,0,0,0,0,0,0,0],"acc_y":[0,0,0,0,0,0,0,0,0,0],"acc_z":[1,1,1,1,1,1,1,1,1,1],"eda":[0,0,0,0,0,0,0,0,0,0],"temp":[36.5,36.5,36.5,36.5,36.5,36.5,36.5,36.5,36.5,36.5],"bvp":[0,0,0,0,0,0,0,0,0,0],"temp_c":22,"humidity":45,"aqi":40,"local_hour":14}
'@ | Set-Content data\unified\all_v10.jsonl -Encoding UTF8
```

