# OEM Dataset Pipeline - Command Reference

## Setup Commands

### 1. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install Requirements
```powershell
pip install -r requirements.txt
```

## API Server Commands

### Start API Server (Option 1: PowerShell Script)
```powershell
.\run_api.ps1
```

### Start API Server (Option 2: Manual)
```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8000/dashboard

## Dataset Pipeline Commands

### 1. Validate 10K Dataset
```powershell
.\venv\Scripts\Activate.ps1
python tools\validate_oem_dataset.py
```

**Options:**
```powershell
# Validate specific files
python tools\validate_oem_dataset.py --csv outputs\oem_sample_windows.csv --parquet outputs\oem_sample_windows.parquet --jsonl outputs\oem_sample_windows.jsonl

# Strict mode (treat warnings as errors)
python tools\validate_oem_dataset.py --strict
```

### 2. Enrich Dataset Context
```powershell
.\venv\Scripts\Activate.ps1
python tools\enrich_context.py
```

**Options:**
```powershell
# Custom ranges
python tools\enrich_context.py --input outputs\oem_sample_windows.csv --output outputs\oem_sample_windows_enriched.csv --temp-min 18 --temp-max 28 --humidity-min 30 --humidity-max 70 --aqi-min 20 --aqi-max 150 --hour-dist realistic

# Uniform hour distribution
python tools\enrich_context.py --hour-dist uniform
```

### 3. Train XGBoost Model v2
```powershell
.\venv\Scripts\Activate.ps1
python training\train_baseline_v2.py
```

**Options:**
```powershell
# Use enriched dataset
python training\train_baseline_v2.py --input outputs\oem_sample_windows_enriched.csv

# Custom model parameters
python training\train_baseline_v2.py --n-estimators 200 --max-depth 8 --learning-rate 0.05

# Custom train/test split
python training\train_baseline_v2.py --test-size 0.2 --val-size 0.2
```

**Output:**
- `models/cav_state_xgb_v2.joblib` - Trained model
- `models/cav_state_scaler_v2.joblib` - Feature scaler
- `models/cav_state_schema_v2.json` - Feature schema and metadata

### 4. Build 100K Dataset (Resumable)
```powershell
.\venv\Scripts\Activate.ps1
python tools\build_100k_dataset.py
```

**Options:**
```powershell
# Resume from checkpoint
python tools\build_100k_dataset.py --resume

# Clear checkpoint and start fresh
python tools\build_100k_dataset.py --clear-checkpoint

# Custom target size
python tools\build_100k_dataset.py --target 50000

# Custom batch size
python tools\build_100k_dataset.py --batch-size 200

# Custom checkpoint interval
python tools\build_100k_dataset.py --checkpoint-interval 5000
```

**Output:**
- `outputs/oem_100k_windows.csv` - CSV format
- `outputs/oem_100k_windows.parquet` - Parquet format
- `outputs/build_100k_checkpoint.json` - Checkpoint file (auto-removed on completion)

## Complete Workflow Example

### Step 1: Start API Server (Terminal 1)
```powershell
.\run_api.ps1
```

### Step 2: Validate Existing 10K Dataset (Terminal 2)
```powershell
.\venv\Scripts\Activate.ps1
python tools\validate_oem_dataset.py
```

### Step 3: Enrich Dataset Context (Terminal 2)
```powershell
python tools\enrich_context.py --input outputs\oem_sample_windows.csv --output outputs\oem_sample_windows_enriched.csv
```

### Step 4: Train Model (Terminal 2)
```powershell
python training\train_baseline_v2.py --input outputs\oem_sample_windows_enriched.csv
```

### Step 5: Build 100K Dataset (Terminal 2)
```powershell
python tools\build_100k_dataset.py --target 100000
```

## File Locations

### Input Files
- **Sensor Data**: `sensors/real_wesad.csv` or `data/real_wesad.csv`
- **10K Dataset**: `outputs/oem_sample_windows.csv`

### Output Files
- **Enriched Dataset**: `outputs/oem_sample_windows_enriched.csv`
- **100K Dataset**: `outputs/oem_100k_windows.csv` / `outputs/oem_100k_windows.parquet`
- **Trained Model**: `models/cav_state_xgb_v2.joblib`
- **Scaler**: `models/cav_state_scaler_v2.joblib`
- **Schema**: `models/cav_state_schema_v2.json`
- **Checkpoint**: `outputs/build_100k_checkpoint.json` (temporary)

## Notes

1. **API Server**: Must be running before building datasets or training models that use the API
2. **Resumable Building**: The 100K builder saves checkpoints every 1000 windows by default. If interrupted, use `--resume` to continue
3. **Windows Paths**: All scripts use Windows-safe path handling (backslashes or pathlib)
4. **Directories**: Scripts automatically create output directories if they don't exist
5. **Virtual Environment**: Always activate venv before running Python scripts

## Troubleshooting

### API Not Available
```powershell
# Check if server is running
curl http://localhost:8000/health

# Start server if not running
.\run_api.ps1
```

### Checkpoint Issues
```powershell
# Clear checkpoint to start fresh
python tools\build_100k_dataset.py --clear-checkpoint
```

### Missing Dependencies
```powershell
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

