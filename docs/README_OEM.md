# EDON CAV Engine v3.2 - OEM SDK Integration Guide

**Version**: v3.2  
**Date**: 2025-11-08  
**Model**: LightGBM Classifier

## Overview

The EDON CAV Engine v3.2 SDK provides a robust, path-agnostic inference system for OEM partners. The SDK automatically discovers model artifacts across multiple directory layouts, making integration seamless regardless of deployment structure.

## Key Features

- **Robust Artifact Discovery**: Automatically finds models in multiple locations:
  1. `$EDON_MODEL_DIR` environment variable (if set)
  2. `./models/` directory (preferred)
  3. Repository root
  4. `cav_engine_v3_2_*` dated folders (fallback)

- **Path-Agnostic Design**: Works regardless of where the SDK is extracted or deployed

- **Helper Scripts**: PowerShell scripts for quick demo and verification

- **Hash Verification**: SHA256 hashes included for integrity checking

## Quick Start

### 1. Extract SDK

Extract the SDK ZIP file to your desired location:

```powershell
Expand-Archive -Path EDON_CAV_v3_2_OEM_SDK_*.zip -DestinationPath .
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run Demo

**Option A: Using helper script (Windows)**
```powershell
.\run_demo.ps1
```

**Option B: Direct Python**
```powershell
python demo_infer.py
```

### 4. Run Dashboard

**Option A: Using helper script**
```powershell
.\run_demo.ps1 -Dashboard
```

**Option B: Direct Streamlit**
```powershell
streamlit run cav_dashboard.py
```

## SDK Structure

```
EDON_CAV_v3_2_OEM_SDK/
├── models/                          # Model artifacts (preferred location)
│   ├── cav_state_v3_2.joblib       # LightGBM model
│   ├── cav_state_scaler_v3_2.joblib # Feature scaler
│   └── cav_state_schema_v3_2.json  # Feature schema & metadata
├── demo_infer.py                    # Robust demo inference script
├── cav_dashboard.py                 # Streamlit dashboard
├── add_v32_features.py              # Feature engineering utility
├── run_demo.ps1                    # Quick demo runner
├── verify_hashes.ps1               # Hash verification script
├── requirements.txt                # Python dependencies
├── HASHES.txt                      # SHA256 hashes for verification
├── MANIFEST.txt                    # File manifest
├── README_SDK.txt                  # Quick reference
└── docs/                           # Documentation
    ├── README_OEM.md              # This file
    ├── OEM_BRIEF.md               # OEM brief
    ├── EVALUATION_LICENSE.md      # License terms
    └── ...
```

## Model Artifacts

### Required Files

- `cav_state_v3_2.joblib` - Trained LightGBM classifier
- `cav_state_scaler_v3_2.joblib` - StandardScaler for feature normalization
- `cav_state_schema_v3_2.json` - Feature order and state mapping

### Schema Format

The schema JSON contains:
- `feature_names`: Ordered list of feature names (used for inference)
- `reverse_state_map`: Maps class indices to state labels
  - `"0"`: "balanced"
  - `"1"`: "focus"
  - `"2"`: "restorative"

## Python API Usage

### Basic Inference

```python
from pathlib import Path
import json
import pandas as pd
from joblib import load

# The SDK scripts handle path discovery automatically
# For custom code, use the same pattern:

ROOT = Path(__file__).resolve().parent

def find_artifact(name: str, also=None) -> Path:
    candidates = []
    env_dir = os.getenv("EDON_MODEL_DIR")
    if env_dir: candidates.append(Path(env_dir) / name)
    candidates.append(ROOT / "models" / name)
    candidates.append(ROOT / name)
    candidates += [(p / name) for p in ROOT.glob("cav_engine_v3_2_*")]
    if also: candidates += [ROOT / a for a in also]
    for c in candidates:
        if c.exists(): return c
    raise FileNotFoundError(f"{name} not found")

# Load artifacts
SCHEMA_PATH = find_artifact("cav_state_schema_v3_2.json")
SCALER_PATH = find_artifact("cav_state_scaler_v3_2.joblib")
MODEL_PATH = find_artifact("cav_state_v3_2.joblib")

# Load schema
with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

FEATURES = schema.get("feature_names", [
    "eda_mean", "eda_deriv_std", "eda_deriv_pos_rate",
    "bvp_std", "acc_magnitude_mean", "acc_var",
])

# Load model and scaler
scaler = load(SCALER_PATH)
model = load(MODEL_PATH)

# Prepare input (6 features in exact order)
row = {
    "eda_mean": -0.80,
    "eda_deriv_std": 0.45,
    "eda_deriv_pos_rate": 0.50,
    "bvp_std": 1.20,
    "acc_magnitude_mean": 2.10,
    "acc_var": 0.90,
}

# Build DataFrame in feature order
X = pd.DataFrame([[row[k] for k in FEATURES]], columns=FEATURES)

# Scale and predict
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)
pred_idx = int(model.predict(X_scaled_df)[0])

# Map to state label
reverse_map = {int(k): v for k, v in schema.get("reverse_state_map", {}).items()}
state = reverse_map.get(pred_idx, "unknown")

print(f"Predicted state: {state}")
```

## Feature Requirements

The v3.2 model requires exactly 6 features in this order:

1. `eda_mean` - EDA signal mean
2. `eda_deriv_std` - EDA derivative standard deviation
3. `eda_deriv_pos_rate` - EDA derivative positive rate
4. `bvp_std` - BVP signal standard deviation
5. `acc_magnitude_mean` - Accelerometer magnitude mean
6. `acc_var` - Accelerometer variance

## State Labels

- **balanced** (class 0): Normal operating state
- **focus** (class 1): High engagement/performance state
- **restorative** (class 2): Recovery/rest state

## Verification

### Verify File Integrity

```powershell
.\verify_hashes.ps1
```

For verbose output:
```powershell
.\verify_hashes.ps1 -Verbose
```

### Manual Verification

```powershell
# Check a specific file
Get-FileHash -Path "models\cav_state_v3_2.joblib" -Algorithm SHA256

# Compare with HASHES.txt
```

## Environment Variables

### EDON_MODEL_DIR

Set this environment variable to override the default model search locations:

```powershell
$env:EDON_MODEL_DIR = "C:\MyModels"
python demo_infer.py
```

```bash
export EDON_MODEL_DIR="/path/to/models"
python demo_infer.py
```

## Troubleshooting

### Model Not Found

If you see `FileNotFoundError`, the SDK is looking in these locations:
1. `$EDON_MODEL_DIR` (if set)
2. `./models/`
3. Repository root
4. `cav_engine_v3_2_*` folders

**Solution**: Ensure model files are in one of these locations, or set `EDON_MODEL_DIR`.

### Missing Features

The model requires all 6 v3.2 features. If your data is missing features:
- Use `add_v32_features.py` to compute missing features
- Or provide default values (see `demo_infer.py` for examples)

### Import Errors

Ensure all dependencies are installed:
```powershell
pip install -r requirements.txt
```

## Integration Checklist

- [ ] Extract SDK to deployment location
- [ ] Install Python dependencies (`pip install -r requirements.txt`)
- [ ] Verify model files exist (check `models/` directory)
- [ ] Run `demo_infer.py` to test inference
- [ ] Verify hashes with `verify_hashes.ps1`
- [ ] Integrate inference code into your application
- [ ] Test with your sensor data format
- [ ] Set `EDON_MODEL_DIR` if using custom model location

## Performance

- **Inference Latency**: ~1-5ms per prediction (CPU)
- **Model Size**: ~500KB (LightGBM)
- **Memory**: ~50MB runtime footprint
- **Throughput**: ~200-500 predictions/second (single-threaded)

## Support

For OEM integration support:
- Email: oem@edon.ai
- Documentation: See `docs/` directory
- API Reference: See `docs/CAV_SPEC.md`

## License

See `docs/EVALUATION_LICENSE.md` for license terms and usage restrictions.

## Version History

- **v3.2** (2025-11-08): Robust artifact discovery, helper scripts, models packaged under `models/`
- **v3.1**: Previous version
- **v2**: Legacy version

---

**EDON CAV Engine v3.2** - Context-Aware Vector scoring for OEM partners.
