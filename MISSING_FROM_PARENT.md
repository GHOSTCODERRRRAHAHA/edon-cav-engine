# Missing Files from Parent EDON Folder

## Critical Files Needed

### 1. Model Artifacts (REQUIRED)
**Location in parent:** `cav_engine_v3_2_LGBM_2025-11-08/`
- `cav_state_v3_2.joblib` - ✅ EXISTS in parent, ❌ MISSING in subfolder
- `cav_state_scaler_v3_2.joblib` - ✅ EXISTS in parent, ❌ MISSING in subfolder  
- `cav_state_schema_v3_2.json` - ✅ EXISTS in both
- `oem_100k_windows.parquet` - ✅ EXISTS in parent (100k windows dataset)

**Why needed:** `app/engine_loader_patch.py` looks in parent directory for these files.

### 2. App Middleware (OPTIONAL - for v0.9 features)
**Location in parent:** `app/middleware/`
- `auth.py` - Token authentication middleware
- `rate_limit.py` - Rate limiting middleware  
- `validation.py` - OpenAPI schema validation
- `__init__.py`

**Status:** ❌ NOT in edon-cav-engine (but may not be needed if running standalone)

### 3. App Core Files (CHECK IF NEEDED)
**Location in parent:** `app/`
- `metrics.py` - Prometheus metrics definitions
- `model_info.py` - Model info utilities
- `edge_bridge.py` - MQTT edge bridge
- `adapt.py` - Adaptation logic
- `memory_lite.py` - Lightweight memory

**Status:** ❌ NOT in edon-cav-engine (check if app uses these)

### 4. Tools (NEEDED for v1.0 SDK)
**Location in parent:** `tools/`
- `oem_dashboard.py` - ✅ EXISTS in parent, ❌ MISSING in subfolder
- `manifest_utils.py` - ✅ EXISTS in parent, ❌ MISSING in subfolder
- `eval_wesad.py` - Evaluation script
- `load_test.py` - Load testing script
- `freeze_openapi.py` - OpenAPI schema freezing
- `generate_hashes.py` - Hash generation

**Status:** Some tools exist in both, but dashboard and manifest_utils are missing

### 5. Configuration Files
**Location in parent:**
- `config/config.yaml` - ✅ EXISTS in parent, ❌ MISSING in subfolder

### 6. Infrastructure Configs
**Location in parent:**
- `prometheus/prometheus.yml` - Prometheus config
- `grafana/` - Grafana dashboards
- `mqtt/config/` - MQTT configuration

**Status:** ❌ NOT in edon-cav-engine (optional for local dev)

### 7. Documentation
**Location in parent:**
- `V1.0_QUICK_START.md` - ✅ EXISTS in parent
- `V1.0_SDK_SUMMARY.md` - ✅ EXISTS in parent

### 8. Raw Datasets
**Location in parent:** `data/raw/wesad/`
- `wesad_wrist_4hz.csv` - ✅ EXISTS in parent (347k rows)
- `WESAD/` directory with full dataset

**Status:** ❌ NOT in edon-cav-engine (but can reference parent)

## Recommendations

### HIGH PRIORITY - Copy These:
1. **Model files** from `cav_engine_v3_2_LGBM_2025-11-08/`:
   ```powershell
   Copy-Item ..\cav_engine_v3_2_LGBM_2025-11-08\cav_state_v3_2.joblib cav_engine_v3_2_LGBM_2025-11-08\
   Copy-Item ..\cav_engine_v3_2_LGBM_2025-11-08\cav_state_scaler_v3_2.joblib cav_engine_v3_2_LGBM_2025-11-08\
   ```

2. **Tools**:
   ```powershell
   Copy-Item ..\tools\oem_dashboard.py tools\
   Copy-Item ..\tools\manifest_utils.py tools\
   ```

3. **Config**:
   ```powershell
   New-Item -ItemType Directory -Force -Path config
   Copy-Item ..\config\config.yaml config\
   ```

### MEDIUM PRIORITY - Check Dependencies:
- Check if `app/main.py` imports from `app.metrics`, `app.edge_bridge`, etc.
- If yes, copy those files or update imports

### LOW PRIORITY - Optional:
- Prometheus/Grafana configs (only if using monitoring)
- MQTT config (only if using edge bridge)

## Quick Copy Script

```powershell
# Run from edon-cav-engine directory
$parent = ".."

# Copy model files
Copy-Item "$parent\cav_engine_v3_2_LGBM_2025-11-08\cav_state_v3_2.joblib" "cav_engine_v3_2_LGBM_2025-11-08\" -ErrorAction SilentlyContinue
Copy-Item "$parent\cav_engine_v3_2_LGBM_2025-11-08\cav_state_scaler_v3_2.joblib" "cav_engine_v3_2_LGBM_2025-11-08\" -ErrorAction SilentlyContinue

# Copy tools
Copy-Item "$parent\tools\oem_dashboard.py" "tools\" -ErrorAction SilentlyContinue
Copy-Item "$parent\tools\manifest_utils.py" "tools\" -ErrorAction SilentlyContinue

# Copy config
New-Item -ItemType Directory -Force -Path config | Out-Null
Copy-Item "$parent\config\config.yaml" "config\" -ErrorAction SilentlyContinue
```

