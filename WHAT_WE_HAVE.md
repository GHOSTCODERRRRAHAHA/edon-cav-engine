# What We Have So Far - EDON v1.0 SDK Summary

## 📁 Project Structure

### Core Application (`app/`)
- **`app/main.py`** - FastAPI application with all routes mounted
- **`app/engine.py`** - CAV computation engine (loads models, computes features, predicts)
- **`app/models.py`** - Pydantic models for API requests/responses
- **`app/adaptive_memory.py`** - Adaptive memory engine (24-hour rolling context)
- **`app/routes/`** - API endpoints:
  - `cav.py` - Single CAV computation (`POST /cav`)
  - `batch.py` - Batch CAV computation (`POST /oem/cav/batch`)
  - `telemetry.py` - Health & telemetry (`GET /health`, `/telemetry`)
  - `models.py` - Model info (`GET /models/info`)
  - `metrics.py` - Prometheus metrics (`GET /metrics`)
  - `streaming.py` - SSE/WebSocket streaming
  - `memory.py` - Memory management
  - `dashboard.py` - Dashboard data
  - `ingest.py` - Data ingestion
  - `debug.py` - Debug endpoints

### Utilities (`app/utils/`)
- **`feature_ingest.py`** - Input normalization utilities:
  - `looks_raw()` - Detects raw windows (240-length arrays)
  - `featurize_raw()` - Converts raw arrays to 6 features
  - `normalize_feature_map()` - Normalizes feature maps
  - `guard_features()` - Feature schema validation (gated by `EDON_STRICT_FEATURES`)
  - `normalize_to_engine_format()` - Normalizes keys for engine

## 🔧 Key Features Implemented

### 1. **Dual Input Format Support**
- ✅ Accepts **raw windows**: `{EDA, TEMP, BVP, ACC_x, ACC_y, ACC_z}` (240-length arrays)
- ✅ Accepts **feature maps**: `{eda_mean, eda_std, bvp_mean, bvp_std, acc_mean, acc_std}`
- ✅ Case-insensitive key handling (uppercase/lowercase/aliases)
- ✅ Automatic featurization of raw windows

### 2. **API Endpoints**
- ✅ `POST /cav` - Single window CAV computation
- ✅ `POST /oem/cav/batch` - Batch processing (1-5 windows per request)
- ✅ `GET /health` - Health check with model info and uptime
- ✅ `GET /models/info` - Model metadata (name, hash, features, window, PCA dims)
- ✅ `GET /telemetry` - System telemetry
- ✅ `GET /metrics` - Prometheus metrics

### 3. **Thread Safety**
- ✅ Sequential batch processing with `threading.Lock()` (engine has shared EMA state)
- ✅ Thread-safe access to shared `CAVEngine` instance

### 4. **Feature Guard**
- ✅ Configurable via `EDON_STRICT_FEATURES` env flag (default: `true`)
- ✅ Only runs on feature-map payloads (skips for raw windows)
- ✅ Validates expected features are present

### 5. **Model Management**
- ✅ Model discovery with priority:
  1. `cav_engine_v3_2_*/cav_state_v3_2.joblib` (production)
  2. `models/*/cav_state_v3_2.joblib`
  3. `models/*/HASHES.txt`
  4. Any other model files (skips `cav_embedder.joblib`)
- ✅ SHA256 hash verification
- ✅ Model info logged on startup

## 📊 Datasets

### In `edon-cav-engine/data/`:
- ✅ `unified/all_v10.jsonl` - Combined dataset (v1.0)
- ✅ `unified/wisdm.jsonl` - WISDM dataset
- ✅ `unified/mobiact.jsonl` - MobiAct dataset
- ✅ `external/mobiact/mobiact.csv` - Raw MobiAct data
- ⚠️ `raw/wesad/wesad_wrist_4hz.csv` - **Needs to be copied** (for evaluation)

### In Parent `EDON/data/`:
- ✅ `raw/wesad/wesad_wrist_4hz.csv` - WESAD data (source)
- ✅ `raw/wesad/WESAD/` - Raw WESAD subject folders (S2-S17)

## 🧪 Validation Tests

### Test Scripts Created:
1. **`test_1_model_info.py`** - Tests `/models/info` endpoint
2. **`test_2_evaluation.py`** - Runs WESAD ground-truth evaluation
3. **`test_3_load_test.py`** - Load tests batch endpoint
4. **`run_validation_tests.ps1`** - PowerShell script to run all 3 tests
5. **`run_tests_one_by_one.py`** - Interactive Python script (runs tests with prompts)

### Test Requirements:
- **Test 1**: Server running, `/models/info` accessible
- **Test 2**: WESAD data at `data/raw/wesad/wesad_wrist_4hz.csv`
- **Test 3**: Server running, batch endpoint working

### Success Criteria:
- ✅ Model info returns valid metadata
- ✅ Evaluation: Accuracy and AUROC metrics
- ✅ Load test: ≥95% success rate AND p95 latency < 120ms

## 🛠️ Tools & Scripts

### Build & Setup:
- ✅ `build_v1.ps1` - Complete v1.0 SDK build pipeline
- ✅ `setup_training_data.ps1` - Sets up training data
- ✅ `COPY_MISSING_FILES.ps1` - Copies models, tools, config, WESAD data
- ✅ `COPY_ALL_DATASETS.ps1` - Copies all required datasets
- ✅ `COPY_WESAD_DATA.ps1` - Copies WESAD data specifically

### Dataset Processing:
- ✅ `tools/parse_wisdm.py` - Parses WISDM dataset
- ✅ `tools/parse_mobiact.py` - Parses MobiAct dataset
- ✅ `tools/train_cav_model.py` - Trains CAV models
- ✅ `tools/eval_wesad.py` - Evaluates on WESAD ground truth
- ✅ `tools/load_test.py` - Load tests batch endpoint
- ✅ `tools/oem_dashboard.py` - Streamlit dashboard

### Utilities:
- ✅ `tools/generate_hashes.py` - Generates `HASHES.txt` for models
- ✅ `tools/freeze_openapi.py` - Freezes OpenAPI schema
- ✅ `tools/manifest_utils.py` - Dataset manifest utilities

## 📝 Documentation

### Status Documents:
- ✅ `VALIDATION_FIXES.md` - Validation test fixes applied
- ✅ `DATASET_LOCATIONS.md` - Where all datasets are located
- ✅ `CAV_NORMALIZATION_IMPLEMENTATION.md` - Input normalization details
- ✅ `BUILD_SUCCESS.md` - Build completion status
- ✅ `PILOT_READINESS.md` - Pilot readiness assessment
- ✅ `CURRENT_STATUS.md` - Current project status

## 🔑 API Keys

- ✅ OpenWeatherMap API key configured
- ✅ AirNow (EPA) API key configured
- ✅ `SET_API_KEYS.ps1` - Script to set environment variables

## ⚙️ Configuration

### Environment Variables:
- `EDON_STRICT_FEATURES` - Enable/disable strict feature guard (default: `true`)
- `EDON_RATE_LIMIT` - Rate limit for batch endpoint (default: `60/minute`)
- `OPENWEATHER_API_KEY` - OpenWeatherMap API key
- `AIRNOW_API_KEY` - AirNow API key

### Model Files:
- ✅ `cav_engine_v3_2_LGBM_2025-11-08/cav_state_v3_2.joblib` - Production model
- ✅ `cav_engine_v3_2_LGBM_2025-11-08/cav_state_scaler_v3_2.joblib` - Scaler
- ✅ `cav_engine_v3_2_LGBM_2025-11-08/cav_state_schema_v3_2.json` - Feature schema

## 🚀 Ready to Run

### Start Server:
```powershell
cd C:\Users\cjbig\Desktop\EDON\edon-cav-engine
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Run Validation Tests:
```powershell
# All tests at once
.\run_validation_tests.ps1

# Or one by one
.\venv\Scripts\python.exe test_1_model_info.py
.\venv\Scripts\python.exe test_2_evaluation.py
.\venv\Scripts\python.exe test_3_load_test.py
```

## ⚠️ Known Issues / TODO

1. **WESAD Data**: Needs to be copied from parent folder (run `COPY_MISSING_FILES.ps1`)
2. **Feature Map Inference**: Engine doesn't support feature-map-only inference yet (requires raw windows)
3. **Batch Method**: `ENGINE.cav_from_features_batch()` doesn't exist yet (uses `cav_from_window()` per window)

## ✅ What's Working

- ✅ Raw window input (240-length arrays)
- ✅ Case-insensitive key handling
- ✅ Automatic featurization
- ✅ Batch processing with thread safety
- ✅ Model discovery and info
- ✅ Health checks
- ✅ Prometheus metrics
- ✅ Load testing framework
- ✅ Evaluation framework

## 📦 Next Steps

1. **Copy WESAD data**: Run `.\COPY_MISSING_FILES.ps1`
2. **Run validation tests**: `.\run_validation_tests.ps1`
3. **Fix any issues** that come up
4. **Verify all 3 tests pass**

