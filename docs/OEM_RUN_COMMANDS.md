# EDON CAV Engine v3.2 - OEM Run Commands

**Quick reference for common operations**

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Activate virtual environment (if using)
.\venv\Scripts\Activate.ps1

# Install/update dependencies
pip install -r requirements.txt
```

### 2. Start API Server

```powershell
# Option 1: Direct Python
python app\main.py

# Option 2: Using PowerShell script
.\run_api.ps1

# Server will be available at: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### 3. Launch Dashboard

```powershell
streamlit run tools\cav_dashboard.py
```

---

## 📦 SDK Operations

### Generate Missing v3.2 Features

```powershell
# Add v3.2 features to existing parquet file
python tools\add_v32_features.py

# Output: outputs\oem_100k_windows_v32.parquet
```

### Package SDK for Distribution

```powershell
# Create versioned SDK zip
.\package_sdk.ps1

# With custom version
.\package_sdk.ps1 -Version "3.2.1"

# Output: dist\EDON_CAV_Engine_v3.2.0_SDK_YYYY-MM-DD.zip
```

---

## 🧪 Testing & Demo

### Run Demo Inference

```powershell
# Clean demo script (no warnings)
python demo_infer.py

# Original demo
python demo_infer_example.py
```

### Test API Endpoints

```powershell
# Health check
curl http://localhost:8000/health

# Single inference
curl -X POST http://localhost:8000/cav -H "Content-Type: application/json" -d @sdk\sample_payload.json

# Memory summary
curl http://localhost:8000/memory/summary
```

---

## 📊 Dataset Operations

### Build OEM Dataset

```powershell
# Test with 10 windows
python tools\build_oem_dataset.py --limit 10

# Full dataset (takes ~197 hours)
python tools\build_oem_dataset.py
```

### Validate Dataset

```powershell
python tools\validate_oem_dataset.py
```

---

## 🔧 Development Commands

### Run Tests

```powershell
pytest tests\ -v

# With coverage
pytest tests\ --cov=app --cov-report=html
```

### Train Model

```powershell
# Train v3.2 model
python training\train_baseline_v3_2.py
```

---

## 📝 Documentation

### View Documentation

- **API Docs**: http://localhost:8000/docs (when server running)
- **OEM Brief**: See `OEM_BRIEF.md`
- **Evaluation License**: See `EVALUATION_LICENSE.md`
- **NDA**: See `NDA.md`

---

## 🎯 Common Workflows

### Complete Setup (First Time)

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python verify_setup.py

# 3. Start server
python app\main.py

# 4. In another terminal, launch dashboard
streamlit run tools\cav_dashboard.py
```

### OEM Evaluation Package

```powershell
# 1. Generate v3.2 features if needed
python tools\add_v32_features.py

# 2. Package SDK
.\package_sdk.ps1

# 3. Deliver dist\EDON_CAV_Engine_v3.2.0_SDK_*.zip
```

### Daily Development

```powershell
# Start server
python app\main.py

# Launch dashboard (separate terminal)
streamlit run tools\cav_dashboard.py

# Test inference
python demo_infer.py
```

---

## 🔍 Troubleshooting

### Port Already in Use

```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Missing Model Files

```powershell
# Check if models exist
Test-Path models\cav_state_v3_2.joblib
Test-Path cav_engine_v3_2_LGBM_2025-11-08\cav_state_v3_2.joblib
```

### Dashboard Not Loading

```powershell
# Clear Streamlit cache
streamlit cache clear

# Restart dashboard
streamlit run tools\cav_dashboard.py
```

---

## 📋 Checklist for OEM Delivery

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Server starts successfully (`python app\main.py`)
- [ ] Demo inference works (`python demo_infer.py`)
- [ ] SDK packaged (`.\package_sdk.ps1`)
- [ ] Documentation reviewed (OEM_BRIEF.md, EVALUATION_LICENSE.md, NDA.md)
- [ ] Sample dataset available (or v3.2 features generated)
- [ ] Dashboard functional (`streamlit run tools\cav_dashboard.py`)

---

**For support or questions, refer to OEM_BRIEF.md or contact technical support.**

