# EDON Commercialization Package - Command Reference

## Setup

### 1. Install Dependencies
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**New packages added:**
- `streamlit` - Dashboard framework
- `lightgbm` - LightGBM model (alternative to XGBoost)
- `seaborn` - Statistical visualization
- `matplotlib` - Plotting library

---

## Training v3 Model

### Train XGBoost v3 (GridSearchCV)
```powershell
.\venv\Scripts\Activate.ps1
python training\train_baseline_v3.py --input outputs\oem_100k_windows.parquet
```

**Options:**
```powershell
# Use LightGBM instead
python training\train_baseline_v3.py --input outputs\oem_100k_windows.parquet --model-type lightgbm

# Custom CV folds
python training\train_baseline_v3.py --input outputs\oem_100k_windows.parquet --cv 5

# Custom test split
python training\train_baseline_v3.py --input outputs\oem_100k_windows.parquet --test-size 0.15

# Use CSV instead of Parquet
python training\train_baseline_v3.py --input outputs\oem_100k_windows.csv
```

**Output:**
- `models/cav_state_xgb_v3.joblib` - Optimized model
- `models/cav_state_scaler_v3.joblib` - Feature scaler
- `models/cav_state_schema_v3.json` - Schema with metrics and confusion matrix

**Note:** GridSearchCV may take 10-30 minutes depending on dataset size and CPU cores.

---

## Launch Dashboard

### Start Streamlit Dashboard
```powershell
.\venv\Scripts\Activate.ps1
streamlit run tools\cav_dashboard.py
```

The dashboard will open automatically in your browser at:
- **URL:** http://localhost:8501

**Features:**
- Dataset overview with sample records
- State distribution (pie chart, bar chart)
- CAV score histograms and box plots
- Feature correlation heatmap
- Interactive filters (state, hour)

**Custom Dataset Path:**
- Default: `outputs/oem_100k_windows.parquet`
- Fallback: `outputs/oem_sample_windows.parquet`
- You can modify the path in the sidebar

---

## Export Dataset Summary

### Generate Summary Document
The summary document is already generated at:
- `docs/EDON_100K_Dataset_Summary.md`

### Update Model Metrics (After Training v3)
After training the v3 model, update the summary document with actual metrics:

1. Open `docs/EDON_100K_Dataset_Summary.md`
2. Update the "Model Performance (v3)" section with actual metrics from training output
3. Update feature importance values if different

### Export to PDF (Optional)
If you have a Markdown-to-PDF converter:

```powershell
# Using pandoc (if installed)
pandoc docs\EDON_100K_Dataset_Summary.md -o docs\EDON_100K_Dataset_Summary.pdf

# Or use online converter
# Upload docs/EDON_100K_Dataset_Summary.md to a Markdown-to-PDF service
```

---

## Complete Workflow

### Step 1: Ensure 100K Dataset Exists
```powershell
# Check if dataset exists
Test-Path outputs\oem_100k_windows.parquet

# If not, build it (requires API server running)
python tools\build_100k_dataset.py --target 100000
```

### Step 2: Train v3 Model
```powershell
python training\train_baseline_v3.py --input outputs\oem_100k_windows.parquet
```

**Expected output:**
- Training progress with GridSearchCV
- Best parameters found
- Test set performance metrics
- Confusion matrix
- Feature importance
- Model files saved

### Step 3: Launch Dashboard
```powershell
streamlit run tools\cav_dashboard.py
```

**Dashboard tabs:**
1. **Overview** - Dataset statistics and sample records
2. **State Distribution** - Pie charts and statistics
3. **CAV Scores** - Histograms and box plots
4. **Correlations** - Feature correlation heatmap

### Step 4: Update Summary Document
1. Copy metrics from training output
2. Update `docs/EDON_100K_Dataset_Summary.md`
3. Export to PDF if needed

---

## File Locations

### Input Files
- **100K Dataset:** `outputs/oem_100k_windows.parquet` (or `.csv`)
- **10K Dataset (fallback):** `outputs/oem_sample_windows.parquet`

### Output Files
- **Trained Model:** `models/cav_state_xgb_v3.joblib`
- **Scaler:** `models/cav_state_scaler_v3.joblib`
- **Schema:** `models/cav_state_schema_v3.json`
- **Summary:** `docs/EDON_100K_Dataset_Summary.md`

---

## Troubleshooting

### Dashboard Not Loading
```powershell
# Check if Streamlit is installed
pip install streamlit

# Check if dataset exists
Test-Path outputs\oem_100k_windows.parquet

# Try with fallback dataset
# The dashboard will automatically try the 10K dataset if 100K is not found
```

### Training Takes Too Long
```powershell
# Reduce CV folds
python training\train_baseline_v3.py --cv 2

# Use fewer parameter combinations (edit script)
# Or use a smaller dataset for testing
python training\train_baseline_v3.py --input outputs\oem_sample_windows_enriched.csv
```

### LightGBM Not Available
```powershell
# Install LightGBM
pip install lightgbm

# Or use XGBoost (default)
python training\train_baseline_v3.py --model-type xgboost
```

---

## Performance Notes

- **GridSearchCV Training:** 10-30 minutes (100K dataset, 3 CV folds)
- **Dashboard Loading:** <5 seconds (100K Parquet file)
- **Dashboard Interactions:** Real-time filtering and visualization

---

## Next Steps

1. **Train v3 model** on 100K dataset
2. **Launch dashboard** to explore data
3. **Update summary document** with actual metrics
4. **Export summary** to PDF for distribution
5. **Package for commercialization** (dataset + model + documentation)

---

*EDON Commercialization Package v1.0*

