# EDON Training v3.1 - Command Reference

## Overview

Training script v3.1 includes:
- **Class weighting** for better balance across classes
- **Macro-F1 scoring** in GridSearchCV
- **Per-class metrics** (precision, recall, F1)
- **Expanded feature set** (9 features including HRV RMSSD and EDA peaks/min)
- **Optional PR curve plotting**
- **Sorted feature importance**

---

## Commands

### 1. Train v3.1 Model (Basic)

```powershell
.\venv\Scripts\Activate.ps1
python training\train_baseline_v3_1.py --input outputs\oem_100k_windows.parquet
```

### 2. Train with PR Curves

```powershell
python training\train_baseline_v3_1.py --input outputs\oem_100k_windows.parquet --plot-pr
```

**Output:**
- `models/cav_state_xgb_v3_1.joblib` - Trained model
- `models/cav_state_scaler_v3_1.joblib` - Feature scaler
- `models/cav_state_schema_v3_1.json` - Schema with metrics
- `models/pr_curves_v3_1.png` - PR curves plot (if --plot-pr used)

### 3. Train with Custom Options

```powershell
# Use LightGBM
python training\train_baseline_v3_1.py --input outputs\oem_100k_windows.parquet --model-type lightgbm

# Custom CV folds
python training\train_baseline_v3_1.py --input outputs\oem_100k_windows.parquet --cv 5

# Custom test split
python training\train_baseline_v3_1.py --input outputs\oem_100k_windows.parquet --test-size 0.15

# Use CSV instead of Parquet
python training\train_baseline_v3_1.py --input outputs\oem_100k_windows.csv
```

---

## Features

### Feature Set (9 features)
1. `eda_mean` - Mean EDA value
2. `eda_peaks_per_min` - EDA peaks per minute (computed from BVP std approximation)
3. `bvp_std` - BVP standard deviation
4. `hrv_rmssd` - HRV RMSSD (computed from BVP std approximation)
5. `acc_magnitude_mean` - Mean acceleration magnitude
6. `temp_c` - Environmental temperature
7. `humidity` - Humidity percentage
8. `aqi` - Air Quality Index
9. `local_hour` - Local hour of day

**Note:** `hrv_rmssd` and `eda_peaks_per_min` are computed from approximations using BVP std. For production, these should be computed from raw signals during dataset building.

### GridSearchCV Parameters
- `n_estimators`: [100, 200, 300]
- `max_depth`: [6, 8, 10]
- `learning_rate`: [0.05, 0.1, 0.2]
- `subsample`: [0.8, 1.0]
- `min_child_weight`: [1, 3]

### Scoring
- **Macro-F1** (unweighted average across classes)
- **Class weighting** applied automatically using `compute_class_weight('balanced')`

---

## Output Metrics

### Overall Metrics
- Accuracy
- F1 (macro) - Unweighted average across classes
- F1 (weighted) - Weighted by class support

### Per-Class Metrics
For each class:
- Precision
- Recall
- F1 score

### Additional Outputs
- Confusion matrix (formatted table)
- Feature importance (sorted descending)
- PR curves plot (optional, with --plot-pr)

---

## View Metrics

### After Training
The script prints all metrics to console:
- Overall performance
- Per-class metrics
- Confusion matrix
- Feature importance

### From Schema File
```powershell
# View saved metrics
python -c "import json; data = json.load(open('models/cav_state_schema_v3_1.json')); print(json.dumps(data['metrics'], indent=2))"
```

### View PR Curves
If `--plot-pr` was used:
```powershell
# Open the PR curves image
start models\pr_curves_v3_1.png
```

---

## Example Output

```
============================================================
XGBoost/LightGBM Baseline Training v3.1 (Class Weighting + Macro-F1)
============================================================

Loading dataset: outputs/oem_100k_windows.parquet
Loaded 100,000 records

Preparing features...
Features: ['eda_mean', 'eda_peaks_per_min', 'bvp_std', 'hrv_rmssd', 'acc_magnitude_mean', 'temp_c', 'humidity', 'aqi', 'local_hour']
Target distribution:
  balanced    :   446 (  4.5%)
  focus       : 1,235 ( 12.3%)
  restorative : 8,319 ( 83.2%)

Computing class weights...
  Class weights:
    balanced    (class 0): 11.210
    focus       (class 1): 4.049
    restorative (class 2): 0.601

Training XGBOOST model with GridSearchCV...
  CV folds: 3
  n_jobs: -1
  Scoring: macro-F1
  Class weights: {0: 11.210, 1: 4.049, 2: 0.601}
  Running GridSearchCV (this may take a while)...
  ...
  Best parameters: {'learning_rate': 0.1, 'max_depth': 8, 'min_child_weight': 1, 'n_estimators': 200, 'subsample': 0.8}
  Best CV score (macro-F1): 0.XXXX

Test Set Performance:
  Accuracy: 0.XXXX
  F1 (macro): 0.XXXX
  F1 (weighted): 0.XXXX

Per-Class Metrics:
  balanced    : precision=0.XXX, recall=0.XXX, f1=0.XXX
  focus       : precision=0.XXX, recall=0.XXX, f1=0.XXX
  restorative : precision=0.XXX, recall=0.XXX, f1=0.XXX

Confusion Matrix:
               balanced    focus       restorative
balanced       XXXX        XXXX        XXXX
focus          XXXX        XXXX        XXXX
restorative    XXXX        XXXX        XXXX

Feature Importance (sorted descending):
  eda_mean            : 0.XXXX
  bvp_std             : 0.XXXX
  acc_magnitude_mean  : 0.XXXX
  ...
```

---

## Performance Notes

- **Training Time:** 15-45 minutes (100K dataset, 3 CV folds, expanded parameter grid)
- **Class Weighting:** Automatically computed to balance classes
- **Macro-F1:** Better for imbalanced datasets than weighted-F1
- **Feature Approximations:** HRV RMSSD and EDA peaks computed from BVP std

---

## Troubleshooting

### Missing Features
If `hrv_rmssd` or `eda_peaks_per_min` are missing from dataset:
- Script automatically computes approximations from BVP std
- For production, compute from raw signals during dataset building

### Class Weighting Issues
- Weights are computed automatically using `compute_class_weight('balanced')`
- Rare classes get higher weights
- XGBoost uses `sample_weight`, LightGBM uses `class_weight`

### PR Curves Not Plotting
```powershell
# Ensure matplotlib is installed
pip install matplotlib

# Use --plot-pr flag
python training\train_baseline_v3_1.py --input outputs\oem_100k_windows.parquet --plot-pr
```

---

## Next Steps

1. **Train model** on 100K dataset
2. **Review metrics** (especially per-class F1 scores)
3. **Compare with v3** to see improvement from class weighting
4. **Export PR curves** for documentation
5. **Update dataset** to include computed HRV RMSSD and EDA peaks from raw signals

---

*EDON Training v3.1 - Enhanced with Class Weighting and Macro-F1 Scoring*

