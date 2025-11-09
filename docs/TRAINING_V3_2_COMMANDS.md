# EDON Training v3.2 - Command Reference

## Overview

Training script v3.2 includes:
- **4 Hz-friendly signal features** (no synthetic HRV approximations)
- **Per-class threshold tuning** for improved minority-class F1
- **LightGBM support** (optional)
- **Environmental features optional** (via `--use_env` flag)
- **9 signal features** optimized for 4 Hz sampling

---

## Commands

### 1. Train v3.2 Model with XGBoost (Default)

```powershell
.\venv\Scripts\Activate.ps1
python training\train_baseline_v3_2.py --input outputs\oem_100k_windows.parquet --model xgb
```

**Features:** 9 signal features (no environmental features)

### 2. Train v3.2 Model with LightGBM

```powershell
python training\train_baseline_v3_2.py --input outputs\oem_100k_windows.parquet --model lgbm
```

**Features:** 9 signal features (no environmental features)

### 3. Train with Environmental Features (XGBoost)

```powershell
python training\train_baseline_v3_2.py --input outputs\oem_100k_windows.parquet --model xgb --use-env
```

**Features:** 9 signal features + 4 environmental features (13 total)

### 4. Train with Environmental Features (LightGBM)

```powershell
python training\train_baseline_v3_2.py --input outputs\oem_100k_windows.parquet --model lgbm --use-env
```

**Features:** 9 signal features + 4 environmental features (13 total)

### 5. Custom Options

```powershell
# Custom CV folds
python training\train_baseline_v3_2.py --input outputs\oem_100k_windows.parquet --model xgb --cv 5

# Custom test split
python training\train_baseline_v3_2.py --input outputs\oem_100k_windows.parquet --model xgb --test-size 0.15

# Use CSV instead of Parquet
python training\train_baseline_v3_2.py --input outputs\oem_100k_windows.csv --model xgb
```

---

## Features

### Signal Features (9) - Always Included

1. `eda_mean` - Mean EDA value
2. `eda_deriv_std` - EDA derivative standard deviation
3. `eda_deriv_pos_rate` - Fraction of positive EDA derivatives
4. `bvp_std` - BVP standard deviation
5. `bvp_diff_std` - BVP difference standard deviation
6. `bvp_diff_mean_abs` - Mean absolute BVP difference
7. `acc_magnitude_mean` - Mean acceleration magnitude
8. `acc_var` - Acceleration variance
9. `acc_energy` - Acceleration energy (sum of squares / N)

### Environmental Features (4) - Optional with `--use_env`

- `temp_c` - Environmental temperature (°C)
- `humidity` - Humidity percentage
- `aqi` - Air Quality Index
- `local_hour` - Local hour of day [0-23]

---

## Improvements over v3.1

1. **No synthetic HRV** - Removed HRV RMSSD and EDA peaks approximations
2. **4 Hz-friendly features** - Derivative and variance features optimized for 4 Hz sampling
3. **Per-class thresholds** - Tuned on validation set to maximize macro-F1
4. **LightGBM support** - Optional LightGBM model via `--model lgbm`
5. **Environmental features optional** - Use `--use_env` flag to include

---

## Output Files

- `models/cav_state_v3_2.joblib` - Trained model
- `models/cav_state_scaler_v3_2.joblib` - Feature scaler
- `models/cav_state_schema_v3_2.json` - Schema with metrics, thresholds, and algorithm

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
- Top 10 feature importance (sorted descending)
- Optimized per-class thresholds

---

## Threshold Tuning

The script automatically tunes per-class thresholds on the validation set to maximize macro-F1.

**Default thresholds:**
- `balanced`: 0.30
- `focus`: 0.35
- `restorative`: 0.50

**Optimized thresholds** are saved in the schema JSON file.

---

## Example Output

```
============================================================
XGBoost/LightGBM Baseline Training v3.2
4 Hz-Friendly Features + Per-Class Thresholds
============================================================

Loading dataset: outputs/oem_100k_windows.parquet
Loaded 100,000 records

Preparing features...
  Computing 4 Hz-friendly signal features...
Features (9): ['eda_mean', 'eda_deriv_std', 'eda_deriv_pos_rate', 'bvp_std', 'bvp_diff_std', 'bvp_diff_mean_abs', 'acc_magnitude_mean', 'acc_var', 'acc_energy']
Target distribution:
  balanced    :   446 (  4.5%)
  focus       : 1,235 ( 12.3%)
  restorative : 8,319 ( 83.2%)

Computing class weights...
  Class weights:
    balanced    (class 0): 11.210
    focus       (class 1): 4.049
    restorative (class 2): 0.601

Training XGB model with GridSearchCV...
  CV folds: 3
  n_jobs: -1
  Scoring: macro-F1
  Class weights: {0: 11.210, 1: 4.049, 2: 0.601}
  Running GridSearchCV (this may take a while)...
  ...
  Best parameters: {'learning_rate': 0.1, 'max_depth': 8, 'min_child_weight': 1, 'n_estimators': 200, 'subsample': 0.8}
  Best CV score (macro-F1): 0.XXXX

Tuning per-class thresholds...
  Searching threshold space...
  Best macro-F1 with tuned thresholds: 0.XXXX
  Optimized thresholds: {'balanced': 0.XX, 'focus': 0.XX, 'restorative': 0.XX}

Evaluating model on test set with tuned thresholds...

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

Top 10 Feature Importance (sorted descending):
   1. eda_mean            : 0.XXXX
   2. bvp_std             : 0.XXXX
   3. acc_magnitude_mean  : 0.XXXX
   ...
```

---

## Performance Notes

- **Training Time:** 15-45 minutes (100K dataset, 3 CV folds, threshold tuning)
- **Class Weighting:** Automatically computed to balance classes
- **Macro-F1 Scoring:** Better for imbalanced datasets
- **Threshold Tuning:** Grid search over threshold space to maximize macro-F1
- **4 Hz Features:** Optimized for 4 Hz sampling rate (no synthetic approximations)

---

## Troubleshooting

### Missing LightGBM
```powershell
# Install LightGBM
pip install lightgbm

# Or use XGBoost (default)
python training\train_baseline_v3_2.py --input outputs\oem_100k_windows.parquet --model xgb
```

### Threshold Tuning Takes Too Long
- Threshold tuning uses grid search which may take time
- Default thresholds are used if tuning fails
- Results are saved in schema JSON

---

## Next Steps

1. **Train model** on 100K dataset
2. **Review metrics** (especially per-class F1 scores)
3. **Compare with v3.1** to see improvement from 4 Hz features and thresholds
4. **Try LightGBM** to see if it performs better than XGBoost
5. **Experiment with `--use_env`** to see if environmental features help

---

*EDON Training v3.2 - 4 Hz-Friendly Features + Per-Class Thresholds*

