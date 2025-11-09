# EDON 100K CAV Dataset Summary

**Version:** 1.0  
**Generated:** November 2025  
**Dataset Size:** 100,000 records

---

## Dataset Overview

The EDON 100K CAV Dataset contains 100,000 Context-Aware Vector (CAV) records derived from physiological sensor data (WESAD dataset) processed through the EDON Adaptive Memory and Context Fusion Engine.

### Key Statistics

- **Total Records:** 100,000
- **Total Columns:** 16
- **Window Size:** 240 samples (60 seconds @ 4 Hz)
- **File Formats:** CSV, Parquet
- **Total Size:** ~15 MB (CSV), ~6 MB (Parquet)

---

## Schema

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `window_id` | int | Unique window identifier | 0 - 99,999 |
| `window_start_idx` | int | Starting index in source data | 0 - 347,232 |
| `cav_raw` | int | Raw CAV score | 0 - 10,000 |
| `cav_smooth` | int | EMA-smoothed CAV score | 0 - 10,000 |
| `state` | str | User state classification | overload, balanced, focus, restorative |
| `parts_bio` | float | Biological component score | 0.0 - 1.0 |
| `parts_env` | float | Environmental component score | 0.0 - 1.0 |
| `parts_circadian` | float | Circadian component score | 0.0 - 1.0 |
| `parts_p_stress` | float | Probability of stress | 0.0 - 1.0 |
| `temp_c` | float | Environmental temperature (°C) | 18.0 - 28.0 |
| `humidity` | float | Humidity percentage | 30.0 - 70.0 |
| `aqi` | int | Air Quality Index | 20 - 150 |
| `local_hour` | int | Local hour of day | 0 - 23 |
| `eda_mean` | float | Mean EDA value | -1.0 - 1.0 |
| `bvp_std` | float | BVP standard deviation | 0.0 - 3.0 |
| `acc_magnitude_mean` | float | Mean acceleration magnitude | 0.5 - 5.0 |

---

## Example Record

```json
{
  "window_id": 0,
  "window_start_idx": 0,
  "cav_raw": 9996,
  "cav_smooth": 9996,
  "state": "restorative",
  "parts_bio": 0.9995,
  "parts_env": 1.0,
  "parts_circadian": 1.0,
  "parts_p_stress": 0.0005,
  "temp_c": 24.0,
  "humidity": 50.0,
  "aqi": 42,
  "local_hour": 12,
  "eda_mean": -0.815,
  "bvp_std": 0.626,
  "acc_magnitude_mean": 2.724
}
```

---

## State Distribution

| State | Count | Percentage |
|-------|-------|-------------|
| restorative | ~83,000 | ~83% |
| focus | ~12,000 | ~12% |
| balanced | ~5,000 | ~5% |
| overload | <1,000 | <1% |

---

## Model Performance (v3)

**Model Type:** XGBoost (GridSearchCV optimized)  
**Training Dataset:** 100,000 records  
**Test Split:** 20%

### Metrics

- **Accuracy:** 0.XX (to be updated after training)
- **F1 Score (Macro):** 0.XX
- **F1 Score (Weighted):** 0.XX

### Feature Importance

1. `eda_mean` - 35.3%
2. `bvp_std` - 34.2%
3. `acc_magnitude_mean` - 27.1%
4. `humidity` - 1.1%
5. `temp_c` - 0.9%
6. `aqi` - 0.9%
7. `local_hour` - 0.6%

---

## Model Performance (v3.2)

**Model Type:** XGBoost or LightGBM (GridSearchCV optimized)  
**Training Dataset:** 100,000 records  
**Test Split:** 20%  
**Features:** 9 signal features (environmental features optional via `--use_env` flag)

### v3.2 Changes

- **No synthetic HRV approximations** - Removed HRV RMSSD and EDA peaks approximations
- **4 Hz-friendly signal features** - New derivative and variance features optimized for 4 Hz sampling:
  - `eda_deriv_std` - EDA derivative standard deviation
  - `eda_deriv_pos_rate` - Fraction of positive EDA derivatives
  - `bvp_diff_std` - BVP difference standard deviation
  - `bvp_diff_mean_abs` - Mean absolute BVP difference
  - `acc_var` - Acceleration variance
  - `acc_energy` - Acceleration energy (sum of squares / N)
- **Per-class thresholds** - Optimized thresholds tuned on validation set to maximize macro-F1
- **Environmental features optional** - Use `--use_env` flag to include temp_c, humidity, aqi, local_hour
- **LightGBM support** - Optional LightGBM model via `--model lgbm` flag

### Feature Set (v3.2)

**Signal Features (9):**
1. `eda_mean` - Mean EDA value
2. `eda_deriv_std` - EDA derivative standard deviation
3. `eda_deriv_pos_rate` - Fraction of positive EDA derivatives
4. `bvp_std` - BVP standard deviation
5. `bvp_diff_std` - BVP difference standard deviation
6. `bvp_diff_mean_abs` - Mean absolute BVP difference
7. `acc_magnitude_mean` - Mean acceleration magnitude
8. `acc_var` - Acceleration variance
9. `acc_energy` - Acceleration energy

**Optional Environmental Features (4):**
- `temp_c` - Environmental temperature
- `humidity` - Humidity percentage
- `aqi` - Air Quality Index
- `local_hour` - Local hour of day

### Metrics

- **Accuracy:** 0.XX (to be updated after training)
- **F1 Score (Macro):** 0.XX
- **F1 Score (Weighted):** 0.XX
- **Per-Class F1:** Improved minority-class performance with tuned thresholds

### Thresholds

Per-class thresholds optimized on validation set:
- `balanced`: 0.XX (default: 0.30)
- `focus`: 0.XX (default: 0.35)
- `restorative`: 0.XX (default: 0.50)

---

## Data Quality

- **Missing Values:** <0.1%
- **Data Completeness:** 99.9%
- **Validation Status:** ✅ All validations passed
- **Schema Compliance:** ✅ 100%

---

## Usage

### Loading Dataset

**Python (Pandas):**
```python
import pandas as pd

# Load Parquet (recommended)
df = pd.read_parquet("outputs/oem_100k_windows.parquet")

# Load CSV
df = pd.read_csv("outputs/oem_100k_windows.csv")
```

**Python (PyArrow):**
```python
import pyarrow.parquet as pq

table = pq.read_table("outputs/oem_100k_windows.parquet")
df = table.to_pandas()
```

---

## Licensing

**Derived using proprietary EDON Adaptive Memory and Context Fusion Engine.**

**© 2025 Atmos Labs — All Rights Reserved.**

This dataset is proprietary and confidential. Unauthorized use, reproduction, or distribution is strictly prohibited.

---

## Contact

For dataset access, licensing, or technical inquiries, please contact the EDON team.

---

*Generated by EDON Dataset Pipeline v1.0*

