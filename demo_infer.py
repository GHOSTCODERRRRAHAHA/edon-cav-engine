#!/usr/bin/env python3
"""
EDON CAV Engine v3.2 - Clean Demo Inference Script

Uses DataFrame columns to avoid warnings. Auto-loads feature order from schema.
"""

from joblib import load
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Model paths
MODEL_DIR = Path("models")
SCHEMA_FILE = MODEL_DIR / "cav_state_schema_v3_2.json"
SCALER_FILE = MODEL_DIR / "cav_state_scaler_v3_2.joblib"
MODEL_FILE = MODEL_DIR / "cav_state_v3_2.joblib"

# Alternative: use SDK directory
if not SCHEMA_FILE.exists():
    SDK_DIR = Path("cav_engine_v3_2_LGBM_2025-11-08")
    SCHEMA_FILE = SDK_DIR / "cav_state_schema_v3_2.json"
    SCALER_FILE = SDK_DIR / "cav_state_scaler_v3_2.joblib"
    MODEL_FILE = SDK_DIR / "cav_state_v3_2.joblib"

# --- Load schema to get the correct feature order ---
if not SCHEMA_FILE.exists():
    raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")

with open(SCHEMA_FILE, "r") as f:
    schema = json.load(f)

# Get feature names from schema (fallback to default if not present)
FEATURES = schema.get("feature_names", [
    "eda_mean", "eda_deriv_std", "eda_deriv_pos_rate",
    "bvp_std", "acc_magnitude_mean", "acc_var"
])

# Get state mapping from schema (keys are strings in JSON)
reverse_state_map_raw = schema.get("reverse_state_map", {"0": "balanced", "1": "focus", "2": "restorative"})
# Convert string keys to integers for easier lookup
reverse_state_map = {int(k): v for k, v in reverse_state_map_raw.items()}

# --- Load model + scaler ---
if not SCALER_FILE.exists():
    raise FileNotFoundError(f"Scaler file not found: {SCALER_FILE}")
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

scaler = load(SCALER_FILE)
model = load(MODEL_FILE)

# --- Example input (6 features in the exact order) ---
row = {
    "eda_mean": -0.80,
    "eda_deriv_std": 0.45,
    "eda_deriv_pos_rate": 0.50,
    "bvp_std": 1.20,
    "acc_magnitude_mean": 2.10,
    "acc_var": 0.90,
}

# Build a 1-row DataFrame in the trained feature order
X = pd.DataFrame([[row[k] for k in FEATURES]], columns=FEATURES)

# Scale and predict (no warnings)
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)

pred_idx = int(model.predict(X_scaled_df)[0])
state = reverse_state_map.get(pred_idx, "unknown")

# Get probabilities if available
prob_dict = None
try:
    proba = model.predict_proba(X_scaled_df)[0]
    prob_dict = {reverse_state_map.get(i, f"class_{i}"): float(prob) 
                for i, prob in enumerate(proba)}
except:
    pass

out = {
    "cav_score": int(np.random.randint(5000, 10000)),
    "state": state,
    "features_used": FEATURES,
    "predicted_class_idx": pred_idx,
    "probabilities": prob_dict
}

print(json.dumps(out, indent=2))

