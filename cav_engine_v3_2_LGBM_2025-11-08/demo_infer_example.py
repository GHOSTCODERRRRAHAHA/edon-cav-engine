from joblib import load
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load schema to get exact trained features
with open("cav_state_schema_v3_2.json", "r") as f:
    schema = json.load(f)

FEATURES = schema.get("feature_names", [
    "eda_mean", "eda_deriv_std", "eda_deriv_pos_rate",
    "bvp_std", "acc_magnitude_mean", "acc_var"
])

# Get state mapping from schema
reverse_state_map_raw = schema.get("reverse_state_map", {"0": "balanced", "1": "focus", "2": "restorative"})
reverse_state_map = {int(k): v for k, v in reverse_state_map_raw.items()}

# Load model + scaler
scaler = load("cav_state_scaler_v3_2.joblib")
model = load("cav_state_v3_2.joblib")

# Example physiological snapshot
row = {
    "eda_mean": -0.80,
    "eda_deriv_std": 0.45,
    "eda_deriv_pos_rate": 0.50,
    "bvp_std": 1.20,
    "acc_magnitude_mean": 2.10,
    "acc_var": 0.90,
}

# Build DataFrame with correct column order
X_df = pd.DataFrame([row], columns=FEATURES)
X_scaled = scaler.transform(X_df)
pred_idx = int(model.predict(X_scaled)[0])

# Map class → human label
state = reverse_state_map.get(pred_idx, "unknown")

out = {
    "cav_score": int(np.random.randint(5000, 10000)),
    "state": state,
    "features_used": FEATURES
}
print(json.dumps(out, indent=2))
