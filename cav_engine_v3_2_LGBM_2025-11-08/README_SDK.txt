EDON CAV Engine v3.2 — LGBM Demo SDK (Nov 2025)
-----------------------------------------------
Files included:
- cav_state_v3_2.joblib ........ Trained LGBM model
- cav_state_scaler_v3_2.joblib .. Feature scaler
- cav_state_schema_v3_2.json .... Schema / feature order
- oem_100k_windows.parquet ...... Example dataset
- demo_infer_example.py .......... Minimal inference demo

Run demo:
  python demo_infer_example.py

Expected Output:
  {
    "cav_score": 9xxx,
    "state": "focus",
    "features_used": [...]
  }

Purpose:
  Demonstrates the CAV Engine's ability to infer human context
  ("balanced", "focus", "restorative") from physiological vectors.
  Optimized for wearable, humanoid, or OEM sensor platforms.
