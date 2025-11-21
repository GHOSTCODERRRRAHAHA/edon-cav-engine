"""Evaluate CAV engine on WESAD ground truth data.

Supports two modes:
  - local : uses LightGBM classifier directly with 6-feature mean pooling
  - api   : calls the running HTTP API (/cav) per window
"""

from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

# Add project root to path so we can import app.*
THIS_FILE = Path(__file__).resolve()
PROJ_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(PROJ_ROOT))

# Optional imports (only needed in specific modes)
import requests  # for --mode api
from joblib import load  # for --mode local
from app.infer_fallback import pool6_from_window  # 6-feature mean pooling


# ---------------------------
# Data loading & windowing
# ---------------------------
def load_wesad_data(wesad_path: Path) -> pd.DataFrame:
    """Load WESAD data from CSV or parquet."""
    if not wesad_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {wesad_path}")

    if wesad_path.is_dir():
        # Try common filenames inside the directory
        cand = None
        for name in ["wesad_wrist_4hz.csv", "wesad_wrist_4hz.parquet"]:
            p = wesad_path / name
            if p.exists():
                cand = p
                break
        if cand is None:
            raise FileNotFoundError(f"No known WESAD file found in {wesad_path}")
        wesad_path = cand

    if wesad_path.suffix.lower() == ".csv":
        return pd.read_csv(wesad_path)
    elif wesad_path.suffix.lower() == ".parquet":
        return pd.read_parquet(wesad_path)
    else:
        raise ValueError(f"Unsupported file format: {wesad_path.suffix}")


def prepare_windows(df: pd.DataFrame, window_size: int = 240) -> List[Dict]:
    """Prepare fixed-size windows grouped by (subject, label).

    Expects df to include columns: subject, label, and sensor channels:
    EDA, TEMP, BVP, ACC_x, ACC_y, ACC_z  (case-insensitive tolerated)
    """
    # Normalize column names for flexible access
    colmap = {c.lower(): c for c in df.columns}
    need = ["subject", "label"]
    for k in need:
        if k not in [x.lower() for x in df.columns]:
            raise KeyError(f"Missing required column: {k}")

    def pick(col_main: str, alt: str, default_val: float) -> List[float]:
        if col_main in df.columns:
            return df[col_main].to_list()
        elif alt in df.columns:
            return df[alt].to_list()
        else:
            # will be handled per-window; return empty sentinel
            return []

    windows: List[Dict] = []
    # Group by subject + label
    for (subject, label), group in df.groupby([colmap["subject"], colmap["label"]]):
        # Slide non-overlapping windows for simplicity
        n = len(group)
        if n < window_size:
            continue

        # Convenience getter in this group with case-fallbacks
        def gget(frame: pd.DataFrame, candA: str, candB: str, fallback: float) -> List[float]:
            if candA in frame.columns:
                return frame[candA].tolist()
            if candB in frame.columns:
                return frame[candB].tolist()
            return [fallback] * len(frame)

        for i in range(0, n - window_size + 1, window_size):
            sl = group.iloc[i : i + window_size]

            # Build window with UPPERCASE keys expected by fallback path
            window = {
                "EDA"  : gget(sl, "EDA",  "eda", 0.0),
                "TEMP" : gget(sl, "TEMP", "temp", 32.0),
                "BVP"  : gget(sl, "BVP",  "bvp", 0.0),
                "ACC_x": gget(sl, "ACC_x","acc_x", 0.0),
                "ACC_y": gget(sl, "ACC_y","acc_y", 0.0),
                "ACC_z": gget(sl, "ACC_z","acc_z", 1.0),
                # env (not used by 6-feature path, ok to include)
                "temp_c": 22.0,
                "humidity": 50.0,
                "aqi": 35,
                "local_hour": 12,
            }

            # Ensure exact length
            for k in ["EDA","TEMP","BVP","ACC_x","ACC_y","ACC_z"]:
                arr = window[k]
                if len(arr) != window_size:
                    if len(arr) == 0:
                        arr = [0.0] * window_size
                    elif len(arr) > window_size:
                        arr = arr[:window_size]
                    else:
                        arr = arr + [arr[-1]] * (window_size - len(arr))
                    window[k] = arr

            windows.append({
                "window": window,
                "label": int(label),
                "subject": subject
            })

    return windows


# ---------------------------
# Evaluation (API mode)
# ---------------------------
def evaluate_cav_api(windows: List[Dict], api_url: str = "http://localhost:8000") -> Dict:
    """Evaluate by calling the running HTTP API /cav."""
    predictions: List[str] = []
    ground_truth: List[int] = []
    cav_scores: List[float] = []

    for item in tqdm(windows, desc="API Eval"):
        win = item["window"]
        label = item["label"]
        try:
            r = requests.post(f"{api_url}/cav", json=win, timeout=5.0)
            if r.status_code == 200:
                out = r.json()
                state = out.get("state", "unknown")
                cav_smooth = float(out.get("cav_smooth", 0.0))
            else:
                state = "unknown"
                cav_smooth = 0.0
        except Exception:
            state = "unknown"
            cav_smooth = 0.0

        predictions.append(state)
        ground_truth.append(label)
        cav_scores.append(cav_smooth)

    return _compute_metrics(predictions, ground_truth, cav_scores)


# ---------------------------
# Evaluation (local 6-feature mode)
# ---------------------------
def evaluate_local_classifier(windows: List[Dict], clf_path: str) -> Dict:
    """Evaluate the LightGBM classifier directly using 6-feature mean pooling."""
    clf = load(clf_path)

    predictions: List[str] = []
    ground_truth: List[int] = []
    cav_scores: List[float] = []  # we’ll just store numeric class as a proxy

    num_to_state = {0: "restorative", 1: "balanced", 2: "overload"}

    for item in tqdm(windows, desc="Local Eval"):
        win = item["window"]
        label = item["label"]

        X = pool6_from_window(win)  # (1,6)
        y = int(clf.predict(X)[0])
        state = num_to_state.get(y, "unknown")

        predictions.append(state)
        ground_truth.append(label)
        cav_scores.append(float(y))

    return _compute_metrics(predictions, ground_truth, cav_scores)


# ---------------------------
# Metrics helper
# ---------------------------
def _compute_metrics(predictions: List[str], ground_truth: List[int], cav_scores: List[float]) -> Dict:
    # Map to numeric for scoring
    state_to_num = {"overload": 2, "balanced": 1, "focus": 0, "restorative": 0}
    pred_numeric = [state_to_num.get(p, 1) for p in predictions]

    accuracy = accuracy_score(ground_truth, pred_numeric)

    binary_truth = [1 if gt == 2 else 0 for gt in ground_truth]   # stress vs non-stress
    binary_pred  = [1 if pn == 2 else 0 for pn in pred_numeric]
    try:
        auroc = roc_auc_score(binary_truth, binary_pred)
    except Exception:
        auroc = 0.0

    cm = confusion_matrix(ground_truth, pred_numeric, labels=[0, 1, 2])
    report = classification_report(
        ground_truth, pred_numeric,
        labels=[0, 1, 2],
        target_names=["non-stress", "baseline", "stress"],
        output_dict=True
    )

    cav_mean = float(np.mean(cav_scores)) if len(cav_scores) else 0.0
    cav_std  = float(np.std(cav_scores)) if len(cav_scores) else 0.0
    drift_scores = [(s - cav_mean) / (cav_std + 1e-6) for s in cav_scores] if len(cav_scores) else [0.0]

    return {
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "cav_stats": {
            "mean": cav_mean, "std": cav_std,
            "min": float(np.min(cav_scores)) if len(cav_scores) else 0.0,
            "max": float(np.max(cav_scores)) if len(cav_scores) else 0.0,
        },
        "drift_stats": {
            "mean": float(np.mean(drift_scores)),
            "std": float(np.std(drift_scores)),
            "min": float(np.min(drift_scores)),
            "max": float(np.max(drift_scores)),
        },
        "n_samples": len(predictions),
        "state_distribution": {s: predictions.count(s) for s in sorted(set(predictions))},
    }


# ---------------------------
# CLI
# ---------------------------
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CAV engine on WESAD data")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to WESAD CSV/Parquet or folder containing wesad_wrist_4hz.(csv|parquet)")
    parser.add_argument("--mode", choices=["local", "api"], default="local",
                        help="local: use joblib classifier with 6-feature pooling; api: call HTTP /cav")
    parser.add_argument("--clf", type=str,
                        default="cav_engine_v3_2_LGBM_2025-11-08/cav_state_v3_2.joblib",
                        help="Path to LightGBM classifier (used in --mode local)")
    parser.add_argument("--api", type=str, default="http://localhost:8000", help="API base URL (for --mode api)")
    parser.add_argument("--output", type=str, default="reports/eval_wesad.json", help="Output JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of windows")

    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    print(f"Loading data from {data_path} ...")
    df = load_wesad_data(data_path)

    # Prepare windows
    print("Preparing windows (240-sample, non-overlapping) ...")
    windows = prepare_windows(df, window_size=240)
    if args.limit:
        windows = windows[: args.limit]
        print(f"Limited to {len(windows)} windows.")

    # Evaluate
    if args.mode == "local":
        print("Mode: LOCAL (6-feature mean pooling -> LightGBM classifier)")
        results = evaluate_local_classifier(windows, args.clf)
    else:
        print(f"Mode: API ({args.api}/cav)")
        results = evaluate_cav_api(windows, args.api)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUROC: {results['auroc']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(results["confusion_matrix"]))
    print(f"\nCAV Stats: mean={results['cav_stats']['mean']:.2f}, std={results['cav_stats']['std']:.2f}")
    print(f"Drift Stats: mean={results['drift_stats']['mean']:.4f}, std={results['drift_stats']['std']:.4f}")
    print(f"\nState Distribution: {results['state_distribution']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
