#!/usr/bin/env python3
"""
Helper script to add v3.2 features to existing parquet files.

Approximates missing v3.2 features using rolling statistics.
For production, compute from per-window raw signals.
"""

import pandas as pd
from pathlib import Path

INP = Path(r"outputs\oem_100k_windows.parquet")
OUT = Path(r"outputs\oem_100k_windows_v32.parquet")


def main():
    if not INP.exists():
        raise SystemExit(f"Input not found: {INP}")

    df = pd.read_parquet(INP)

    need_base = {"eda_mean", "bvp_std", "acc_magnitude_mean"}
    if not need_base.issubset(df.columns):
        raise SystemExit("Missing base columns: need eda_mean, bvp_std, acc_magnitude_mean.")

    # Approximate v3.2 features using rolling stats across windows.
    # NOTE: For production, compute from per-window raw signals.
    w = 25
    df2 = df.copy()

    # eda_deriv_std: volatility of eda_mean change
    eda_diff = df2["eda_mean"].diff()
    df2["eda_deriv_std"] = eda_diff.rolling(w, min_periods=2).std()
    if pd.isna(df2["eda_deriv_std"]).all():
        df2["eda_deriv_std"] = 0.35
    df2["eda_deriv_std"] = df2["eda_deriv_std"].fillna(df2["eda_deriv_std"].median()).clip(0.0, 2.0)

    # eda_deriv_pos_rate: fraction of positive diffs
    pos_rate = (eda_diff > 0).rolling(w, min_periods=2).mean()
    df2["eda_deriv_pos_rate"] = pos_rate.fillna(0.5).clip(0.0, 1.0)

    # acc_var: rolling variance proxy of acc_magnitude_mean
    accv = df2["acc_magnitude_mean"].rolling(w, min_periods=2).var()
    if pd.isna(accv).all():
        accv = 0.5
    df2["acc_var"] = accv.fillna(df2["acc_magnitude_mean"].var() or 0.5).clip(0.0, 2.0)

    df2.to_parquet(OUT, index=False)
    print(f"[OK] Wrote {OUT} with v3.2 features (approx/compat).")


if __name__ == "__main__":
    main()

