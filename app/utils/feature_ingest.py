# app/utils/feature_ingest.py
from __future__ import annotations

from typing import Any, Dict, List

import math

import os


MODEL_FEATURE_ORDER = ["eda_mean", "eda_std", "bvp_mean", "bvp_std", "acc_mean", "acc_std"]


def looks_raw(win: Dict[str, Any], n: int = 240) -> bool:
    """Heuristic: raw windows have time-series arrays for these keys."""
    need = ["EDA", "TEMP", "BVP", "ACC_x", "ACC_y", "ACC_z"]
    keys_lower = {k.lower() for k in win.keys()}
    if not all(k.lower() in keys_lower for k in need):
        return False
    
    def get(k): 
        return win.get(k) or win.get(k.lower()) or win.get(k.upper())
    
    try:
        return all(isinstance(get(k), (list, tuple)) and len(get(k)) == n for k in need)
    except Exception:
        return False


def featurize_raw(win: Dict[str, Any]) -> Dict[str, float]:
    def get(k): 
        return win.get(k) or win.get(k.lower()) or win.get(k.upper())
    
    EDA = [float(x) for x in get("EDA")]
    BVP = [float(x) for x in get("BVP")]
    ACCx = [float(x) for x in get("ACC_x")]
    ACCy = [float(x) for x in get("ACC_y")]
    ACCz = [float(x) for x in get("ACC_z")]
    ACCm = [math.sqrt(x*x + y*y + z*z) for x, y, z in zip(ACCx, ACCy, ACCz)]
    
    def mean(a): 
        return sum(a)/len(a) if a else 0.0
    
    def sstd(a):
        if len(a) <= 1: 
            return 0.0
        m = mean(a)
        return (sum((x-m)*(x-m) for x in a) / (len(a)-1))**0.5
    
    return {
        "eda_mean": mean(EDA),  "eda_std":  sstd(EDA),
        "bvp_mean": mean(BVP),  "bvp_std":  sstd(BVP),
        "acc_mean": mean(ACCm), "acc_std":  sstd(ACCm),
    }


def normalize_feature_map(win: Dict[str, Any]) -> Dict[str, float]:
    """Lowercase keys and coerce to float; ignore extras."""
    f = {k.lower(): win[k] for k in win}
    out = {}
    for k in MODEL_FEATURE_ORDER:
        v = f.get(k, 0.0)
        out[k] = float(v)
    return out


def to_vector(fmap: Dict[str, float]) -> List[float]:
    return [float(fmap.get(k, 0.0)) for k in MODEL_FEATURE_ORDER]


def guard_features(fmaps: List[Dict[str, Any]]) -> None:
    """Existing overlap/schema guard, gated by env flag."""
    if os.getenv("EDON_STRICT_FEATURES", "true").lower() != "true":
        return
    
    # Only guard feature maps, not raw payloads.
    RAW_HINTS = {"eda", "bvp", "acc_x", "acc_y", "acc_z", "temp", "temp_c", "humidity", "aqi"}
    for fm in fmaps:
        keys = {k.lower() for k in fm.keys()}
        if keys & RAW_HINTS:
            return  # raw-ish → upstream will featurize; skip guard
    
    # Keep it simple: require all expected keys present
    for fm in fmaps:
        for k in MODEL_FEATURE_ORDER:
            if k not in {x.lower() for x in fm.keys()}:
                raise ValueError(f"Missing expected feature: {k}")


def normalize_to_engine_format(win: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize window dict to engine format (uppercase keys for raw signals)."""
    normalized = {}
    # Handle raw signal keys (case-insensitive)
    signal_map = {
        "EDA": ["EDA", "eda"],
        "TEMP": ["TEMP", "temp"],
        "BVP": ["BVP", "bvp"],
        "ACC_x": ["ACC_x", "acc_x"],
        "ACC_y": ["ACC_y", "acc_y"],
        "ACC_z": ["ACC_z", "acc_z"],
    }
    
    for upper_key, variants in signal_map.items():
        for variant in variants:
            if variant in win:
                normalized[upper_key] = win[variant]
                break
    
    # Handle environmental params (case-insensitive)
    env_map = {
        "temp_c": ["temp_c", "TEMP_C"],
        "humidity": ["humidity", "HUMIDITY"],
        "aqi": ["aqi", "AQI", "air_quality"],
        "local_hour": ["local_hour", "LOCAL_HOUR"],
    }
    
    for key, variants in env_map.items():
        for variant in variants:
            if variant in win:
                normalized[key] = win[variant]
                break
    
    return normalized

