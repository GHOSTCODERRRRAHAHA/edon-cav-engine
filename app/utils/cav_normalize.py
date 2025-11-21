"""CAV input normalization utilities."""

from typing import Any, Dict
import math

RAW_UPPER = {"EDA", "TEMP", "BVP", "ACC_x", "ACC_y", "ACC_z"}
RAW_LOWER = {"eda", "temp", "bvp", "acc_x", "acc_y", "acc_z"}


def normalize_keys(win: Dict[str, Any]) -> Dict[str, Any]:
    """Lowercase all keys and expose common aliases; do not mutate original."""
    lower = {str(k).lower(): v for k, v in win.items()}
    # Accept ambient context in multiple spellings
    if "temp" not in lower:
        if "temp_c" in lower:
            lower["temp"] = lower["temp_c"]
    if "aqi" not in lower and "air_quality" in lower:
        lower["aqi"] = lower["air_quality"]
    return lower


def is_raw_window(win: Dict[str, Any], window_len: int = 240) -> bool:
    """True if it looks like raw series: has the 6 keys (any case) and each is a list/tuple of length window_len."""
    keys = set(win.keys())
    # allow either casing
    has_upper = RAW_UPPER.issubset(keys)
    has_lower = RAW_LOWER.issubset({k.lower() for k in keys})
    if not (has_upper or has_lower):
        return False
    # Pick values regardless of casing
    def _val(name: str):
        return win.get(name) or win.get(name.lower()) or win.get(name.upper())
    
    for name in ["EDA", "TEMP", "BVP", "ACC_x", "ACC_y", "ACC_z"]:
        v = _val(name)
        if not isinstance(v, (list, tuple)) or len(v) != window_len:
            return False
        # quick type check
        try:
            _ = float(v[0]) if v else 0.0
        except (ValueError, TypeError, IndexError):
            return False
    return True


def featurize_raw(win: Dict[str, Any]) -> Dict[str, float]:
    """Compute minimal, stable features from raw arrays. Uses numpy if present, else pure python."""
    try:
        import numpy as np
        def asf(x): return np.asarray(x, dtype=float)
        get = lambda k: win.get(k) or win.get(k.lower()) or win.get(k.upper())
        
        eda = asf(get("EDA"))
        bvp = asf(get("BVP"))
        accx = asf(get("ACC_x"))
        accy = asf(get("ACC_y"))
        accz = asf(get("ACC_z"))
        accm = (accx**2 + accy**2 + accz**2) ** 0.5
        
        def sstd(x):  # sample std
            if x.size <= 1:
                return 0.0
            return float(x.std(ddof=1))
        
        return {
            "eda_mean": float(eda.mean()), "eda_std": sstd(eda),
            "bvp_mean": float(bvp.mean()), "bvp_std": sstd(bvp),
            "acc_mean": float(accm.mean()), "acc_std": sstd(accm),
        }
    except Exception:
        # numpy-less fallback
        import statistics
        get = lambda k: win.get(k) or win.get(k.lower()) or win.get(k.upper())
        
        eda = [float(x) for x in get("EDA")]
        bvp = [float(x) for x in get("BVP")]
        accx = [float(x) for x in get("ACC_x")]
        accy = [float(x) for x in get("ACC_y")]
        accz = [float(x) for x in get("ACC_z")]
        accm = [math.sqrt(x*x + y*y + z*z) for x, y, z in zip(accx, accy, accz)]
        
        def mean(a): return sum(a)/len(a) if a else 0.0
        def sstd(a):
            if len(a) <= 1:
                return 0.0
            return float(statistics.stdev(a))
        
        return {
            "eda_mean": mean(eda), "eda_std": sstd(eda),
            "bvp_mean": mean(bvp), "bvp_std": sstd(bvp),
            "acc_mean": mean(accm), "acc_std": sstd(accm),
        }


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
        "temp_c": ["temp_c", "temp_c", "TEMP_C"],
        "humidity": ["humidity", "HUMIDITY"],
        "aqi": ["aqi", "AQI", "air_quality"],
        "local_hour": ["local_hour", "LOCAL_HOUR", "local_hour"],
    }
    
    for key, variants in env_map.items():
        for variant in variants:
            if variant in win:
                normalized[key] = win[variant]
                break
    
    return normalized

