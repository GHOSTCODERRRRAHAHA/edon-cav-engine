import numpy as np
FEATURE_ORDER = ["EDA", "TEMP", "BVP", "ACC_x", "ACC_y", "ACC_z"]
def pool6_from_window(win240):
    """Take a 240-sample window dict and return a (1,6) pooled feature vector."""
    X = np.zeros((1, 6), dtype=float)
    for i, key in enumerate(FEATURE_ORDER):
        arr = np.asarray(win240.get(key, np.zeros(240)), dtype=float)
        if arr.size == 0:
            arr = np.zeros(240)
        X[0, i] = float(np.nanmean(arr))
    return X
