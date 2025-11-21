import pandas as pd
import numpy as np
from joblib import dump
from lightgbm import LGBMClassifier
DATA = r"data/raw/wesad/wesad_wrist_4hz.csv"
OUT  = r"cav_engine_v3_3_LGBM.joblib"
WIN  = 240  # 60s @ 4 Hz
df = pd.read_csv(DATA)
cols = ["EDA","TEMP","BVP","ACC_x","ACC_y","ACC_z","label","subject"]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise RuntimeError(f"CSV missing columns: {missing}")
df = df.sort_values(["subject","label"]).reset_index(drop=True)
X_list, y_list = [], []
for (subj, lab), grp in df.groupby(["subject","label"]):
    n = len(grp)
    if n < WIN:
        continue
    for i in range(0, n - WIN + 1, WIN):
        chunk = grp.iloc[i:i+WIN]
        x = [
            float(chunk["EDA"].mean()),
            float(chunk["TEMP"].mean()),
            float(chunk["BVP"].mean()),
            float(chunk["ACC_x"].mean()),
            float(chunk["ACC_y"].mean()),
            float(chunk["ACC_z"].mean()),
        ]
        X_list.append(x)
        y_list.append(int(lab))
X = np.array(X_list, dtype=float)
y = np.array(y_list, dtype=int)
print("Training set:", X.shape, "labels:", {k:int(v) for k,v in zip(*np.unique(y, return_counts=True))})
if X.shape[0] == 0:
    raise RuntimeError("No training windows built. Check CSV and column names.")
clf = LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    class_weight="balanced",
    random_state=42,
)
clf.fit(X, y)
dump(clf, OUT)
print(f"Saved -> {OUT}")
