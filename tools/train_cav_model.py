import argparse, json, hashlib, joblib
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

CHANNELS = ["eda","temp","bvp","acc_x","acc_y","acc_z"]

def load_jsonl(p: Path):
    with p.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def summarize(rec):
    feats=[]
    for k in CHANNELS:
        arr = np.asarray(rec[k], dtype=float)
        feats += [float(arr.mean()), float(arr.std())]
    return np.array(feats, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=128)
    a = ap.parse_args()

    X=[]; y=[]
    for rec in load_jsonl(Path(a.data)):
        X.append(summarize(rec))
        # simple pseudo-label from motion variance (placeholder)
        var = np.var(np.array(rec["acc_x"]+rec["acc_y"]+rec["acc_z"]))
        y.append(1 if var < 0.05 else 0)
    X=np.vstack(X); y=np.array(y)

    # Fit PCA (clip to feature count)
    pca = PCA(n_components=min(a.pca, X.shape[1]))
    Z = pca.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Z, y)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    out_file = out/"cav_engine_v4_0.pkl"
    joblib.dump({"pca":pca,"clf":clf,"channels":CHANNELS}, out_file)

    sha = hashlib.sha256(out_file.read_bytes()).hexdigest()
    (out/"HASHES.txt").write_text(f"{out_file.name}  {sha}\n")
    print(f"Saved model to {out_file}  sha256={sha}")

if __name__ == "__main__":
    main()

