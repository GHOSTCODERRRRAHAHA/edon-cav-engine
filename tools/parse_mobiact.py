import argparse, json
from pathlib import Path
import pandas as pd

def to_windows(xs, ys, zs, win=240, step=120):
    i=0
    n=len(xs)
    while i+win<=n:
        yield xs[i:i+win], ys[i:i+win], zs[i:i+win]
        i+=step
    if i< n and n>=win:
        yield xs[-win:], ys[-win:], zs[-win:]

def synth_missing(n):
    return {"eda":[0.0]*n,"temp":[36.5]*n,"bvp":[0.0]*n,
            "temp_c":22.0,"humidity":45,"aqi":40,"local_hour":14}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=int, default=240)
    ap.add_argument("--step", type=int, default=120)
    a = ap.parse_args()

    src = Path(a.src)
    files = [src] if src.is_file() else list(src.rglob("*.csv"))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    count=0
    with out.open("w") as w:
        for f in files:
            df = pd.read_csv(f)
            # Handle both lowercase and uppercase column names
            acc_x_col = "acc_x" if "acc_x" in df.columns else "ACC_x" if "ACC_x" in df.columns else None
            acc_y_col = "acc_y" if "acc_y" in df.columns else "ACC_y" if "ACC_y" in df.columns else None
            acc_z_col = "acc_z" if "acc_z" in df.columns else "ACC_z" if "ACC_z" in df.columns else None
            
            if not all([acc_x_col, acc_y_col, acc_z_col]):
                print(f"Warning: Could not find acc columns in {f}. Available: {list(df.columns)}")
                continue
                
            for ax, ay, az in to_windows(df[acc_x_col].tolist(), df[acc_y_col].tolist(), df[acc_z_col].tolist(), a.win, a.step):
                rec = {"acc_x":ax,"acc_y":ay,"acc_z":az, **synth_missing(a.win)}
                w.write(json.dumps(rec)+"\n"); count+=1
    print(f"Wrote {count} windows to {out}")

if __name__ == "__main__":
    main()

