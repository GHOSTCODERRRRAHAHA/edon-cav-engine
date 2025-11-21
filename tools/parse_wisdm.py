import argparse, csv, json
from pathlib import Path

def read_wisdm_txt(p: Path):
    with p.open() as f:
        r = csv.reader(f, delimiter=',')
        for row in r:
            if len(row) < 6: 
                continue
            # <user>,<activity>,<ts>,<x>,<y>,<z> (typical WISDM txt)
            _, _, _, x, y, z = row[:6]
            yield float(x), float(y), float(z)

def to_windows(samples, win=240, step=120):
    buf=[]
    for x,y,z in samples:
        buf.append((x,y,z))
        if len(buf) == win:
            yield buf[:]
            buf = buf[step:]
    if len(buf) >= win:
        yield buf[:win]

def synth_missing(n):
    return {"eda":[0.0]*n, "temp":[36.5]*n, "bvp":[0.0]*n,
            "temp_c":22.0, "humidity":45, "aqi":40, "local_hour":14}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=int, default=240)
    ap.add_argument("--step", type=int, default=120)
    a = ap.parse_args()

    src = Path(a.src)
    files = [src] if src.is_file() else list(src.rglob("*.txt"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    count=0
    with out.open("w") as w:
        for f in files:
            for win in to_windows(read_wisdm_txt(f), a.win, a.step):
                ax=[t[0] for t in win]; ay=[t[1] for t in win]; az=[t[2] for t in win]
                rec = {"acc_x":ax,"acc_y":ay,"acc_z":az, **synth_missing(a.win)}
                w.write(json.dumps(rec)+"\n"); count+=1
    print(f"Wrote {count} windows to {out}")

if __name__ == "__main__":
    main()

