import os
import math
import time
import json
from typing import Dict, Any, List

import requests
import pandas as pd

TOTAL = 10_000
WINDOW = 240

BASE = os.getenv("EDON_API_BASE", "http://127.0.0.1:8001")
TOKEN = os.getenv("EDON_API_TOKEN")
AUTH_ENABLED = os.getenv("EDON_AUTH_ENABLED", "false").lower() == "true"


def make_window(seed: int) -> Dict[str, Any]:
    """Create one synthetic but structured 240-sample window."""
    return {
        "EDA": [k * 0.01 for k in range(WINDOW)],
        "TEMP": [36.5 for _ in range(WINDOW)],
        "BVP": [math.sin((k + seed) / 12.0) for k in range(WINDOW)],
        "ACC_x": [0.0] * WINDOW,
        "ACC_y": [0.0] * WINDOW,
        "ACC_z": [1.0] * WINDOW,
        "temp_c": 22.0,
        "humidity": 50.0,
        "aqi": 35,
        "local_hour": seed % 24,
    }


def get_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if AUTH_ENABLED and TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    return headers


def main() -> None:
    os.makedirs("data", exist_ok=True)

    records: List[Dict[str, Any]] = []
    ok = 0
    failed = 0

    print(f"[BUILD] Base={BASE} total_windows={TOTAL}")
    headers = get_headers()

    for seed in range(TOTAL):
        payload = {"windows": [make_window(seed)]}

        retries = 0
        last_err: Any = None
        while retries < 3:
            try:
                resp = requests.post(
                    f"{BASE}/oem/cav/batch",
                    headers=headers,
                    json=payload,
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                item = data[0] if isinstance(data, list) and data else data

                # Store a compact row; keep full response as JSON string
                records.append(
                    {
                        "seed": seed,
                        "response": json.dumps(item),
                    }
                )
                ok += 1
                break
            except Exception as e:  # noqa: BLE001
                retries += 1
                last_err = e
                time.sleep(0.2)

        if retries == 3:
            failed += 1
            print(f"[WARN] seed={seed} failed after retries. last_err={last_err}")

        # lightweight progress
        if (seed + 1) % 500 == 0 or seed == TOTAL - 1:
            print(f"[PROGRESS] {seed + 1}/{TOTAL} windows processed (ok={ok}, failed={failed})")

    print(f"[BUILD] Done. ok={ok}, failed={failed}")

    if records:
        df = pd.DataFrame(records)
        parquet_path = "data/edon_cav_10k.parquet"
        jsonl_path = "data/edon_cav_10k.jsonl"

        df.to_parquet(parquet_path, index=False)
        df.to_json(jsonl_path, orient="records", lines=True)

        print(f"[BUILD] Saved Parquet → {parquet_path}")
        print(f"[BUILD] Saved JSONL  → {jsonl_path}")
    else:
        print("[BUILD] No records collected; nothing saved.")


if __name__ == "__main__":
    main()
