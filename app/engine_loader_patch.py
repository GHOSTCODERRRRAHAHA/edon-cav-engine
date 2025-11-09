from pathlib import Path
import os
import json
from joblib import load

ROOT = Path(__file__).resolve().parents[1]  # repo root
# Also check parent directory (in case we're in a subdirectory like edon-cav-engine)
PARENT_ROOT = ROOT.parent if ROOT.name in ["edon-cav-engine", "tools", "temp_sdk"] else ROOT

def _find_artifact(name: str, also=None) -> Path:
    cands = []
    env_dir = os.getenv("EDON_MODEL_DIR")
    if env_dir:
        cands.append(Path(env_dir) / name)
    
    # Check current repo root
    cands.append(ROOT / "models" / name)
    cands.append(ROOT / name)
    for p in ROOT.glob("cav_engine_v3_2_*"):
        cands.append(p / name)
    
    # Check parent directory if different
    if PARENT_ROOT != ROOT:
        cands.append(PARENT_ROOT / "models" / name)
        cands.append(PARENT_ROOT / name)
        for p in PARENT_ROOT.glob("cav_engine_v3_2_*"):
            cands.append(p / name)
    
    if also:
        cands += [ROOT / a for a in also]
        if PARENT_ROOT != ROOT:
            cands += [PARENT_ROOT / a for a in also]
    
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"{name} not found. Checked: " + ", ".join(map(str, cands[:10])))  # Limit error message length

def load_artifacts():
    """
    Returns: (model, scaler, schema_dict)
    Looks for:
      - cav_state_v3_2.joblib
      - cav_state_scaler_v3_2.joblib
      - cav_state_schema_v3_2.json
    """
    schema_path = _find_artifact("cav_state_schema_v3_2.json")
    scaler_path = _find_artifact("cav_state_scaler_v3_2.joblib")
    model_path  = _find_artifact("cav_state_v3_2.joblib")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    scaler = load(scaler_path)
    model  = load(model_path)
    return model, scaler, schema

