# Installing MuJoCo

MuJoCo is required for the demo. Here's how to install it:

## Option 1: Direct pip install (Recommended)

```bash
pip install mujoco
```

## Option 2: If Option 1 fails (Windows)

On Windows, you might need to install the pre-built wheel:

```bash
pip install mujoco --upgrade
```

Or try the older mujoco-py (if mujoco doesn't work):

```bash
pip install mujoco-py
```

## Option 3: Manual installation

1. Download MuJoCo from: https://github.com/google-deepmind/mujoco/releases
2. Extract to a folder (e.g., `C:\Users\YourName\.mujoco\mujoco-3.x.x`)
3. Set environment variable: `MUJOCO_PY_MUJOCO_PATH=C:\Users\YourName\.mujoco\mujoco-3.x.x`
4. Then: `pip install mujoco`

## Verify Installation

Run:
```bash
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
```

If this works, you're good to go!

## Troubleshooting

**Error: "No module named 'mujoco'"**
- Make sure you're in the virtual environment: `(.venv) PS ...`
- Try: `pip install --upgrade mujoco`
- Check Python version (MuJoCo requires Python 3.8+)

**Error: "DLL load failed" (Windows)**
- Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Or install Microsoft Visual C++ Build Tools

**Still having issues?**
- Check MuJoCo docs: https://mujoco.readthedocs.io/en/latest/installation.html
- Try the mujoco-py package instead (older but more compatible)

