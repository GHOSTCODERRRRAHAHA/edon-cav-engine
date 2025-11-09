# Building OEM Dataset

## Quick Start

1. **Navigate to project root:**
   ```powershell
   cd C:\Users\cjbig\Desktop\EDON
   ```

2. **Activate virtual environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   (If you don't have a venv, run `.\setup_venv.ps1` first)

3. **Ensure FastAPI server is running:**
   ```powershell
   uvicorn app.main:app --reload
   ```
   (In a separate terminal)

4. **Run the dataset builder:**
   ```powershell
   python tools\build_oem_dataset.py
   ```

## Output Files

The script will create three files in the `outputs/` directory:
- `oem_sample_windows.csv` - Analytics and results (compact)
- `oem_sample_windows.parquet` - Analytics and results (efficient)
- `oem_sample_windows.jsonl` - Full records with raw signal arrays

## Troubleshooting

**Error: "can't open file"**
- Make sure you're in the project root directory (`C:\Users\cjbig\Desktop\EDON`)

**Error: "ModuleNotFoundError"**
- Activate your virtual environment: `.\venv\Scripts\Activate.ps1`
- Install dependencies: `pip install -r requirements.txt`
- Also install: `pip install requests tqdm`

**Error: "CAV API is not available"**
- Start the FastAPI server first: `uvicorn app.main:app --reload`

