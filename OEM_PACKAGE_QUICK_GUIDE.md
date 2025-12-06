# Quick Guide: What to Give OEMs

**Contact:** Charlie Biggins - charlie@edoncore.com

---

## Essential Package Contents

### 1. Docker Image
- **File:** `edon-server-v1.0.1.tar`
- **Location:** `release/edon-server-v1.0.1.tar` or build it
- **Contains:** Full EDON Core server with v8 robot stability models

### 2. Python SDK
- **File:** `edon-0.1.0-py3-none-any.whl`
- **Location:** `sdk/python/dist/` (build first if needed)
- **Contains:** `EdonClient` with `cav()` and `robot_stability()` methods

### 3. Documentation (IP-Safe)
- `docs/OEM_ONBOARDING.md`
- `docs/OEM_INTEGRATION.md`
- `docs/OEM_ROBOT_STABILITY.md` ⭐ NEW
- `docs/OEM_API_CONTRACT.md`
- `docs/OEM_BRIEF.md`

### 4. Examples
- `examples/robot_stability_example.py` ⭐ NEW
- Other integration examples

### 5. Setup Scripts
- `setup_oem_sdk.ps1` (Windows)
- `setup_oem_sdk.sh` (Linux/macOS - create if missing)

### 6. Release Files
- `RELEASE_NOTES.md` - What's new
- `LICENSE.txt` - License agreement
- `README.md` - Package overview
- `SHA256SUMS.txt` - File checksums

---

## Manual Package Creation (If Script Fails)

### Step 1: Create Directory
```powershell
New-Item -ItemType Directory -Path "EDON_v1.0.1_OEM_RELEASE" -Force
cd EDON_v1.0.1_OEM_RELEASE
New-Item -ItemType Directory -Path "docker","sdk/python","docs","examples","scripts" -Force
```

### Step 2: Copy Files
```powershell
# Docker image
Copy-Item ..\release\edon-server-v1.0.1.tar docker\

# Python SDK
Copy-Item ..\sdk\python\dist\edon-*.whl sdk\python\

# Documentation
Copy-Item ..\docs\OEM_*.md docs\

# Examples
Copy-Item ..\examples\robot_stability_example.py examples\

# Setup scripts
Copy-Item ..\setup_oem_sdk.ps1 scripts\
```

### Step 3: Create Release Notes
Create `RELEASE_NOTES.md` with version info and new features.

### Step 4: Create README
Create `README.md` with quick start instructions.

### Step 5: Generate Checksums
```powershell
Get-ChildItem -Recurse -File | ForEach-Object {
    $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    "$hash  $($_.FullName.Replace((Get-Location).Path + '\', ''))"
} | Set-Content SHA256SUMS.txt
```

### Step 6: Create ZIP
```powershell
cd ..
Compress-Archive -Path "EDON_v1.0.1_OEM_RELEASE\*" -DestinationPath "EDON_v1.0.1_OEM_RELEASE.zip" -Force
```

---

## What OEMs Receive

### ✅ Included
1. Docker image (pre-built server)
2. Python SDK (client library)
3. Documentation (IP-safe, usage only)
4. Examples (working code)
5. Setup scripts (automated installation)

### ✅ Capabilities
1. Human state prediction (`/oem/cav/batch`)
2. Robot stability control (`/oem/robot/stability`) ⭐ NEW
3. Health monitoring (`/health`)
4. Adaptive memory (`/memory/summary`)

### ❌ NOT Included
1. Source code
2. Training scripts
3. Research code (v8 training)
4. Implementation details
5. Model internals

---

## Verification

Before shipping, verify:
- [ ] Docker image loads: `docker load < docker/edon-server-v1.0.1.tar`
- [ ] SDK installs: `pip install sdk/python/edon-*.whl`
- [ ] Server starts: `docker run -p 8002:8000 edon-server:v1.0.1`
- [ ] Health check works: `curl http://localhost:8002/health`
- [ ] Robot stability endpoint works: Test with `test_robot_stability_api.ps1`
- [ ] All docs are IP-safe (no implementation details)

---

**Contact:** Charlie Biggins - charlie@edoncore.com

