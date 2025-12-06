# Git Commit Commands

## Step-by-Step Commands to Run

### 1. Check Current Status
```powershell
git status
```

### 2. Add Files
```powershell
# Add demo_mujoco directory
git add demo_mujoco/

# Add updated .gitignore
git add .gitignore
```

### 3. Verify What Will Be Committed
```powershell
git status
```

### 4. Commit
```powershell
git commit -m "feat: Add MuJoCo demo with zero-shot and training modes

- Add side-by-side comparison demo (baseline vs EDON)
- Implement zero-shot EDON integration with safety mechanisms
- Add training mode using OEM API endpoints
- Include comprehensive documentation for OEMs
- Add web UI for real-time visualization
- Implement state-aware modulation fixes for zero-shot
- Add adaptive memory integration
- Include verification and metrics tracking
- Update .gitignore to exclude log and database files

Features:
- Zero-shot performance: 25-50% intervention reduction
- Training mode: 90%+ improvement after training
- Safety mechanism: Prevents worse-than-baseline performance
- Real-time UI: WebSocket-based visualization
- OEM-ready: Uses same API endpoints as production"
```

### 5. Set Remote (if not already set)
```powershell
git remote set-url origin https://github.com/GHOSTCODERRRRAHAHA/Edc.git
```

### 6. Check Remote
```powershell
git remote -v
```

### 7. Push to Repository
```powershell
# Try main branch first
git push -u origin main

# If that fails, try master
git push -u origin master

# Or push current branch
git push -u origin HEAD
```

---

## If Repository Not Initialized

If you get an error about repository not being initialized:

```powershell
# Initialize repository
git init

# Add remote
git remote add origin https://github.com/GHOSTCODERRRRAHAHA/Edc.git

# Then follow steps 1-7 above
```

---

## If Authentication Required

If you need to authenticate:

```powershell
# Use GitHub CLI (if installed)
gh auth login

# Or use personal access token
# GitHub will prompt for username and token
```

---

## Quick One-Liner (If Everything is Ready)

```powershell
git add demo_mujoco/ .gitignore && git commit -m "feat: Add MuJoCo demo with zero-shot and training modes" && git remote set-url origin https://github.com/GHOSTCODERRRRAHAHA/Edc.git && git push -u origin main
```

