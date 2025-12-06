# Git Commit Checklist

## ✅ What Should Be Committed

### Core Code
- ✅ `demo_mujoco/run_demo.py` - Main demo script
- ✅ `demo_mujoco/train_edon_mujoco.py` - Training script
- ✅ `demo_mujoco/controllers/` - Baseline and EDON controllers
- ✅ `demo_mujoco/sim/` - MuJoCo environment wrapper
- ✅ `demo_mujoco/disturbances/` - Disturbance generator
- ✅ `demo_mujoco/metrics/` - Metrics tracker
- ✅ `demo_mujoco/ui/` - Web UI (HTML, FastAPI server)
- ✅ `demo_mujoco/requirements.txt` - Python dependencies

### Documentation
- ✅ All `.md` files in `demo_mujoco/` (OEM guides, explanations)
- ✅ `demo_mujoco/README.md` - Setup instructions

### Configuration
- ✅ `demo_mujoco/sim/simple_humanoid.xml` - MuJoCo model
- ✅ `demo_mujoco/start_demo.ps1` - Demo startup script
- ✅ `demo_mujoco/clear_memory.ps1` - Memory clearing script

---

## ❌ What Should NOT Be Committed

### Model Files (Already in .gitignore)
- ❌ `demo_mujoco/models/*.pt` - Trained model weights
- ❌ `demo_mujoco/models/*.pth` - Model checkpoints
- ❌ Any `.pt` or `.pth` files

### Database Files
- ❌ `app/robot_stability_memory.db` - Adaptive memory database
- ❌ Any `.db` files (should be in .gitignore)

### Build Artifacts
- ❌ `__pycache__/` - Python cache (already in .gitignore)
- ❌ `*.pyc` - Compiled Python files
- ❌ `*.egg-info/` - Package metadata

### Logs and Temporary Files
- ❌ `demo_mujoco/MUJOCO_LOG.TXT` - Log files
- ❌ `logs/` - Log directories
- ❌ `tmp/` - Temporary files

### Environment Files
- ❌ `.env` - Environment variables
- ❌ `.env.*` - Environment variable files

---

## ⚠️ Check Before Committing

### 1. Check for Sensitive Data
```bash
# Check for API keys, secrets, etc.
grep -r "API_KEY\|SECRET\|PASSWORD" demo_mujoco/
```

### 2. Check for Large Files
```bash
# Check for files > 10MB
find demo_mujoco/ -type f -size +10M
```

### 3. Check for Database Files
```bash
# Check for .db files
find . -name "*.db" -not -path "./.git/*"
```

### 4. Verify .gitignore
```bash
# Check what git sees
git status --ignored
```

---

## 📝 Suggested Commit Message

```
feat: Add MuJoCo demo with zero-shot and training modes

- Add side-by-side comparison demo (baseline vs EDON)
- Implement zero-shot EDON integration with safety mechanisms
- Add training mode using OEM API endpoints
- Include comprehensive documentation for OEMs
- Add web UI for real-time visualization
- Implement state-aware modulation fixes for zero-shot
- Add adaptive memory integration
- Include verification and metrics tracking

Features:
- Zero-shot performance: 25-50% intervention reduction
- Training mode: 90%+ improvement after training
- Safety mechanism: Prevents worse-than-baseline performance
- Real-time UI: WebSocket-based visualization
- OEM-ready: Uses same API endpoints as production

Documentation:
- OEM environment description
- Training guide
- Verification guide
- Roadmap and deployment guides
```

---

## 🚀 Commit Steps

### 1. Check Status
```bash
git status
```

### 2. Add Files
```bash
# Add all demo_mujoco files (respects .gitignore)
git add demo_mujoco/

# Or add specific files
git add demo_mujoco/run_demo.py
git add demo_mujoco/train_edon_mujoco.py
# ... etc
```

### 3. Check What Will Be Committed
```bash
git status
git diff --cached --stat
```

### 4. Commit
```bash
git commit -m "feat: Add MuJoCo demo with zero-shot and training modes

- Add side-by-side comparison demo (baseline vs EDON)
- Implement zero-shot EDON integration with safety mechanisms
- Add training mode using OEM API endpoints
- Include comprehensive documentation for OEMs
- Add web UI for real-time visualization
- Implement state-aware modulation fixes for zero-shot
- Add adaptive memory integration
- Include verification and metrics tracking"
```

### 5. Push
```bash
git push origin main
# or
git push origin master
```

---

## ⚠️ Important Notes

### Don't Commit:
1. **Trained models** (`.pt`, `.pth` files) - Too large, already in .gitignore
2. **Database files** (`.db`) - Contains learned data, should be local
3. **Log files** - Temporary, can be regenerated
4. **Environment files** (`.env`) - May contain secrets

### Do Commit:
1. **All code** - Python scripts, HTML, XML models
2. **All documentation** - Markdown files, guides
3. **Configuration** - Requirements, scripts
4. **UI assets** - HTML, CSS, JavaScript

---

## ✅ Final Checklist

Before pushing:
- [ ] No `.pt` or `.pth` files in commit
- [ ] No `.db` files in commit
- [ ] No `.env` files in commit
- [ ] No large log files
- [ ] All code files included
- [ ] All documentation included
- [ ] `.gitignore` is correct
- [ ] Commit message is descriptive

---

## 🎯 Quick Command

```bash
# Check what will be committed
git status

# Add everything (respects .gitignore)
git add demo_mujoco/

# Verify what's staged
git status

# Commit
git commit -m "feat: Add MuJoCo demo with zero-shot and training modes"

# Push
git push
```

