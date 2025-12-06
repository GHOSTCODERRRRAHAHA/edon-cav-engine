# Training vs EDON Core: What OEMs Access

## Quick Answer: **NO, OEMs Do NOT Touch EDON Core**

During training, OEMs:
- ✅ Train a **policy network** (separate from EDON Core)
- ✅ Use EDON Core API for **inference only** (read-only, black box)
- ❌ **Do NOT modify** EDON Core engine
- ❌ **Do NOT see** EDON Core source code

---

## What is EDON Core?

**EDON Core** = The proprietary intelligence engine (IP-protected):

```
app/
├── engine.py              # CAV Engine (physiological state prediction)
├── adaptive_memory.py     # Adaptive Memory (unsupervised learning)
└── routes/
    └── robot_stability.py # Robot stability API endpoint
```

**What it does:**
- Processes robot state (roll, pitch, COM, etc.)
- Computes fail-risk prediction
- Outputs strategy + modulations via API
- **Black box** - OEMs only see API interface

---

## What OEMs Access During Training

### 1. Training Scripts (Public/Research Code)

```
training/
├── train_edon_v8_strategy.py  # PPO training loop
├── edon_v8_policy.py          # Policy network (PyTorch)
└── fail_risk_model.py        # Fail-risk model (PyTorch)
```

**What OEMs see:**
- ✅ Training loop code
- ✅ Policy network architecture
- ✅ Reward computation
- ❌ **NOT EDON Core** (separate system)

### 2. Environment Wrapper (Public/Research Code)

```
env/
└── edon_humanoid_env_v8.py    # v8 environment wrapper
```

**What OEMs see:**
- ✅ Environment interface
- ✅ Reward shaping
- ✅ Episode management
- ❌ **NOT EDON Core** (uses API, doesn't modify it)

### 3. EDON Core API (Read-Only Interface)

```
POST /oem/robot/stability
```

**What OEMs see:**
- ✅ API endpoint (HTTP interface)
- ✅ Request/response format
- ✅ Input/output schemas
- ❌ **NOT implementation** (black box)

---

## Training Flow (What OEMs Actually Do)

```
┌─────────────────────────────────────────────────────────┐
│  OEM's Training Script                                  │
│  (training/train_edon_v8_strategy.py)                  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  1. Create Environment                                  │
│     - env.edon_humanoid_env_v8.py                      │
│     - Uses baseline_controller()                       │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  2. Train Policy Network                                │
│     - training.edon_v8_policy.py                        │
│     - PyTorch neural network                            │
│     - Learns strategy selection + modulations           │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  3. Use EDON Core API (Read-Only)                       │
│     - POST /oem/robot/stability                         │
│     - Gets fail-risk, strategy suggestions              │
│     - Does NOT modify EDON Core                         │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  4. Save Trained Model                                  │
│     - models/edon_v8_oem.pt                             │
│     - Policy network weights only                       │
│     - Does NOT include EDON Core                        │
└─────────────────────────────────────────────────────────┘
```

---

## What Gets Trained

### ✅ Policy Network (What OEMs Train)

```python
# training/edon_v8_policy.py
class EdonV8StrategyPolicy(nn.Module):
    """Policy network that learns strategy selection + modulations."""
    
    def forward(self, obs):
        # Input: Robot state (roll, pitch, COM, etc.)
        # Output: Strategy ID + modulations (gain_scale, compliance, bias)
        return strategy_id, modulations
```

**This is:**
- ✅ Public code (OEMs can see it)
- ✅ Trainable (OEMs modify weights)
- ✅ Robot-specific (different per OEM)
- ❌ **NOT EDON Core** (separate system)

### ❌ EDON Core (What OEMs DON'T Touch)

```python
# app/engine.py (IP-protected, NOT in training)
class EdonCore:
    """CAV Engine - proprietary intelligence."""
    
    def compute_cav(self, sensor_data):
        # Proprietary algorithms
        # OEMs never see this
        pass
```

**This is:**
- ❌ IP-protected (OEMs never see source)
- ❌ Read-only (OEMs only use API)
- ❌ Unchanged (training doesn't modify it)
- ✅ Black box (OEMs treat as service)

---

## API Interface (What OEMs See)

### Request (OEMs Send This)

```python
POST /oem/robot/stability
{
    "roll": 0.1,
    "pitch": 0.05,
    "roll_velocity": 0.02,
    "pitch_velocity": 0.01,
    "com_x": 0.0,
    "com_y": 0.0,
    # ... other robot state
}
```

### Response (OEMs Receive This)

```python
{
    "strategy_id": 1,
    "strategy_name": "HIGH_DAMPING",
    "intervention_risk": 0.75,
    "modulations": {
        "gain_scale": 1.2,
        "compliance": 0.6,
        "bias": [0.1, -0.05, ...]
    }
}
```

**OEMs see:**
- ✅ API contract (input/output)
- ✅ Response format
- ❌ **NOT implementation** (black box)

---

## IP Protection Summary

| Component | OEMs See? | OEMs Modify? | IP Status |
|-----------|-----------|--------------|-----------|
| **EDON Core** (`app/engine.py`) | ❌ No | ❌ No | 🔒 IP-Protected |
| **Training Scripts** (`training/`) | ✅ Yes | ✅ Yes | 📖 Public/Research |
| **Policy Network** (`edon_v8_policy.py`) | ✅ Yes | ✅ Yes | 📖 Public/Research |
| **EDON Core API** (`/oem/robot/stability`) | ✅ Yes (interface) | ❌ No | 🔒 IP-Protected (impl) |

---

## What OEMs Actually Train

**OEMs train:**
1. **Policy network weights** (PyTorch model)
   - Learns: "When to use which strategy"
   - Learns: "How to modulate actions"
   - Saved as: `models/edon_v8_oem.pt`

**OEMs do NOT train:**
1. **EDON Core engine** (remains unchanged)
2. **Fail-risk model** (pre-trained, optional to retrain)
3. **CAV Engine** (separate system, not used in robot training)

---

## Example: OEM Training Process

```python
# 1. OEM creates environment (their robot/simulator)
env = OEMHumanoidEnv()

# 2. OEM trains policy network (public code)
policy = EdonV8StrategyPolicy()
trainer = PPO(policy)

# 3. During training, calls EDON Core API (read-only)
for episode in range(300):
    obs = env.reset()
    while not done:
        # Get baseline action
        baseline = baseline_controller(obs)
        
        # Call EDON Core API (read-only, black box)
        edon_response = edon_client.robot_stability(obs)
        # Response: {strategy_id, modulations, fail_risk}
        
        # Policy network learns from EDON Core suggestions
        action = policy.compute_action(baseline, obs, edon_response)
        
        # Step environment
        obs, reward, done, info = env.step(action)
    
    # Update policy network (only thing being trained)
    trainer.update(trajectory)

# 4. Save trained policy (does NOT include EDON Core)
torch.save(policy.state_dict(), "models/edon_v8_oem.pt")
```

**Key point:** EDON Core is called as a **service** (API), not modified.

---

## Conclusion

**OEMs:**
- ✅ Train policy networks (public code)
- ✅ Use EDON Core API (read-only interface)
- ❌ **Do NOT touch EDON Core** (IP-protected black box)

**EDON Core:**
- 🔒 Remains unchanged during training
- 🔒 IP-protected (source not visible)
- 🔒 Used as service (API calls only)

This separation ensures:
1. **IP protection** (EDON Core stays proprietary)
2. **Flexibility** (OEMs can train on their robots)
3. **Modularity** (training code separate from core engine)

