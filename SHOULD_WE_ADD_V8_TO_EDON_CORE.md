# Should We Add v8 to EDON Core?

## The Question

**"So we should add v8 into EDON Core?"**

## Short Answer: **YES, but with careful architecture**

Adding v8 capabilities to EDON Core makes sense for OEMs, but requires careful design to maintain separation of concerns.

---

## Arguments FOR Adding v8 to EDON Core

### 1. OEMs Need Robot Stability Control

**Current Gap:**
- EDON Core: Human state prediction ✅
- EDON Core: Robot stability control ❌
- v8: Robot stability control ✅ (but not accessible via API)

**OEM Requirement:**
- OEMs deploying robots need **both**:
  - Human-robot interaction adaptation (EDON Core)
  - Robot stability control (v8)
- Currently, they can only get one through the API

### 2. Unified API Experience

**Benefits:**
- Single API endpoint for all EDON capabilities
- Unified authentication, versioning, deployment
- Easier integration for OEMs (one SDK, one service)
- Consistent response format

**Example:**
```python
# Unified API
client = EdonClient()

# Human state
human_state = client.cav(physio_window)

# Robot stability (NEW)
robot_control = client.robot_stability(robot_state)

# Combined
final_action = combine_edon_controls(human_state, robot_control)
```

### 3. Proven Value

**v8's Track Record:**
- 97.5% intervention reduction (40.30 → 1.00 interventions/episode)
- Validated architecture (temporal memory + early-warning)
- Production-ready model (`edon_v8_strategy_memory_features.pt`)

**Why This Matters:**
- OEMs need this capability
- It's already built and tested
- Just needs to be exposed via API

### 4. Complementary Capabilities

**They Work Together:**
- EDON Core: Adapts robot to human state (speed/torque scales)
- v8: Maintains robot stability (prevents interventions)
- Combined: Robot is both stable AND adaptive to human

**Example Use Case:**
```
Human is stressed (EDON Core) → Reduce speed to 0.4
Robot is tilting (v8) → Apply recovery strategy
Result: Robot maintains stability while being gentle to human
```

---

## Arguments AGAINST Adding v8 to EDON Core

### 1. Different Domains

**Fundamental Difference:**
- **EDON Core**: Human physiological state (cognitive/emotional)
- **v8**: Robot physical state (stability/balance)

**Concern:**
- Mixing domains in one service could be confusing
- Different use cases, different customers
- May violate separation of concerns

### 2. Different Requirements

**Technical Differences:**

| Aspect | EDON Core | v8 |
|--------|-----------|-----|
| **Input frequency** | 60-second windows | Real-time (milliseconds) |
| **Latency requirement** | <100ms acceptable | <10ms critical |
| **Model type** | LightGBM classifier | Learned neural policy |
| **Inference type** | Batch-friendly | Real-time streaming |
| **State management** | 24-hour memory | 8-frame temporal buffer |

**Concern:**
- v8 needs real-time performance (critical for control)
- EDON Core is designed for batch processing
- May need separate infrastructure

### 3. v8 is Research Platform

**Current Status:**
- v8 is research/validation platform
- Not fully productized (no API, no versioning)
- May need refactoring for production

**Concern:**
- Adding research code to production service
- May need significant engineering work
- Risk of instability

### 4. Deployment Complexity

**Infrastructure Concerns:**
- Different model serving requirements
- Different scaling needs
- Different resource requirements (GPU vs CPU)
- May complicate deployment

---

## Recommended Approach: **Hybrid Architecture**

### Option 1: Add v8 as Separate Endpoint (Recommended)

**Architecture:**
```
EDON Core Service
├── /oem/cav/batch          (existing - human state)
├── /oem/robot/stability    (NEW - robot stability)
└── /health                  (unified health check)
```

**Benefits:**
- ✅ Unified API (single service)
- ✅ Clear separation (different endpoints)
- ✅ Can scale independently
- ✅ OEMs get both capabilities

**Implementation:**
```python
# app/routes/robot_stability.py
@router.post("/oem/robot/stability")
async def robot_stability(req: RobotStabilityRequest):
    """
    Compute robot stability control from robot state.
    
    Input: Robot state (roll, pitch, velocities, history)
    Output: Strategy + modulations (gain_scale, compliance, bias)
    """
    # Load v8 model
    policy = load_v8_policy()
    
    # Compute control
    strategy, modulations = policy.compute_action(robot_state)
    
    return RobotStabilityResponse(
        strategy_id=strategy,
        modulations=modulations,
        intervention_risk=fail_risk_model.predict(robot_state)
    )
```

### Option 2: Keep Separate Services (Alternative)

**Architecture:**
```
EDON Core Service (human state)
└── /oem/cav/batch

EDON Robot Service (robot stability)
└── /oem/robot/stability
```

**Benefits:**
- ✅ Complete separation
- ✅ Independent scaling
- ✅ Different deployment strategies

**Drawbacks:**
- ❌ Two services to manage
- ❌ Two SDKs/APIs for OEMs
- ❌ More complex integration

---

## Implementation Plan

### Phase 1: Add v8 Endpoint to EDON Core

**Steps:**
1. **Create robot stability route** (`app/routes/robot_stability.py`)
   - Load v8 model on startup
   - Create endpoint `/oem/robot/stability`
   - Handle robot state input
   - Return strategy + modulations

2. **Add request/response models** (`app/models.py`)
   ```python
   class RobotStabilityRequest(BaseModel):
       robot_state: RobotState
       history: Optional[List[RobotState]] = None
       fail_risk: Optional[float] = None
   
   class RobotStabilityResponse(BaseModel):
       strategy_id: int
       modulations: Modulations
       intervention_risk: float
   ```

3. **Integrate v8 model loading** (`app/main.py`)
   - Load v8 policy on startup
   - Load fail-risk model
   - Handle model versioning

4. **Update SDK** (`sdk/python/edon/client.py`)
   ```python
   def robot_stability(self, robot_state: Dict) -> Dict:
       """Get robot stability control from v8."""
       response = self._post("/oem/robot/stability", {
           "robot_state": robot_state
       })
       return response
   ```

### Phase 2: Testing & Validation

**Tests:**
- Unit tests for v8 endpoint
- Integration tests with real robot state
- Performance tests (latency <10ms)
- Comparison with standalone v8

### Phase 3: Documentation & Release

**Documentation:**
- API documentation for `/oem/robot/stability`
- Integration guide for OEMs
- Example code showing combined usage

---

## Technical Considerations

### 1. Model Loading

**Challenge:** v8 model is large (~5MB), needs to be loaded on startup

**Solution:**
```python
# app/main.py
V8_POLICY = None
V8_FAIL_RISK = None

@app.on_event("startup")
async def load_v8_models():
    global V8_POLICY, V8_FAIL_RISK
    V8_POLICY = load_v8_policy("models/edon_v8_strategy_memory_features.pt")
    V8_FAIL_RISK = load_fail_risk_model("models/edon_fail_risk_v1_fixed_v2.pt")
```

### 2. Latency Requirements

**Challenge:** v8 needs <10ms latency for real-time control

**Solution:**
- Pre-load models in memory
- Use efficient inference (torch.jit.script if needed)
- Cache temporal buffers per session
- Consider async processing for non-critical paths

### 3. State Management

**Challenge:** v8 needs 8-frame temporal buffer per robot

**Solution:**
```python
# Per-robot session state
robot_sessions: Dict[str, RobotSession] = {}

class RobotSession:
    obs_history: deque
    obs_vec_history: deque
    near_fail_history: deque
```

### 4. Versioning

**Challenge:** v8 model may be updated independently

**Solution:**
- Version v8 models separately
- Support multiple model versions
- Allow OEMs to specify model version in request

---

## Recommendation

### ✅ **YES - Add v8 to EDON Core**

**But with this architecture:**

1. **Separate endpoint**: `/oem/robot/stability` (not mixed with `/oem/cav/batch`)
2. **Unified service**: Same EDON Core service, different endpoints
3. **Clear separation**: Different models, different use cases, same API
4. **Productized**: Refactor v8 for production (proper error handling, versioning, etc.)

**Why This Works:**
- ✅ OEMs get unified API experience
- ✅ Clear separation of concerns (different endpoints)
- ✅ Can scale independently if needed
- ✅ Complementary capabilities (human state + robot stability)
- ✅ Proven value (97% intervention reduction)

**Implementation Priority:**
1. **High**: Add v8 endpoint to EDON Core
2. **Medium**: Productize v8 (error handling, versioning)
3. **Low**: Optimize for latency (if needed)

---

## Next Steps

1. **Create design document** for `/oem/robot/stability` endpoint
2. **Implement endpoint** in EDON Core
3. **Add to SDK** for OEM integration
4. **Test with real robot state** data
5. **Document for OEMs** with examples

---

## Conclusion

**YES, we should add v8 to EDON Core** because:
- OEMs need robot stability control
- v8 has proven value (97% improvement)
- Unified API is better for OEMs
- They're complementary (human state + robot stability)

**But do it right:**
- Separate endpoint (clear separation)
- Productize v8 (not just research code)
- Maintain performance (real-time requirements)
- Version properly (model versioning)

**The architecture should be:**
```
EDON Core Service
├── Human State API (/oem/cav/batch)
└── Robot Stability API (/oem/robot/stability) ← NEW
```

**This gives OEMs everything they need in one unified API.**

---

*Last Updated: After analyzing whether to add v8 to EDON Core*

