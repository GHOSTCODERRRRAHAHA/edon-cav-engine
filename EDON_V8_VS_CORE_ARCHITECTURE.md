# EDON v8 vs EDON Core: Research vs Product

## Overview

**EDON v8** = Internal validation/prototype of the EDON nervous-system architecture  
**EDON Core** = Productized version for portable deployment across robots and OEMs

---

## Architecture Relationship

```
┌─────────────────────────────────────────────────────────────┐
│              EDON NERVOUS-SYSTEM ARCHITECTURE               │
│         (The Intelligence Layer Concept)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│   EDON v8        │          │   EDON Core      │
│   (Research)      │          │   (Product)      │
├──────────────────┤          ├──────────────────┤
│ • Inline         │          │ • Portable       │
│ • Experimental   │          │ • API/Service     │
│ • Full control   │          │ • OEM-ready      │
│ • Validation     │          │ • Deployable     │
└──────────────────┘          └──────────────────┘
```

---

## EDON v8: Internal Validation Platform

### Purpose
- **Research & Development**: Validate EDON nervous-system concepts
- **Experimental Control**: Everything inline for maximum flexibility
- **Architecture Validation**: Test temporal memory, early-warning features, fail-risk prediction
- **Proof of Concept**: Demonstrate 97.5% intervention reduction

### Characteristics
- **Inline Implementation**: All components in one codebase
- **Tight Integration**: Direct function calls, no API boundaries
- **Full Control**: Can modify any component during research
- **Research-Focused**: Optimized for experimentation, not deployment

### Key Components (v8)
1. **Temporal Memory**: 8-frame stacking (248 dims)
2. **Early-Warning Features**: Variance, oscillation, density
3. **Fail-Risk Model**: Standalone neural network
4. **Strategy Policy**: Learned PPO policy
5. **Environment Wrapper**: Direct integration

### Results (v8)
- **Interventions**: 1.00/episode (97.5% reduction)
- **Stability**: 0.0215 (within ±5%)
- **Status**: ✅ Validates EDON nervous-system architecture

---

## EDON Core: Productized Intelligence

### Purpose
- **Portable Deployment**: Runs as service/API across robots and OEMs
- **OEM Integration**: Standardized interface for partners
- **Production-Ready**: Optimized for deployment, not experimentation
- **Cross-Platform**: Works with different robot platforms

### Characteristics
- **Service-Based**: REST/gRPC API endpoints
- **Portable**: Docker containers, SDKs (Python, C++)
- **Standardized**: Consistent interface across OEMs
- **Deployment-Focused**: Optimized for production use

### Key Components (EDON Core)
1. **CAV Engine**: Context-Aware Vector computation
2. **Adaptive Memory**: 24-hour rolling context
3. **State Detection**: `restorative`, `balanced`, `focus`, `overload`
4. **API Layer**: REST (`/cav`, `/oem/cav/batch`) and gRPC
5. **SDKs**: Python and C++ client libraries

### Deployment
- **Docker**: `edon-server:v1.0.1`
- **REST API**: Port 8000
- **gRPC**: Port 50051
- **SDKs**: `pip install edon` or C++ library

---

## How v8 Validates EDON Core Concepts

### 1. Temporal Memory → Adaptive Memory

**v8 (Research)**:
- 8-frame stacking (248 dims)
- Direct observation history buffers
- Inline computation

**EDON Core (Product)**:
- 24-hour rolling context
- Adaptive memory engine
- Service-based storage

**Validation**: v8 proves temporal context is critical (97.5% improvement)

### 2. Early-Warning Features → State Detection

**v8 (Research)**:
- Rolling variance (trend detection)
- Oscillation energy (wobble detection)
- Near-fail density (persistent danger)

**EDON Core (Product)**:
- State detection (`restorative`, `balanced`, `focus`, `overload`)
- `p_stress`, `p_chaos` probabilities
- CAV score computation

**Validation**: v8 proves predictive features enable preventive control

### 3. Fail-Risk Prediction → Risk Assessment

**v8 (Research)**:
- Standalone fail-risk model (15 → 1)
- Predicts failure 0.5-1.0s ahead
- Used for early-warning features

**EDON Core (Product)**:
- Risk assessment via CAV engine
- `p_stress` probability
- State-based risk signals

**Validation**: v8 proves failure prediction enables intervention prevention

### 4. Layered Control → Adaptive Modulation

**v8 (Research)**:
- Strategy selection (4 discrete options)
- Continuous modulations (gain_scale, compliance, bias)
- Modulates baseline controller

**EDON Core (Product)**:
- Control scales (speed, torque, safety margins)
- Adaptive gain based on state
- OEM-configurable modulation

**Validation**: v8 proves layered control maintains stability while adapting

---

## Why v8 Doesn't Use EDON Core

### Design Decision

**v8 is the research platform** that validates concepts **before** they're productized:

1. **Experimental Control**: v8 needs full control to test new ideas
2. **Rapid Iteration**: Inline code allows fast experimentation
3. **Concept Validation**: Proves the architecture works
4. **Independence**: Validates that the concepts work standalone

### The Flow

```
Research Phase (v8):
  └─> Validate concepts inline
      └─> Prove 97.5% improvement
          └─> Productize concepts
              └─> EDON Core (portable service)
```

**v8 validates the concepts, EDON Core productizes them.**

---

## What v8 Proves for EDON Core

### 1. Temporal Context is Critical ✅

**v8 Evidence**: 8-frame stacking enables 97.5% intervention reduction

**EDON Core Implication**: Adaptive memory (24-hour context) is valuable for production

### 2. Predictive Features Enable Prevention ✅

**v8 Evidence**: Early-warning features detect problems before they become critical

**EDON Core Implication**: State detection and risk assessment enable preventive control

### 3. Failure Prediction Works ✅

**v8 Evidence**: Fail-risk model predicts failures 0.5-1.0s ahead

**EDON Core Implication**: CAV engine's risk assessment can prevent interventions

### 4. Layered Control Maintains Stability ✅

**v8 Evidence**: Learned modulations on baseline maintain stability while adapting

**EDON Core Implication**: Adaptive modulation can improve performance without breaking stability

---

## Integration Path: v8 → EDON Core

### Current State

**v8 (Research)**:
- ✅ Validates concepts inline
- ✅ Achieves 97.5% improvement
- ✅ Proves architecture works

**EDON Core (Product)**:
- ✅ Portable service/API
- ✅ OEM-ready deployment
- ⏳ Can incorporate v8 learnings

### Future Integration

**Option 1: v8 Concepts → EDON Core Features**
- Add temporal memory to EDON Core adaptive memory
- Add early-warning features to CAV engine
- Add fail-risk prediction to risk assessment

**Option 2: EDON Core → v8 Validation**
- Use EDON Core API in v8 for production-like testing
- Validate EDON Core features in v8 research platform
- Ensure productized version maintains v8 performance

**Option 3: Hybrid Approach**
- v8 continues as research platform
- EDON Core productizes validated concepts
- Both evolve independently but share learnings

---

## Key Insights

### 1. v8 is the Research Platform

**Purpose**: Validate EDON nervous-system architecture concepts
- Temporal memory ✅
- Early-warning features ✅
- Fail-risk prediction ✅
- Layered control ✅

**Result**: 97.5% intervention reduction proves the architecture works

### 2. EDON Core is the Product

**Purpose**: Portable, deployable intelligence for OEMs
- Service-based architecture
- Standardized APIs
- Cross-platform support
- Production-ready

**Status**: Ready for OEM deployment

### 3. v8 Validates EDON Core Concepts

**What v8 proves**:
- Temporal context is critical (8-frame stacking)
- Predictive features enable prevention (early-warning)
- Failure prediction works (fail-risk model)
- Layered control maintains stability (modulations)

**What this means for EDON Core**:
- Adaptive memory is valuable (validated by v8)
- State detection enables prevention (validated by v8)
- Risk assessment prevents interventions (validated by v8)
- Adaptive modulation improves performance (validated by v8)

### 4. They Serve Different Purposes

| Aspect | v8 (Research) | EDON Core (Product) |
|--------|---------------|---------------------|
| **Purpose** | Validate concepts | Deploy to OEMs |
| **Architecture** | Inline | Service/API |
| **Control** | Full experimental | Standardized |
| **Focus** | Research | Production |
| **Deployment** | Single codebase | Portable service |

---

## Conclusion

**v8 and EDON Core are complementary:**

- **v8** = Internal validation platform that proves EDON nervous-system architecture works (97.5% improvement)
- **EDON Core** = Productized version that makes this intelligence portable across robots and OEMs

**v8 doesn't use EDON Core because:**
- v8 is the research platform that validates concepts
- EDON Core is the product that gets deployed
- v8 proves the concepts work, EDON Core makes them portable

**The relationship:**
```
v8 (Research) → Validates Concepts → EDON Core (Product) → Deploys to OEMs
```

**v8's 97.5% improvement validates that the EDON nervous-system architecture works**, which gives confidence that EDON Core (the productized version) will provide value to OEMs.

---

*Last Updated: After understanding v8 as research platform and EDON Core as product*

