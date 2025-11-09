# EDON - The Adaptive Intelligence OS
**EDON** is a full-stack, real-time **Adaptive Intelligence OS** for devices that interact with people and the physical world. It fuses **physiology** (EDA, BVP, TEMP, ACC) with **environmental context** (time, air, space) to drive **emotionally-aware** and **situation-adaptive** behavior.
---
## What EDON Is
- **CAV Engine** - Context-Aware Vector scoring across states: *balanced*, *focus*, *restorative*  
- **Adaptive Memory** - Learns sensitivity, context, and environment over time  
- **APIs & SDK** - FastAPI endpoints + Python client for OEM integrations  
- **Realtime UI** - Streamlit dashboard (Tesla/Neuralink-style console)
---
## Why It Matters
- Converts raw biosignals into an **intelligence layer** deployable on edge devices and robots  
- Enables **emotionally-aware control** and **environment-adaptive UX**  
- Provides a **verifiable, versioned** platform for OEM partners
---
## System Layout
Sensors (EDA, BVP, ACC, TEMP) + Env → Feature Windows → CAV Engine (LGBM v3.2)
                                           ↓
                                     Adaptive Memory
                                           ↓
                             FastAPI (inference + memory + telemetry)
                                           ↓
                       SDK / Dashboard / OEM Integrations / Edge Apps
# Windows (PowerShell)
.\setup_oem_sdk.ps1
# Backend → http://localhost:8000  |  Docs → /docs
# Dashboard → http://localhost:8501
## Core Endpoints
- `POST /cav` - single-window inference (arrays: EDA, TEMP, BVP, ACC_x/y/z + env fields)  
- `POST /oem/cav/batch` - batch windows  
- `GET /memory/summary`, `POST /memory/clear` - adaptive memory  
- `GET /telemetry`, `GET /health` - service observability
---
## Versioning & Provenance
- Tagged releases (e.g., `v3.2`)  
- SHA-256 `HASHES.txt` in packaged SDKs  
- Reproducible setup via `setup_oem_sdk.ps1`
---
## Roadmap (Preview)
- v4.0: multi-modal OS hooks (voice/context I/O), policy adapters, edge profiles  
- v4.x: hardware abstraction for wearables/robotics, multi-agent coordination
---
**Developed by Charlie Biggins**  
*EDON - The Adaptive Intelligence OS powering emotional and environmental AI.*
