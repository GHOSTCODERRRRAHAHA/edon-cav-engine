# EDON CAV Engine – Release v1.0.1

**Date**: 2025-11-20

## Overview

EDON v1.0.1 is the first production-ready release of the EDON CAV Engine for OEMs and robotics partners.  
It provides a frozen v1 API contract, Docker image, Python SDK wheel, and example fake humanoid client.

## Highlights

- ✅ **Frozen OEM API contract (v1)**
  - REST: `/health`, `/oem/cav/batch`
  - gRPC: service using `edon.v1` package and `CavRequest` / `CavResponse` messages
  - Versioning policy: breaking changes will be released as `v2` and above.

- ✅ **Production Docker image**
  - Image: `edon-server:v1.0.1`
  - Exposes:
    - REST API on port `8000` inside the container
    - gRPC on port `50051` inside the container
  - Health endpoint: `GET /health`

- ✅ **Python SDK wheel**
  - Package: `edon`
  - Wheel: `edon-0.1.0-py3-none-any.whl`
  - Supports REST (and optionally gRPC if configured)

- ✅ **Fake humanoid client**
  - `clients/robot_example.py` simulates a simple control loop
  - Demonstrates using EDON CAV output to scale speed/torque and adjust safety

## Artifacts

- **Docker image**:
  - Built locally as: `edon-server:v1.0.1`
  - Saved as tarball (example): `release/edon-server-v1.0.1.tar`

- **Python SDK**:
  - Built from `sdk/python`
  - Wheel: `sdk/python/dist/edon-0.1.0-py3-none-any.whl`

- **C++ SDK**:
  - Library and headers under `sdk/cpp`
  - See `sdk/cpp/README.md` for build instructions

## How to run (local Docker)

```bash
docker run --rm \
  -p 8002:8000 \
  -p 50052:50051 \
  edon-server:v1.0.1
```

**REST health check**:
```bash
curl http://localhost:8002/health
```

## Known limitations

- v1 model trained on synthetic + WESAD-style data; OEMs should validate on their own sensor stack.
- C++ SDK is provided as an initial build; integration details may be customized per partner.
- All breaking interface changes will ship behind a new API version (e.g. `/v2/` and `edon.v2` package).

## Git tag commands

From the repo root:

```bash
git status
git add .
git commit -m "Release v1.0.1"
git tag -a v1.0.1 -m "EDON CAV Engine v1.0.1"
git push origin main
git push origin v1.0.1
```

