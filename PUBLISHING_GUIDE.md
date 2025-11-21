# EDON v1.0.1 Publishing Guide

Complete guide for publishing EDON v1.0.1 as a vendor release.

---

## 📦 Step 1: Create Release Bundle

### Run Bundle Script

**Windows**:
```powershell
.\scripts\create_release_bundle.ps1
```

**Linux/macOS**:
```bash
chmod +x scripts/create_release_bundle.sh
./scripts/create_release_bundle.sh
```

This creates:
```
EDON_v1.0.1_OEM_RELEASE/
├── docker/
│   └── edon-server-v1.0.1.tar (or README.txt if not built)
├── sdk/
│   ├── python/
│   │   └── edon-0.1.0-py3-none-any.whl
│   └── cpp/
│       └── edon-cpp-sdk-v1.0.1.zip (or README.txt if not built)
├── docs/
│   ├── OEM_ONBOARDING.md
│   ├── OEM_API_CONTRACT.md
│   └── RELEASE_NOTES.md
└── README.md

EDON_v1.0.1_OEM_RELEASE.zip
```

---

## 🐳 Step 2: Push Docker Image to Registry

### Option A: GitHub Container Registry (GHCR)

**Prerequisites**:
- GitHub Personal Access Token with `write:packages` permission
- Repository: `edonlabs/edon-cav-engine` (or your org/repo)

**Commands**:

```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Tag the image
docker tag edon-server:v1.0.1 ghcr.io/edonlabs/edon-server:v1.0.1

# Push the image
docker push ghcr.io/edonlabs/edon-server:v1.0.1

# Optional: Also tag as latest
docker tag edon-server:v1.0.1 ghcr.io/edonlabs/edon-server:latest
docker push ghcr.io/edonlabs/edon-server:latest
```

**Pull from GHCR**:
```bash
docker pull ghcr.io/edonlabs/edon-server:v1.0.1
docker run --rm -p 8002:8000 -p 50052:50051 ghcr.io/edonlabs/edon-server:v1.0.1
```

### Option B: Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag the image
docker tag edon-server:v1.0.1 edonlabs/edon-server:v1.0.1

# Push the image
docker push edonlabs/edon-server:v1.0.1
```

**Pull from Docker Hub**:
```bash
docker pull edonlabs/edon-server:v1.0.1
```

---

## 🚀 Step 3: Create GitHub Release

### Using GitHub CLI (Recommended)

**Prerequisites**:
- Install: https://cli.github.com/
- Authenticate: `gh auth login`

**Create Release**:

```bash
# Create release with all assets
gh release create v1.0.1 \
  --title "EDON CAV Engine v1.0.1" \
  --notes-file release/v1.0.1/RELEASE_NOTES.md \
  EDON_v1.0.1_OEM_RELEASE.zip \
  release/v1.0.1/edon-0.1.0-py3-none-any.whl

# If Docker image is available
gh release upload v1.0.1 release/v1.0.1/edon-server-v1.0.1.docker

# If C++ SDK is available
gh release upload v1.0.1 release/v1.0.1/edon-cpp-sdk-v1.0.1.zip
```

### Using GitHub Web UI

1. Go to: `https://github.com/YOUR_ORG/edon-cav-engine/releases/new`
2. **Tag**: `v1.0.1` (create new tag)
3. **Release title**: `EDON CAV Engine v1.0.1`
4. **Description**: Copy from `release/v1.0.1/RELEASE_NOTES.md`
5. **Attach files**:
   - `EDON_v1.0.1_OEM_RELEASE.zip` (full bundle)
   - `release/v1.0.1/edon-0.1.0-py3-none-any.whl` (Python SDK)
   - `release/v1.0.1/edon-server-v1.0.1.docker` (Docker image, if available)
   - `release/v1.0.1/edon-cpp-sdk-v1.0.1.zip` (C++ SDK, if available)
6. Click **Publish release**

---

## ✅ Release Checklist

### Pre-Release
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version numbers correct
- [ ] Release notes complete

### Build Artifacts
- [ ] Python wheel built (`edon-0.1.0-py3-none-any.whl`)
- [ ] Docker image built (`edon-server:v1.0.1`)
- [ ] C++ SDK packaged (if applicable)
- [ ] Release bundle created (`EDON_v1.0.1_OEM_RELEASE.zip`)

### Publishing
- [ ] Docker image pushed to registry
- [ ] GitHub release created
- [ ] All assets uploaded
- [ ] Release notes published

### Post-Release
- [ ] Verify Docker image pulls correctly
- [ ] Verify Python wheel installs correctly
- [ ] Update main README with registry info
- [ ] Notify OEM partners
- [ ] Update changelog

---

## 📋 Release Assets Summary

| Asset | Location | Size (approx) |
|-------|----------|---------------|
| **Full Bundle** | `EDON_v1.0.1_OEM_RELEASE.zip` | ~50-100 MB |
| **Python SDK** | `edon-0.1.0-py3-none-any.whl` | ~50 KB |
| **Docker Image** | `edon-server-v1.0.1.docker` | ~500 MB - 1 GB |
| **C++ SDK** | `edon-cpp-sdk-v1.0.1.zip` | ~5-10 MB |

---

## 🔗 Quick Links

**Docker Registry**:
- GHCR: `ghcr.io/edonlabs/edon-server:v1.0.1`
- Docker Hub: `edonlabs/edon-server:v1.0.1`

**GitHub Release**:
- URL: `https://github.com/YOUR_ORG/edon-cav-engine/releases/tag/v1.0.1`

**Documentation**:
- OEM Onboarding: `docs/OEM_ONBOARDING.md`
- API Contract: `docs/OEM_API_CONTRACT.md`
- Release Notes: `release/v1.0.1/RELEASE_NOTES.md`

---

## 🧪 Verification

### Test Docker Image

```bash
# Pull from registry
docker pull ghcr.io/edonlabs/edon-server:v1.0.1

# Run
docker run --rm -p 8002:8000 -p 50052:50051 ghcr.io/edonlabs/edon-server:v1.0.1

# Test
curl http://localhost:8002/health
```

### Test Python Wheel

```bash
# Install from GitHub release
pip install https://github.com/YOUR_ORG/edon-cav-engine/releases/download/v1.0.1/edon-0.1.0-py3-none-any.whl[grpc]

# Test
python -c "from edon import EdonClient; print('OK')"
```

---

## 📝 Notes

- Docker images are large; consider using `.tar.gz` compression for GitHub releases
- Python wheel is small and can be uploaded directly
- Full bundle includes all documentation for offline use
- C++ SDK is optional but recommended for robotics partners

---

**See also**: `scripts/publish_release.md` for detailed commands

