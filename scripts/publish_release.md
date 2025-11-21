# Publishing EDON v1.0.1 Release

This guide covers publishing the EDON v1.0.1 release to registries and GitHub.

---

## 1. Create Release Bundle

### Windows (PowerShell)

```powershell
.\scripts\create_release_bundle.ps1
```

### Linux/macOS (Bash)

```bash
chmod +x scripts/create_release_bundle.sh
./scripts/create_release_bundle.sh
```

This creates:
- `EDON_v1.0.1_OEM_RELEASE/` - Bundle directory
- `EDON_v1.0.1_OEM_RELEASE.zip` - Zip archive

---

## 2. Push Docker Image to Registry

### GitHub Container Registry (GHCR)

**Prerequisites**:
- GitHub Personal Access Token with `write:packages` permission
- Docker logged in to GHCR

**Steps**:

```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag the image
docker tag edon-server:v1.0.1 ghcr.io/edonlabs/edon-server:v1.0.1

# Push the image
docker push ghcr.io/edonlabs/edon-server:v1.0.1

# Also tag and push as latest (optional)
docker tag edon-server:v1.0.1 ghcr.io/edonlabs/edon-server:latest
docker push ghcr.io/edonlabs/edon-server:latest
```

**Alternative: Docker Hub**

```bash
# Login to Docker Hub
docker login

# Tag the image
docker tag edon-server:v1.0.1 edonlabs/edon-server:v1.0.1

# Push the image
docker push edonlabs/edon-server:v1.0.1
```

**Pull from registry**:

```bash
# GHCR
docker pull ghcr.io/edonlabs/edon-server:v1.0.1

# Docker Hub
docker pull edonlabs/edon-server:v1.0.1
```

---

## 3. Create GitHub Release

### Using GitHub CLI (gh)

**Prerequisites**:
- Install GitHub CLI: https://cli.github.com/
- Authenticate: `gh auth login`

**Steps**:

```bash
# Create release with notes
gh release create v1.0.1 \
  --title "EDON CAV Engine v1.0.1" \
  --notes-file release/v1.0.1/RELEASE_NOTES.md \
  EDON_v1.0.1_OEM_RELEASE.zip \
  release/v1.0.1/edon-0.1.0-py3-none-any.whl \
  release/v1.0.1/edon-server-v1.0.1.docker
```

### Using GitHub Web UI

1. Go to your repository on GitHub
2. Click **Releases** → **Draft a new release**
3. **Tag**: `v1.0.1`
4. **Release title**: `EDON CAV Engine v1.0.1`
5. **Description**: Copy contents from `release/v1.0.1/RELEASE_NOTES.md`
6. **Attach binaries**:
   - `EDON_v1.0.1_OEM_RELEASE.zip` (full bundle)
   - `release/v1.0.1/edon-0.1.0-py3-none-any.whl` (Python SDK)
   - `release/v1.0.1/edon-server-v1.0.1.docker` (Docker image, if available)
7. Click **Publish release**

### Release Assets Checklist

- [ ] `EDON_v1.0.1_OEM_RELEASE.zip` - Full OEM bundle
- [ ] `edon-0.1.0-py3-none-any.whl` - Python SDK wheel
- [ ] `edon-server-v1.0.1.docker` or `.tar.gz` - Docker image (optional, if large)
- [ ] `edon-cpp-sdk-v1.0.1.zip` - C++ SDK (if available)

---

## 4. Update Documentation Links

After publishing, update any documentation that references:
- Docker image registry location
- Download URLs
- Installation instructions

---

## 5. Verify Release

### Test Docker Image

```bash
# Pull from registry
docker pull ghcr.io/edonlabs/edon-server:v1.0.1

# Run container
docker run --rm -p 8002:8000 -p 50052:50051 ghcr.io/edonlabs/edon-server:v1.0.1

# Test health
curl http://localhost:8002/health
```

### Test Python Wheel

```bash
# Download from GitHub release
pip install https://github.com/edonlabs/edon-cav-engine/releases/download/v1.0.1/edon-0.1.0-py3-none-any.whl[grpc]

# Test import
python -c "from edon import EdonClient; print('OK')"
```

---

## 6. Announce Release

- Update main README with registry information
- Notify OEM partners
- Update changelog/roadmap

---

## Quick Reference

**Docker Registry**:
- GHCR: `ghcr.io/edonlabs/edon-server:v1.0.1`
- Docker Hub: `edonlabs/edon-server:v1.0.1`

**GitHub Release**:
- URL: `https://github.com/edonlabs/edon-cav-engine/releases/tag/v1.0.1`

**Bundle**:
- `EDON_v1.0.1_OEM_RELEASE.zip`

