# Release Build Status - v1.0.0

**Date**: 2025-11-20

## Build Status

### ✅ Python Wheel
- **Status**: ✅ Built successfully
- **Location**: `sdk/python/dist/edon-0.1.0-py3-none-any.whl`
- **Action Required**: Move to `release/v1.0.0/` manually or re-run script

### ⏳ C++ SDK
- **Status**: ⏳ Not built (requires CMake)
- **Action Required**: 
  ```bash
  cd sdk/cpp
  mkdir build && cd build
  cmake ..
  cmake --build . --config Release
  # Then package manually
  ```

### ⏳ Docker Image
- **Status**: ⏳ Not built (Docker not running)
- **Action Required**: 
  ```bash
  # Start Docker Desktop, then:
  docker build -t edon-server:v1.0.0 .
  docker save edon-server:v1.0.0 > release/v1.0.0/edon-server-v1.0.0.docker
  ```

## Quick Build Commands

### Complete Release Build

**Windows (PowerShell)**:
```powershell
# 1. Python wheel (already built)
Move-Item sdk\python\dist\*.whl release\v1.0.0\ -Force

# 2. C++ SDK (if CMake available)
cd sdk\cpp
mkdir build; cd build
cmake ..
cmake --build . --config Release
# Package manually

# 3. Docker (if Docker running)
docker build -t edon-server:v1.0.0 .
docker save edon-server:v1.0.0 > ..\..\release\v1.0.0\edon-server-v1.0.0.docker
```

**Linux/macOS**:
```bash
# Use scripts/build_release.sh
chmod +x scripts/build_release.sh
./scripts/build_release.sh
```

## Files Created

- ✅ `scripts/build_release.sh` - Bash release script
- ✅ `scripts/build_release.ps1` - PowerShell release script
- ✅ `scripts/README_RELEASE.md` - Release build documentation
- ✅ `release/v1.0.0/RELEASE_NOTES.md` - Release notes

## Next Steps

1. **Move Python wheel** to release directory
2. **Build C++ SDK** (if needed for distribution)
3. **Build Docker image** (when Docker is available)
4. **Test installation** of Python wheel
5. **Create release tag** in git

## Testing Release

### Test Python Wheel

```bash
# Install wheel
pip install release/v1.0.0/edon-0.1.0-py3-none-any.whl[grpc]

# Test import
python -c "from edon import EdonClient; print('OK')"
```

### Test Docker Image

```bash
# Load image
docker load < release/v1.0.0/edon-server-v1.0.0.docker

# Run container
docker run -p 8001:8001 -p 50051:50051 edon-server:v1.0.0

# Test health
curl http://localhost:8001/health
```

