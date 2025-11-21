# Release Build Scripts

Scripts to build EDON v1.0.0 release packages.

## Usage

### Windows (PowerShell)

```powershell
.\scripts\build_release.ps1
```

### Linux/macOS (Bash)

```bash
chmod +x scripts/build_release.sh
./scripts/build_release.sh
```

## What Gets Built

1. **Python Wheel** (`edon-0.1.0-py3-none-any.whl`)
   - Built using `python -m build --wheel`
   - Includes REST and gRPC transport support
   - Install with: `pip install edon-0.1.0-py3-none-any.whl[grpc]`

2. **C++ SDK** (`edon-sdk-cpp-v1.0.0.tar.gz`)
   - Built using CMake
   - Includes headers, libraries, and examples
   - Platform-specific (Windows/Linux/macOS)

3. **Docker Image** (`edon-server-v1.0.0.docker`)
   - Built from `Dockerfile`
   - Includes REST API and gRPC server
   - Load with: `docker load < edon-server-v1.0.0.docker`

## Prerequisites

- Python 3.10+ with `build` and `wheel` packages
- CMake 3.15+ (for C++ SDK)
- Docker (for Docker image)
- tar/gzip (for packaging)

## Output

All files are placed in `release/v1.0.0/`:

```
release/v1.0.0/
├── edon-0.1.0-py3-none-any.whl
├── edon-sdk-cpp-v1.0.0.tar.gz
├── edon-server-v1.0.0.docker
└── RELEASE_NOTES.md
```

## Manual Build Steps

If the script fails, you can build manually:

### Python Wheel

```bash
cd sdk/python
python -m build --wheel
mv dist/*.whl ../../release/v1.0.0/
```

### C++ SDK

```bash
cd sdk/cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
cd ../..
tar -czf release/v1.0.0/edon-sdk-cpp-v1.0.0.tar.gz \
    -C sdk/cpp include lib examples CMakeLists.txt
```

### Docker Image

```bash
docker build -t edon-server:v1.0.0 .
docker save edon-server:v1.0.0 > release/v1.0.0/edon-server-v1.0.0.docker
```

## Troubleshooting

**Python build warnings**: The license format warning is non-critical. Update `pyproject.toml` to use `license = "MIT"` (string) instead of `license = {text = "MIT"}`.

**C++ build fails**: Ensure CMake and gRPC/Protobuf are installed. See `sdk/cpp/README.md`.

**Docker build fails**: Ensure Docker Desktop is running and Docker daemon is accessible.

