#!/bin/bash
# Create EDON v1.0.1 OEM Release Bundle

set -e

VERSION="v1.0.1"
BUNDLE_NAME="EDON_${VERSION}_OEM_RELEASE"
BUNDLE_DIR="${BUNDLE_NAME}"

echo "=========================================="
echo "Creating EDON ${VERSION} OEM Release Bundle"
echo "=========================================="

# Clean previous bundle
if [ -d "${BUNDLE_DIR}" ]; then
    echo "Removing previous bundle..."
    rm -rf "${BUNDLE_DIR}"
fi

if [ -f "${BUNDLE_NAME}.zip" ]; then
    echo "Removing previous zip..."
    rm -f "${BUNDLE_NAME}.zip"
fi

# Create bundle structure
echo "Creating bundle structure..."
mkdir -p "${BUNDLE_DIR}/docker"
mkdir -p "${BUNDLE_DIR}/sdk/python"
mkdir -p "${BUNDLE_DIR}/sdk/cpp"
mkdir -p "${BUNDLE_DIR}/docs"

# Copy Docker image (if exists)
if [ -f "release/${VERSION}/edon-server-${VERSION}.docker" ]; then
    echo "Copying Docker image..."
    cp "release/${VERSION}/edon-server-${VERSION}.docker" "${BUNDLE_DIR}/docker/edon-server-${VERSION}.tar"
elif [ -f "release/${VERSION}/edon-server-${VERSION}.docker.tar.gz" ]; then
    echo "Copying Docker image (compressed)..."
    cp "release/${VERSION}/edon-server-${VERSION}.docker.tar.gz" "${BUNDLE_DIR}/docker/edon-server-${VERSION}.tar.gz"
else
    echo "Warning: Docker image not found. Creating placeholder..."
    echo "Docker image: edon-server:${VERSION}" > "${BUNDLE_DIR}/docker/README.txt"
    echo "Build with: docker build -t edon-server:${VERSION} ." >> "${BUNDLE_DIR}/docker/README.txt"
    echo "Save with: docker save edon-server:${VERSION} > edon-server-${VERSION}.tar" >> "${BUNDLE_DIR}/docker/README.txt"
fi

# Copy Python SDK wheel
if [ -f "release/${VERSION}/edon-0.1.0-py3-none-any.whl" ]; then
    echo "Copying Python SDK wheel..."
    cp "release/${VERSION}/edon-0.1.0-py3-none-any.whl" "${BUNDLE_DIR}/sdk/python/"
elif [ -f "sdk/python/dist/edon-0.1.0-py3-none-any.whl" ]; then
    echo "Copying Python SDK wheel from dist..."
    cp "sdk/python/dist/edon-0.1.0-py3-none-any.whl" "${BUNDLE_DIR}/sdk/python/"
else
    echo "Warning: Python wheel not found. Creating placeholder..."
    echo "Python SDK wheel: edon-0.1.0-py3-none-any.whl" > "${BUNDLE_DIR}/sdk/python/README.txt"
    echo "Build with: cd sdk/python && python -m build --wheel" >> "${BUNDLE_DIR}/sdk/python/README.txt"
fi

# Copy C++ SDK (if exists)
if [ -f "release/${VERSION}/edon-cpp-sdk-${VERSION}.zip" ]; then
    echo "Copying C++ SDK..."
    cp "release/${VERSION}/edon-cpp-sdk-${VERSION}.zip" "${BUNDLE_DIR}/sdk/cpp/"
elif [ -f "release/${VERSION}/edon-sdk-cpp-${VERSION}.tar.gz" ]; then
    echo "Copying C++ SDK (tar.gz)..."
    cp "release/${VERSION}/edon-sdk-cpp-${VERSION}.tar.gz" "${BUNDLE_DIR}/sdk/cpp/"
else
    echo "Warning: C++ SDK not found. Creating placeholder..."
    echo "C++ SDK: edon-cpp-sdk-${VERSION}.zip" > "${BUNDLE_DIR}/sdk/cpp/README.txt"
    echo "Build with: cd sdk/cpp && mkdir build && cd build && cmake .. && cmake --build ." >> "${BUNDLE_DIR}/sdk/cpp/README.txt"
fi

# Copy documentation
echo "Copying documentation..."
cp "docs/OEM_ONBOARDING.md" "${BUNDLE_DIR}/docs/" 2>/dev/null || echo "Warning: OEM_ONBOARDING.md not found"
cp "docs/OEM_API_CONTRACT.md" "${BUNDLE_DIR}/docs/" 2>/dev/null || echo "Warning: OEM_API_CONTRACT.md not found"
cp "release/${VERSION}/RELEASE_NOTES.md" "${BUNDLE_DIR}/docs/" 2>/dev/null || echo "Warning: RELEASE_NOTES.md not found"

# Create bundle README
cat > "${BUNDLE_DIR}/README.md" << EOF
# EDON CAV Engine ${VERSION} - OEM Release Bundle

**Release Date**: 2025-11-20

## Contents

This bundle contains all artifacts for EDON CAV Engine ${VERSION}:

- **docker/**: Docker image tarball
- **sdk/python/**: Python SDK wheel
- **sdk/cpp/**: C++ SDK archive
- **docs/**: OEM documentation

## Quick Start

1. **Load Docker image**:
   \`\`\`bash
   docker load < docker/edon-server-${VERSION}.tar
   \`\`\`

2. **Install Python SDK**:
   \`\`\`bash
   pip install sdk/python/edon-0.1.0-py3-none-any.whl[grpc]
   \`\`\`

3. **Extract C++ SDK**:
   \`\`\`bash
   unzip sdk/cpp/edon-cpp-sdk-${VERSION}.zip
   \`\`\`

4. **Read documentation**:
   - Start with \`docs/OEM_ONBOARDING.md\`
   - See \`docs/OEM_API_CONTRACT.md\` for API details
   - See \`docs/RELEASE_NOTES.md\` for release information

## Support

For integration questions, see the documentation or contact the EDON team.
EOF

# Create zip archive
echo ""
echo "Creating zip archive..."
zip -r "${BUNDLE_NAME}.zip" "${BUNDLE_DIR}" -x "*.git*" "*.DS_Store" "*__pycache__*"

echo ""
echo "=========================================="
echo "Release bundle created!"
echo "=========================================="
echo "Bundle directory: ${BUNDLE_DIR}/"
echo "Zip archive: ${BUNDLE_NAME}.zip"
echo ""
echo "Contents:"
ls -lh "${BUNDLE_DIR}"/* "${BUNDLE_DIR}"/*/* 2>/dev/null | head -20
echo ""
echo "Bundle size:"
du -sh "${BUNDLE_DIR}"
du -sh "${BUNDLE_NAME}.zip"

