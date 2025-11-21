#!/bin/bash
# Build release package for EDON v1.0.0

set -e  # Exit on error

VERSION="v1.0.0"
RELEASE_DIR="release/${VERSION}"

echo "=========================================="
echo "Building EDON Release ${VERSION}"
echo "=========================================="

# Create release directory
echo "Creating release directory..."
mkdir -p "${RELEASE_DIR}"

# Build Python wheel
echo ""
echo "Building Python wheel..."
cd sdk/python

# Check if build tools are available
if ! python -c "import build" 2>/dev/null; then
    echo "Installing build tools..."
    pip install build wheel
fi

# Build wheel
python -m build --wheel

# Move wheel to release directory
mv dist/*.whl "../../${RELEASE_DIR}/"
echo "Python wheel built: $(ls ../../${RELEASE_DIR}/*.whl)"

# Build C++ SDK
echo ""
echo "Building C++ SDK..."
cd ../../sdk/cpp

# Clean previous build
if [ -d "build" ]; then
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
cmake --build . --config Release

# Create C++ SDK package
cd ../..
SDK_PACKAGE_DIR="${RELEASE_DIR}/edon-sdk-cpp-${VERSION}"
mkdir -p "${SDK_PACKAGE_DIR}"

# Copy headers
cp -r sdk/cpp/include "${SDK_PACKAGE_DIR}/"

# Copy libraries (platform-specific)
if [ -f "sdk/cpp/build/libedon_sdk.a" ]; then
    mkdir -p "${SDK_PACKAGE_DIR}/lib"
    cp sdk/cpp/build/libedon_sdk.a "${SDK_PACKAGE_DIR}/lib/"
    echo "Static library copied"
elif [ -f "sdk/cpp/build/libedon_sdk.so" ]; then
    mkdir -p "${SDK_PACKAGE_DIR}/lib"
    cp sdk/cpp/build/libedon_sdk.so "${SDK_PACKAGE_DIR}/lib/"
    echo "Shared library copied"
fi

# Copy example
if [ -f "sdk/cpp/build/bin/edon_robot_example" ] || [ -f "sdk/cpp/build/edon_robot_example" ]; then
    mkdir -p "${SDK_PACKAGE_DIR}/bin"
    if [ -f "sdk/cpp/build/bin/edon_robot_example" ]; then
        cp sdk/cpp/build/bin/edon_robot_example "${SDK_PACKAGE_DIR}/bin/"
    else
        cp sdk/cpp/build/edon_robot_example "${SDK_PACKAGE_DIR}/bin/"
    fi
    echo "Example binary copied"
fi

# Copy CMake files
cp sdk/cpp/CMakeLists.txt "${SDK_PACKAGE_DIR}/"
cp -r sdk/cpp/examples "${SDK_PACKAGE_DIR}/" 2>/dev/null || true

# Create tarball
cd "${RELEASE_DIR}"
tar -czf "edon-sdk-cpp-${VERSION}.tar.gz" "edon-sdk-cpp-${VERSION}"
rm -rf "edon-sdk-cpp-${VERSION}"
echo "C++ SDK package created: edon-sdk-cpp-${VERSION}.tar.gz"

cd ../..

# Build Docker image
echo ""
echo "Building Docker image..."
docker build -t edon-server:${VERSION} .

# Save Docker image
echo "Saving Docker image..."
docker save edon-server:${VERSION} | gzip > "${RELEASE_DIR}/edon-server-${VERSION}.docker.tar.gz"

# Also save without compression for compatibility
docker save edon-server:${VERSION} > "${RELEASE_DIR}/edon-server-${VERSION}.docker" 2>/dev/null || {
    echo "Note: Large Docker image saved as .tar.gz (use: docker load < edon-server-${VERSION}.docker.tar.gz)"
}

echo ""
echo "=========================================="
echo "Release build complete!"
echo "=========================================="
echo "Release files in: ${RELEASE_DIR}/"
ls -lh "${RELEASE_DIR}/"
echo ""
echo "To load Docker image:"
echo "  docker load < ${RELEASE_DIR}/edon-server-${VERSION}.docker.tar.gz"
echo ""
echo "To install Python wheel:"
echo "  pip install ${RELEASE_DIR}/*.whl"
echo ""

