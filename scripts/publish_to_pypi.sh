#!/bin/bash
# Publish EDON Python SDK to PyPI

set -e

echo "=========================================="
echo "Publishing EDON Python SDK to PyPI"
echo "=========================================="

# Check if in correct directory
if [ ! -f "sdk/python/pyproject.toml" ]; then
    echo "ERROR: Must run from repo root or sdk/python directory"
    exit 1
fi

cd sdk/python

# Check if build tools are installed
if ! python -c "import build" 2>/dev/null; then
    echo "Installing build tools..."
    pip install build twine
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Build wheel and source distribution
echo "Building wheel and source distribution..."
python -m build

# Check what was built
echo ""
echo "Built artifacts:"
ls -lh dist/

# Verify with twine
echo ""
echo "Verifying package..."
twine check dist/*

# Ask for confirmation
echo ""
echo "=========================================="
echo "Ready to upload to PyPI"
echo "=========================================="
echo "Files to upload:"
ls -lh dist/*
echo ""
read -p "Upload to PyPI? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Upload cancelled."
    exit 0
fi

# Upload to PyPI
echo ""
echo "Uploading to PyPI..."
twine upload dist/*

echo ""
echo "=========================================="
echo "✅ Published to PyPI!"
echo "=========================================="
echo "Install with: pip install edon"
echo "Or with gRPC: pip install edon[grpc]"

