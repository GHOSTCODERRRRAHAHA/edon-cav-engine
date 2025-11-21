#!/bin/bash
# Publish EDON Python SDK to private PyPI

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
SDK_DIR="$PROJECT_ROOT/sdk/python"

echo "=========================================="
echo "Publishing EDON Python SDK to PyPI"
echo "=========================================="
echo ""

# Check prerequisites
if ! command -v twine &> /dev/null; then
    echo "ERROR: twine not installed. Install with: pip install twine"
    exit 1
fi

# Build wheel
echo "1. Building wheel..."
cd "$SDK_DIR"
python -m build --wheel

# Check if wheel was created
WHEEL_FILE=$(ls -t dist/edon-*.whl | head -1)
if [ -z "$WHEEL_FILE" ]; then
    echo "ERROR: Wheel file not found"
    exit 1
fi

echo "✓ Wheel built: $WHEEL_FILE"
echo ""

# Check for PyPI configuration
if [ -z "$TWINE_REPOSITORY_URL" ] && [ ! -f ~/.pypirc ]; then
    echo "WARNING: PyPI configuration not found."
    echo "Set environment variables:"
    echo "  export TWINE_USERNAME=your-username"
    echo "  export TWINE_PASSWORD=your-token"
    echo "  export TWINE_REPOSITORY_URL=https://your-pypi-server.com/simple"
    echo ""
    echo "Or create ~/.pypirc (see PUBLISH_PYPI.md)"
    exit 1
fi

# Upload
echo "2. Uploading to PyPI..."
twine upload "$WHEEL_FILE"

echo ""
echo "=========================================="
echo "✓ SDK published successfully!"
echo "=========================================="
echo ""
echo "Install with:"
echo "  pip install --index-url \$TWINE_REPOSITORY_URL edon==2.0.0"
echo ""

