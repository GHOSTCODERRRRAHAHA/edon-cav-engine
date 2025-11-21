#!/bin/bash
# Create GitHub Release for EDON v1.0.1

set -e

VERSION="v1.0.1"
REPO="edonlabs/edon-cav-engine"  # Update with your org/repo

echo "=========================================="
echo "Creating GitHub Release ${VERSION}"
echo "=========================================="

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "ERROR: GitHub CLI (gh) is not installed."
    echo "Install from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "ERROR: Not authenticated with GitHub CLI."
    echo "Run: gh auth login"
    exit 1
fi

# Check if files exist
ZIP_FILE="EDON_v1.0.1_OEM_RELEASE.zip"
WHEEL_FILE="release/${VERSION}/edon-0.1.0-py3-none-any.whl"
DOCKER_FILE="release/${VERSION}/edon-server-${VERSION}.docker"
RELEASE_NOTES="release/${VERSION}/RELEASE_NOTES.md"

FILES_TO_UPLOAD=()

if [ -f "$ZIP_FILE" ]; then
    echo "✅ Found: $ZIP_FILE"
    FILES_TO_UPLOAD+=("$ZIP_FILE")
else
    echo "⚠️  Missing: $ZIP_FILE"
fi

if [ -f "$WHEEL_FILE" ]; then
    echo "✅ Found: $WHEEL_FILE"
    FILES_TO_UPLOAD+=("$WHEEL_FILE")
else
    echo "⚠️  Missing: $WHEEL_FILE"
    # Try alternative location
    if [ -f "sdk/python/dist/edon-0.1.0-py3-none-any.whl" ]; then
        echo "✅ Found: sdk/python/dist/edon-0.1.0-py3-none-any.whl"
        FILES_TO_UPLOAD+=("sdk/python/dist/edon-0.1.0-py3-none-any.whl")
    fi
fi

if [ -f "$DOCKER_FILE" ]; then
    echo "✅ Found: $DOCKER_FILE"
    FILES_TO_UPLOAD+=("$DOCKER_FILE")
else
    echo "⚠️  Missing: $DOCKER_FILE (optional)"
fi

if [ ! -f "$RELEASE_NOTES" ]; then
    echo "ERROR: Release notes not found: $RELEASE_NOTES"
    exit 1
fi

echo ""
echo "Creating release with ${#FILES_TO_UPLOAD[@]} asset(s)..."

# Create release
gh release create "${VERSION}" \
    --title "EDON CAV Engine ${VERSION}" \
    --notes-file "$RELEASE_NOTES" \
    "${FILES_TO_UPLOAD[@]}"

echo ""
echo "=========================================="
echo "✅ GitHub Release created!"
echo "=========================================="
echo "Release URL: https://github.com/${REPO}/releases/tag/${VERSION}"
echo ""
echo "To view: gh release view ${VERSION}"

