#!/bin/bash
# Sign EDON Docker image with SHA256 and GPG/cosign

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RELEASE_DIR="$SCRIPT_DIR/../release/EDON_v2.0.0_OEM_EVAL/bin"
TAR_FILE="edon-server-v2.0.0.tar"

cd "$RELEASE_DIR"

echo "=========================================="
echo "Signing EDON Docker Image"
echo "=========================================="
echo ""

# Check if tar exists
if [ ! -f "$TAR_FILE" ]; then
    echo "ERROR: Docker tar not found: $TAR_FILE"
    exit 1
fi

# Generate SHA256
echo "1. Generating SHA256 checksum..."
sha256sum "$TAR_FILE" > "${TAR_FILE}.sha256"
echo "✓ Created: ${TAR_FILE}.sha256"

# GPG signing (if GPG is available)
if command -v gpg &> /dev/null; then
    echo ""
    echo "2. Signing with GPG..."
    
    # Check for GPG key
    if [ -z "$GPG_KEY_ID" ]; then
        echo "WARNING: GPG_KEY_ID not set. Using default key."
        echo "Set GPG_KEY_ID environment variable to use specific key."
    fi
    
    if [ -n "$GPG_KEY_ID" ]; then
        gpg --detach-sign --armor --local-user "$GPG_KEY_ID" "${TAR_FILE}.sha256"
    else
        gpg --detach-sign --armor "${TAR_FILE}.sha256"
    fi
    
    echo "✓ Created: ${TAR_FILE}.sha256.asc"
    
    # Export public key if requested
    if [ "$EXPORT_GPG_KEY" = "true" ]; then
        echo ""
        echo "3. Exporting GPG public key..."
        if [ -n "$GPG_KEY_ID" ]; then
            gpg --export --armor "$GPG_KEY_ID" > EDON_SIGNING_KEY.asc
        else
            gpg --export --armor > EDON_SIGNING_KEY.asc
        fi
        echo "✓ Created: EDON_SIGNING_KEY.asc"
    fi
else
    echo "2. GPG not found, skipping GPG signing"
fi

# Cosign signing (if cosign is available)
if command -v cosign &> /dev/null; then
    echo ""
    echo "4. Signing with cosign..."
    
    # Check for cosign key
    if [ ! -f "cosign.key" ]; then
        echo "WARNING: cosign.key not found. Generating new key pair..."
        cosign generate-key-pair
        echo "✓ Generated: cosign.key and cosign.pub"
    fi
    
    # Load and sign image
    echo "Loading Docker image..."
    docker load -i "$TAR_FILE"
    
    echo "Signing image..."
    cosign sign --key cosign.key edon-server:v2.0.0-oem-eval
    
    echo "✓ Image signed with cosign"
    
    # Create attestation if build info exists
    if [ -f "build-attestation.json" ]; then
        echo "Creating attestation..."
        cosign attest --key cosign.key --predicate build-attestation.json edon-server:v2.0.0-oem-eval
        echo "✓ Attestation created"
    fi
else
    echo "4. cosign not found, skipping cosign signing"
    echo "   Install: https://github.com/sigstore/cosign"
fi

echo ""
echo "=========================================="
echo "✓ Signing complete!"
echo "=========================================="
echo ""
echo "Files created:"
ls -lh "${TAR_FILE}"* *.asc *.pub 2>/dev/null | grep -E "\.(sha256|asc|pub)$" || true
echo ""
echo "Verify signatures:"
echo "  sha256sum -c ${TAR_FILE}.sha256"
if [ -f "${TAR_FILE}.sha256.asc" ]; then
    echo "  gpg --verify ${TAR_FILE}.sha256.asc ${TAR_FILE}.sha256"
fi
if command -v cosign &> /dev/null && [ -f "cosign.pub" ]; then
    echo "  cosign verify --key cosign.pub edon-server:v2.0.0-oem-eval"
fi
echo ""

