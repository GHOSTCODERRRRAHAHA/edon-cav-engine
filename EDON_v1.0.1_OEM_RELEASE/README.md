# EDON CAV Engine v1.0.1 - OEM Release Bundle

**Release Date**: 2025-11-20

## Contents

This bundle contains all artifacts for EDON CAV Engine v1.0.1:

- **docker/**: Docker image tarball
- **sdk/python/**: Python SDK wheel
- **sdk/cpp/**: C++ SDK archive
- **docs/**: OEM documentation

## Quick Start

1. **Load Docker image**:
   `docker load < docker/edon-server-v1.0.1.tar`

2. **Install Python SDK**:
   `pip install sdk/python/edon-0.1.0-py3-none-any.whl[grpc]`

3. **Extract C++ SDK**:
   `unzip sdk/cpp/edon-cpp-sdk-v1.0.1.zip`

4. **Read documentation**:
   - Start with `docs/OEM_ONBOARDING.md`
   - See `docs/OEM_API_CONTRACT.md` for API details
   - See `docs/RELEASE_NOTES.md` for release information

## Support

For integration questions, see the documentation or contact the EDON team.
