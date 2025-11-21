# Release v1.0.1 - Summary

**Date**: 2025-11-20

## ✅ Completed Tasks

### 1. Version Metadata Updates
- ✅ Updated `README.md` with version badge: "Current engine release: **v1.0.1**"
- ✅ Updated `sdk/cpp/CMakeLists.txt` to require CMake 3.16+ and set version to 1.0.1

### 2. Release Notes
- ✅ Created `release/v1.0.1/RELEASE_NOTES.md` with:
  - Overview and highlights
  - Artifacts (Docker, Python SDK, C++ SDK)
  - How to run instructions
  - Known limitations
  - **Git tag commands** (documented, not executed)

### 3. C++ SDK Documentation
- ✅ Updated `sdk/cpp/README.md` with:
  - Requirements (CMake 3.16+, C++17, gRPC/Protobuf)
  - Build instructions (Linux/WSL2, Windows, macOS)
  - Install/package instructions
  - Integration guide
  - Example usage code

### 4. OEM Onboarding Documentation
- ✅ Created `docs/OEM_ONBOARDING.md` with:
  - What EDON does
  - Deployment overview
  - Docker run instructions
  - REST API examples
  - Python SDK integration
  - C++ SDK integration
  - Versioning and compatibility
  - Support information

### 5. README.md Updates
- ✅ Added version badge at top
- ✅ Added "C++ SDK" section pointing to `sdk/cpp/README.md`
- ✅ Added "OEM Onboarding" section pointing to `docs/OEM_ONBOARDING.md`
- ✅ Updated version section to show v1.0.1

## 📋 Git Tag Commands (Documented)

The following commands are documented in `release/v1.0.1/RELEASE_NOTES.md` but **NOT executed**:

```bash
git status
git add .
git commit -m "Release v1.0.1"
git tag -a v1.0.1 -m "EDON CAV Engine v1.0.1"
git push origin main
git push origin v1.0.1
```

## 📁 Files Created/Modified

### Created
- `release/v1.0.1/RELEASE_NOTES.md`
- `docs/OEM_ONBOARDING.md`
- `sdk/cpp/README.md` (replaced existing)

### Modified
- `README.md` (version badge, C++ SDK section, OEM Onboarding section)
- `sdk/cpp/CMakeLists.txt` (CMake version requirement, project version)

## 🎯 Next Steps

1. **Review changes**: Check all modified files
2. **Test builds**: Verify C++ SDK builds correctly
3. **Run git commands**: Execute the documented git tag commands when ready
4. **Build artifacts**: Build Docker image and Python wheel for v1.0.1
5. **Update VERSION file**: Consider updating `VERSION` file to match (currently 0.1.0)

## 📝 Notes

- The `VERSION` file still shows `0.1.0` - this may be intentional (SDK version vs engine release version)
- All documentation now references v1.0.1 as the current release
- Git tag commands are documented but not executed as requested

