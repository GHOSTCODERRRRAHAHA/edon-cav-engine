#!/usr/bin/env python3
"""Test EDON mode switching."""

import os
import sys

# Test v1 mode
print("=" * 70)
print("Testing EDON Mode Switching")
print("=" * 70)

print("\n[Test 1] v1 Mode (default)")
os.environ["EDON_MODE"] = "v1"
# Clear any cached imports
if 'app.engine_state' in sys.modules:
    del sys.modules['app.engine_state']
if 'app.main' in sys.modules:
    del sys.modules['app.main']
if 'app.routes.telemetry' in sys.modules:
    del sys.modules['app.routes.telemetry']

from app import engine_state
print(f"   [OK] Mode: {engine_state.EDON_MODE}")
assert engine_state.EDON_MODE == "v1", "Should be v1 mode"

print("\n[Test 2] v2 Mode")
os.environ["EDON_MODE"] = "v2"
# Clear cached imports
if 'app.engine_state' in sys.modules:
    del sys.modules['app.engine_state']

# Re-import to get new mode
import importlib
importlib.reload(engine_state)
print(f"   [OK] Mode: {engine_state.EDON_MODE}")
assert engine_state.EDON_MODE == "v2", "Should be v2 mode"

print("\n[Test 3] Invalid Mode (should default to v1)")
os.environ["EDON_MODE"] = "invalid"
# Clear cached imports
if 'app.engine_state' in sys.modules:
    del sys.modules['app.engine_state']

importlib.reload(engine_state)
print(f"   [OK] Mode: {engine_state.EDON_MODE} (defaulted from invalid)")
assert engine_state.EDON_MODE == "v1", "Should default to v1"

print("\n" + "=" * 70)
print("Mode switching tests passed!")
print("=" * 70)
print("\nTo test with server:")
print("1. Set environment: $env:EDON_MODE = 'v2'")
print("2. Start server: python -m uvicorn app.main:app --host 127.0.0.1 --port 8002")
print("3. Check health: curl http://127.0.0.1:8002/health")
print("\nExpected output:")
print('{"ok": true, "mode": "v2", "engine": "v2", "neural_loaded": false, "pca_loaded": false, "uptime_s": ...}')

