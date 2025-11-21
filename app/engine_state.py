"""Shared engine state for EDON mode switching."""

import os

# Determine EDON mode from environment variable
EDON_MODE = os.getenv("EDON_MODE", "v1").lower()
if EDON_MODE not in ["v1", "v2"]:
    EDON_MODE = "v1"  # Default to v1 if invalid

# Engine instances (initialized in main.py)
default_engine_v2 = None
neural_loaded = False
pca_loaded = False

