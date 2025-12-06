#!/usr/bin/env python3
"""
Quick EDON License Activation Script

Creates a valid offline evaluation license for development/testing.
"""

import json
import time
import hmac
import hashlib
from pathlib import Path
import os

# Configuration
SECRET_KEY = os.getenv("EDON_LICENSE_SECRET_V1", "eval-secret-key-v1-change-in-production")
KEY_VERSION = "v1"
EVAL_PERIOD_DAYS = 30

def create_license(org_id: str = "dev-org", project_id: str = "dev-project", days: int = 30):
    """Create a valid offline evaluation license."""
    
    # Create license data
    license_data = {
        "type": "evaluation",
        "version": "2.0.0",
        "key_version": KEY_VERSION,
        "activated_at": time.time(),
        "expires_at": time.time() + (days * 86400),
        "activation_id": "offline-dev",
        "org_id": org_id,
        "project_id": project_id,
        "plan": "evaluation",
        "hostname": os.getenv("HOSTNAME", "localhost"),
        "revoked": False,
        "offline": True
    }
    
    # Compute signature
    data_copy = {k: v for k, v in license_data.items() if k != "signature"}
    canonical = json.dumps(data_copy, sort_keys=True, separators=(',', ':'))
    signature = hmac.new(
        SECRET_KEY.encode(),
        canonical.encode(),
        hashlib.sha256
    ).hexdigest()
    
    license_data["signature"] = signature
    
    # Save license file
    license_file = Path.home() / ".edon" / "license.json"
    license_file.parent.mkdir(parents=True, exist_ok=True)
    with open(license_file, 'w') as f:
        json.dump(license_data, f, indent=2)
    
    print("=" * 70)
    print("EDON License Activated (Offline Mode)")
    print("=" * 70)
    print(f"License file: {license_file}")
    print(f"Organization: {org_id}")
    print(f"Project: {project_id}")
    print(f"Expires in: {days} days")
    print(f"Activation ID: {license_data['activation_id']}")
    print("=" * 70)
    print("\nNote: This is an offline development license.")
    print("For production, use cloud activation via the activation server.")
    
    return license_file

if __name__ == "__main__":
    import sys
    
    org_id = sys.argv[1] if len(sys.argv) > 1 else "dev-org"
    project_id = sys.argv[2] if len(sys.argv) > 2 else "dev-project"
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    create_license(org_id, project_id, days)

