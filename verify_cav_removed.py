#!/usr/bin/env python3
"""Verify that /cav endpoint is completely removed."""

import sys
import os

# Skip dashboard to avoid import errors
os.environ['EDON_SKIP_DASHBOARD'] = '1'

# Temporarily mock dash to avoid import errors
import sys
from unittest.mock import MagicMock
sys.modules['dash'] = MagicMock()
sys.modules['dash.dependencies'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objs'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()

try:
    from app.main import app
    
    # Check routes
    routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            methods = list(route.methods) if hasattr(route, 'methods') else []
            routes.append({'path': route.path, 'methods': methods})
    
    # Find any legacy /cav routes (not /oem/cav)
    cav_routes = [r for r in routes if '/cav' in r['path'] and '/oem/cav' not in r['path']]
    
    print("=" * 60)
    print("Verification: /cav Endpoint Removal")
    print("=" * 60)
    
    if cav_routes:
        print(f"\n[FAIL] Found {len(cav_routes)} legacy /cav route(s):")
        for r in cav_routes:
            print(f"  - {r['path']} {r['methods']}")
        sys.exit(1)
    else:
        print("\n[OK] No legacy /cav routes found")
    
    # Check OpenAPI spec
    openapi = app.openapi()
    paths = list(openapi['paths'].keys())
    cav_paths = [p for p in paths if '/cav' in p and '/oem/cav' not in p]
    
    if cav_paths:
        print(f"\n[FAIL] Found {len(cav_paths)} legacy /cav path(s) in OpenAPI:")
        for p in cav_paths:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print("[OK] No legacy /cav paths in OpenAPI spec")
    
    # Show remaining routes
    print(f"\n[OK] Total routes: {len(routes)}")
    print("\nAvailable routes:")
    for r in sorted(routes, key=lambda x: x['path']):
        methods_str = ', '.join(r['methods']) if r['methods'] else 'N/A'
        print(f"  {r['path']:30s} [{methods_str}]")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] /cav endpoint completely removed!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] Failed to verify: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

