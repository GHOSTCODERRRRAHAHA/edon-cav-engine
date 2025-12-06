"""Quick check to see if robot_stability route is accessible."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app.routes import robot_stability
    print(f"✅ robot_stability route imported successfully")
    print(f"   Router prefix: {robot_stability.router.prefix}")
    print(f"   Router tags: {robot_stability.router.tags}")
    
    # Check if router has the endpoint
    for route in robot_stability.router.routes:
        print(f"   Route: {route.path} ({route.methods if hasattr(route, 'methods') else 'N/A'})")
        
except Exception as e:
    print(f"❌ Error importing robot_stability: {e}")
    import traceback
    traceback.print_exc()

