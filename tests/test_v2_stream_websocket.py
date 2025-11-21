"""Test v2 WebSocket streaming endpoint."""

import asyncio
import json
import websockets
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import math
import random


def create_test_window():
    """Create a test v2 window."""
    return {
        "physio": {
            "EDA": [0.25] * 240,
            "BVP": [0.5 + 0.1 * math.sin(i / 20) for i in range(240)]
        },
        "motion": {
            "ACC_x": [random.gauss(0, 0.1) for _ in range(240)],
            "ACC_y": [random.gauss(0, 0.1) for _ in range(240)],
            "ACC_z": [1.0 + random.gauss(0, 0.05) for _ in range(240)]
        },
        "env": {
            "temp_c": 22.0,
            "humidity": 45.0,
            "aqi": 20
        },
        "task": {
            "id": "test",
            "complexity": 0.5
        }
    }


async def test_stream():
    """Test WebSocket streaming."""
    uri = "ws://127.0.0.1:8002/v2/stream/cav"
    
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected")
            
            # Send a few windows
            for i in range(3):
                window = create_test_window()
                window["task"]["id"] = f"test_{i}"
                
                print(f"\nSending window {i+1}...")
                await websocket.send(json.dumps(window))
                
                # Receive response
                response = await websocket.recv()
                result = json.loads(response)
                
                if result.get("ok"):
                    print(f"  ✓ State: {result['state_class']}")
                    print(f"  ✓ p_stress: {result['p_stress']:.3f}")
                    print(f"  ✓ p_chaos: {result['p_chaos']:.3f}")
                else:
                    print(f"  ✗ Error: {result.get('error')}")
                
                await asyncio.sleep(0.5)
            
            print("\n✓ Streaming test completed")
            return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("EDON v2 WebSocket Streaming Test")
    print("=" * 70)
    print()
    print("Prerequisites:")
    print("  1. Start REST server: python -m uvicorn app.main:app --port 8002")
    print("  2. Set EDON_MODE=v2")
    print()
    
    try:
        result = asyncio.run(test_stream())
        if result:
            print("\n✓ Test passed!")
        else:
            print("\n✗ Test failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(1)

