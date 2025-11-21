#!/usr/bin/env python3
"""Simple example: Using EDON SDK with gRPC transport."""

from edon import EdonClient, TransportType

def main():
    # Initialize client with gRPC transport
    client = EdonClient(
        transport=TransportType.GRPC,
        grpc_host="localhost",
        grpc_port=50051
    )
    
    # Create sensor window
    window = {
        "EDA": [0.1] * 240,
        "TEMP": [36.5] * 240,
        "BVP": [0.5] * 240,
        "ACC_x": [0.0] * 240,
        "ACC_y": [0.0] * 240,
        "ACC_z": [1.0] * 240,
        "temp_c": 22.0,
        "humidity": 50.0,
        "aqi": 35,
        "local_hour": 14,
    }
    
    # Compute CAV via gRPC
    print("Computing CAV via gRPC...")
    result = client.cav(window)
    
    print(f"State: {result['state']}")
    print(f"CAV Smooth: {result['cav_smooth']}")
    print(f"P-Stress: {result['parts']['p_stress']:.3f}")
    
    # Control scales (gRPC only)
    if 'controls' in result:
        print(f"\nControl Scales:")
        print(f"  Speed: {result['controls']['speed']:.2f}")
        print(f"  Torque: {result['controls']['torque']:.2f}")
        print(f"  Safety: {result['controls']['safety']:.2f}")
    
    # Streaming (server push)
    print("\nStreaming updates (press Ctrl+C to stop)...")
    try:
        count = 0
        for update in client.stream(window):
            count += 1
            print(f"Update #{count}: State={update['state']}, CAV={update['cav_smooth']}")
            if count >= 3:  # Just show a few updates
                break
    except KeyboardInterrupt:
        print("\nStopped streaming")
    
    client.close()
    print("\nDone!")

if __name__ == "__main__":
    main()

