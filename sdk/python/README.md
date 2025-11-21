# EDON Python SDK

Product-ready Python SDK for the EDON CAV Engine.

## Installation

```bash
# Basic installation (REST only)
pip install -e sdk/python

# With gRPC support
pip install -e "sdk/python[grpc]"

# Future: Install from PyPI
pip install edon[grpc]
```

## Quick Start

```python
from edon import EdonClient, TransportType

# REST transport (default, uses EDON_BASE_URL env var)
client = EdonClient()

# Or explicitly
client = EdonClient(
    base_url="http://127.0.0.1:8001",
    transport=TransportType.REST
)

# Create sensor window (240 samples per signal)
window = {
    "EDA": [0.1] * 240,          # Electrodermal activity
    "TEMP": [36.5] * 240,        # Temperature
    "BVP": [0.5] * 240,          # Blood volume pulse
    "ACC_x": [0.0] * 240,        # Accelerometer X
    "ACC_y": [0.0] * 240,        # Accelerometer Y
    "ACC_z": [1.0] * 240,        # Accelerometer Z
    "temp_c": 22.0,              # Environmental temperature (°C)
    "humidity": 50.0,            # Humidity (%)
    "aqi": 35,                   # Air Quality Index
    "local_hour": 14,            # Local hour [0-23]
}

# Compute CAV
result = client.cav(window)
print(f"State: {result['state']}")
print(f"CAV: {result['cav_smooth']}")
print(f"P-Stress: {result['parts']['p_stress']:.3f}")

# Classify state (convenience method)
state = client.classify(window)
print(f"State: {state}")
```

## Robot Integration Example

```python
from edon import EdonClient
import time

client = EdonClient()

while True:
    # Collect sensor data (4 seconds = 240 samples)
    window = build_window_from_robot_sensors()
    
    # Get EDON state
    result = client.cav(window)
    state = result['state']
    p_stress = result['parts']['p_stress']
    
    # Map to control scales
    if state == "restorative":
        speed, torque, safety = 0.7, 0.7, 0.95
    elif state == "balanced":
        speed, torque, safety = 1.0, 1.0, 0.85
    elif state == "focus":
        speed, torque, safety = 1.2, 1.1, 0.8
    elif state == "overload":
        speed, torque, safety = 0.4, 0.4, 1.0
    
    # Apply to robot controllers
    apply_scales_to_controllers(speed, torque, safety)
    
    time.sleep(4.0)  # Wait for next window
```

## gRPC Transport

```python
from edon import EdonClient, TransportType

# gRPC transport
client = EdonClient(
    transport=TransportType.GRPC,
    grpc_host="localhost",
    grpc_port=50051
)

result = client.cav(window)

# Control scales included in gRPC response
if 'controls' in result:
    print(f"Speed: {result['controls']['speed']:.2f}")
    print(f"Torque: {result['controls']['torque']:.2f}")
    print(f"Safety: {result['controls']['safety']:.2f}")

# Streaming (server push)
for update in client.stream(window):
    print(f"State: {update['state']}, CAV: {update['cav_smooth']}")
    # Process update...
    
client.close()  # Close gRPC channel
```

## API Reference

### EdonClient

**Methods**:
- `cav(window)` - Compute CAV from sensor window
- `cav_batch(windows)` - Batch CAV computation (REST only, 1-5 windows)
- `classify(window)` - Classify state (convenience method)
- `stream(window)` - Stream CAV updates (gRPC only)
- `health()` - Check service health
- `close()` - Close connections (gRPC only)

**Parameters**:
- `base_url` - REST API base URL (default: from `EDON_BASE_URL` env var)
- `api_key` - API token (default: from `EDON_API_TOKEN` env var)
- `transport` - `TransportType.REST` or `TransportType.GRPC`
- `grpc_host` - gRPC server host (default: "localhost")
- `grpc_port` - gRPC server port (default: 50051)

## Environment Variables

- `EDON_BASE_URL` - Base URL for REST API (default: http://127.0.0.1:8000)
- `EDON_API_TOKEN` - API token for authentication (optional)

## Examples

See `examples/python/` and `clients/robot_example.py` for complete examples.

## API Contract

See `docs/OEM_API_CONTRACT.md` for exact request/response schemas and versioning policy.
