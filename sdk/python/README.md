# EDON Python SDK

Python SDK for the **EDON CAV Engine** — an adaptive state engine for physical AI (humanoids, wearables, and smart environments).

EDON processes physiological sensor data (EDA, temperature, BVP, accelerometer) combined with environmental context (temperature, humidity, air quality, time of day) to compute **Context-Aware Vectors (CAV)** and predict adaptive states: `restorative`, `balanced`, `focus`, or `overload`.

This SDK provides a clean Python interface to the EDON REST API, making it easy for robotics, wearable, and smart-home developers to integrate EDON into their applications.

## Installation

### Local Development (Editable Install)

```bash
# Clone the repository
git clone <repo-url>
cd edon-cav-engine

# Install the SDK in editable mode
cd sdk/python
pip install -e .
```

### Production Install

```bash
pip install edon-sdk
```

## Quick Start

### Basic Usage

```python
from edon_sdk import EdonClient

# Initialize client (defaults to http://127.0.0.1:8000)
client = EdonClient(base_url="http://127.0.0.1:8000")

# Create a sensor window (240 samples per signal)
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
print(f"CAV (raw): {result['cav_raw']}")
print(f"CAV (smooth): {result['cav_smooth']}")
print(f"Parts: {result['parts']}")  # bio, env, circadian, p_stress
```

### Batch Processing

```python
# Process multiple windows at once
windows = [window1, window2, window3]
results = client.cav_batch(windows)

for i, result in enumerate(results):
    if result.get("ok"):
        print(f"Window {i}: {result['state']} (CAV={result['cav_smooth']})")
    else:
        print(f"Window {i} error: {result.get('error')}")
```

### Authentication

If your EDON API requires authentication:

```python
# Option 1: Pass API key directly
client = EdonClient(
    base_url="https://api.edon.example.com",
    api_key="your-api-token-here"
)

# Option 2: Use environment variable
import os
os.environ["EDON_API_TOKEN"] = "your-api-token-here"
client = EdonClient(base_url="https://api.edon.example.com")
```

### Health Check

```python
health = client.health()
print(f"Status: {health['ok']}")
print(f"Model: {health['model']}")
print(f"Uptime: {health['uptime_s']:.1f}s")
```

### Ingest & Debug State

```python
# Ingest sensor frames
payload = {
    "frames": [
        {
            "ts": 1234567890.0,
            "user_id": "user123",
            "env": {"co2": 600, "dba": 40},
        }
    ]
}
response = client.ingest(payload)

# Get debug state (if available)
state = client.debug_state()
if state:
    print(f"Current mode: {state.get('mode')}")
```

## Configuration

The `EdonClient` supports the following configuration options:

- **base_url**: API base URL (default: `http://127.0.0.1:8000`, or `EDON_BASE_URL` env var)
- **api_key**: API token for authentication (default: `EDON_API_TOKEN` env var)
- **timeout**: Request timeout in seconds (default: `5.0`)
- **max_retries**: Maximum retries on 5xx/connection errors (default: `2`)
- **verbose**: Enable request logging to stdout (default: `False`)

## Error Handling

The SDK provides custom exceptions for different error scenarios:

```python
from edon_sdk import EdonClient, EdonAuthError, EdonHTTPError, EdonConnectionError

try:
    result = client.cav(window)
except EdonAuthError:
    print("Authentication failed - check your API key")
except EdonHTTPError as e:
    print(f"API error {e.status_code}: {e}")
except EdonConnectionError:
    print("Connection failed - is the server running?")
```

## Example Script

See `examples/basic_infer.py` for a complete example that:
- Creates a synthetic sensor window
- Calls the CAV API
- Prints state, CAV scores, and component parts

Run it with:

```bash
python examples/basic_infer.py --base-url http://127.0.0.1:8000
```

## Versioning

This SDK is currently tied to engine model `cav_state_v3_2`. Future releases will align SDK versions with engine tags (e.g., v1.0.0, v1.1.0, etc.).

## Requirements

- Python >= 3.10
- `requests >= 2.31.0`
- `urllib3 >= 2.0.0`

## License

MIT

## Support

For issues, questions, or contributions, please open an issue on the repository.

