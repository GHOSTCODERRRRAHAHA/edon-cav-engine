# EDON CAV Engine

**EDON** is an adaptive state engine for physical AI (humanoids, wearables, and smart environments). It processes physiological sensor data (EDA, temperature, BVP, accelerometer) combined with environmental context to compute **Context-Aware Vectors (CAV)** and predict adaptive states: `restorative`, `balanced`, `focus`, or `overload`.

## Overview

EDON CAV Engine provides:
- **REST API** for real-time CAV computation
- **Python SDK** for easy integration
- **Adaptive memory engine** with 24-hour rolling context
- **Production-ready** inference pipeline

**All inference flows through `/oem/cav/batch`. The Python SDK's `EdonClient.cav()` is a convenience wrapper over the batch API.**

---

## How to Run the Engine

### Prerequisites

- Python 3.10+
- Model files (see `cav_engine_v3_2_LGBM_2025-11-08/`)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd edon-cav-engine

# Install dependencies
pip install -r requirements.txt

# Optional: Set API keys for environmental data
# Copy .env.example to .env and configure
```

### Start the Server

```bash
# Start FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001

# Or use the provided script
.\run_api.ps1  # Windows
```

The API will be available at `http://127.0.0.1:8001`

### Verify Server is Running

```bash
# Health check
curl http://127.0.0.1:8001/health

# View API documentation
# Open http://127.0.0.1:8001/docs in your browser
```

### API Endpoints

- **`POST /oem/cav/batch`** - Batch CAV computation (1-5 windows per request)
- **`GET /health`** - Health check with model info
- **`GET /telemetry`** - System telemetry
- **`GET /memory/summary`** - Adaptive memory statistics
- **`POST /v1/ingest`** - Sensor frame ingestion
- **`GET /_debug/state`** - Debug state information

See `/docs` for full OpenAPI documentation.

---

## How to Install and Use the Python SDK

### Installation

```bash
# From the repository root
cd sdk/python
pip install -e .

# Or install from PyPI (when published)
pip install edon-sdk
```

### Quick Start

```python
from edon_sdk import EdonClient

# Initialize client
client = EdonClient(base_url="http://127.0.0.1:8001")

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

# Compute CAV (uses batch endpoint internally)
result = client.cav(window)
print(f"State: {result['state']}")
print(f"CAV: {result['cav_smooth']}")

# Batch processing
windows = [window1, window2, window3]
results = client.cav_batch(windows)
```

### SDK Methods

- **`client.health()`** - Check API health
- **`client.cav(window)`** - Single window CAV computation (wrapper over batch API)
- **`client.cav_batch(windows)`** - Batch CAV computation (1-5 windows)
- **`client.ingest(payload)`** - Ingest sensor frames
- **`client.debug_state()`** - Get debug state

### Configuration

```python
# Environment variables
import os
os.environ["EDON_BASE_URL"] = "http://127.0.0.1:8001"
os.environ["EDON_API_TOKEN"] = "your-token"  # Optional

# Or pass directly
client = EdonClient(
    base_url="http://127.0.0.1:8001",
    api_key="your-token",  # Optional
    timeout=5.0,
    max_retries=2,
    verbose=False
)
```

### Example Script

```bash
# Run the example
cd sdk/python
python examples/basic_infer.py --base-url http://127.0.0.1:8001
```

---

## Architecture

### Inference Flow

All CAV computation flows through the batch endpoint:

```
Client Request → POST /oem/cav/batch → CAV Engine → Response
```

The SDK's `client.cav()` method is a convenience wrapper that:
1. Wraps a single window in a batch request
2. Calls `POST /oem/cav/batch`
3. Extracts and returns the single result

### Model

- **Model**: `cav_state_v3_2` (LightGBM)
- **Training**: 100K windows from WESAD dataset
- **Features**: 6 physiological + environmental features
- **Output**: CAV score [0-10000] and state classification

---

## Documentation

- **API Documentation**: `http://localhost:8001/docs` (when server is running)
- **SDK Documentation**: See `sdk/python/README.md`
- **Project Documentation**: See `docs/` directory

---

## License

MIT

---

## Support

For issues, questions, or contributions, please open an issue on the repository.

