# EDON CAV Engine

**Current engine release: v1.1.0** (REST + gRPC, Docker image `edon-server:v1.1.0`)

**EDON** is an adaptive state engine for physical AI (humanoids, wearables, and smart environments). It processes physiological sensor data (EDA, temperature, BVP, accelerometer) combined with environmental context to compute **Context-Aware Vectors (CAV)** and predict adaptive states: `restorative`, `balanced`, `focus`, or `overload`.

## Overview

EDON CAV Engine provides:
- **REST API** for real-time CAV computation
- **gRPC Service** for high-performance robotics integration
- **ROS2 Node** for real-time robot state management
- **Python SDK** with REST and gRPC transport support
- **C++ SDK** for native integration
- **Adaptive memory engine** with 24-hour rolling context
- **Production-ready** inference pipeline

**All inference flows through `/oem/cav/batch` (REST) or gRPC `GetState` (gRPC). The Python SDK's `EdonClient.cav()` supports both transports.**

---

## Quick Start (Docker - Recommended)

**One-command deployment**:

```bash
git clone <repo-url>
cd edon-cav-engine
docker compose up --build
```

**Access**:
- REST API: `http://localhost:8001`
- gRPC: `localhost:50051`
- Health: `http://localhost:8001/health`
- Metrics: `http://localhost:8001/metrics`

See `README_DOCKER.md` for detailed Docker instructions.

## Manual Installation

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

# For gRPC support
pip install grpcio grpcio-tools

# Install SDK
pip install -e sdk/python

# Optional: Set API keys for environmental data
# Copy .env.example to .env and configure
```

### Start the REST API Server

```bash
# Start FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001

# Or use the provided script
.\run_api.ps1  # Windows
```

The REST API will be available at `http://127.0.0.1:8001`

### Start the gRPC Server

```bash
# Generate protobuf files first
cd integrations/grpc/edon_grpc_service
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. edon.proto

# Run gRPC server
cd ../../..
chmod +x scripts/run_grpc_server.sh
./scripts/run_grpc_server.sh --port 50051
```

The gRPC server will be available at `localhost:50051`

### Verify Server is Running

```bash
# Health check
curl http://127.0.0.1:8001/health

# View API documentation
# Open http://127.0.0.1:8001/docs in your browser
```

### REST API Endpoints

- **`POST /oem/cav/batch`** - Batch CAV computation (1-5 windows per request)
- **`GET /health`** - Health check with model info
- **`GET /telemetry`** - System telemetry
- **`GET /memory/summary`** - Adaptive memory statistics
- **`POST /v1/ingest`** - Sensor frame ingestion
- **`GET /_debug/state`** - Debug state information

See `/docs` for full OpenAPI documentation.

### gRPC Service

- **`GetState(StreamDataRequest)`** - Single CAV computation
- **`StreamState(StreamDataRequest)`** - Server-side streaming

See `integrations/grpc/edon_grpc_service/edon.proto` for protocol definitions.

---

## ROS2 Integration

### Prerequisites

- ROS2 Foxy or later
- Python 3.10+

### Installation

```bash
# Source ROS2 environment
source /opt/ros/foxy/setup.bash  # Adjust for your ROS2 distribution

# Install dependencies
pip install rclpy numpy scipy pandas scikit-learn lightgbm
```

### Running the ROS2 Node

```bash
# From repository root
chmod +x scripts/run_ros2_node.sh
./scripts/run_ros2_node.sh
```

The node will:
- Subscribe to `/edon/sensors/physiology` (sensor_msgs/msg/PointCloud2)
- Subscribe to `/edon/sensors/environment` (sensor_msgs/msg/Temperature)
- Publish `/edon/state` (std_msgs/msg/String)
- Publish `/edon/controls` (control scales for robot behavior)

### ROS2 Topics

**Subscriptions:**
- `/edon/sensors/physiology` - Physiological sensor data (EDA, BVP, ACC, TEMP)
- `/edon/sensors/environment` - Environmental data (temp_c, humidity, aqi)

**Publications:**
- `/edon/state` - Current EDON state: `restorative`, `balanced`, `focus`, or `overload`
- `/edon/controls` - Robot control scales (speed, torque, safety)

---

## gRPC Service

### Prerequisites

- Python 3.10+
- grpcio and grpcio-tools

### Installation

```bash
pip install grpcio grpcio-tools
```

### Generate Protobuf Code

```bash
cd integrations/grpc/edon_grpc_service
chmod +x generate_proto.sh
./generate_proto.sh
```

### Running the gRPC Server

```bash
# From repository root
chmod +x scripts/run_grpc_server.sh
./scripts/run_grpc_server.sh --port 50051
```

### gRPC API

- **`GetState(StreamDataRequest)`** - Single request/response for CAV computation
- **`StreamState(StreamDataRequest)`** - Server-side streaming (push updates)

See `integrations/grpc/edon_grpc_service/edon.proto` for message definitions.

---

## Python SDK

### Installation

```bash
# From the repository root
cd sdk/python
pip install -e .

# Install with gRPC support
pip install -e ".[grpc]"

# Or install from PyPI (when published)
pip install edon-sdk[grpc]
```

### Quick Start (REST)

```python
from edon_sdk import EdonClient, TransportType

# Initialize client with REST transport (default)
client = EdonClient(
    base_url="http://127.0.0.1:8001",
    transport=TransportType.REST
)

# Create a sensor window (240 samples per signal)
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

# Compute CAV
result = client.cav(window)
print(f"State: {result['state']}")
print(f"CAV: {result['cav_smooth']}")

# Classify state (convenience method)
state = client.classify(window)
```

### Quick Start (gRPC)

```python
from edon_sdk import EdonClient, TransportType

# Initialize client with gRPC transport
client = EdonClient(
    transport=TransportType.GRPC,
    grpc_host="localhost",
    grpc_port=50051
)

# Compute CAV via gRPC
result = client.cav(window)

# Stream updates (server push)
for update in client.stream(window):
    print(f"State: {update['state']}, CAV: {update['cav_smooth']}")
    # Includes control scales: update['controls']['speed'], etc.

client.close()  # Close gRPC channel
```

### SDK Methods

- **`client.health()`** - Check service health
- **`client.cav(window)`** - Single window CAV computation
- **`client.stream(window)`** - Stream CAV updates (gRPC only)
- **`client.classify(window)`** - Classify state (convenience method)
- **`client.cav_batch(windows)`** - Batch CAV computation (REST only)
- **`client.ingest(payload)`** - Ingest sensor frames (REST only)
- **`client.debug_state()`** - Get debug state (REST only)

### Examples

```bash
# REST example
python examples/python/rest_example.py

# gRPC example
python examples/python/grpc_example.py
```

---

## C++ SDK

### Prerequisites

- C++17 or later
- CMake 3.15+
- gRPC and Protobuf
- pkg-config

### Building

```bash
cd sdk/cpp
mkdir build
cd build
cmake ..
make
```

### Usage

```cpp
#include "edon/edon.hpp"

edon::EdonClient client("localhost", 50051);

edon::SensorWindow window;
window.eda = std::vector<float>(240, 0.1f);
// ... fill other fields ...

edon::CAVResponse response = client.computeCAV(window);
std::cout << "State: " << response.state << std::endl;

// Stream updates
client.stream(window, [](const edon::CAVResponse& resp) {
    std::cout << "Update: " << resp.state << std::endl;
}, true);
```

### Example

```bash
cd sdk/cpp/build
make edon_example
./edon_example
```

See `sdk/cpp/README.md` for detailed documentation.

---

## Architecture

### Inference Flow

**REST:**
```
Client Request → POST /oem/cav/batch → CAV Engine → Response
```

**gRPC:**
```
Client Request → GetState/StreamState → CAV Engine → Response
```

**ROS2:**
```
Sensor Topics → ROS2 Node → CAV Engine → State/Control Topics
```

The Python SDK's `client.cav()` method supports both REST and gRPC transports:
- **REST**: Wraps a single window in a batch request
- **gRPC**: Direct gRPC call to `GetState`

### Model

- **Model**: `cav_state_v3_2` (LightGBM)
- **Training**: 100K windows from WESAD dataset
- **Features**: 6 physiological + environmental features
- **Output**: CAV score [0-10000] and state classification

**State Classification** (v1.1.0):
- **restorative**: `p_stress < 0.2` - Very low stress, optimal recovery
- **focus**: `0.2 ≤ p_stress ≤ 0.5` AND `env ≥ 0.8` AND `circadian ≥ 0.9` - Moderate stress with strong alignment
- **balanced**: `0.2 ≤ p_stress < 0.8` (when focus conditions not met) - Normal operation
- **overload**: `p_stress ≥ 0.8` - High stress requiring intervention

See [`MODEL_CARD.md`](MODEL_CARD.md) for detailed state mapping explanation.

---

## Documentation

- **OEM Onboarding**: `docs/OEM_ONBOARDING.md` - Complete OEM integration overview
- **OEM Integration Guide**: `docs/OEM_INTEGRATION.md` - Detailed integration guide
- **API Contract**: `docs/OEM_API_CONTRACT.md` - FROZEN v1 API contract
- **REST API Docs**: `http://localhost:8001/docs` (when server is running)
- **Python SDK**: `sdk/python/README.md`
- **C++ SDK**: `sdk/cpp/README.md`
- **Docker Deployment**: `README_DOCKER.md`
- **gRPC Service**: `integrations/grpc/edon_grpc_service/README.md`
- **ROS2 Node**: `integrations/ros2/edon_ros2_node/`

### C++ SDK

For robotics and embedded integrations, a C++ SDK is provided under `sdk/cpp`.

See [`sdk/cpp/README.md`](sdk/cpp/README.md) for build and packaging instructions.

### OEM Onboarding

For a full OEM integration overview (Docker, REST, gRPC, Python, C++), see:

- [`docs/OEM_ONBOARDING.md`](docs/OEM_ONBOARDING.md)

## Performance Benchmarks

Run benchmarks to get performance numbers:

```bash
# REST API benchmark
python tests/latency_benchmark.py --n 1000

# gRPC benchmark
python tests/load_test_grpc.py --n 1000
```

**Typical Performance** (single CPU core):
- **Throughput**: 50-100 windows/sec
- **Latency**: 10-20ms median (p50), 30-50ms p99

## Version

Current engine release: **v1.1.0** (REST + gRPC, Docker image `edon-server:v1.1.0`)

**API Contract**: v1 is FROZEN - see `docs/OEM_API_CONTRACT.md` for versioning policy

---

## License

MIT

---

## Support

For issues, questions, or contributions, please open an issue on the repository.

