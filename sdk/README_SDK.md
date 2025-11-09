# EDON SDK - Quick Start Guide

## Overview

The EDON SDK provides a simple Python client for interacting with the EDON CAV API. It enables OEM partners to easily integrate EDON's adaptive intelligence into their applications.

## Installation

No installation required! The SDK is a single Python file with minimal dependencies:

```bash
pip install requests
```

## Quick Start

```python
from edon_client import EDONClient
import json

# Initialize client
client = EDONClient(base_url="http://localhost:8000")

# Load sample payload
with open("sample_payload.json", "r") as f:
    payload = json.load(f)

# Compute CAV score
result = client.post_cav_from_dict(payload)

print(f"CAV Score: {result['cav_smooth']}")
print(f"State: {result['state']}")
print(f"Adaptive: {result['adaptive']}")
```

## API Endpoints

### POST `/cav`

Compute CAV score from sensor window.

**Request:**
```json
{
  "EDA": [0.01, 0.02, ...240 samples],
  "TEMP": [36.5, 36.4, ...240 samples],
  "BVP": [0.3, 0.31, ...240 samples],
  "ACC_x": [0.01, -0.01, ...240 samples],
  "ACC_y": [0.02, -0.02, ...240 samples],
  "ACC_z": [0.03, -0.03, ...240 samples],
  "temp_c": 24.0,
  "humidity": 50.0,
  "aqi": 40,
  "local_hour": 10
}
```

**Response:**
```json
{
  "cav_raw": 9389,
  "cav_smooth": 9245,
  "state": "restorative",
  "parts": {
    "bio": 0.95,
    "env": 0.97,
    "circadian": 0.73,
    "p_stress": 0.05
  },
  "adaptive": {
    "z_cav": -0.83,
    "sensitivity": 1.28,
    "env_weight_adj": 0.88
  }
}
```

### GET `/memory/summary`

Get 24-hour memory statistics.

**Response:**
```json
{
  "total_records": 1234,
  "window_hours": 24,
  "hourly_stats": {...},
  "overall_stats": {
    "cav_mean": 5234.1,
    "cav_std": 1456.7,
    "state_distribution": {...}
  }
}
```

### POST `/memory/clear`

Clear all memory (for testing).

## Example Usage

### Basic CAV Computation

```python
from edon_client import EDONClient
import numpy as np

client = EDONClient()

# Generate sample sensor data
eda = np.random.normal(0, 0.2, 240).tolist()
temp = np.random.normal(32, 0.2, 240).tolist()
bvp = np.random.normal(0, 0.5, 240).tolist()
acc_x = np.random.normal(0, 0.05, 240).tolist()
acc_y = np.random.normal(0, 0.05, 240).tolist()
acc_z = np.random.normal(1, 0.05, 240).tolist()

# Compute CAV
result = client.post_cav(
    eda=eda,
    temp=temp,
    bvp=bvp,
    acc_x=acc_x,
    acc_y=acc_y,
    acc_z=acc_z,
    temp_c=24.0,
    humidity=50.0,
    aqi=40,
    local_hour=10
)

print(f"CAV: {result['cav_smooth']}")
print(f"State: {result['state']}")
print(f"Z-score: {result['adaptive']['z_cav']}")
```

### Monitor Memory

```python
summary = client.get_memory_summary()
print(f"Total records: {summary['total_records']}")
print(f"CAV mean: {summary['overall_stats']['cav_mean']}")
```

## Response Schema

### CAV Response

- **cav_raw**: Raw CAV score [0-10000]
- **cav_smooth**: Smoothed CAV score [0-10000]
- **state**: State string (`overload`, `balanced`, `focus`, `restorative`)
- **parts**: Component scores
  - `bio`: Biological component [0-1]
  - `env`: Environmental component [0-1]
  - `circadian`: Circadian component [0-1]
  - `p_stress`: Stress probability [0-1]
- **adaptive**: Adaptive adjustments
  - `z_cav`: Z-score relative to baseline
  - `sensitivity`: Sensitivity multiplier (1.0 = normal, >1.0 = increased)
  - `env_weight_adj`: Environment weight adjustment (1.0 = normal, <1.0 = reduced)

## Dashboard Integration

Results can be visualized in the dashboard at:
```
http://localhost:8000/dashboard
```

The dashboard shows:
- Real-time CAV over time
- State frequency distribution
- Hourly CAV baseline heatmap
- Adaptive sensitivity gauge
- Memory statistics

## Error Handling

The SDK raises `requests.exceptions.RequestException` on API errors:

```python
try:
    result = client.post_cav(...)
except requests.exceptions.RequestException as e:
    print(f"API error: {e}")
```

## Requirements

- Python 3.7+
- `requests` library

## License

See main EDON project license.

