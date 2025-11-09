# EDON Adaptive Memory Engine (Soul Layer v1)

## Overview

The Adaptive Memory Engine transforms EDON from a static analyzer into an adaptive intelligence core. It maintains rolling 24-hour context of CAV responses and computes adaptive adjustments based on historical patterns, enabling the system to learn and self-adjust.

## Architecture

### Components

1. **Memory Engine** (`app/adaptive_memory.py`)
   - Maintains rolling 24-hour buffer of CAV responses
   - Persists data to SQLite database
   - Computes hourly EWMA statistics
   - Calculates adaptive adjustments

2. **CAV Integration** (`app/routes/cav.py`)
   - Records each CAV response in memory
   - Computes adaptive adjustments
   - Returns adaptive information in response

3. **Memory Management** (`app/routes/memory.py`)
   - Provides summary of memory statistics
   - Allows clearing memory for testing

## How It Works

### 1. Memory Storage

Each CAV response is stored with:
- Timestamp
- CAV scores (raw and smoothed)
- State (overload, balanced, focus, restorative)
- Component parts (bio, env, circadian, p_stress)
- Environmental context (temp_c, humidity, aqi, local_hour)

**Storage Strategy:**
- **In-memory buffer**: Rolling deque (last 24 hours, max 10,000 records)
- **SQLite database**: Persistent storage (last 7 days)
- **Automatic cleanup**: Records older than 7 days are removed

### 2. Baseline Calculation

The engine computes hourly baselines using **Exponential Weighted Moving Average (EWMA)**:

#### Hourly Statistics (per hour 0-23)

For each hour, the engine maintains:
- **`cav_mu[hour]`**: Mean CAV (EWMA with α=0.3)
- **`cav_var[hour]`**: Variance CAV (EWMA with α=0.3)
- **`state_probs[hour]`**: State frequency distribution

**EWMA Formula:**
```
new_mean = α * current_mean + (1 - α) * previous_mean
new_var = α * current_var + (1 - α) * previous_var
```

**Update Frequency:**
- Statistics updated every 10 records or hourly
- Ensures responsive adaptation without excessive computation

### 3. Adaptive Adjustments

The engine computes three adaptive adjustments:

#### A. Z-Score (`z_cav`)

Measures how far the current CAV deviates from the baseline:

```
z_cav = (cav_smooth - baseline_mu[hour]) / baseline_std[hour]
```

**Interpretation:**
- `|z_cav| < 1.0`: Normal variation
- `1.0 ≤ |z_cav| < 1.5`: Moderate deviation
- `|z_cav| ≥ 1.5`: Significant deviation (triggers sensitivity adjustment)

#### B. Sensitivity (`sensitivity`)

Controls how quickly the system responds to state changes:

```
if |z_cav| > 1.5:
    sensitivity = 1.0 + min(|z_cav| - 1.5, 0.5) * 0.5  # Max 1.25x
else:
    sensitivity = 1.0
```

**Behavior:**
- `sensitivity = 1.0`: Normal response speed
- `sensitivity > 1.0`: Increased sensitivity (faster state transitions)
- Higher sensitivity when CAV deviates significantly from baseline

#### C. Environment Weight Adjustment (`env_weight_adj`)

Adjusts environment component weighting based on AQI patterns:

```
if bad_aqi_ratio > 0.5:  # >50% of recent readings are bad AQI
    env_weight_adj = 0.8  # Reduce by 20%
elif bad_aqi_ratio > 0.3:  # >30% bad AQI
    env_weight_adj = 0.9  # Reduce by 10%
else:
    env_weight_adj = 1.0  # No adjustment
```

**Bad AQI Definition:** AQI > 100 (unhealthy for sensitive groups)

**Logic:**
- If AQI is consistently poor, reduce environment component influence
- Prevents environment from dominating CAV when conditions are chronically bad
- Evaluates last 6 hours of data

## API Endpoints

### POST `/cav`

Compute CAV score with adaptive adjustments.

**Request:** Same as before (CAVRequest)

**Response:** Enhanced CAVResponse with `adaptive` field:

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
    "sensitivity": 1.0,
    "env_weight_adj": 0.88
  }
}
```

### GET `/memory/summary`

Get 24-hour summary of memory statistics.

**Response:**

```json
{
  "total_records": 1234,
  "window_hours": 24,
  "hourly_stats": {
    "12": {
      "cav_mean": 5123.4,
      "cav_std": 1234.5,
      "state_probs": {
        "overload": 0.15,
        "balanced": 0.35,
        "focus": 0.30,
        "restorative": 0.20
      }
    },
    ...
  },
  "overall_stats": {
    "cav_mean": 5234.1,
    "cav_std": 1456.7,
    "state_distribution": {
      "overload": 0.12,
      "balanced": 0.38,
      "focus": 0.32,
      "restorative": 0.18
    }
  }
}
```

### POST `/memory/clear`

Clear all memory (buffer and database).

**WARNING:** This deletes all stored CAV history. Use for testing or resetting.

**Response:**

```json
{
  "status": "success",
  "message": "Memory cleared successfully"
}
```

## Algorithm Details

### EWMA Parameters

- **Alpha (α)**: 0.3
  - Balances responsiveness vs. stability
  - Lower α = more stable, slower adaptation
  - Higher α = more responsive, faster adaptation

### Z-Score Thresholds

- **Normal**: `|z| < 1.0`
- **Moderate**: `1.0 ≤ |z| < 1.5`
- **Significant**: `|z| ≥ 1.5` (triggers sensitivity adjustment)

### Environment Adjustment Logic

- **Evaluation Window**: Last 6 hours
- **Bad AQI Threshold**: AQI > 100
- **Adjustment Levels**:
  - >50% bad AQI: 20% reduction
  - >30% bad AQI: 10% reduction
  - Otherwise: No adjustment

## OEM Benefits

### 1. Personalized Baselines

Each user's CAV patterns are learned over time, creating personalized baselines that account for:
- Individual physiological patterns
- Circadian rhythms
- Environmental adaptation

### 2. Contextual Adaptation

The system adapts to:
- **Time of day**: Hourly baselines capture circadian patterns
- **Environmental conditions**: Adjusts weighting when conditions are chronically poor
- **Physiological deviations**: Increases sensitivity during significant deviations

### 3. Monitoring & Insights

OEMs can monitor adaptation through `/memory/summary`:
- Track hourly patterns
- Identify state distribution trends
- Monitor baseline stability
- Detect anomalies

### 4. Production-Ready

- **Efficient**: In-memory buffer for fast access
- **Persistent**: SQLite database for reliability
- **Scalable**: Automatic cleanup prevents unbounded growth
- **Thread-safe**: Single shared instance across requests

## Usage Examples

### Basic CAV Request

```python
import requests

response = requests.post("http://localhost:8000/cav", json={
    "EDA": [...240 samples...],
    "TEMP": [...240 samples...],
    "BVP": [...240 samples...],
    "ACC_x": [...240 samples...],
    "ACC_y": [...240 samples...],
    "ACC_z": [...240 samples...],
    "temp_c": 24.0,
    "humidity": 50.0,
    "aqi": 42,
    "local_hour": 12
})

result = response.json()
print(f"CAV: {result['cav_smooth']}")
print(f"State: {result['state']}")
print(f"Z-score: {result['adaptive']['z_cav']}")
print(f"Sensitivity: {result['adaptive']['sensitivity']}")
```

### Monitor Memory

```python
import requests

summary = requests.get("http://localhost:8000/memory/summary").json()
print(f"Total records: {summary['total_records']}")
print(f"Overall CAV mean: {summary['overall_stats']['cav_mean']}")
print(f"State distribution: {summary['overall_stats']['state_distribution']}")
```

### Reset Memory

```python
import requests

response = requests.post("http://localhost:8000/memory/clear")
print(response.json()["message"])
```

## Database Schema

The SQLite database (`data/memory.db`) contains:

**Table: `cav_memory`**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| timestamp | REAL | Unix timestamp |
| cav_raw | INTEGER | Raw CAV score |
| cav_smooth | INTEGER | Smoothed CAV score |
| state | TEXT | State string |
| parts_bio | REAL | Biological component |
| parts_env | REAL | Environmental component |
| parts_circadian | REAL | Circadian component |
| parts_p_stress | REAL | Stress probability |
| temp_c | REAL | Temperature |
| humidity | REAL | Humidity |
| aqi | INTEGER | Air Quality Index |
| local_hour | INTEGER | Hour [0-23] |

**Index:** `idx_timestamp` on `timestamp` for efficient time-based queries.

## Performance Considerations

- **Memory Buffer**: Max 10,000 records (approximately 24 hours at 4 Hz)
- **Database Cleanup**: Automatic removal of records older than 7 days
- **Statistics Update**: Every 10 records or hourly (whichever comes first)
- **Query Performance**: Indexed timestamp column for fast time-based queries

## Future Enhancements

Potential extensions:
- Multi-user support (user-specific baselines)
- Advanced anomaly detection
- Predictive modeling
- Customizable EWMA parameters
- Environment band classification (good/moderate/bad)

## Technical Notes

- **Thread Safety**: Single shared instance ensures thread-safe operations
- **Error Handling**: Graceful degradation if memory operations fail
- **Backward Compatibility**: `adaptive` field is optional in response
- **PEP8 Compliant**: Follows Python style guidelines
- **Production Ready**: Comprehensive error handling and logging

