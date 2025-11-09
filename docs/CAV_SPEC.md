# EDON CAV (Context-Aware Vectors) Specification

## Overview

EDON CAV generates 128-dimensional embeddings that encode physiological, environmental, and contextual signals into a unified vector space. These vectors enable similarity search, clustering, and downstream applications in context-aware systems.

## JSON Schema

Each CAV record follows this structure:

```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "geo": {
    "lat": 40.7128,
    "lon": -74.0060
  },
  "emotion": {
    "valence": 0.65,
    "arousal": 0.42
  },
  "bio": {
    "hr": 72.5,
    "hrv_rmssd": 45.2,
    "eda_mean": 2.3,
    "eda_var": 0.8,
    "resp_bpm": 16.0,
    "accel_mag": 1.2
  },
  "env": {
    "temp_c": 22.5,
    "humidity": 65,
    "cloud": 30,
    "aqi": 45,
    "pm25": 12.3,
    "ozone": 0.08,
    "hour": 14,
    "is_daylight": 1
  },
  "activity": "walking",
  "cav128": [0.123, -0.456, ..., 0.789]
}
```

## Field Definitions

### Root Level

- **timestamp** (string, ISO8601): UTC timestamp of the measurement
- **geo** (object): Geographic coordinates
  - `lat` (float): Latitude in degrees [-90, 90]
  - `lon` (float): Longitude in degrees [-180, 180]
- **emotion** (object): Derived emotional state (optional, can be inferred)
  - `valence` (float): Valence score [0.0, 1.0]
  - `arousal` (float): Arousal score [0.0, 1.0]
- **bio** (object): Physiological measurements
  - `hr` (float): Heart rate in BPM
  - `hrv_rmssd` (float): Heart rate variability (RMSSD) in ms
  - `eda_mean` (float): Electrodermal activity mean (μS)
  - `eda_var` (float): Electrodermal activity variance
  - `resp_bpm` (float): Respiration rate in breaths per minute
  - `accel_mag` (float): Accelerometer magnitude (g)
- **env** (object): Environmental context
  - `temp_c` (float): Temperature in Celsius
  - `humidity` (int): Relative humidity percentage [0, 100]
  - `cloud` (int): Cloud coverage percentage [0, 100]
  - `aqi` (int): Air Quality Index
  - `pm25` (float): PM2.5 concentration (μg/m³)
  - `ozone` (float): Ozone concentration (ppm)
  - `hour` (int): Hour of day [0, 23]
  - `is_daylight` (int): Binary flag (1 = daylight, 0 = night)
- **activity** (string): Activity label (e.g., "walking", "sitting", "running", "standing")
- **cav128** (array[float]): 128-dimensional embedding vector

## Data Sources

### Physiological Data
- **WESAD**: Stress vs neutral states, HR, EDA, respiration, accelerometer
- **MobiAct/WISDM**: Activity classification labels

### Environmental Data
- **OpenWeatherMap API**: Temperature, humidity, cloud coverage
- **AirNow API (EPA)**: AQI, PM2.5, ozone
- **WorldTimeAPI**: Hour, daylight status

## Feature Extraction

### HRV (RMSSD)
Root Mean Square of Successive Differences between R-R intervals:
```
RMSSD = sqrt(mean((RR[i+1] - RR[i])²))
```

### EDA Statistics
- Mean: Average electrodermal activity over window
- Variance: Variability measure

### Accelerometer Magnitude
```
mag = sqrt(accel_x² + accel_y² + accel_z²)
```

## Embedding Generation

1. **Feature Normalization**: StandardScaler (zero mean, unit variance)
2. **Dimensionality Reduction**: PCA to 128 dimensions
3. **Vector Storage**: Normalized L2 vectors for cosine similarity

## Similarity Metric

Cosine similarity is used for nearest neighbor search:
```
similarity = (A · B) / (||A|| × ||B||)
```

## Version

- **v0.1.0**: Initial specification with PCA-based embeddings
- **v0.2.0** (planned): Autoencoder-based embeddings

## License Notes

- WESAD: Research use, cite original paper
- MobiAct/WISDM: Public datasets, check individual licenses
- OpenWeatherMap: Free tier available, commercial use requires subscription
- AirNow: Public domain, no restrictions
- WorldTimeAPI: Free, no API key required

