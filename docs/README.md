# EDON CAV (Context-Aware Vectors)

A lean data pipeline and micro-API for generating 128-dimensional CAV embeddings from physiological and environmental datasets.

## 🎯 Overview

EDON CAV combines:
- **Physiological signals** (HR, HRV, EDA, Respiration, Accelerometer) from WESAD
- **Environmental data** (Weather, Air Quality) from public APIs
- **Circadian context** (Time, Daylight) from WorldTimeAPI
- **Activity labels** from MobiAct/WISDM

All normalized and embedded into 128-dimensional vectors using PCA for similarity search and clustering.

## 📋 Requirements

- Python 3.11+
- API keys (optional, for real environmental data):
  - [OpenWeatherMap](https://openweathermap.org/api) (free tier available)
  - [AirNow](https://www.airnow.gov/developers/) (EPA, free)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd EDON

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys (optional)
```

### Build Dataset

```bash
# Using CLI
python cli.py build-cav --n 10000

# Or using Make
make build
```

This generates:
- `data/edon_cav.json` - 10,000+ CAV records
- `models/scaler.joblib` - Feature scaler
- `models/pca.joblib` - PCA model

### Launch API

```bash
# Using Make
make api

# Or directly
cd api && python main.py
```

API will be available at `http://localhost:8000`

### Run Tests

```bash
pytest tests/ -v
```

## 📚 Usage

### CLI

```bash
# Build dataset with custom parameters
python cli.py build-cav --n 50000 --output data/large_cav.json --lat 37.7749 --lon -122.4194

# Options:
#   --n          Number of samples (default: 10000)
#   --output     Output JSON path (default: data/edon_cav.json)
#   --lat        Latitude (default: 40.7128)
#   --lon        Longitude (default: -74.0060)
#   --model-dir  Model directory (default: models)
```

### API Endpoints

#### `POST /generate_cav`

Generate a 128-D embedding from features.

**Request:**
```json
{
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
  }
}
```

**Response:**
```json
{
  "cav128": [0.123, -0.456, ..., 0.789]
}
```

#### `POST /similar`

Find top-k similar vectors.

**Request:**
```json
{
  "cav128": [0.123, -0.456, ..., 0.789],
  "k": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "timestamp": "2024-01-15T14:30:00Z",
      "bio": {...},
      "env": {...},
      "activity": "walking",
      "cav128": [...],
      "similarity": 0.95
    },
    ...
  ]
}
```

#### `GET /sample?n=5`

Get random sample records.

#### `GET /health`

Health check endpoint.

### Python API

```python
from src.pipeline import build_cav_dataset
from src.embedding import CAVEmbedder
from src.api_clients import get_weather_data, get_air_quality

# Build dataset
df = build_cav_dataset(n_samples=10000)

# Generate embedding
embedder = CAVEmbedder(n_components=128, model_dir="models")
embedder.load()
embedding = embedder.transform(feature_df)
```

## 📁 Project Structure

```
EDON/
├── api/
│   └── main.py              # FastAPI service
├── src/
│   ├── features.py          # Feature extraction
│   ├── api_clients.py       # External API clients
│   ├── embedding.py         # PCA embedding generation
│   └── pipeline.py          # Main pipeline
├── notebooks/
│   └── 01_build_cav.ipynb   # Data pipeline notebook
├── tests/
│   ├── test_features.py
│   ├── test_embedding.py
│   └── test_api_clients.py
├── data/                    # Generated datasets
├── models/                  # Saved models
├── docs/
│   └── CAV_SPEC.md          # CAV specification
├── cli.py                   # CLI tool
├── requirements.txt
├── Makefile
├── Dockerfile
└── README.md
```

## 🔬 Data Pipeline

1. **Physiological Features**: Extract HR, HRV (RMSSD), EDA stats, respiration rate, accelerometer magnitude from WESAD
2. **Environmental Data**: Fetch weather, air quality, and circadian data from APIs
3. **Feature Normalization**: StandardScaler (zero mean, unit variance)
4. **Embedding**: PCA to 128 dimensions, L2 normalized
5. **Output**: JSON with full records + embeddings

## 📊 Data Schema

See [docs/CAV_SPEC.md](docs/CAV_SPEC.md) for complete schema definition.

Each record contains:
- `timestamp`: ISO8601 timestamp
- `geo`: Latitude/longitude
- `emotion`: Valence/arousal (optional)
- `bio`: Physiological features
- `env`: Environmental context
- `activity`: Activity label
- `cav128`: 128-dimensional embedding vector

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🐳 Docker

```bash
# Build image
docker build -t edon-cav .

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models edon-cav
```

## 📝 License Notes

- **WESAD**: Research use, cite original paper
- **MobiAct/WISDM**: Public datasets, check individual licenses
- **OpenWeatherMap**: Free tier available, commercial use requires subscription
- **AirNow**: Public domain, no restrictions
- **WorldTimeAPI**: Free, no API key required

## 🔒 Privacy

- Only derived features are stored (no personal identifiers)
- No raw signal data is persisted
- API keys stored in `.env` (never committed)

## 🎯 Acceptance Criteria

✅ `data/edon_cav.json` has ≥10,000 rows with valid 128-D vectors  
✅ API endpoints respond in <100ms locally  
✅ Cosine similarity returns logical neighbors  
✅ Codebase is modular, PCA easily swappable with autoencoder  
✅ Full reproducibility with requirements.txt and Dockerfile

## 🚀 Stretch Goals

- [ ] UMAP/t-SNE visualization notebook (`notebooks/02_visualize.ipynb`)
- [ ] Autoencoder-based embeddings (v0.2.0)
- [ ] POST `/recommend` endpoint with policy outputs
- [ ] Real-time streaming pipeline

## 📖 Documentation

- [CAV Specification](docs/CAV_SPEC.md)
- [API Documentation](http://localhost:8000/docs) (when server is running)

## 🤝 Contributing

This is a prototype. For production use, consider:
- Proper error handling and retries for API calls
- Database storage instead of JSON files
- Authentication for API endpoints
- More sophisticated feature engineering
- Autoencoder-based embeddings for better representations

## 📧 Contact

For questions or issues, please open an issue on the repository.

