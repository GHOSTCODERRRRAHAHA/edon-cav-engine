"""FastAPI microservice for EDON CAV."""

import os
import json
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding import CAVEmbedder, cosine_similarity

load_dotenv()

app = FastAPI(
    title="EDON CAV API",
    description="Context-Aware Vectors microservice for generating and querying 128-D embeddings",
    version="0.1.0"
)

# Global state
embedder: Optional[CAVEmbedder] = None
cav_data: List[Dict] = []
cav_embeddings: np.ndarray = None


# Pydantic models
class BioFeatures(BaseModel):
    hr: float = Field(..., description="Heart rate in BPM")
    hrv_rmssd: float = Field(..., description="HRV RMSSD in ms")
    eda_mean: float = Field(..., description="EDA mean in μS")
    eda_var: float = Field(..., description="EDA variance")
    resp_bpm: float = Field(..., description="Respiration rate in BPM")
    accel_mag: float = Field(..., description="Accelerometer magnitude in g")


class EnvFeatures(BaseModel):
    temp_c: float = Field(..., description="Temperature in Celsius")
    humidity: int = Field(..., description="Humidity percentage [0-100]")
    cloud: int = Field(..., description="Cloud coverage [0-100]")
    aqi: int = Field(..., description="Air Quality Index")
    pm25: float = Field(..., description="PM2.5 concentration")
    ozone: float = Field(..., description="Ozone concentration")
    hour: int = Field(..., description="Hour of day [0-23]")
    is_daylight: int = Field(..., description="Daylight flag [0 or 1]")


class GenerateCAVRequest(BaseModel):
    bio: BioFeatures
    env: EnvFeatures


class GenerateCAVResponse(BaseModel):
    cav128: List[float] = Field(..., description="128-dimensional CAV embedding")


class SimilarRequest(BaseModel):
    cav128: List[float] = Field(..., description="128-dimensional query vector")
    k: int = Field(5, ge=1, le=100, description="Number of nearest neighbors")


class SimilarResponse(BaseModel):
    results: List[Dict] = Field(..., description="Top-k similar records with similarity scores")


@app.on_event("startup")
async def startup_event():
    """Load models and data on startup."""
    global embedder, cav_data, cav_embeddings
    
    # Load embedder
    model_dir = os.getenv("MODEL_DIR", "models")
    embedder = CAVEmbedder(n_components=128, model_dir=model_dir)
    
    try:
        embedder.load()
        print(f"✓ Loaded embedding models from {model_dir}/")
    except FileNotFoundError:
        print(f"⚠ Warning: Models not found in {model_dir}/. Run build-cav first.")
        embedder = None
    
    # Load CAV data if available
    data_path = os.getenv("CAV_DATA_PATH", "data/edon_cav.json")
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            cav_data = json.load(f)
        cav_embeddings = np.array([r['cav128'] for r in cav_data])
        print(f"✓ Loaded {len(cav_data)} CAV records from {data_path}")
    else:
        print(f"⚠ Warning: CAV data not found at {data_path}. Similarity search will be unavailable.")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "EDON CAV API",
        "version": "0.1.0",
        "endpoints": {
            "POST /generate_cav": "Generate 128-D embedding from features",
            "POST /similar": "Find similar CAV vectors",
            "GET /sample": "Get random sample records"
        }
    }


@app.post("/generate_cav", response_model=GenerateCAVResponse)
async def generate_cav(request: GenerateCAVRequest):
    """
    Generate a 128-dimensional CAV embedding from physiological and environmental features.
    
    Returns the embedding vector that can be used for similarity search.
    """
    if embedder is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model not loaded. Run build-cav first."
        )
    
    # Create feature DataFrame
    import pandas as pd
    feature_df = pd.DataFrame([{
        'hr': request.bio.hr,
        'hrv_rmssd': request.bio.hrv_rmssd,
        'eda_mean': request.bio.eda_mean,
        'eda_var': request.bio.eda_var,
        'resp_bpm': request.bio.resp_bpm,
        'accel_mag': request.bio.accel_mag,
        'temp_c': request.env.temp_c,
        'humidity': request.env.humidity,
        'cloud': request.env.cloud,
        'aqi': request.env.aqi,
        'pm25': request.env.pm25,
        'ozone': request.env.ozone,
        'hour': request.env.hour,
        'is_daylight': request.env.is_daylight
    }])
    
    # Generate embedding
    embedding = embedder.transform(feature_df)[0]
    
    return GenerateCAVResponse(cav128=embedding.tolist())


@app.post("/similar", response_model=SimilarResponse)
async def find_similar(request: SimilarRequest):
    """
    Find the top-k most similar CAV vectors using cosine similarity.
    
    Returns the most similar records with their similarity scores.
    """
    if cav_embeddings is None or len(cav_data) == 0:
        raise HTTPException(
            status_code=503,
            detail="CAV data not loaded. Run build-cav first."
        )
    
    if len(request.cav128) != 128:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 128-dimensional vector, got {len(request.cav128)}"
        )
    
    # Convert query to numpy array
    query_vec = np.array(request.cav128)
    
    # Compute cosine similarities
    similarities = np.dot(cav_embeddings, query_vec)  # Already L2 normalized
    
    # Get top-k indices
    top_k_indices = np.argsort(similarities)[::-1][:request.k]
    
    # Build results
    results = []
    for idx in top_k_indices:
        record = cav_data[idx].copy()
        record['similarity'] = float(similarities[idx])
        results.append(record)
    
    return SimilarResponse(results=results)


@app.get("/sample")
async def get_sample(n: int = 5):
    """
    Get random sample records from the CAV dataset.
    
    Args:
        n: Number of samples to return (default: 5, max: 100)
    """
    if len(cav_data) == 0:
        raise HTTPException(
            status_code=503,
            detail="CAV data not loaded. Run build-cav first."
        )
    
    n = min(n, 100, len(cav_data))
    
    # Random sampling
    import random
    samples = random.sample(cav_data, n)
    
    return {"samples": samples, "total": len(cav_data)}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "embedder_loaded": embedder is not None,
        "data_loaded": len(cav_data) > 0,
        "data_count": len(cav_data)
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

