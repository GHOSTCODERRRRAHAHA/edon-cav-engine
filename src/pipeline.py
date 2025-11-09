"""Main pipeline for building CAV dataset."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os

from .features import extract_wesad_features
from .api_clients import get_weather_data, get_air_quality, get_circadian_data, get_activity_label
from .embedding import CAVEmbedder


def build_cav_dataset(
    n_samples: int = 10000,
    output_path: str = "data/edon_cav.json",
    wesad_data: Optional[Dict] = None,
    lat: float = 40.7128,  # NYC default
    lon: float = -74.0060,
    model_dir: str = "models"
) -> pd.DataFrame:
    """
    Build CAV dataset from physiological and environmental data.
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save JSON output
        wesad_data: Optional WESAD dataset dictionary
        lat: Latitude for environmental data
        lon: Longitude for environmental data
        model_dir: Directory for saving models
        
    Returns:
        DataFrame with CAV records
    """
    print(f"Building CAV dataset with {n_samples} samples...")
    
    # Step 1: Extract physiological features
    print("Step 1: Extracting physiological features...")
    bio_df = extract_wesad_features(wesad_data or {}, window_size=60)
    
    # If we don't have enough samples, replicate with noise
    if len(bio_df) < n_samples:
        n_replicate = (n_samples // len(bio_df)) + 1
        bio_df = pd.concat([bio_df] * n_replicate, ignore_index=True)
        # Add small noise to avoid exact duplicates
        for col in bio_df.columns:
            bio_df[col] += np.random.normal(0, bio_df[col].std() * 0.01, len(bio_df))
    
    bio_df = bio_df.head(n_samples).copy()
    
    # Step 2: Fetch environmental data (with caching to avoid rate limits)
    print("Step 2: Fetching environmental data...")
    env_data_cache = {}
    
    def get_env_data(idx: int) -> Dict:
        # Cache environmental data (update every 100 samples to simulate time progression)
        cache_key = idx // 100
        if cache_key not in env_data_cache:
            # Simulate slight location variation
            lat_offset = np.random.uniform(-0.1, 0.1)
            lon_offset = np.random.uniform(-0.1, 0.1)
            env_lat = lat + lat_offset
            env_lon = lon + lon_offset
            
            weather = get_weather_data(env_lat, env_lon)
            air = get_air_quality(env_lat, env_lon)
            circadian = get_circadian_data(env_lat, env_lon)
            
            env_data_cache[cache_key] = {**weather, **air, **circadian}
        
        return env_data_cache[cache_key].copy()
    
    # Step 3: Combine features
    print("Step 3: Combining features...")
    records = []
    
    base_time = datetime.now() - timedelta(days=30)  # Start 30 days ago
    
    for idx in range(n_samples):
        # Get environmental data
        env = get_env_data(idx)
        
        # Get bio features
        bio = bio_df.iloc[idx].to_dict()
        
        # Determine activity
        activity = get_activity_label(bio['accel_mag'], bio['hr'])
        
        # Create timestamp
        timestamp = (base_time + timedelta(minutes=idx * 5)).isoformat() + "Z"
        
        # Create record
        record = {
            "timestamp": timestamp,
            "geo": {"lat": lat, "lon": lon},
            "emotion": {
                "valence": np.random.uniform(0.3, 0.8),  # Placeholder - would be derived from signals
                "arousal": np.random.uniform(0.2, 0.7)
            },
            "bio": {
                "hr": float(bio['hr']),
                "hrv_rmssd": float(bio['hrv_rmssd']),
                "eda_mean": float(bio['eda_mean']),
                "eda_var": float(bio.get('eda_var', 0.0)),
                "resp_bpm": float(bio['resp_bpm']),
                "accel_mag": float(bio['accel_mag'])
            },
            "env": {
                "temp_c": float(env['temp_c']),
                "humidity": int(env['humidity']),
                "cloud": int(env['cloud']),
                "aqi": int(env['aqi']),
                "pm25": float(env['pm25']),
                "ozone": float(env['ozone']),
                "hour": int(env['hour']),
                "is_daylight": int(env['is_daylight'])
            },
            "activity": activity
        }
        
        records.append(record)
    
    # Step 4: Create feature matrix for embedding
    print("Step 4: Generating embeddings...")
    feature_df = pd.DataFrame([
        {
            'hr': r['bio']['hr'],
            'hrv_rmssd': r['bio']['hrv_rmssd'],
            'eda_mean': r['bio']['eda_mean'],
            'eda_var': r['bio']['eda_var'],
            'resp_bpm': r['bio']['resp_bpm'],
            'accel_mag': r['bio']['accel_mag'],
            'temp_c': r['env']['temp_c'],
            'humidity': r['env']['humidity'],
            'cloud': r['env']['cloud'],
            'aqi': r['env']['aqi'],
            'pm25': r['env']['pm25'],
            'ozone': r['env']['ozone'],
            'hour': r['env']['hour'],
            'is_daylight': r['env']['is_daylight']
        }
        for r in records
    ])
    
    # Generate embeddings
    embedder = CAVEmbedder(n_components=128, model_dir=model_dir)
    embeddings = embedder.fit_transform(feature_df)
    
    # Add embeddings to records
    for i, record in enumerate(records):
        record['cav128'] = embeddings[i].tolist()
    
    # Step 5: Save to JSON
    print(f"Step 5: Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"✓ Generated {len(records)} CAV records")
    print(f"✓ Saved to {output_path}")
    print(f"✓ Embedding models saved to {model_dir}/")
    
    return pd.DataFrame(records)

