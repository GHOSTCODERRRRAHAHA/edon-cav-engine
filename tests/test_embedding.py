"""Tests for embedding generation."""

import numpy as np
import pandas as pd
import pytest
import os
import shutil
from src.embedding import CAVEmbedder, cosine_similarity


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame."""
    return pd.DataFrame({
        'hr': np.random.normal(72, 10, 100),
        'hrv_rmssd': np.random.normal(45, 15, 100),
        'eda_mean': np.random.normal(2.5, 0.8, 100),
        'eda_var': np.random.normal(0.5, 0.3, 100),
        'resp_bpm': np.random.normal(16, 3, 100),
        'accel_mag': np.random.normal(1.0, 0.5, 100),
        'temp_c': np.random.normal(22, 5, 100),
        'humidity': np.random.randint(30, 80, 100),
        'cloud': np.random.randint(0, 100, 100),
        'aqi': np.random.randint(20, 100, 100),
        'pm25': np.random.normal(15, 5, 100),
        'ozone': np.random.normal(0.05, 0.02, 100),
        'hour': np.random.randint(0, 24, 100),
        'is_daylight': np.random.randint(0, 2, 100)
    })


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    yield str(model_dir)
    shutil.rmtree(model_dir, ignore_errors=True)


def test_embedder_fit_transform(sample_features, temp_model_dir):
    """Test embedder fit and transform."""
    embedder = CAVEmbedder(n_components=128, model_dir=temp_model_dir)
    
    embeddings = embedder.fit_transform(sample_features)
    
    assert embeddings.shape == (100, 128)
    assert embedder.is_fitted
    
    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedder_save_load(sample_features, temp_model_dir):
    """Test model saving and loading."""
    embedder = CAVEmbedder(n_components=128, model_dir=temp_model_dir)
    embedder.fit(sample_features)
    
    # Save
    embedder.save()
    
    # Load in new embedder
    embedder2 = CAVEmbedder(n_components=128, model_dir=temp_model_dir)
    embedder2.load()
    
    # Transform should produce same results
    embeddings1 = embedder.transform(sample_features)
    embeddings2 = embedder2.transform(sample_features)
    
    assert np.allclose(embeddings1, embeddings2, atol=1e-5)


def test_cosine_similarity():
    """Test cosine similarity computation."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    
    sim = cosine_similarity(vec1, vec2)
    assert sim == pytest.approx(1.0)
    
    vec3 = np.array([0.0, 1.0, 0.0])
    sim = cosine_similarity(vec1, vec3)
    assert sim == pytest.approx(0.0)
    
    # Test with normalized vectors
    vec4 = np.array([1.0, 1.0, 0.0])
    vec4 = vec4 / np.linalg.norm(vec4)
    sim = cosine_similarity(vec1, vec4)
    assert 0 < sim < 1


def test_embedder_shape_consistency(sample_features, temp_model_dir):
    """Test that embeddings have consistent shapes."""
    embedder = CAVEmbedder(n_components=128, model_dir=temp_model_dir)
    embedder.fit(sample_features)
    
    # Transform single sample
    single_sample = sample_features.iloc[[0]]
    embedding = embedder.transform(single_sample)
    
    assert embedding.shape == (1, 128)
    
    # Transform multiple samples
    multi_sample = sample_features.iloc[:5]
    embeddings = embedder.transform(multi_sample)
    
    assert embeddings.shape == (5, 128)

