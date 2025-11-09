"""Integration tests for the full pipeline."""

import pytest
import os
import json
import tempfile
import shutil
from src.pipeline import build_cav_dataset


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for data and models."""
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    model_dir.mkdir()
    
    yield {
        "data": str(data_dir),
        "models": str(model_dir)
    }
    
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_build_cav_dataset_small(temp_dirs):
    """Test building a small CAV dataset."""
    output_path = os.path.join(temp_dirs["data"], "test_cav.json")
    
    df = build_cav_dataset(
        n_samples=100,
        output_path=output_path,
        model_dir=temp_dirs["models"]
    )
    
    # Check output file exists
    assert os.path.exists(output_path)
    
    # Load and verify
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    assert len(data) == 100
    assert all("cav128" in record for record in data)
    assert all(len(record["cav128"]) == 128 for record in data)
    
    # Check required fields
    required_fields = ["timestamp", "geo", "bio", "env", "activity", "cav128"]
    for record in data:
        for field in required_fields:
            assert field in record
    
    # Check models were saved
    assert os.path.exists(os.path.join(temp_dirs["models"], "scaler.joblib"))
    assert os.path.exists(os.path.join(temp_dirs["models"], "pca.joblib"))


def test_build_cav_dataset_embeddings_normalized(temp_dirs):
    """Test that embeddings are properly normalized."""
    import numpy as np
    
    output_path = os.path.join(temp_dirs["data"], "test_cav.json")
    
    build_cav_dataset(
        n_samples=50,
        output_path=output_path,
        model_dir=temp_dirs["models"]
    )
    
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    embeddings = np.array([record["cav128"] for record in data])
    
    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

