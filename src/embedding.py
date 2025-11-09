"""CAV embedding generation using PCA."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Dict, Optional
import joblib
import os


class CAVEmbedder:
    """Generate 128-dimensional CAV embeddings from features."""
    
    def __init__(self, n_components: int = 128, model_dir: str = "models"):
        """
        Initialize the embedder.
        
        Args:
            n_components: Target dimensionality (default 128)
            model_dir: Directory to save/load models
        """
        self.n_components = n_components
        self.model_dir = model_dir
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
    
    def fit(self, features_df: pd.DataFrame) -> None:
        """
        Fit scaler and PCA on training features.
        
        Args:
            features_df: DataFrame with feature columns
        """
        # Select feature columns (exclude metadata)
        feature_cols = [
            'hr', 'hrv_rmssd', 'eda_mean', 'eda_var', 'resp_bpm', 'accel_mag',
            'temp_c', 'humidity', 'cloud', 'aqi', 'pm25', 'ozone', 'hour', 'is_daylight'
        ]
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in features_df.columns]
        
        if len(available_cols) == 0:
            raise ValueError("No valid feature columns found")
        
        X = features_df[available_cols].values
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        self.pca.fit(X_scaled)
        
        self.is_fitted = True
        
        # Save models
        self.save()
    
    def transform(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Transform features to 128-D embeddings.
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            Array of shape (n_samples, 128)
        """
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted before transform")
        
        # Select feature columns
        feature_cols = [
            'hr', 'hrv_rmssd', 'eda_mean', 'eda_var', 'resp_bpm', 'accel_mag',
            'temp_c', 'humidity', 'cloud', 'aqi', 'pm25', 'ozone', 'hour', 'is_daylight'
        ]
        
        available_cols = [col for col in feature_cols if col in features_df.columns]
        X = features_df[available_cols].values
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Transform with PCA
        embeddings = self.pca.transform(X_scaled)
        
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings
    
    def fit_transform(self, features_df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features_df)
        return self.transform(features_df)
    
    def save(self) -> None:
        """Save scaler and PCA models."""
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        pca_path = os.path.join(self.model_dir, "pca.joblib")
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.pca, pca_path)
    
    def load(self) -> None:
        """Load scaler and PCA models."""
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        pca_path = os.path.join(self.model_dir, "pca.joblib")
        
        if not os.path.exists(scaler_path) or not os.path.exists(pca_path):
            raise FileNotFoundError("Model files not found. Run fit() first.")
        
        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        self.is_fitted = True


def generate_cav_from_features(features: Dict) -> List[float]:
    """
    Generate a single CAV embedding from feature dictionary.
    
    This is a convenience function that requires a pre-fitted embedder.
    
    Args:
        features: Dictionary with feature values
        
    Returns:
        128-dimensional embedding vector
    """
    # This would typically use a loaded embedder
    # For now, return a placeholder
    raise NotImplementedError("Use CAVEmbedder class with fit/transform")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score [-1, 1]
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have same shape")
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))

