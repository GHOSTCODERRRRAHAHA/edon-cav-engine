"""PCA-based fusion for 128-dimensional CAV embeddings."""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
import warnings

warnings.filterwarnings("ignore")


class PCAFusion:
    """PCA-based fusion to 128-dimensional embeddings."""
    
    def __init__(self, n_components: int = 128, random_state: int = 42):
        """
        Initialize PCA fusion.
        
        Args:
            n_components: Target dimensionality (default 128)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.rproj: Optional[GaussianRandomProjection] = None
        self.feature_order: Optional[List[str]] = None
        self.is_fitted = False
    
    def fit(self, feature_vectors: List[Dict[str, float]]) -> None:
        """
        Fit PCA on feature vectors.
        
        Args:
            feature_vectors: List of feature dictionaries
        """
        if not feature_vectors:
            raise ValueError("Cannot fit on empty feature vectors")
        
        # Convert to DataFrame-like structure
        all_features = set()
        for fv in feature_vectors:
            all_features.update(fv.keys())
        
        self.feature_order = sorted(all_features)
        
        # Build matrix
        X = np.array([
            [fv.get(feat, 0.0) for feat in self.feature_order]
            for fv in feature_vectors
        ])
        
        if X.shape[0] < 2:
            # Need at least 2 samples to fit
            # Use identity-like transformation
            self.pca = None
            self.rproj = None
            self.is_fitted = True
            return
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle NaN/Inf
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # PCA - cannot exceed n_features
        n_features = X_scaled.shape[1]
        pca_dim = min(self.n_components, n_features, X_scaled.shape[0] - 1)
        
        if pca_dim > 0:
            self.pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=self.random_state)
            X_pca = self.pca.fit_transform(X_scaled)
            
            # If we need more dimensions, use random projection
            if self.n_components > pca_dim:
                self.rproj = GaussianRandomProjection(
                    n_components=self.n_components,
                    random_state=self.random_state
                )
                self.rproj.fit(X_pca)
            else:
                self.rproj = None
        else:
            # Fallback: use random projection directly
            self.pca = None
            self.rproj = GaussianRandomProjection(
                n_components=self.n_components,
                random_state=self.random_state
            )
            self.rproj.fit(X_scaled)
        
        self.is_fitted = True
    
    def transform(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Transform feature dictionary to 128-dim embedding.
        
        Args:
            feature_dict: Feature dictionary
            
        Returns:
            128-dimensional embedding vector
        """
        if not self.is_fitted:
            raise ValueError("PCA fusion not fitted. Call fit() first.")
        
        if self.feature_order is None:
            # Fallback: use features from dict
            features = sorted(feature_dict.keys())
            X = np.array([[feature_dict.get(f, 0.0) for f in features]])
        else:
            X = np.array([[
                feature_dict.get(feat, 0.0) for feat in self.feature_order
            ]])
        
        # Standardize
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply PCA
        if self.pca is not None:
            X_pca = self.pca.transform(X_scaled)
            
            # Apply random projection if needed
            if self.rproj is not None:
                X_embed = self.rproj.transform(X_pca)
            else:
                X_embed = X_pca
        elif self.rproj is not None:
            X_embed = self.rproj.transform(X_scaled)
        else:
            # Fallback: pad or truncate to target dimension
            if X_scaled.shape[1] < self.n_components:
                padding = np.zeros((1, self.n_components - X_scaled.shape[1]))
                X_embed = np.hstack([X_scaled, padding])
            else:
                X_embed = X_scaled[:, :self.n_components]
        
        # L2 normalize
        norm = np.linalg.norm(X_embed, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        X_embed = X_embed / norm
        
        return X_embed[0]  # Return first (and only) row
    
    def fit_transform(self, feature_vectors: List[Dict[str, float]]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(feature_vectors)
        if len(feature_vectors) == 1:
            return self.transform(feature_vectors[0])
        # For multiple vectors, return all
        return np.array([self.transform(fv) for fv in feature_vectors])


def create_default_pca_fusion() -> PCAFusion:
    """Create a default PCA fusion instance."""
    return PCAFusion(n_components=128, random_state=42)

