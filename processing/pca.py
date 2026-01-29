import numpy as np
import joblib
from pathlib import Path
from typing import Optional
from sklearn.decomposition import PCA
from .base import BaseProcessor
from .config import PCAConfig

class PCAProcessor(BaseProcessor):
    def __init__(self, config: PCAConfig):
        self.config = config
        self.model = None
        
        if config.n_components is not None:
            self.model = PCA(n_components=config.n_components, whiten=config.whiten)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit PCA. 'y' is ignored but accepted for pipeline compatibility.
        """
        if self.model is None:
            return # PCA disabled
            
        print(f"   [PCA] Fitting PCA (n_components={self.config.n_components})...")
        self.model.fit(X)
        
        # Log explained variance
        explained_variance = np.sum(self.model.explained_variance_ratio_)
        print(f"   [PCA] Explained Variance: {explained_variance:.4f}")
        print(f"   [PCA] Components selected: {self.model.n_components_}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return X # Pass-through
        return self.model.transform(X)

    def save(self, path: Path):
        if self.model:
            joblib.dump(self.model, path)

    def load(self, path: Path):
        if path.exists():
            self.model = joblib.load(path)
