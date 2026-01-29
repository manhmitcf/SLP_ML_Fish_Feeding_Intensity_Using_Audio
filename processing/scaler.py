import numpy as np
import joblib
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .base import BaseProcessor
from .config import ScalerConfig

class ScalerProcessor(BaseProcessor):
    def __init__(self, config: ScalerConfig):
        self.config = config
        self.model = None
        
        if config.type == 'standard':
            self.model = StandardScaler()
        elif config.type == 'minmax':
            self.model = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {config.type}")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the scaler. 'y' is ignored but accepted for pipeline compatibility.
        """
        print(f"   [Scaler] Fitting {self.config.type} scaler...")
        self.model.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Scaler not fitted yet!")
        return self.model.transform(X)

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)
