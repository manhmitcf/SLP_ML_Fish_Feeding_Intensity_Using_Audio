import numpy as np
import joblib
from pathlib import Path
from typing import Optional
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from .base import BaseProcessor
from .config import SelectionConfig

class SelectionProcessor(BaseProcessor):
    def __init__(self, config: SelectionConfig):
        self.config = config
        self.model = None
        
        if config.method == 'variance':
            self.model = VarianceThreshold(threshold=config.threshold)
        elif config.method == 'k_best':
            # Using f_classif for classification tasks
            self.model = SelectKBest(score_func=f_classif, k=config.k)
        else:
            raise ValueError(f"Unknown selection method: {config.method}")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        print(f"   [Selector] Fitting {self.config.method} selector...")
        if self.config.method == 'k_best' and y is None:
            raise ValueError("SelectKBest requires 'y' labels for fitting.")
        
        self.model.fit(X, y)
        
        # Log selected features
        if hasattr(self.model, 'get_support'):
            n_original = X.shape[1]
            n_selected = np.sum(self.model.get_support())
            print(f"   [Selector] Selected {n_selected} features out of {n_original}.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Selector not fitted yet!")
        return self.model.transform(X)

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)
