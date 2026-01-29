from abc import ABC, abstractmethod
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

class BaseProcessor(ABC):
    """
    Abstract Base Class for all processors (Scaler, PCA, Selector, etc.).
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the processor to the data.
        y is optional, required for supervised methods like LDA or SelectKBest.
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def save(self, path: Path):
        """Save the fitted model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path):
        """Load the fitted model from disk."""
        pass
