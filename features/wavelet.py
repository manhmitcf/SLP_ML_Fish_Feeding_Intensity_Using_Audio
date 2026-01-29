import numpy as np
import pywt
from .base import BaseFeatureExtractor
from .config import WaveletConfig

class WaveletExtractor(BaseFeatureExtractor):
    def __init__(self, config: WaveletConfig):
        super().__init__(config)
        self.config = config

    def extract(self, signal: np.ndarray) -> np.ndarray:
        # Decompose signal
        coeffs = pywt.wavedec(
            signal, 
            self.config.wavelet, 
            mode=self.config.mode, 
            level=self.config.level
        )
        
        # Calculate statistics for each level's coefficients
        features = []
        for c in coeffs:
            features.extend([np.mean(c), np.std(c), np.max(c), np.min(c)])
            
        return np.array(features)
