import numpy as np
import librosa
from .base import BaseFeatureExtractor
from .config import LPCConfig

class LPCExtractor(BaseFeatureExtractor):
    def __init__(self, config: LPCConfig):
        super().__init__(config)
        self.config = config

    def extract(self, signal: np.ndarray) -> np.ndarray:
        # LPC calculation
        # librosa.lpc takes the signal and order
        lpc_coeffs = librosa.lpc(signal, order=self.config.order)
        # Skip the first coefficient (always 1)
        return lpc_coeffs[1:]
