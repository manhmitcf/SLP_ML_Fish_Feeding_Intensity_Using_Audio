import numpy as np
import librosa
from .base import BaseFeatureExtractor
from .config import EnergyConfig

class EnergyExtractor(BaseFeatureExtractor):
    def __init__(self, config: EnergyConfig):
        super().__init__(config)
        self.config = config

    def extract(self, signal: np.ndarray) -> np.ndarray:
        # Calculate RMSE (Root Mean Square Energy)
        rmse = librosa.feature.rms(
            y=signal, 
            frame_length=self.config.frame_length, 
            hop_length=self.config.hop_length
        )
        # Return mean RMSE across frames
        return np.mean(rmse, axis=1)
