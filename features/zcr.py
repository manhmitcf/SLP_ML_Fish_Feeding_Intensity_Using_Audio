import numpy as np
import librosa
from .base import BaseFeatureExtractor
from .config import ZCRConfig

class ZCRExtractor(BaseFeatureExtractor):
    def __init__(self, config: ZCRConfig):
        super().__init__(config)
        self.config = config

    def extract(self, signal: np.ndarray) -> np.ndarray:
        zcr = librosa.feature.zero_crossing_rate(
            y=signal, 
            frame_length=self.config.frame_length, 
            hop_length=self.config.hop_length,
            center=self.config.center
        )
        return np.mean(zcr, axis=1)
