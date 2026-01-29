import numpy as np
import librosa
from .base import BaseFeatureExtractor
from .config import CQTConfig

class CQTExtractor(BaseFeatureExtractor):
    def __init__(self, config: CQTConfig):
        super().__init__(config)
        self.config = config

    def extract(self, signal: np.ndarray) -> np.ndarray:
        cqt = librosa.cqt(
            y=signal,
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
            fmin=self.config.fmin,
            n_bins=self.config.n_bins,
            bins_per_octave=self.config.bins_per_octave,
            tuning=self.config.tuning,
            filter_scale=self.config.filter_scale,
            norm=self.config.norm,
            sparsity=self.config.sparsity,
            window=self.config.window,
            scale=self.config.scale,
            pad_mode=self.config.pad_mode
        )
        # Magnitude and Mean
        cqt_mag = np.abs(cqt)
        return np.mean(cqt_mag, axis=1)
