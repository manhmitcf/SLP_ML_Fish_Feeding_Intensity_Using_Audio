import numpy as np
import librosa
from .base import BaseFeatureExtractor
from .config import STFTConfig

class STFTExtractor(BaseFeatureExtractor):
    """
    Feature Extractor for Short-Time Fourier Transform (STFT).
    Supports advanced scaling (Log, dB) and pooling (Mean, Max, Mean+Std).
    """

    def __init__(self, config: STFTConfig):
        if not isinstance(config, STFTConfig):
            raise TypeError("Config must be an instance of STFTConfig")
        super().__init__(config)
        self.config = config

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """
        Computes STFT features with configured post-processing.
        """
        # Convert string dtype from config to numpy dtype object
        dtype = np.dtype(self.config.dtype) if self.config.dtype else None

        # 1. Compute STFT (Complex)
        stft_matrix = librosa.stft(
            y=signal,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.config.window,
            center=self.config.center,
            pad_mode=self.config.pad_mode,
            dtype=dtype
        )
        
        # 2. Get Magnitude
        magnitude = np.abs(stft_matrix)
        
        # 3. Apply Scaling
        if self.config.scaling == 'log':
            magnitude = np.log(magnitude + 1e-8)
        elif self.config.scaling == 'db':
            magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # 4. Apply Pooling
        if self.config.pooling == 'mean':
            feature = np.mean(magnitude, axis=1)
        elif self.config.pooling == 'max':
            feature = np.max(magnitude, axis=1)
        elif self.config.pooling == 'mean_std':
            mean_feat = np.mean(magnitude, axis=1)
            std_feat = np.std(magnitude, axis=1)
            feature = np.concatenate([mean_feat, std_feat])
        else:
            feature = np.mean(magnitude, axis=1)
            
        return feature
