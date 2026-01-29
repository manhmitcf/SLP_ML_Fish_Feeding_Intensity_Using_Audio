import numpy as np
import librosa
from .base import BaseFeatureExtractor
from .config import MFCCConfig

class MFCCExtractor(BaseFeatureExtractor):
    """
    Feature Extractor for Mel-frequency cepstral coefficients (MFCCs).
    Supports full customization of librosa.feature.mfcc parameters.
    """

    def __init__(self, config: MFCCConfig):
        if not isinstance(config, MFCCConfig):
            raise TypeError("Config must be an instance of MFCCConfig")
        super().__init__(config)
        self.config = config

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """
        Computes MFCCs from the audio signal using the detailed configuration.
        Returns the mean of MFCCs across time (Global Average Pooling).
        Output shape: (n_mfcc,)
        """
        
        # Prepare arguments for melspectrogram (passed as kwargs to mfcc)
        melspec_kwargs = {
            'n_fft': self.config.n_fft,
            'hop_length': self.config.hop_length,
            'win_length': self.config.win_length,
            'window': self.config.window,
            'center': self.config.center,
            'pad_mode': self.config.pad_mode,
            'power': self.config.power,
            'n_mels': self.config.n_mels,
            'fmin': self.config.fmin,
            'fmax': self.config.fmax,
            'htk': self.config.htk
        }

        # Filter out None values (e.g., win_length, fmax) to let librosa use defaults
        melspec_kwargs = {k: v for k, v in melspec_kwargs.items() if v is not None}

        # Compute MFCC
        mfcc = librosa.feature.mfcc(
            y=signal, 
            sr=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            dct_type=self.config.dct_type,
            norm=self.config.norm,
            lifter=self.config.lifter,
            **melspec_kwargs
        )
        
        # Global Average Pooling (Mean across time)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        return mfcc_mean
