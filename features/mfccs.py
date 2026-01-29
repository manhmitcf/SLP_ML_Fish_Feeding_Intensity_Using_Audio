import numpy as np
import librosa
from .base import BaseFeatureExtractor
from .config import MFCCsConfig

class MFCCsExtractor(BaseFeatureExtractor):
    """
    Feature Extractor for MFCCs (MFCC with Pre-emphasis and manual log scaling).
    """

    def __init__(self, config: MFCCsConfig):
        if not isinstance(config, MFCCsConfig):
            raise TypeError("Config must be an instance of MFCCsConfig")
        super().__init__(config)
        self.config = config

    def _pre_emphasis(self, signal_in):
        """
        Apply pre-emphasis to emphasize high frequencies.
        """
        return np.append(signal_in[0], signal_in[1:] - self.config.pre_emph * signal_in[:-1])

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute MFCCs features with full parameter set.
        """
        # 1. Pre-emphasis
        emphasized_signal = self._pre_emphasis(signal)
        
        # Prepare kwargs for melspectrogram part of mfcc
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
        melspec_kwargs = {k: v for k, v in melspec_kwargs.items() if v is not None}
        
        # 2. Compute MFCC
        mfcc_matrix = librosa.feature.mfcc(
            y=emphasized_signal, 
            sr=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            dct_type=self.config.dct_type,
            norm=self.config.norm,
            lifter=self.config.lifter,
            **melspec_kwargs
        )
        
        # 3. Mean across time
        mfcc_feature = np.mean(mfcc_matrix, axis=1)
        
        # 4. Log scaling
        if self.config.apply_log:
            mfcc_feature = np.log(np.maximum(mfcc_feature, 1e-8))
            
        return mfcc_feature
