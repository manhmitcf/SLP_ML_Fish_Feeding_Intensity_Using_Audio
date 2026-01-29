import numpy as np
import librosa
from .base import BaseFeatureExtractor
from .config import FFTConfig

class FFTExtractor(BaseFeatureExtractor):
    """
    Feature Extractor for Fast Fourier Transform (FFT).
    Computes the magnitude spectrum of the first n_fft samples (or padded).
    Supports scaling (Log, dB).
    """

    def __init__(self, config: FFTConfig):
        if not isinstance(config, FFTConfig):
            raise TypeError("Config must be an instance of FFTConfig")
        super().__init__(config)
        self.config = config

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """
        Computes standard FFT with configurable options.
        """
        n_fft = self.config.n_fft
        
        # Compute FFT
        fft = np.fft.fft(
            signal, 
            n=n_fft, 
            axis=self.config.axis,
            norm=self.config.norm
        )
        
        # Process output based on config
        if self.config.use_magnitude:
            result = np.abs(fft)
            
            if self.config.power != 1.0:
                result = result ** self.config.power
                
            if self.config.scaling == 'log':
                result = np.log(result + 1e-8)
            elif self.config.scaling == 'db':
                result = librosa.amplitude_to_db(result, ref=np.max)
                
        else:
            result = np.real(fft)
        
        # Return one-sided or full spectrum
        if self.config.return_onesided:
            half_length = n_fft // 2
            return result[:half_length]
        else:
            return result
