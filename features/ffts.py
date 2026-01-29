import numpy as np
from .base import BaseFeatureExtractor
from .config import FFTSConfig

class FFTSExtractor(BaseFeatureExtractor):
    """
    Feature Extractor for FFTS (FFT with Pre-emphasis, Framing, Windowing).
    """

    def __init__(self, config: FFTSConfig):
        if not isinstance(config, FFTSConfig):
            raise TypeError("Config must be an instance of FFTSConfig")
        super().__init__(config)
        self.config = config

    def _pre_emphasis(self, signal_in):
        """
        Apply pre-emphasis to emphasize high frequencies.
        """
        return np.append(signal_in[0], signal_in[1:] - self.config.pre_emph * signal_in[:-1])

    def _framing(self, signal_in):
        """
        Split signal into frames.
        """
        sample_rate = self.config.sample_rate
        frame_size = self.config.frame_size
        frame_stride = self.config.frame_stride
        
        frame_length = int(round(frame_size * sample_rate))
        frame_step = int(round(frame_stride * sample_rate))
        signal_length = len(signal_in)
        
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

        pad_signal_length = num_frames * frame_step + frame_length
        pad_signal = np.pad(signal_in, (0, pad_signal_length - signal_length), mode='constant')

        indices = np.arange(0, frame_length) + np.arange(0, num_frames * frame_step, frame_step)[:, None]
        frames = pad_signal[indices.astype(np.int32)]
        return frames

    def _windowing(self, frames):
        """
        Apply window function to each frame.
        """
        frame_length = frames.shape[1]
        win_type = self.config.window_type
        
        if win_type == 'hamming':
            window = np.hamming(frame_length)
        elif win_type == 'hann':
            window = np.hanning(frame_length)
        elif win_type == 'blackman':
            window = np.blackman(frame_length)
        elif win_type == 'bartlett':
            window = np.bartlett(frame_length)
        else:
            # Fallback or raise error. For now, default to hamming if unknown
            print(f"Warning: Unknown window type '{win_type}'. Using Hamming.")
            window = np.hamming(frame_length)
            
        return frames * window

    def _fft_frames(self, frames):
        """
        Compute FFT for each frame and take magnitude.
        """
        n_fft = self.config.n_fft
        # Only take positive frequencies
        return np.abs(np.fft.fft(frames, n_fft))[:, :n_fft//2+1]

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute FFTS features.
        """
        # 1. Pre-emphasis
        emphasized_signal = self._pre_emphasis(signal)
        
        # 2. Framing
        frames = self._framing(emphasized_signal)
        
        # 3. Windowing
        windowed_frames = self._windowing(frames)
        
        # 4. FFT
        mag_frames = self._fft_frames(windowed_frames)
        
        # 5. Mean across frames
        fft_feature = np.mean(mag_frames, axis=0)
        
        # 6. Log scaling
        if self.config.apply_log:
            fft_feature = np.log(np.maximum(fft_feature, 1e-8))
            
        return fft_feature
