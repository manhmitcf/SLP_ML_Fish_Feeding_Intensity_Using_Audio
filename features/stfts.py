import numpy as np
from .base import BaseFeatureExtractor
from .config import STFTSConfig

class STFTSExtractor(BaseFeatureExtractor):
    """
    Feature Extractor for STFTS (Manual STFT using rfft).
    """

    def __init__(self, config: STFTSConfig):
        if not isinstance(config, STFTSConfig):
            raise TypeError("Config must be an instance of STFTSConfig")
        super().__init__(config)
        self.config = config

    def _pre_emphasis(self, signal_in):
        return np.append(signal_in[0], signal_in[1:] - self.config.pre_emph * signal_in[:-1])

    def _framing(self, signal_in):
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
        frame_length = frames.shape[1]
        win_type = self.config.window_type
        
        if win_type == 'hamming':
            window = np.hamming(frame_length)
        elif win_type == 'hann':
            window = np.hanning(frame_length)
        elif win_type == 'blackman':
            window = np.blackman(frame_length)
        else:
            window = np.hamming(frame_length)
            
        return frames * window

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute STFTS features using np.fft.rfft.
        """
        emphasized_signal = self._pre_emphasis(signal)
        frames = self._framing(emphasized_signal)
        windowed_frames = self._windowing(frames)
        
        # Use rfft for efficiency on real input, pass norm
        mag_frames = np.abs(np.fft.rfft(windowed_frames, n=self.config.n_fft, axis=1, norm=self.config.norm))
        
        stfts_feature = np.mean(mag_frames, axis=0)
        
        if self.config.apply_log:
            stfts_feature = np.log(stfts_feature + 1e-8)
            
        return stfts_feature
