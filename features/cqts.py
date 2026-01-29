import numpy as np
import librosa
from .base import BaseFeatureExtractor
from .config import CQTsConfig

class CQTsExtractor(BaseFeatureExtractor):
    def __init__(self, config: CQTsConfig):
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
        else:
            window = np.hamming(frame_length)
            
        return frames * window

    def extract(self, signal: np.ndarray) -> np.ndarray:
        # 1. Pre-emphasis
        emphasized_signal = self._pre_emphasis(signal)
        
        # 2. Framing
        frames = self._framing(emphasized_signal)
        
        # 3. Windowing
        windowed_frames = self._windowing(frames)
        
        # 4. CQT on each frame
        # Note: CQT on short frames (25ms) might have poor frequency resolution
        cqt_features = []
        for frame in windowed_frames:
            # We treat each frame as a short signal
            # hop_length must be < frame length. 
            # Since frame is short, we might just get 1 or few columns of CQT
            try:
                cqt = librosa.cqt(
                    y=frame,
                    sr=self.config.sample_rate,
                    hop_length=len(frame), # One hop per frame effectively
                    fmin=self.config.fmin,
                    n_bins=self.config.n_bins,
                    bins_per_octave=self.config.bins_per_octave
                )
                cqt_mag = np.abs(cqt)
                cqt_features.append(np.mean(cqt_mag, axis=1))
            except Exception:
                # Fallback if frame is too short for CQT config
                cqt_features.append(np.zeros(self.config.n_bins))

        cqt_features = np.array(cqt_features)
        
        # 5. Mean across frames
        final_feature = np.mean(cqt_features, axis=0)
        
        # 6. Log
        if self.config.apply_log:
            final_feature = np.log(final_feature + 1e-8)
            
        return final_feature
