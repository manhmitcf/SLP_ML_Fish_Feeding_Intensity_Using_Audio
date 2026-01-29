from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import Union, Optional

class BasePreprocessor(ABC):
    """
    Abstract Base Class for Audio Preprocessing.
    All preprocessing steps (denoising, trimming, normalization, etc.) 
    should inherit from this class.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Args:
            sample_rate (int): Target sample rate for processing.
        """
        self.sample_rate = sample_rate

    @abstractmethod
    def process(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Process a single audio file.

        Args:
            audio_path (str|Path): Path to the input audio file.

        Returns:
            np.ndarray: The processed audio signal (waveform).
        """
        pass

    def process_batch(self, audio_paths: list) -> list:
        """
        Process a batch of audio files.
        Can be overridden for parallel processing optimization.
        """
        return [self.process(p) for p in audio_paths]
