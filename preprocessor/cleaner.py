import numpy as np
import librosa
from pathlib import Path
from typing import Union
from .base import BasePreprocessor

class DummyPreprocessor(BasePreprocessor):
    """
    A placeholder preprocessor that currently performs basic loading.
    Future implementation: Add noise reduction, silence removal, etc.
    """

    def process(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Loads the audio file and returns the raw waveform.
        Currently does NO modification (Pass-through).
        """
        # Ensure path is string
        path_str = str(audio_path)
        
        try:
            # Load audio using librosa
            # sr=self.sample_rate ensures all audio is resampled to the same rate
            y, _ = librosa.load(path_str, sr=self.sample_rate)
            
            # TODO: Add noise reduction logic here later
            # e.g., y = some_denoising_function(y)
            
            return y
        except Exception as e:
            print(f"Error processing {path_str}: {e}")
            return np.array([])
