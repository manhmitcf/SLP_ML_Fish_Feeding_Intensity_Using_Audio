# Audio Preprocessing Module

This module handles all audio signal processing tasks *before* feature extraction. It is designed to be modular, allowing for easy integration of noise reduction, silence removal, normalization, and other cleaning techniques.

## Structure

- **`base.py`**: Defines the `BasePreprocessor` abstract class. All custom preprocessors must inherit from this class and implement the `process()` method.
- **`cleaner.py`**: Contains concrete implementations of preprocessors.
  - `DummyPreprocessor`: A basic implementation that currently loads and resamples audio (pass-through). It serves as a placeholder for future cleaning logic.

## Usage

```python
from preprocessor import DummyPreprocessor

# Initialize with a target sample rate (default: 16kHz)
cleaner = DummyPreprocessor(sample_rate=16000)

# Process a single file
# Returns: numpy array containing the audio waveform
signal = cleaner.process("path/to/audio.wav")

# Process a batch of files
signals = cleaner.process_batch(["file1.wav", "file2.wav"])
```

## Future Improvements

To add noise reduction or other enhancements:

1. Modify `cleaner.py` or create a new file (e.g., `denoiser.py`).
2. Implement the logic inside the `process` method.
3. Example:

```python
# In cleaner.py
class NoiseReducer(BasePreprocessor):
    def process(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Apply noise reduction algorithm
        y_clean = some_denoising_lib.reduce_noise(y)
        
        return y_clean
```

## Dependencies

- `librosa`: Used for audio loading and resampling.
- `numpy`: Used for signal representation.
