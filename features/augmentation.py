import numpy as np
import librosa
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation.
    Allows enabling/disabling specific strategies.
    """
    enable: bool = False
    
    # Strategy toggles
    enable_noise: bool = True
    enable_time_shift: bool = True
    enable_pitch_shift: bool = False
    
    # Parameters
    noise_factor: float = 0.005
    shift_max: float = 0.2
    pitch_steps: float = 2.0

class BaseAugmentor(ABC):
    """
    Abstract Base Class for Audio Augmentation strategies.
    """
    
    def __init__(self, config: AugmentationConfig):
        self.config = config

    @abstractmethod
    def augment(self, signal: np.ndarray, sr: int) -> List[np.ndarray]:
        pass

class NoiseInjection(BaseAugmentor):
    def augment(self, signal: np.ndarray, sr: int) -> List[np.ndarray]:
        if not self.config.enable or not self.config.enable_noise:
            return []
            
        noise = np.random.randn(len(signal))
        augmented_signal = signal + self.config.noise_factor * noise
        return [augmented_signal]

class TimeShift(BaseAugmentor):
    def augment(self, signal: np.ndarray, sr: int) -> List[np.ndarray]:
        if not self.config.enable or not self.config.enable_time_shift:
            return []
            
        shift_amount = int(self.config.shift_max * len(signal))
        direction = np.random.choice([-1, 1])
        shift = direction * np.random.randint(low=1, high=shift_amount)
        
        augmented_signal = np.roll(signal, shift)
        return [augmented_signal]

class PitchShift(BaseAugmentor):
    def augment(self, signal: np.ndarray, sr: int) -> List[np.ndarray]:
        if not self.config.enable or not self.config.enable_pitch_shift:
            return []
            
        aug_up = librosa.effects.pitch_shift(signal, sr=sr, n_steps=self.config.pitch_steps)
        aug_down = librosa.effects.pitch_shift(signal, sr=sr, n_steps=-self.config.pitch_steps)
        
        return [aug_up, aug_down]

class CompositeAugmentor(BaseAugmentor):
    """
    Combines multiple augmentation strategies.
    """
    def __init__(self, config: AugmentationConfig):
        super().__init__(config)
        self.strategies: List[BaseAugmentor] = [
            NoiseInjection(config),
            TimeShift(config),
            PitchShift(config)
        ]

    def augment(self, signal: np.ndarray, sr: int) -> List[np.ndarray]:
        if not self.config.enable:
            return []
            
        augmented_signals = []
        for strategy in self.strategies:
            # Each strategy checks its own enable flag internally
            augmented_signals.extend(strategy.augment(signal, sr))
        return augmented_signals
