from dataclasses import dataclass, asdict, field
import hashlib
import json
from typing import Optional, Type, TypeVar, Literal, get_type_hints
from .augmentation import AugmentationConfig

# Generic type for class method return
T = TypeVar('T', bound='BaseFeatureConfig')

@dataclass
class BaseFeatureConfig:
    """
    Base configuration for all feature extractors.
    Supports loading from dictionary or JSON file.
    """
    name: str
    sample_rate: int = 16000
    description: str = ""
    n_workers: int = 4
    
    # Augmentation Configuration
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def get_hash(self) -> str:
        data = self.to_dict()
        if 'description' in data:
            del data['description']
        if 'n_workers' in data:
            del data['n_workers']
            
        config_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        # Handle nested AugmentationConfig manually
        aug_data = data.pop('augmentation', None)
        
        # FIX: Use get_type_hints to retrieve annotations from parent classes as well
        valid_keys = get_type_hints(cls).keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        instance = cls(**filtered_data)
        
        if aug_data:
            instance.augmentation = AugmentationConfig(**aug_data)
            
        return instance

    @classmethod
    def from_json(cls: Type[T], json_path: str) -> T:
        """
        Creates a config instance from a JSON file.
        Supports new structure with 'active_config' key or legacy flat structure.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Check for new structure
        if 'active_config' in data:
            print(f"Loading configuration from 'active_config' section in {json_path}")
            data = data['active_config']
            
        return cls.from_dict(data)


@dataclass
class MFCCConfig(BaseFeatureConfig):
    """
    Comprehensive configuration for MFCC extraction.
    """
    name: str = "mfcc"

    # MFCC-specific parameters
    n_mfcc: int = 13
    dct_type: Literal[1, 2, 3] = 2
    norm: Optional[Literal['ortho']] = None
    lifter: int = 0

    # Melspectrogram parameters
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None
    window: str = 'hann'
    center: bool = True
    pad_mode: str = 'constant'
    power: float = 2.0
    
    # Mel filter bank parameters
    n_mels: int = 128
    fmin: float = 0.0
    fmax: Optional[float] = None
    htk: bool = False

@dataclass
class STFTConfig(BaseFeatureConfig):
    """
    Configuration for Short-Time Fourier Transform (STFT) extraction.
    """
    name: str = "stft"
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None
    window: str = 'hann'
    center: bool = True
    pad_mode: str = 'constant'
    dtype: Optional[str] = None 
    scaling: Literal['none', 'log', 'db'] = 'log'
    pooling: Literal['mean', 'max', 'mean_std'] = 'mean'

@dataclass
class FFTConfig(BaseFeatureConfig):
    """
    Configuration for Fast Fourier Transform (FFT).
    """
    name: str = "fft"
    n_fft: int = 2048
    axis: int = -1
    norm: Optional[Literal["ortho", "forward", "backward"]] = None
    return_onesided: bool = True
    use_magnitude: bool = True
    power: float = 1.0
    scaling: Literal['none', 'log', 'db'] = 'none'
