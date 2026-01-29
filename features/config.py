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
        aug_data = data.pop('augmentation', None)
        valid_keys = get_type_hints(cls).keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        instance = cls(**filtered_data)
        if aug_data:
            instance.augmentation = AugmentationConfig(**aug_data)
        return instance

    @classmethod
    def from_json(cls: Type[T], json_path: str) -> T:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if 'active_config' in data:
            data = data['active_config']
            
        return cls.from_dict(data)

@dataclass
class MFCCConfig(BaseFeatureConfig):
    name: str = "mfcc"
    n_mfcc: int = 13
    dct_type: Literal[1, 2, 3] = 2
    norm: Optional[Literal['ortho']] = None
    lifter: int = 0
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None
    window: str = 'hann'
    center: bool = True
    pad_mode: str = 'constant'
    power: float = 2.0
    n_mels: int = 128
    fmin: float = 0.0
    fmax: Optional[float] = None
    htk: bool = False

@dataclass
class STFTConfig(BaseFeatureConfig):
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
    name: str = "fft"
    n_fft: int = 2048
    axis: int = -1
    norm: Optional[Literal["ortho", "forward", "backward"]] = None
    return_onesided: bool = True
    use_magnitude: bool = True
    power: float = 1.0
    scaling: Literal['none', 'log', 'db'] = 'none'

@dataclass
class FFTSConfig(BaseFeatureConfig):
    name: str = "ffts"
    pre_emph: float = 0.97
    frame_size: float = 0.025
    frame_stride: float = 0.01
    n_fft: int = 512
    window_type: str = 'hamming'
    norm: Optional[Literal["ortho", "forward", "backward"]] = None # Added norm
    apply_log: bool = True

@dataclass
class STFTSConfig(BaseFeatureConfig):
    name: str = "stfts"
    pre_emph: float = 0.97
    frame_size: float = 0.025
    frame_stride: float = 0.01
    n_fft: int = 2048
    window_type: str = 'hamming'
    norm: Optional[Literal["ortho", "forward", "backward"]] = None # Added norm
    apply_log: bool = True

@dataclass
class MFCCsConfig(BaseFeatureConfig):
    name: str = "mfccs"
    pre_emph: float = 0.97
    apply_log: bool = True
    n_mfcc: int = 13
    dct_type: Literal[1, 2, 3] = 2
    norm: Optional[Literal['ortho']] = None
    lifter: int = 0
    n_fft: int = 2048
    hop_length: int = 1024
    win_length: Optional[int] = None
    window: str = 'hann'
    center: bool = True
    pad_mode: str = 'constant'
    power: float = 2.0
    n_mels: int = 128
    fmin: float = 0.0
    fmax: Optional[float] = None
    htk: bool = False
