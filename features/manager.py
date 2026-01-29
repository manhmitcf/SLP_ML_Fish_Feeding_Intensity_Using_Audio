import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Type, Dict

from .config import BaseFeatureConfig, MFCCConfig, STFTConfig, FFTConfig, FFTSConfig
from .base import BaseFeatureExtractor
from .mfcc import MFCCExtractor
from .stft import STFTExtractor
from .fft import FFTExtractor
from .ffts import FFTSExtractor
from utils.dataloader import BaseDataLoader

class FeatureManager:
    """
    Manages the feature extraction pipeline with Caching and Augmentation support.
    Uses a Registry pattern to manage extractors dynamically.
    """
    
    # Registry mapping Config types to Extractor types
    _EXTRACTOR_REGISTRY: Dict[Type[BaseFeatureConfig], Type[BaseFeatureExtractor]] = {
        MFCCConfig: MFCCExtractor,
        STFTConfig: STFTExtractor,
        FFTConfig: FFTExtractor,
        FFTSConfig: FFTSExtractor,
    }

    def __init__(
        self, 
        config: BaseFeatureConfig, 
        base_dir: str = "features_cache"
    ):
        self.config = config
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.extractor_class = self._get_extractor_class(config)

    @classmethod
    def register_extractor(cls, config_cls: Type[BaseFeatureConfig], extractor_cls: Type[BaseFeatureExtractor]):
        """
        Registers a new extractor dynamically.
        """
        cls._EXTRACTOR_REGISTRY[config_cls] = extractor_cls

    def _get_extractor_class(self, config: BaseFeatureConfig) -> Type[BaseFeatureExtractor]:
        """
        Factory method to get the correct extractor class using the registry.
        """
        config_type = type(config)
        if config_type in self._EXTRACTOR_REGISTRY:
            return self._EXTRACTOR_REGISTRY[config_type]
            
        for registered_conf, extractor_cls in self._EXTRACTOR_REGISTRY.items():
            if isinstance(config, registered_conf):
                return extractor_cls
                
        raise ValueError(f"Unsupported config type: {config_type}. Please register it using FeatureManager.register_extractor()")

    def _get_cache_paths(self) -> Tuple[Path, Path]:
        config_hash = self.config.get_hash()
        filename = f"{self.config.name}_{config_hash}"
        data_path = self.base_dir / f"{filename}.npz"
        meta_path = self.base_dir / f"{filename}.json"
        return data_path, meta_path

    def _save_to_cache(self, X_train, y_train, X_test, y_test):
        data_path, meta_path = self._get_cache_paths()
        print(f"--- Saving features to cache: {data_path} ---")
        np.savez_compressed(
            data_path, 
            X_train=X_train, y_train=y_train, 
            X_test=X_test, y_test=y_test
        )
        with open(meta_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=4)

    def _load_from_cache(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data_path, _ = self._get_cache_paths()
        print(f"--- Cache HIT! Loading from: {data_path} ---")
        loaded = np.load(data_path)
        return loaded['X_train'], loaded['y_train'], loaded['X_test'], loaded['y_test']

    def get_data(
        self, 
        data_loader: BaseDataLoader, 
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        data_path, _ = self._get_cache_paths()

        if data_path.exists() and not force_recompute:
            return self._load_from_cache()

        print(f"--- Cache MISS (or forced). Starting extraction pipeline... ---")
        
        extractor = self.extractor_class(self.config)

        train_paths, train_labels = data_loader.load_train_data()
        test_paths, test_labels = data_loader.load_test_data()

        print(">>> Processing Training Set...")
        X_train_feat, y_train_feat = extractor.extract_from_paths(
            train_paths, train_labels, 
            is_training=True,
            max_workers=self.config.n_workers
        )
        
        print(">>> Processing Test Set...")
        X_test_feat, y_test_feat = extractor.extract_from_paths(
            test_paths, test_labels, 
            is_training=False,
            max_workers=self.config.n_workers
        )

        self._save_to_cache(X_train_feat, y_train_feat, X_test_feat, y_test_feat)

        return X_train_feat, y_train_feat, X_test_feat, y_test_feat
