import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Type, Optional, Any
import joblib
import hashlib
import json

from .config import PipelineConfig, BaseStepConfig
from .base import BaseProcessor
from .scaler import ScalerProcessor
from .pca import PCAProcessor
from .selection import SelectionProcessor

class ProcessingManager:
    """
    Manages the raw feature processing pipeline with Caching support.
    Uses Registry pattern for extensibility.
    """
    
    _PROCESSOR_REGISTRY: Dict[str, Type[BaseProcessor]] = {
        'scaler': ScalerProcessor,
        'pca': PCAProcessor,
        'selection': SelectionProcessor,
    }

    def __init__(self, config: PipelineConfig, cache_dir: str = "features_cache/processed"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.processors = {}
        self._init_processors()

    @classmethod
    def register_processor(cls, step_name: str, processor_cls: Type[BaseProcessor]):
        cls._PROCESSOR_REGISTRY[step_name] = processor_cls

    def _init_processors(self):
        for step_name in self.config.steps:
            if step_name not in self._PROCESSOR_REGISTRY:
                raise ValueError(f"Unknown processing step: '{step_name}'.")
            
            processor_cls = self._PROCESSOR_REGISTRY[step_name]
            step_config = getattr(self.config, step_name)
            self.processors[step_name] = processor_cls(step_config)

    def _concatenate_features(self, feature_list: List[np.ndarray]) -> np.ndarray:
        if not feature_list:
            raise ValueError("Feature list is empty.")
        if len(feature_list) == 1:
            return feature_list[0]
        
        print(f"--- Concatenating {len(feature_list)} feature sets ---")
        return np.concatenate(feature_list, axis=1)

    def _get_pipeline_hash(self, input_hashes: List[str]) -> str:
        combined_hash_str = "".join(sorted(input_hashes))
        pipeline_config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        final_hash_str = combined_hash_str + pipeline_config_str
        return hashlib.md5(final_hash_str.encode('utf-8')).hexdigest()

    def _get_cache_paths(self, pipeline_hash: str) -> Dict[str, Path]:
        return {
            "data": self.cache_dir / f"{pipeline_hash}.npz",
            "config": self.cache_dir / f"{pipeline_hash}_config.json"
        }

    def process_data(
        self, 
        feature_sets: Dict[str, Dict[str, Any]],
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        
        input_hashes = [data['config_hash'] for data in feature_sets.values()]
        pipeline_hash = self._get_pipeline_hash(input_hashes)
        cache_paths = self._get_cache_paths(pipeline_hash)

        if cache_paths['data'].exists() and not force_recompute:
            print(f"--- Pipeline Cache HIT! Loading from: {cache_paths['data']} ---")
            loaded = np.load(cache_paths['data'])
            return {
                'X_train': loaded['X_train'], 'y_train': loaded['y_train'],
                'X_test': loaded['X_test'], 'y_test': loaded['y_test'],
                'pipeline_hash': pipeline_hash
            }

        print(f"--- Pipeline Cache MISS. Running processing pipeline... ---")
        
        # Prepare data
        X_train_list = [data['X_train'] for data in feature_sets.values()]
        X_test_list = [data['X_test'] for data in feature_sets.values()]
        
        first_key = list(feature_sets.keys())[0]
        y_train = feature_sets[first_key]['y_train']
        y_test = feature_sets[first_key]['y_test']

        # Concatenate
        X_train_combined = self._concatenate_features(X_train_list)
        X_test_combined = self._concatenate_features(X_test_list)
        
        # Fit pipeline on training data
        print("--- Fitting Processing Pipeline ---")
        current_X_train = X_train_combined
        
        for step_name in self.config.steps:
            processor = self.processors[step_name]
            processor.fit(current_X_train, y_train)
            current_X_train = processor.transform(current_X_train)
        
        # Transform test data
        print("--- Transforming Test Data ---")
        current_X_test = X_test_combined
        for step_name in self.config.steps:
            processor = self.processors[step_name]
            current_X_test = processor.transform(current_X_test)
            
        # Save to cache
        self._save_cache(
            cache_paths,
            {'X_train': current_X_train, 'y_train': y_train, 'X_test': current_X_test, 'y_test': y_test},
            feature_sets
        )
        
        return {
            'X_train': current_X_train, 'y_train': y_train,
            'X_test': current_X_test, 'y_test': y_test,
            'pipeline_hash': pipeline_hash
        }

    def _save_cache(self, cache_paths: Dict[str, Path], data: Dict[str, np.ndarray], input_feature_sets: Dict[str, Dict]):
        # Save data
        print(f"--- Saving processed features to cache: {cache_paths['data']} ---")
        np.savez_compressed(cache_paths['data'], **data)
        
        # Save full experiment config
        print(f"--- Saving full experiment config to: {cache_paths['config']} ---")
        full_config = {
            "pipeline_config": self.config.to_dict(),
            "input_features": {
                name: fs['config'].to_dict() for name, fs in input_feature_sets.items()
            }
        }
        with open(cache_paths['config'], 'w') as f:
            json.dump(full_config, f, indent=4)
