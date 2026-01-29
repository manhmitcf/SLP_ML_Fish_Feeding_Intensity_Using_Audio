import joblib
from pathlib import Path
from typing import Type, Dict, Any, Optional, Union
import json
import time
import numpy as np

from .config import BaseModelConfig, KNNConfig, SVMConfig, RFConfig, ETConfig, LRConfig
from .base import BaseModel
from .classifiers import KNNClassifier, SVMClassifier, RFClassifier, ETClassifier, LRClassifier

class ModelManager:
    """
    Manages model creation, training, and persistence.
    """
    
    _MODEL_REGISTRY: Dict[Type[BaseModelConfig], Type[BaseModel]] = {
        KNNConfig: KNNClassifier,
        SVMConfig: SVMClassifier,
        RFConfig: RFClassifier,
        ETConfig: ETClassifier,
        LRConfig: LRClassifier,
    }
    
    _CONFIG_NAME_MAP: Dict[str, Type[BaseModelConfig]] = {
        "knn": KNNConfig,
        "svm": SVMConfig,
        "rf": RFConfig,
        "et": ETConfig,
        "lr": LRConfig,
    }

    def __init__(self, config: Optional[BaseModelConfig] = None, save_dir: str = "models_cache"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_instance: Optional[BaseModel] = None
        self.upstream_config: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Any] = {} # Store metrics
        
        if config:
            self._init_model(config)

    def _init_model(self, config: BaseModelConfig):
        model_class = self._get_model_class(config)
        self.model_instance = model_class(config)

    def _get_model_class(self, config: BaseModelConfig) -> Type[BaseModel]:
        config_type = type(config)
        if config_type in self._MODEL_REGISTRY:
            return self._MODEL_REGISTRY[config_type]
        
        for registered_conf, model_cls in self._MODEL_REGISTRY.items():
            if isinstance(config, registered_conf):
                return model_cls
                
        raise ValueError(f"Unsupported config type: {config_type}")

    def set_upstream_config(self, config_path: Union[str, Path]):
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.upstream_config = json.load(f)
        else:
            print(f"Warning: Upstream config not found at {config_path}")

    def train(self, X_train, y_train):
        if not self.model_instance:
            raise RuntimeError("Model not initialized.")
            
        print(f"--- Training {self.config.name.upper()} ---")
        
        start_time = time.time()
        self.model_instance.fit(X_train, y_train)
        end_time = time.time()
        
        self.model_instance.training_time = end_time - start_time
        
        print(f"--- Training Completed in {self.model_instance.training_time:.4f} seconds ---")

    def evaluate(self, X_test, y_test, class_names=None) -> Dict[str, Any]:
        if not self.model_instance:
            raise RuntimeError("Model not initialized.")
            
        print(f"--- Evaluating {self.config.name.upper()} ---")
        self.evaluation_results = self.model_instance.evaluate(X_test, y_test, class_names)
        return self.evaluation_results

    def save(self, filename: Optional[str] = None):
        if not self.model_instance:
            raise RuntimeError("No model to save.")
            
        if filename is None:
            filename = f"{self.config.name}_model.joblib"
        
        save_path = self.save_dir / filename
        print(f"--- Saving model to: {save_path} ---")
        self.model_instance.save(save_path)
        
        # Save FULL config (Model + Upstream + Results)
        config_path = save_path.with_suffix('.config.json')
        
        full_config = {
            "model_config": self.config.to_dict(),
            "evaluation_results": self.evaluation_results, # Added results
            "data_processing_config": self.upstream_config
        }
        
        with open(config_path, 'w') as f:
            # Custom encoder to handle numpy types in metrics
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            json.dump(full_config, f, indent=4, cls=NpEncoder)
            
        # Save params
        params_path = save_path.with_suffix('.params.json')
        with open(params_path, 'w') as f:
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            json.dump(self.model_instance.get_params(), f, indent=4, cls=NpEncoder)

    def load(self, filename: str):
        """
        Loads a model and its full configuration from disk.
        """
        load_path = self.save_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found at {load_path}")
            
        print(f"--- Loading model and config from: {load_path.parent} ---")
        
        # 1. Load Config
        config_path = load_path.with_suffix('.config.json')
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            full_config = json.load(f)
            
        model_conf_dict = full_config.get('model_config', {})
        model_name = model_conf_dict.get('name')
        
        if model_name not in self._CONFIG_NAME_MAP:
            raise ValueError(f"Unknown model name '{model_name}' in config file.")
            
        config_cls = self._CONFIG_NAME_MAP[model_name]
        self.config = config_cls.from_dict(model_conf_dict)
        self.upstream_config = full_config.get('data_processing_config', {})
        self.evaluation_results = full_config.get('evaluation_results', {}) # Load results
        
        # 2. Initialize Model
        self._init_model(self.config)
        
        # 3. Load Model State
        self.model_instance.load(load_path)
        print("--- Model loaded successfully ---")
