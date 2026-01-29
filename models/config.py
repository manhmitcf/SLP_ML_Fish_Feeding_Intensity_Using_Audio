from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Union, List
import json

@dataclass
class BaseModelConfig:
    """Base configuration for all models."""
    name: str
    random_state: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        # Filter keys to match dataclass fields
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'active_config' in data:
            data = data['active_config']
        return cls.from_dict(data)

@dataclass
class KNNConfig(BaseModelConfig):
    """Configuration for K-Nearest Neighbors."""
    name: str = "knn"
    n_neighbors: int = 5
    weights: Literal['uniform', 'distance'] = 'uniform'
    algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = 'auto'
    leaf_size: int = 30
    p: int = 2
    metric: str = 'minkowski'
    n_jobs: int = -1 # Added n_jobs

@dataclass
class SVMConfig(BaseModelConfig):
    """Configuration for Support Vector Machine."""
    name: str = "svm"
    C: float = 1.0
    kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] = 'rbf'
    degree: int = 3
    gamma: Union[str, float] = 'scale'
    coef0: float = 0.0
    shrinking: bool = True
    probability: bool = True # Enable probability for AUC
    tol: float = 1e-3
    class_weight: Optional[Union[dict, str]] = None
    max_iter: int = -1

@dataclass
class RFConfig(BaseModelConfig):
    """Configuration for Random Forest."""
    name: str = "rf"
    n_estimators: int = 100
    criterion: Literal['gini', 'entropy', 'log_loss'] = 'gini'
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[str, int, float, None] = 'sqrt'
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: int = -1 # Use all cores
    class_weight: Optional[Union[dict, str]] = None
    ccp_alpha: float = 0.0

@dataclass
class ETConfig(RFConfig):
    """Configuration for Extra Trees (Inherits from RF as params are similar)."""
    name: str = "et"
    # Extra Trees specific defaults can be overridden here if needed

@dataclass
class LRConfig(BaseModelConfig):
    """Configuration for Logistic Regression."""
    name: str = "lr"
    penalty: Literal['l1', 'l2', 'elasticnet', 'none'] = 'l2'
    dual: bool = False
    tol: float = 1e-4
    C: float = 1.0
    fit_intercept: bool = True
    intercept_scaling: float = 1
    class_weight: Optional[Union[dict, str]] = None
    solver: Literal['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] = 'lbfgs'
    max_iter: int = 100
    # multi_class removed
    n_jobs: int = -1
    l1_ratio: Optional[float] = None
