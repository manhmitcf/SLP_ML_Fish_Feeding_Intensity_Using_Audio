from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Union
import json

@dataclass
class BaseStepConfig:
    """Base configuration for a processing step."""
    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class ScalerConfig(BaseStepConfig):
    """
    Configuration for Feature Scaling.
    type: 'standard', 'minmax', or None.
    """
    type: Literal['standard', 'minmax'] = 'standard'

@dataclass
class PCAConfig(BaseStepConfig):
    """
    Configuration for PCA.
    n_components: float (variance ratio) or int (number of components).
    """
    n_components: Optional[Union[float, int]] = 0.95
    whiten: bool = False

@dataclass
class SelectionConfig(BaseStepConfig):
    """
    Configuration for Feature Selection.
    method: 'variance' (VarianceThreshold) or 'k_best' (SelectKBest).
    threshold: For VarianceThreshold (default 0.0).
    k: For SelectKBest (number of top features to select).
    """
    method: Literal['variance', 'k_best'] = 'variance'
    threshold: float = 0.0
    k: int = 10

@dataclass
class PipelineConfig:
    """
    Master configuration for the processing pipeline.
    """
    # List of steps to execute in order. 
    # Example: ["scaler", "selection", "pca"]
    steps: List[str] = field(default_factory=lambda: ["scaler", "pca"])
    
    # Configs for each step
    scaler: ScalerConfig = field(default_factory=ScalerConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    def to_dict(self) -> dict:
        """Converts the entire pipeline config to a dictionary."""
        return asdict(self)

    @classmethod
    def from_json(cls, json_path: str) -> 'PipelineConfig':
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        scaler_data = data.get('scaler', {})
        pca_data = data.get('pca', {})
        selection_data = data.get('selection', {})
        steps = data.get('steps', ["scaler", "pca"])
        
        return cls(
            steps=steps,
            scaler=ScalerConfig(**scaler_data),
            pca=PCAConfig(**pca_data),
            selection=SelectionConfig(**selection_data)
        )
