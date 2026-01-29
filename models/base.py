from abc import ABC, abstractmethod
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize

class BaseModel(ABC):
    """
    Abstract Base Class for all Machine Learning models.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.training_time: float = 0.0 # Store training time in seconds

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Required for AUC/mAP calculation."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Returns the actual parameters of the underlying sklearn model."""
        if self.model is hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return self.config.to_dict() # Fallback

    def save(self, path: Path):
        # Save the model object and the training time
        data = {
            'model': self.model,
            'training_time': self.training_time,
            'config': self.config
        }
        joblib.dump(data, path)

    def load(self, path: Path):
        data = joblib.load(path)
        # Handle backward compatibility if loading old models that are just the model object
        if isinstance(data, dict) and 'model' in data:
            self.model = data['model']
            self.training_time = data.get('training_time', 0.0)
            # self.config = data.get('config') # Optional: restore config
        else:
            self.model = data

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the model.
        Returns a dictionary of metrics.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        # 1. Predictions
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        
        # 2. Basic Metrics
        metrics = {
            "training_time": self.training_time, # Include training time
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        # 3. Advanced Metrics (AUC & mAP for Multiclass)
        classes = np.unique(y_test)
        n_classes = len(classes)
        
        if n_classes == 2:
            if y_prob.ndim == 2:
                y_prob_pos = y_prob[:, 1]
            else:
                y_prob_pos = y_prob
            
            try:
                metrics["auc"] = roc_auc_score(y_test, y_prob_pos)
                metrics["map"] = average_precision_score(y_test, y_prob_pos)
            except Exception as e:
                metrics["auc"] = 0.0
                metrics["map"] = 0.0
        else:
            y_test_bin = label_binarize(y_test, classes=classes)
            try:
                metrics["auc_ovr"] = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
                metrics["map"] = average_precision_score(y_test_bin, y_prob, average='macro')
            except Exception as e:
                metrics["auc_ovr"] = 0.0
                metrics["map"] = 0.0

        # 4. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # 5. Classification Report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        metrics["classification_report"] = report

        return metrics

    def plot_confusion_matrix(self, X_test, y_test, class_names=None, save_path=None):
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.config.name.upper()}')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
