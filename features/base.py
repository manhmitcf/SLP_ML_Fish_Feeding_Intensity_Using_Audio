from abc import ABC, abstractmethod
import numpy as np
import librosa
from pathlib import Path
from typing import List, Union, Tuple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from .config import BaseFeatureConfig
from .augmentation import CompositeAugmentor

class BaseFeatureExtractor(ABC):
    """
    Abstract Base Class for all feature extractors.
    Handles audio loading, augmentation, and parallel processing.
    """

    def __init__(self, config: BaseFeatureConfig):
        self.config = config
        # Initialize Augmentor
        self.augmentor = CompositeAugmentor(config.augmentation)

    @abstractmethod
    def extract(self, signal: np.ndarray) -> np.ndarray:
        pass

    def _process_single_item(self, args):
        """
        Helper for parallel processing.
        Loads audio -> (Augments) -> Extracts features.
        """
        file_path, label, is_training = args
        results = []
        
        try:
            # 1. Load Original Audio
            signal, sr = librosa.load(str(file_path), sr=self.config.sample_rate)
            
            # 2. Extract Feature for Original
            feat_original = self.extract(signal)
            results.append((feat_original, label))
            
            # 3. Augmentation (Only for Training data)
            if is_training and self.config.augmentation.enable:
                augmented_signals = self.augmentor.augment(signal, sr)
                for aug_sig in augmented_signals:
                    feat_aug = self.extract(aug_sig)
                    results.append((feat_aug, label))
                    
            return results
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def extract_from_paths(
        self, 
        file_paths: List[Path], 
        labels: List[int], 
        is_training: bool = False,
        max_workers: int = 4
    ) -> tuple:
        """
        Extracts features for a batch of files using parallel processing.
        Supports Data Augmentation for training set.
        """
        # Correctly determine the mode string
        if is_training:
            mode_str = "TRAIN"
            if self.config.augmentation.enable:
                mode_str += " (with Augmentation)"
        else:
            mode_str = "TEST"

        print(f"--- Extracting features using {self.config.name.upper()} [{mode_str}] (Workers: {max_workers}) ---")
        
        # Prepare arguments: (path, label, is_training)
        tasks = [(p, l, is_training) for p, l in zip(file_paths, labels)]
        
        X_features = []
        y_labels = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map returns a list of lists (because one file can produce multiple features due to augmentation)
            results_list = list(tqdm(executor.map(self._process_single_item, tasks), total=len(tasks), unit="file"))

        # Flatten the results
        for batch_result in results_list:
            for feat, lbl in batch_result:
                X_features.append(feat)
                y_labels.append(lbl)

        return np.array(X_features), np.array(y_labels)
