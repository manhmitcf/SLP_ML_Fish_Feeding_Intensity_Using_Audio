import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np


class BaseDataLoader(ABC):
    """
    Abstract Base Class for loading audio datasets.
    Defines the standard interface for retrieving file paths and labels.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the data loader.

        Args:
            data_dir (str): Path to the root directory of the split dataset 
                            (containing 'train' and 'test' subfolders).
        """
        self.data_dir = Path(data_dir)
        self.classes = ['none', 'weak', 'middle', 'strong']
        # Mapping class names to integer labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    @abstractmethod
    def load_train_data(self) -> Tuple[List[Path], List[int]]:
        """
        Loads training data.
        
        Returns:
            Tuple containing:
            - List of file paths (X_train)
            - List of integer labels (y_train)
        """
        pass

    @abstractmethod
    def load_test_data(self) -> Tuple[List[Path], List[int]]:
        """
        Loads testing data.
        
        Returns:
            Tuple containing:
            - List of file paths (X_test)
            - List of integer labels (y_test)
        """
        pass

    def get_class_names(self) -> List[str]:
        """Returns the list of class names."""
        return self.classes

    def get_class_mapping(self) -> Dict[str, int]:
        """Returns the dictionary mapping class names to integers."""
        return self.class_to_idx


class StandardDataLoader(BaseDataLoader):
    """
    Standard implementation of a data loader.
    Assumes the directory structure:
        root/
          train/
            class_A/
            class_B/
          test/
            class_A/
            class_B/
    """

    def _load_subset(self, subset_name: str) -> Tuple[List[Path], List[int]]:
        """
        Helper function to load data from a specific subset folder (train or test).
        """
        subset_dir = self.data_dir / subset_name
        
        if not subset_dir.exists():
            raise FileNotFoundError(f"Directory not found: {subset_dir}")

        file_paths = []
        labels = []

        print(f"--- Loading '{subset_name}' data from: {subset_dir} ---")

        for cls_name, label_idx in self.class_to_idx.items():
            cls_dir = subset_dir / cls_name
            
            if not cls_dir.exists():
                print(f"Warning: Class folder '{cls_name}' not found in {subset_name}. Skipping.")
                continue

            # Find all .wav files
            files = list(cls_dir.glob("*.wav"))
            count = len(files)
            
            if count == 0:
                print(f"Warning: No .wav files found for class '{cls_name}' in {subset_name}.")
            
            file_paths.extend(files)
            labels.extend([label_idx] * count)
            
            print(f"Class '{cls_name}': Loaded {count} files.")

        print(f"Total '{subset_name}' samples: {len(file_paths)}")
        return file_paths, labels

    def load_train_data(self) -> Tuple[List[Path], List[int]]:
        return self._load_subset('train')

    def load_test_data(self) -> Tuple[List[Path], List[int]]:
        return self._load_subset('test')
