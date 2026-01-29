import os
import shutil
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor


class BaseSplitter(ABC):
    """
    Abstract Base Class for data splitting.
    Optimized for performance with Multithreading and reduced I/O overhead.
    """

    def __init__(self, source_dir: str, dest_dir: Optional[str] = None, mode: str = 'ratio', test_val: Union[float, int] = 0.2):
        self.source_dir = Path(source_dir)
        self.mode = mode
        self.test_val = test_val
        
        self.classes = ['middle', 'none', 'strong', 'weak']
        self.class_mapping = {'medium': 'middle'}

        if dest_dir:
            self.dest_dir = Path(dest_dir)
        else:
            source_name = self.source_dir.name
            self.dest_dir = Path("datasplit") / f"datasplit_{source_name}"

    def _prepare_directories(self):
        for split_type in ['train', 'test']:
            for cls in self.classes:
                (self.dest_dir / split_type / cls).mkdir(parents=True, exist_ok=True)

    def _get_wav_files(self, directory: Path) -> List[Path]:
        # Using generator to list conversion is fine, but rglob is fast enough.
        return list(directory.rglob("*.wav"))

    @abstractmethod
    def _generate_unique_name(self, src_file: Path) -> str:
        pass

    def _copy_single_file(self, args):
        """Helper function for threading."""
        src_file, dest_path = args
        try:
            # shutil.copy is faster than copy2 (skips some metadata)
            shutil.copy(src_file, dest_path)
        except Exception as e:
            print(f"Error copying {src_file.name}: {e}")

    def _copy_files(self, files: List[Path], split_type: str, cls: str):
        """
        Copies files using ThreadPoolExecutor for parallel I/O.
        """
        tasks = []
        for src_file in files:
            unique_name = self._generate_unique_name(src_file)
            dest_path = self.dest_dir / split_type / cls / unique_name
            tasks.append((src_file, dest_path))

        total = len(tasks)
        # Use max_workers=8 or similar to avoid freezing the OS, 
        # but enough to saturate Disk I/O.
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks
            list(executor.map(self._copy_single_file, tasks))

    def _validate_inputs(self):
        if self.mode == 'ratio':
            if not (0.0 < self.test_val < 1.0):
                raise ValueError(f"Invalid ratio: {self.test_val}. Must be between 0.0 and 1.0.")
        elif self.mode == 'fixed':
            if self.test_val <= 0:
                raise ValueError(f"Invalid fixed count: {self.test_val}. Must be > 0.")
        else:
            raise ValueError(f"Invalid mode: '{self.mode}'.")

    def split(self):
        print(f"--- Processing data from: {self.source_dir} ---")
        print(f"--- Destination: {self.dest_dir} ---")
        
        self._validate_inputs()

        if not self.source_dir.exists():
            print(f"Error: Source directory '{self.source_dir}' does not exist.")
            return

        self._prepare_directories()
        
        # Optimized: Use a dictionary of lists directly
        class_data: Dict[str, List[Path]] = {cls: [] for cls in self.classes}
        
        print("Scanning files... (This may take a moment)")
        all_wav_files = self._get_wav_files(self.source_dir)
        
        total_ignored = 0
        for f in all_wav_files:
            found_class = None
            # Optimization: Check parent name first (most common case) to avoid loop
            parent_name = f.parent.name.lower()
            if parent_name in self.class_mapping:
                parent_name = self.class_mapping[parent_name]
            
            if parent_name in self.classes:
                found_class = parent_name
            else:
                # Fallback: Check full path parts
                for part in f.parts:
                    p_lower = part.lower()
                    if p_lower in self.class_mapping:
                        p_lower = self.class_mapping[p_lower]
                    
                    if p_lower in self.classes:
                        found_class = p_lower
                        break
            
            if found_class:
                class_data[found_class].append(f)
            else:
                total_ignored += 1

        fixed_test_per_class = 0
        if self.mode == 'fixed':
            fixed_test_per_class = int(self.test_val) // len(self.classes)

        total_processed = 0

        for cls in self.classes:
            files = class_data[cls]
            random.shuffle(files)
            
            count = len(files)
            if count == 0:
                continue

            if self.mode == 'ratio':
                n_test = int(count * self.test_val)
            elif self.mode == 'fixed':
                if count < fixed_test_per_class:
                    raise ValueError(f"Not enough files in '{cls}' (Found {count}, Need {fixed_test_per_class}).")
                n_test = fixed_test_per_class
            
            test_files = files[:n_test]
            train_files = files[n_test:]
            
            if len(train_files) == 0:
                raise ValueError(f"Training set empty for '{cls}'.")

            total_processed += count

            print(f"Class '{cls.upper()}': {count} files. Copying {len(train_files)} Train, {len(test_files)} Test...")
            
            # Copying is now silent per-file to prevent lag, but faster
            self._copy_files(train_files, 'train', cls)
            self._copy_files(test_files, 'test', cls)

        print(f"\n--- DONE. Total files processed: {total_processed} ---")
        if total_ignored > 0:
            print(f"Note: Ignored {total_ignored} files.")


class UniversalSplitter(BaseSplitter):
    def _generate_unique_name(self, src_file: Path) -> str:
        try:
            rel_path = src_file.relative_to(self.source_dir)
            return str(rel_path).replace(os.sep, "_")
        except ValueError:
            return f"{src_file.parent.name}_{src_file.name}"
