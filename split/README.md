# Data Splitting Module

This module provides a high-performance, universal tool for splitting audio datasets into training and testing sets. It is optimized for large datasets (e.g., 27k+ files) using multithreading and smart caching.

## Key Features

- **High Performance**:
  - **Multithreading**: Uses `ThreadPoolExecutor` to copy files in parallel, significantly reducing processing time for large datasets.
  - **Optimized I/O**: Uses `shutil.copy` for faster file transfer and minimizes console logging to prevent UI lag.
- **Universal Compatibility**: Handles any folder structure depth automatically.
- **Label Normalization**: Automatically maps `medium` folders to the `middle` class.
- **Dual Modes**: Supports both Ratio-based (percentage) and Fixed-count splitting.
- **Robust Validation**: Prevents invalid splits (e.g., empty training sets, insufficient source files).

## `UniversalSplitter` Class

### Parameters

- `source_dir` (str): Path to the source dataset.
- `dest_dir` (str, optional): Output path. Defaults to `datasplit/datasplit_<source_name>`.
- `mode` (str): `'ratio'` or `'fixed'`.
- `test_val` (float|int): Split value (0.0-1.0 for ratio, >0 for fixed).

---

## Usage Examples

### 1. High-Performance Split for Large Datasets (27k)

```python
from split import UniversalSplitter

# Splits 5600 files to Test set (1400 per class)
# Uses multithreading for speed
splitter = UniversalSplitter(
    source_dir="dataset_27k",
    mode='fixed',
    test_val=5600
)
splitter.split()
```

### 2. Standard Split for Small Datasets (3k)

```python
from split import UniversalSplitter

# Splits 6.67% to Test set
splitter = UniversalSplitter(
    source_dir="dataset_3k",
    mode='ratio',
    test_val=0.0667
)
splitter.split()
```

## Performance Notes

- The script uses **8 worker threads** by default for file copying.
- Console output is minimized to show only class-level progress, preventing terminal lag during large batch operations.
