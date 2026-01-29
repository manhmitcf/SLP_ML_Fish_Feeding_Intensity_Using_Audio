# Feature Extraction Module

This module converts raw audio signals into numerical feature vectors (e.g., MFCCs). The system is designed with a **Config-Driven** architecture, supporting **Smart Caching**, **Data Augmentation**, and **Parallel Processing** to maximize performance and model accuracy.

## Directory Structure

- **`config.py`**: Defines configuration classes (`MFCCConfig`, `AugmentationConfig`).
- **`base.py`**: Abstract base class `BaseFeatureExtractor`. Contains logic for parallel processing and data augmentation.
- **`mfcc.py`**: Concrete `MFCCExtractor` class for MFCC calculation.
- **`augmentation.py`**: Contains data augmentation strategies (noise injection, time shifting, etc.).
- **`manager.py`**: `FeatureManager` - The core orchestrator. Manages caching, augmentation, and pipeline execution.
- **`../configs_json/`**: Directory containing sample `.json` configuration files.

## Key Features

1.  **Data Augmentation**:
    - Automatically generates variations of the **Training** data.
    - **Granular Control**: Enable/disable specific strategies (Noise, Time Shift, Pitch Shift) individually.
    - **Note**: Augmentation is NOT applied to the **Test** set.

2.  **Flexible Configuration**:
    - All parameters (for both extraction and augmentation) are managed via JSON files or Config objects.
    - Easily switch between experiments without modifying the core code.

3.  **Smart Caching System**:
    - The configuration (including augmentation settings) generates a unique Hash.
    - Changing any parameter results in a new cache file, ensuring data consistency.

4.  **High Performance**:
    - Utilizes `ProcessPoolExecutor` to leverage multi-core CPUs.

## Usage Guide

### Step 1: Prepare Configuration
Edit `configs_json/mfcc_default.json` to configure augmentation in detail.

```json
{
    "name": "mfcc",
    "sample_rate": 16000,
    "n_mfcc": 13,
    "augmentation": {
        "enable": true,
        "enable_noise": true,       // Enable Noise Injection
        "enable_time_shift": false, // Disable Time Shift
        "enable_pitch_shift": false,// Disable Pitch Shift
        "noise_factor": 0.005,
        "shift_max": 0.2
    },
    "description": "MFCC experiment with Noise Injection only"
}
```

### Step 2: Retrieve Data
The process remains the same. `FeatureManager` handles augmentation automatically based on the config.

```python
from features import FeatureManager, MFCCConfig
from utils import StandardDataLoader

# 1. Load config from JSON
config = MFCCConfig.from_json("configs_json/mfcc_default.json")

# 2. Setup components
loader = StandardDataLoader("datasplit/datasplit_dataset_27k")

# 3. Initialize Manager and get data
manager = FeatureManager(config=config)
X_train, y_train, X_test, y_test = manager.get_data(loader)

# If augmentation is enabled, X_train will have more samples
print(f"Train samples: {len(X_train)}") 
print(f"Test samples: {len(X_test)}")
```

## Extension
- **Add Augmentation Method**: Create a new class inheriting from `BaseAugmentor` in `augmentation.py`.
- **Add Extraction Method**: Create a new `Config` and `Extractor`, then register them in `FeatureManager`.
