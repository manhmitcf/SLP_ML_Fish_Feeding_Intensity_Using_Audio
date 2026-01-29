# Configuration Guide

This directory contains JSON configuration files for feature extraction. Each file follows a structured format to separate your custom settings from the library's default reference values.

## JSON Structure

Each configuration file (e.g., `mfcc_default.json`) consists of two main sections:

1.  **`active_config`**: **[EDIT THIS]**
    *   This is the configuration that the code actually uses.
    *   Modify values in this section to change how features are extracted.
    *   Includes settings for the extractor, data augmentation, and system performance.

2.  **`defaults_reference`**: **[READ ONLY]**
    *   This section lists the default parameter values used by the underlying libraries (`librosa` or `numpy`).
    *   Use this as a reference to know what the "standard" values are if you want to revert your changes.
