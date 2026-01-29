import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from features import FeatureManager, MFCCConfig, STFTConfig, FFTConfig, FFTSConfig, STFTSConfig, MFCCsConfig
from processing import ProcessingManager, PipelineConfig, ScalerConfig, PCAConfig
from models import ModelManager, KNNConfig, SVMConfig, RFConfig, ETConfig
from utils import StandardDataLoader

# --- CONFIGURATION ---
FEATURE_CONFIG_MAP = {
    "mfcc": (MFCCConfig, "experiment/exp3/configs/mfcc_exp3.json"),
    "stft": (STFTConfig, "experiment/exp3/configs/stft_exp3.json"),
    "fft": (FFTConfig, "experiment/exp3/configs/fft_exp3.json"),
    "ffts": (FFTSConfig, "experiment/exp3/configs/ffts_exp3.json"),
    "stfts": (STFTSConfig, "experiment/exp3/configs/stfts_exp3.json"),
    "mfccs": (MFCCsConfig, "experiment/exp3/configs/mfccs_exp3.json"),
}

MODELS_CONFIG = [
    (KNNConfig, {"name": "knn", "n_neighbors": 5, "n_jobs": 6}),
    (SVMConfig, {"name": "svm", "C": 10, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42}),
    (RFConfig, {"name": "rf", "n_estimators": 100, "n_jobs": 6, "random_state": 42}),
    (ETConfig, {"name": "et", "n_estimators": 100, "n_jobs": 6, "random_state": 42})
]

def get_pipeline(scaled=False, pca=False):
    steps = []
    scaler_conf = None
    pca_conf = None
    
    if scaled:
        steps.append("scaler")
        scaler_conf = ScalerConfig(type="standard")
        
    if pca:
        steps.append("pca")
        pca_conf = PCAConfig(n_components=0.95)
        
    return PipelineConfig(steps=steps, scaler=scaler_conf, pca=pca_conf)

def run_model_evaluation(X_train, y_train, X_test, y_test, feature_name, pipeline_name, timestamp, results_list, class_names, pipeline_hash):
    # Dummy upstream config path
    processing_config_path = f"features_cache/processed/{pipeline_hash}_config.json"
    
    for model_cls, params in MODELS_CONFIG:
        print(f"      Running Model: {params['name'].upper()}")
        
        model_conf = model_cls(**params)
        manager = ModelManager(model_conf)
        # Try to set upstream config if exists, otherwise it will warn and skip
        manager.set_upstream_config(processing_config_path)
        
        manager.train(X_train, y_train)
        metrics = manager.evaluate(X_test, y_test, class_names=class_names)
        
        save_name = f"exp3_{feature_name}_{pipeline_name}_{model_conf.name}_{timestamp}.joblib"
        manager.save(save_name)
        
        row = {
            "Timestamp": timestamp,
            "Feature": feature_name,
            "Pipeline": pipeline_name,
            "Model": model_conf.name.upper(),
            "Shape": str(X_train.shape),
            "Accuracy": metrics['accuracy'],
            "Precision_Macro": metrics['precision_macro'],
            "Recall_Macro": metrics['recall_macro'],
            "F1_Macro": metrics['f1_macro'],
            "Precision_Weighted": metrics['precision_weighted'],
            "Recall_Weighted": metrics['recall_weighted'],
            "F1_Weighted": metrics['f1_weighted'],
            "AUC_OVR": metrics.get('auc_ovr', 0.0),
            "mAP": metrics.get('map', 0.0),
            "Train_Time": metrics['training_time'],
            "Model_File": save_name
        }
        results_list.append(row)

def run_master_experiment(feature_names):
    print("\n" + "="*60)
    print(f"EXP3 MASTER RUN: {feature_names}")
    print("="*60)

    loader = StandardDataLoader("datasplit/datasplit_dataset_3k")
    class_names = loader.get_class_names()
    results_list = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load Features
    print("\n--- Step 1: Loading Features ---")
    loaded_features = {}
    for name in feature_names:
        if name not in FEATURE_CONFIG_MAP:
            print(f"Warning: Feature '{name}' not found in config map. Skipping.")
            continue
            
        config_cls, config_path = FEATURE_CONFIG_MAP[name]
        conf = config_cls.from_json(config_path)
        print(f"   Loading {name.upper()}...")
        loaded_features[name] = FeatureManager(conf).get_data(loader)

    if not loaded_features:
        print("No features loaded. Exiting.")
        return

    # 2. Single Feature Experiments
    print("\n--- Step 2: Single Feature Experiments ---")
    for name, data in loaded_features.items():
        for scaled in [False, True]:
            pipe_name = "Scaled" if scaled else "Raw"
            print(f"   >>> {name.upper()} - {pipe_name}")
            
            pipeline = get_pipeline(scaled=scaled, pca=False)
            processor = ProcessingManager(pipeline)
            processed = processor.process_data({name: data})
            
            run_model_evaluation(
                processed['X_train'], processed['y_train'], 
                processed['X_test'], processed['y_test'],
                name.upper(), pipe_name, timestamp, results_list, class_names, processed['pipeline_hash']
            )

    # 3. Fusion Experiments (2 and 3 combinations)
    print("\n--- Step 3: Fusion Experiments ---")
    
    # Generate combinations
    combos_2 = list(combinations(loaded_features.items(), 2))
    combos_3 = list(combinations(loaded_features.items(), 3))
    all_combos = combos_2 + combos_3
    
    for combo_items in all_combos:
        combo_names = [item[0] for item in combo_items]
        combo_data = [item[1] for item in combo_items]
        combo_name_str = "+".join(combo_names)
        
        print(f"\n   >>> Fusion: {combo_name_str}")
        
        # Scenarios: 
        # 1. Raw (No Scale, No PCA)
        # 2. Scaled (Scale Separate, No PCA)
        # 3. Raw + PCA (No Scale, PCA Separate)
        # 4. Scaled + PCA (Scale Separate, PCA Separate)
        
        scenarios = [
            ("Raw", False, False),
            ("Scaled", True, False),
            ("Raw+PCA", False, True),
            ("Scaled+PCA", True, True)
        ]
        
        for pipe_label, do_scale, do_pca in scenarios:
            print(f"      Pipeline: {pipe_label}")
            
            # Define pipeline for individual components
            pipeline = get_pipeline(scaled=do_scale, pca=do_pca)
            
            # Process each component separately then concatenate
            processed_parts_train = []
            processed_parts_test = []
            component_hashes = []
            
            for feat_name, feat_data in zip(combo_names, combo_data):
                proc = ProcessingManager(pipeline)
                res = proc.process_data({feat_name: feat_data})
                processed_parts_train.append(res['X_train'])
                processed_parts_test.append(res['X_test'])
                component_hashes.append(res['pipeline_hash'])
            
            X_train = np.hstack(processed_parts_train)
            X_test = np.hstack(processed_parts_test)
            y_train = combo_data[0]['y_train']
            y_test = combo_data[0]['y_test']
            
            # Create a dummy hash for the fusion
            fusion_hash = f"fusion_{combo_name_str}_{pipe_label}_{timestamp}"
            
            # Save dummy config for traceability
            upstream_info = {
                "description": f"Fusion {combo_name_str} ({pipe_label})",
                "components": component_hashes,
                "pipeline": pipe_label
            }
            dummy_config_path = f"features_cache/processed/{fusion_hash}_config.json"
            with open(dummy_config_path, 'w') as f: json.dump(upstream_info, f, indent=4)
            
            run_model_evaluation(
                X_train, y_train, X_test, y_test,
                combo_name_str, pipe_label, timestamp, results_list, class_names, fusion_hash
            )

    # 4. Save Results
    csv_path = os.path.join(os.path.dirname(__file__), "exp3_master_results.csv")
    df = pd.DataFrame(results_list)
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
        
    print("\n" + "="*60)
    print(f"EXP3 Completed. Results saved to: {csv_path}")
    print("="*60)

if __name__ == "__main__":
    # Example usage: Pass feature names as arguments or input
    if len(sys.argv) > 1:
        features_input = sys.argv[1:]
    else:
        # Default input if no args provided
        print("Enter feature names separated by space (e.g., mfcc stft fft):")
        user_input = input().strip()
        if user_input:
            features_input = user_input.split()
        else:
            print("No input provided. Using default: ['mfcc', 'stft']")
            features_input = ['mfcc', 'stft']
            
    run_master_experiment(features_input)
