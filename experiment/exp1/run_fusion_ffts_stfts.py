import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from features import FeatureManager, FFTSConfig, STFTSConfig
from processing import ProcessingManager, PipelineConfig, ScalerConfig
from models import ModelManager, KNNConfig, SVMConfig, RFConfig, ETConfig
from utils import StandardDataLoader

def run_experiment():
    print("\n" + "="*60)
    print("EXPERIMENT: Fusion (FFTs + STFTs) - Raw vs Scaled (Separate Scaling)")
    print("="*60)

    # 1. Setup Data Loader
    loader = StandardDataLoader("datasplit/datasplit_dataset_3k")
    class_names = loader.get_class_names()

    # 2. Extract Features
    print("\n--- Step 1: Extracting Features ---")
    
    # FFTs (Manual)
    ffts_conf = FFTSConfig(
        name="ffts", sample_rate=128000, pre_emph=0.97, 
        frame_size=0.025, frame_stride=0.008, n_fft=2048, apply_log=True
    )
    ffts_data = FeatureManager(ffts_conf).get_data(loader)
    
    # STFTs (Manual)
    stfts_conf = STFTSConfig(
        name="stfts", sample_rate=128000, pre_emph=0.97, 
        frame_size=0.025, frame_stride=0.008, n_fft=2048, apply_log=True
    )
    stfts_data = FeatureManager(stfts_conf).get_data(loader)

    # 3. Prepare Datasets (Raw & Scaled)
    print("\n--- Step 2: Preparing Datasets ---")
    
    # A. Raw (No Scaling)
    X_train_raw = np.hstack((ffts_data['X_train'], stfts_data['X_train']))
    X_test_raw = np.hstack((ffts_data['X_test'], stfts_data['X_test']))
    y_train = ffts_data['y_train']
    y_test = ffts_data['y_test']
    
    print(f"Raw Combined Shape: {X_train_raw.shape}")

    # B. Scaled (Separate Scaling before Concatenation)
    scaler_conf = PipelineConfig(steps=["scaler"], scaler=ScalerConfig(type="standard"))
    
    proc_ffts = ProcessingManager(scaler_conf)
    ffts_scaled = proc_ffts.process_data({'ffts': ffts_data})
    
    proc_stfts = ProcessingManager(scaler_conf)
    stfts_scaled = proc_stfts.process_data({'stfts': stfts_data})
    
    X_train_scaled = np.hstack((ffts_scaled['X_train'], stfts_scaled['X_train']))
    X_test_scaled = np.hstack((ffts_scaled['X_test'], stfts_scaled['X_test']))
    
    print(f"Scaled Combined Shape: {X_train_scaled.shape}")

    datasets = {
        "Raw": (X_train_raw, X_test_raw),
        "Scaled": (X_train_scaled, X_test_scaled)
    }

    # 4. Define Models
    models_config = [
        (KNNConfig, {"name": "knn", "n_neighbors": 5, "n_jobs": 6}),
        (SVMConfig, {"name": "svm", "C": 100, "kernel": "rbf", "gamma": 0.1, "probability": True, "random_state": 42}),
        (RFConfig, {"name": "rf", "n_estimators": 100, "n_jobs": 6, "random_state": 42}),
        (ETConfig, {"name": "et", "n_estimators": 200, "n_jobs": 6, "random_state": 42})
    ]

    # Prepare CSV results
    results_list = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(os.path.dirname(__file__), "fusion_ffts_stfts_results.csv")

    # 5. Run Loop
    for data_name, (X_tr, X_te) in datasets.items():
        print(f"\n>>> Running on Dataset: {data_name}")
        
        # Dummy upstream config for traceability
        upstream_info = {
            "description": f"Fusion FFTs+STFTs ({data_name})",
            "components": [ffts_data['config_hash'], stfts_data['config_hash']],
            "scaling": "Separate" if data_name == "Scaled" else "None"
        }
        dummy_config_path = f"features_cache/processed/fusion_ffts_stfts_{data_name.lower()}_{timestamp}.json"
        import json
        with open(dummy_config_path, 'w') as f:
            json.dump(upstream_info, f, indent=4)

        for config_cls, params in models_config:
            print(f"\n   --- Model: {params['name'].upper()} ---")
            
            model_conf = config_cls(**params)
            manager = ModelManager(model_conf)
            manager.set_upstream_config(dummy_config_path)
            
            manager.train(X_tr, y_train)
            metrics = manager.evaluate(X_te, y_test, class_names=class_names)
            
            save_name = f"fusion_ffts_stfts_{data_name.lower()}_{model_conf.name}_{timestamp}.joblib"
            manager.save(save_name)
            
            row = {
                "Timestamp": timestamp,
                "Feature": "Fusion_FFTs_STFTs",
                "Pipeline": data_name,
                "Model": model_conf.name.upper(),
                "Accuracy": metrics['accuracy'],
                "F1_Macro": metrics['f1_macro'],
                "AUC_OVR": metrics.get('auc_ovr', 0.0),
                "mAP": metrics.get('map', 0.0),
                "Train_Time": metrics['training_time'],
                "Model_File": save_name
            }
            results_list.append(row)
            print(f"   Accuracy: {metrics['accuracy']:.4f}")

    # 6. Save to CSV
    df = pd.DataFrame(results_list)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
    
    print("\n" + "="*60)
    print(f"Experiment Completed. Results saved to: {csv_path}")
    print("="*60)
    print(df[["Pipeline", "Model", "Accuracy", "F1_Macro", "Train_Time"]])

if __name__ == "__main__":
    run_experiment()
