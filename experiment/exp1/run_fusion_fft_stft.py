import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from features import FeatureManager, FFTConfig, STFTConfig
from processing import ProcessingManager, PipelineConfig, ScalerConfig
from models import ModelManager, KNNConfig, SVMConfig, RFConfig, ETConfig
from utils import StandardDataLoader

def run_experiment():
    print("\n" + "="*60)
    print("EXPERIMENT: Fusion (FFT + STFT) - Raw vs Scaled (Separate Scaling)")
    print("="*60)

    # 1. Setup Data Loader
    loader = StandardDataLoader("datasplit/datasplit_dataset_3k")
    class_names = loader.get_class_names()

    # 2. Extract Features
    print("\n--- Step 1: Extracting Features ---")
    
    # FFT (Origin)
    fft_conf = FFTConfig(name="fft_origin", sample_rate=128000, n_fft=2048)
    fft_data = FeatureManager(fft_conf).get_data(loader)
    
    # STFT
    stft_conf = STFTConfig(name="stft", sample_rate=128000, n_fft=2048, hop_length=1024, scaling='log')
    stft_data = FeatureManager(stft_conf).get_data(loader)

    # 3. Prepare Datasets (Raw & Scaled)
    print("\n--- Step 2: Preparing Datasets ---")
    
    # A. Raw (No Scaling)
    # Concatenate raw features directly
    X_train_raw = np.hstack((fft_data['X_train'], stft_data['X_train']))
    X_test_raw = np.hstack((fft_data['X_test'], stft_data['X_test']))
    y_train = fft_data['y_train'] # Labels are same
    y_test = fft_data['y_test']
    
    print(f"Raw Combined Shape: {X_train_raw.shape}")

    # B. Scaled (Separate Scaling before Concatenation)
    # Scale FFT
    scaler_conf = PipelineConfig(steps=["scaler"], scaler=ScalerConfig(type="standard"))
    proc_fft = ProcessingManager(scaler_conf)
    fft_scaled = proc_fft.process_data({'fft': fft_data})
    
    # Scale STFT
    proc_stft = ProcessingManager(scaler_conf)
    stft_scaled = proc_stft.process_data({'stft': stft_data})
    
    # Concatenate scaled features
    X_train_scaled = np.hstack((fft_scaled['X_train'], stft_scaled['X_train']))
    X_test_scaled = np.hstack((fft_scaled['X_test'], stft_scaled['X_test']))
    
    print(f"Scaled Combined Shape: {X_train_scaled.shape}")

    # Define datasets dictionary for loop
    datasets = {
        "Raw": (X_train_raw, X_test_raw),
        "Scaled": (X_train_scaled, X_test_scaled)
    }

    # 4. Define Models
    models_config = [
        (KNNConfig, {"name": "knn", "n_neighbors": 3, "n_jobs": 6}),
        (SVMConfig, {"name": "svm", "C": 100, "kernel": "rbf", "gamma": 0.1, "probability": True, "random_state": 42}),
        (RFConfig, {"name": "rf", "n_estimators": 100, "n_jobs": 6, "random_state": 42}),
        (ETConfig, {"name": "et", "n_estimators": 100, "n_jobs": 6, "random_state": 42})
    ]

    # Prepare CSV results
    results_list = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(os.path.dirname(__file__), "fusion_fft_stft_results.csv")

    # 5. Run Loop: Dataset -> Model
    for data_name, (X_tr, X_te) in datasets.items():
        print(f"\n>>> Running on Dataset: {data_name}")
        
        # Construct a virtual pipeline config path for traceability
        # Since we manually combined, we point to the component configs
        upstream_info = {
            "description": f"Fusion FFT+STFT ({data_name})",
            "components": [fft_data['config_hash'], stft_data['config_hash']],
            "scaling": "Separate" if data_name == "Scaled" else "None"
        }
        # We can save this to a temp json if we want strict ModelManager compatibility,
        # or just rely on the model config saving. For now, we skip set_upstream_config 
        # or create a dummy one. Let's create a dummy one.
        dummy_config_path = f"features_cache/processed/fusion_{data_name.lower()}_{timestamp}.json"
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
            
            save_name = f"fusion_fft_stft_{data_name.lower()}_{model_conf.name}_{timestamp}.joblib"
            manager.save(save_name)
            
            row = {
                "Timestamp": timestamp,
                "Feature": "Fusion_FFT_STFT",
                "Pipeline": data_name, # Raw or Scaled
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
