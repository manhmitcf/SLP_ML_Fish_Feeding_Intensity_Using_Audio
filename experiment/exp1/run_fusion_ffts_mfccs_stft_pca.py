import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from features import FeatureManager, FFTSConfig, MFCCConfig, STFTConfig
from processing import ProcessingManager, PipelineConfig, PCAConfig
from models import ModelManager, SVMConfig, RFConfig, ETConfig
from utils import StandardDataLoader

def run_experiment():
    print("\n" + "="*60)
    print("EXPERIMENT: Fusion (FFTs + MFCCs + STFT) with Individual PCA")
    print("="*60)

    # 1. Setup Data Loader
    loader = StandardDataLoader("datasplit/datasplit_dataset_3k")
    class_names = loader.get_class_names()

    # 2. Extract Raw Features
    print("\n--- Step 1: Extracting Raw Features ---")
    
    # FFT (Manual - FFTS)
    fft_conf = FFTSConfig(name="ffts_pca", sample_rate=128000, n_fft=512, frame_size=0.025, apply_log=True)
    fft_data = FeatureManager(fft_conf).get_data(loader)
    
    # MFCC
    mfcc_conf = MFCCConfig(name="mfcc_pca", sample_rate=128000, n_mfcc=13, n_fft=512)
    mfcc_data = FeatureManager(mfcc_conf).get_data(loader)
    
    # STFT
    stft_conf = STFTConfig(name="stft_pca", sample_rate=128000, n_fft=2048, hop_length=1024, scaling='log')
    stft_data = FeatureManager(stft_conf).get_data(loader)

    # 3. Apply PCA Individually
    print("\n--- Step 2: Applying PCA to each feature set ---")
    
    # PCA for FFT (130 components)
    pca_fft_conf = PipelineConfig(steps=["pca"], pca=PCAConfig(n_components=130))
    proc_fft = ProcessingManager(pca_fft_conf)
    fft_pca = proc_fft.process_data({'fft': fft_data})
    
    # PCA for MFCC (13 components)
    pca_mfcc_conf = PipelineConfig(steps=["pca"], pca=PCAConfig(n_components=13))
    proc_mfcc = ProcessingManager(pca_mfcc_conf)
    mfcc_pca = proc_mfcc.process_data({'mfcc': mfcc_data})
    
    # PCA for STFT (80 components)
    pca_stft_conf = PipelineConfig(steps=["pca"], pca=PCAConfig(n_components=80))
    proc_stft = ProcessingManager(pca_stft_conf)
    stft_pca = proc_stft.process_data({'stft': stft_data})

    # 4. Combine PCA-transformed Features
    print("\n--- Step 3: Combining PCA-transformed features ---")
    X_train = np.hstack((fft_pca['X_train'], mfcc_pca['X_train'], stft_pca['X_train']))
    X_test = np.hstack((fft_pca['X_test'], mfcc_pca['X_test'], stft_pca['X_test']))
    y_train = fft_pca['y_train']
    y_test = fft_pca['y_test']
    
    print(f"Final Combined Shape: {X_train.shape}")

    # 5. Define Models
    models_config = [
        (SVMConfig, {"name": "svm", "C": 10, "kernel": "rbf", "gamma": 0.0075, "probability": True, "random_state": 42}),
        (RFConfig, {"name": "rf", "n_estimators": 100, "n_jobs": 6, "random_state": 42}),
        (ETConfig, {"name": "et", "n_estimators": 200, "criterion": "log_loss", "n_jobs": 6, "random_state": 42})
    ]

    # Prepare CSV results
    results_list = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(os.path.dirname(__file__), "fusion_ffts_mfccs_stft_pca_results.csv")

    # 6. Run Loop
    for config_cls, params in models_config:
        print(f"\n   --- Model: {params['name'].upper()} ---")
        
        model_conf = config_cls(**params)
        manager = ModelManager(model_conf)
        
        # Dummy upstream config for traceability
        upstream_info = {
            "description": "Fusion of individual PCA components (FFTs+MFCCs+STFT)",
            "components": {
                "fft_pca": fft_pca['pipeline_hash'],
                "mfcc_pca": mfcc_pca['pipeline_hash'],
                "stft_pca": stft_pca['pipeline_hash']
            }
        }
        dummy_config_path = f"features_cache/processed/fusion_ffts_mfccs_stft_pca_{timestamp}.json"
        import json
        with open(dummy_config_path, 'w') as f:
            json.dump(upstream_info, f, indent=4)
        manager.set_upstream_config(dummy_config_path)
        
        manager.train(X_train, y_train)
        metrics = manager.evaluate(X_test, y_test, class_names=class_names)
        
        save_name = f"fusion_ffts_mfccs_stft_pca_{model_conf.name}_{timestamp}.joblib"
        manager.save(save_name)
        
        row = {
            "Timestamp": timestamp,
            "Feature": "Fusion_PCA(FFTs+MFCCs+STFT)",
            "Pipeline": "Individual_PCA",
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

    # 7. Save to CSV
    df = pd.DataFrame(results_list)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
    
    print("\n" + "="*60)
    print(f"Experiment Completed. Results saved to: {csv_path}")
    print("="*60)
    print(df[["Model", "Accuracy", "F1_Macro", "Train_Time"]])

if __name__ == "__main__":
    run_experiment()
