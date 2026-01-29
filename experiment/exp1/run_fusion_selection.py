import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from features import FeatureManager, FFTSConfig, MFCCsConfig, STFTConfig
from processing import ProcessingManager, PipelineConfig, ScalerConfig, SelectionConfig
from models import ModelManager, SVMConfig, RFConfig, ETConfig, KNNConfig
from utils import StandardDataLoader

@dataclass
class DummyConfig:
    """Helper class to mimic BaseFeatureConfig for manual combinations."""
    name: str
    description: str = ""
    
    def to_dict(self):
        return asdict(self)

def run_experiment():
    print("\n" + "=" * 60)
    print("EXPERIMENT: Fusion (FFTs + MFCCs + STFT) with Feature Selection")
    print("=" * 60)

    # 1. Setup Data Loader
    loader = StandardDataLoader("datasplit/datasplit_dataset_3k")
    class_names = loader.get_class_names()

    # 2. Load Existing Features (from cache)
    print("\n--- Step 1: Loading Features from Cache ---")

    # FFTS
    ffts_conf = FFTSConfig(
        name="ffts",
        sample_rate=128000,
        pre_emph=0.97,
        frame_size=0.025,
        frame_stride=0.008,
        n_fft=2048,
        apply_log=True
    )

    ffts_manager = FeatureManager(ffts_conf)
    ffts_data = ffts_manager.get_data(loader)

    # MFCCs
    mfccs_conf = MFCCsConfig(
        name="mfccs",
        sample_rate=128000,
        pre_emph=0.97,
        n_mfcc=20,
        n_fft=2048,
        hop_length=512,
        apply_log=False
    )

    mfccs_manager = FeatureManager(mfccs_conf)
    mfccs_data = mfccs_manager.get_data(loader)

    # STFT
    stft_conf = STFTConfig(
        name="stft",
        sample_rate=128000,
        n_fft=2048,
        hop_length=1024,
        scaling='log'
    )

    stft_manager = FeatureManager(stft_conf)
    stft_data = stft_manager.get_data(loader)
    
    # Combine all features before processing
    X_train_combined = np.hstack((ffts_data['X_train'], mfccs_data['X_train'], stft_data['X_train']))
    X_test_combined = np.hstack((ffts_data['X_test'], mfccs_data['X_test'], stft_data['X_test']))
    y_train = ffts_data['y_train']
    y_test = ffts_data['y_test']
    
    print(f"\nInitial Combined Shape: {X_train_combined.shape}")

    # 3. Define Selection Pipelines
    pipelines = {
        "Variance_Threshold": PipelineConfig(
            steps=["selection", "scaler"],
            scaler=ScalerConfig(type="standard"),
            selection=SelectionConfig(method="variance", threshold=0.086)
        ),
        "Select_K_Best_1000": PipelineConfig(
            steps=["selection", "scaler"],
            scaler=ScalerConfig(type="standard"),
            selection=SelectionConfig(method="k_best", k=2066)
        )
    }

    # 4. Define Models
    models_config = [
        (KNNConfig, {"name": "knn", "n_neighbors": 5, "n_jobs": 6}),
        (SVMConfig, {"name": "svm", "C": 100, "kernel": "rbf", "gamma": 0.1, "probability": True, "random_state": 42}),
        (RFConfig, {"name": "rf", "n_estimators": 100, "n_jobs": 6, "random_state": 42}),
        (ETConfig, {"name": "et", "n_estimators": 200, "max_depth": 30, "n_jobs": 6, "random_state": 42})
    ]


    # Prepare CSV results
    results_list = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(os.path.dirname(__file__), "fusion_selection_results.csv")

    # 5. Run Loop
    for pipe_name, pipe_conf in pipelines.items():
        print(f"\n>>> Running Pipeline: {pipe_name}")

        processor = ProcessingManager(pipe_conf)

        # Create a dummy feature_set with a proper config object
        dummy_config = DummyConfig(name="manual_combination_ffts_mfccs_stft")
        
        dummy_feature_set = {
            'combined': {
                'X_train': X_train_combined, 'y_train': y_train,
                'X_test': X_test_combined, 'y_test': y_test,
                'config_hash': 'manual_combination', 
                'config': dummy_config # Now an object with to_dict()
            }
        }
        
        processed = processor.process_data(dummy_feature_set)

        X_train = processed['X_train']
        y_train = processed['y_train']
        X_test = processed['X_test']
        y_test = processed['y_test']

        print(f"Shape AFTER {pipe_name}: {X_train.shape}")

        pipeline_hash = processed['pipeline_hash']
        processing_config_path = f"features_cache/processed/{pipeline_hash}_config.json"

        for config_cls, params in models_config:
            print(f"\n   --- Model: {params['name'].upper()} ---")

            model_conf = config_cls(**params)
            manager = ModelManager(model_conf)
            manager.set_upstream_config(processing_config_path)

            manager.train(X_train, y_train)
            metrics = manager.evaluate(X_test, y_test, class_names=class_names)

            save_name = f"fusion_sel_{pipe_name.lower()}_{model_conf.name}_{timestamp}.joblib"
            manager.save(save_name)

            row = {
                "Timestamp": timestamp,
                "Feature": "Fusion_FFTs_MFCCs_STFT",
                "Pipeline": pipe_name,
                "Model": model_conf.name.upper(),
                "Accuracy": metrics['accuracy'],
                "F1_Macro": metrics['f1_macro'],
                "AUC_OVR": metrics.get('auc_ovr', 0.0),
                "mAP": metrics.get('map', 0.0),
                "Train_Time": metrics['training_time'],
                "Num_Features": X_train.shape[1],
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

    print("\n" + "=" * 60)
    print(f"Experiment Completed. Results saved to: {csv_path}")
    print("=" * 60)
    print(df[["Pipeline", "Model", "Num_Features", "Accuracy", "F1_Macro"]])


if __name__ == "__main__":
    run_experiment()
