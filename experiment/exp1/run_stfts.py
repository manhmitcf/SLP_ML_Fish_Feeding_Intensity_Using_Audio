import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from features import FeatureManager, STFTSConfig
from processing import ProcessingManager, PipelineConfig, ScalerConfig
from models import ModelManager, KNNConfig, SVMConfig, RFConfig, ETConfig
from utils import StandardDataLoader

def run_experiment():
    print("\n" + "="*60)
    print("EXPERIMENT: STFTs (Manual) - Raw vs Scaled")
    print("="*60)

    # 1. Setup Data Loader
    loader = StandardDataLoader("datasplit/datasplit_dataset_3k")
    class_names = loader.get_class_names()

    # 2. Extract STFTs Features
    print("\n--- Step 1: Extracting STFTs Features ---")
    # Config from notebook
    stfts_conf = STFTSConfig(
        name="stfts",
        sample_rate=128000,
        pre_emph=0.97,
        frame_size=0.025,
        frame_stride=0.01,
        n_fft=2048,
        apply_log=True
    )
    
    stfts_manager = FeatureManager(stfts_conf)
    stfts_data = stfts_manager.get_data(loader)
    
    # 3. Define Processing Pipelines
    pipelines = {
        "Raw": PipelineConfig(steps=[]),
        "Scaled": PipelineConfig(steps=["scaler"], scaler=ScalerConfig(type="standard"))
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
    csv_path = os.path.join(os.path.dirname(__file__), "stfts_results.csv")

    # 5. Run Loop: Pipeline -> Model
    for pipe_name, pipe_conf in pipelines.items():
        print(f"\n>>> Running Pipeline: {pipe_name}")
        
        processor = ProcessingManager(pipe_conf)
        processed = processor.process_data({'stfts': stfts_data})
        
        X_train = processed['X_train']
        y_train = processed['y_train']
        X_test = processed['X_test']
        y_test = processed['y_test']
        
        pipeline_hash = processed['pipeline_hash']
        processing_config_path = f"features_cache/processed/{pipeline_hash}_config.json"

        for config_cls, params in models_config:
            print(f"\n   --- Model: {params['name'].upper()} ---")
            
            model_conf = config_cls(**params)
            manager = ModelManager(model_conf)
            manager.set_upstream_config(processing_config_path)
            
            manager.train(X_train, y_train)
            metrics = manager.evaluate(X_test, y_test, class_names=class_names)
            
            save_name = f"stfts_{pipe_name.lower()}_{model_conf.name}_{timestamp}.joblib"
            manager.save(save_name)
            
            row = {
                "Timestamp": timestamp,
                "Feature": "STFTs",
                "Pipeline": pipe_name,
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
