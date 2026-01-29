import os
import json
import pandas as pd
from pathlib import Path

def update_summary_with_full_metrics():
    print("\n" + "="*60)
    print("UPDATING SUMMARY WITH FULL METRICS FROM CACHE")
    print("="*60)

    # Paths
    exp_dir = os.path.dirname(__file__)
    summary_path = os.path.join(exp_dir, "final_summary.csv")
    models_cache_dir = os.path.abspath(os.path.join(exp_dir, '..', '..', 'models_cache'))

    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found at {summary_path}")
        print("Please run 'summarize_results.py' first.")
        return

    print(f"Reading summary from: {summary_path}")
    df = pd.read_csv(summary_path)
    
    if "Model_File" not in df.columns:
        print("Error: 'Model_File' column missing in summary CSV. Cannot link to cache.")
        return

    # List of metrics to extract
    metrics_to_extract = [
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "auc_ovr", "map", "training_time"
    ]

    updated_rows = []
    
    for index, row in df.iterrows():
        model_file = row.get("Model_File")
        if pd.isna(model_file):
            updated_rows.append(row)
            continue

        # Construct path to config json
        # Model file: name.joblib -> Config file: name.config.json
        config_filename = Path(model_file).with_suffix('.config.json')
        config_path = os.path.join(models_cache_dir, config_filename)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    full_config = json.load(f)
                
                eval_results = full_config.get("evaluation_results", {})
                
                # Update row with all metrics found in cache
                for metric in metrics_to_extract:
                    if metric in eval_results:
                        row[metric] = eval_results[metric]
                
                # Also try to get confusion matrix if needed (as string)
                if "confusion_matrix" in eval_results:
                    row["confusion_matrix"] = str(eval_results["confusion_matrix"])

            except Exception as e:
                print(f"Warning: Failed to read config for {model_file}: {e}")
        else:
            print(f"Warning: Config file not found for {model_file}")
        
        updated_rows.append(row)

    # Create updated DataFrame
    full_df = pd.DataFrame(updated_rows)
    
    # Save to new file
    output_path = os.path.join(exp_dir, "final_summary_full.csv")
    full_df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print(f"Successfully created FULL summary at: {output_path}")
    print(f"Columns: {list(full_df.columns)}")
    print("="*60)

if __name__ == "__main__":
    update_summary_with_full_metrics()
