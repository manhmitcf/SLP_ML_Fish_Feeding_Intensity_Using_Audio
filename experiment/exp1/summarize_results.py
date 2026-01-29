import os
import pandas as pd
import glob

def summarize_all_results():
    """
    Finds all '*_results.csv' files in the current directory,
    concatenates them, and saves a final summary file.
    """
    
    current_dir = os.path.dirname(__file__)
    output_filename = "final_summary.csv"
    
    # Find all individual result files, excluding the final summary file itself
    csv_files = [f for f in glob.glob(os.path.join(current_dir, "*_results.csv")) 
                 if os.path.basename(f) != output_filename]
    
    if not csv_files:
        print("No result CSV files found to summarize.")
        return

    print("Found the following result files to combine:")
    for f in csv_files:
        print(f" - {os.path.basename(f)}")

    # Read and concatenate all found CSVs
    df_list = [pd.read_csv(f) for f in csv_files]
    summary_df = pd.concat(df_list, ignore_index=True)

    # Sort by best accuracy
    summary_df = summary_df.sort_values(by="Accuracy", ascending=False)

    # Save the final summary
    output_path = os.path.join(current_dir, output_filename)
    summary_df.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print(f"Successfully created summary file at: {output_path}")
    print(f"Total experiments summarized: {len(summary_df)}")
    print("="*60)
    
    # Display the top 10 results
    print("\nTop 10 Results by Accuracy:")
    print(summary_df.head(10).to_string())

if __name__ == "__main__":
    summarize_all_results()
