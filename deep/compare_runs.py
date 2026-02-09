import os
import pandas as pd
from pathlib import Path

def aggregate_all_summaries(results_dir="results"):
    results_path = Path(results_dir)
    all_dfs = []

    # This looks in any subfolder for a CSV that ends in '_summary.csv'
    summary_files = list(results_path.glob("**/*_summary.csv"))

    if not summary_files:
        print("No summary CSVs found. Ensure your batch folders start with 'batch_'.")
        return None

    for csv_path in summary_files:
        # Extract metadata from the path
        # Path: results/algo_name/batch_id/algo_summary.csv
        algo_name = csv_path.parent.parent.name
        batch_id = csv_path.parent.name
        
        # Load the CSV
        df = pd.read_csv(csv_path)
        
        # We only need Environment and Mean_Ret for the comparison
        # We rename Mean_Ret to include the algo and timestamp for uniqueness
        column_name = f"{algo_name} ({batch_id})"
        df = df[["Environment", "Mean_Ret"]].rename(columns={"Mean_Ret": column_name})
        
        # Set index to Environment for easier joining
        df.set_index("Environment", inplace=True)
        all_dfs.append(df)

    # Join all dataframes on the 'Environment' index
    # 'outer' join ensures we keep all environments found across all files
    comparison_df = pd.concat(all_dfs, axis=1, join="outer")
    
    # Fill missing values (envs not run in a specific batch) with NaN or 0
    comparison_df = comparison_df.sort_index()
    
    return comparison_df

if __name__ == "__main__":
    df_comparison = aggregate_all_summaries()
    
    if df_comparison is not None:
        print("\n=== Cross-Run Comparison (Mean Returns) ===")
        print(df_comparison.to_string())
        
        # Save the master comparison
        df_comparison.to_csv("results/master_comparison.csv")
        print("\nSaved master comparison to results/master_comparison.csv")