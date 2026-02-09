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
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping corrupt or unreadable file {csv_path}: {e}")
            continue
            
        # Define the base name for this run
        base_col_name = f"{algo_name} ({batch_id})"
        
        # Determine which columns to keep and how to rename them
        # We check if Mean_Disc_Ret exists to support older summary files that might not have it
        cols_to_keep = ["Environment", "Mean_Ret"]
        rename_map = {"Mean_Ret": f"{base_col_name} [Ret]"}
        
        if "Mean_Disc_Ret" in df.columns:
            cols_to_keep.append("Mean_Disc_Ret")
            rename_map["Mean_Disc_Ret"] = f"{base_col_name} [Disc]"
            
        # Select and Rename
        df = df[cols_to_keep].rename(columns=rename_map)
        
        # Set index to Environment for easier joining
        df.set_index("Environment", inplace=True)
        all_dfs.append(df)

    if not all_dfs:
        print("No valid data found to aggregate.")
        return None

    # Join all dataframes on the 'Environment' index
    # 'outer' join ensures we keep all environments found across all files
    comparison_df = pd.concat(all_dfs, axis=1, join="outer")
    
    # Sort by Environment name
    comparison_df = comparison_df.sort_index()
    
    return comparison_df

if __name__ == "__main__":
    df_comparison = aggregate_all_summaries()
    
    if df_comparison is not None:
        print("\n=== Cross-Run Comparison (Returns & Discounted Returns) ===")
        # Using to_string() makes sure pandas prints the whole table without truncating
        print(df_comparison.to_string())
        
        # Save the master comparison
        output_path = "results/master_comparison.csv"
        df_comparison.to_csv(output_path)
        print(f"\nSaved master comparison to {output_path}")