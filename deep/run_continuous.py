# run_all.py
from core.configs import CONTINUOUS_REGISTRY
import subprocess
import datetime
import argparse
import datetime
import argparse
import os
import jax.numpy as jnp
import pandas as pd
from core.utils import load_run_data
from notebooks.mail import email_results_file

def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="algos/cov_lstd.py")
    parser.add_argument("--config", type=str, default=None) # Capture the override here
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()

    batch_id = args.suffix if args.suffix else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Determine result root based on script name (e.g., results/cov_lstd)
    script_base = os.path.basename(args.script).replace(".py", "")
    results_root = os.path.join("results", script_base)

    print(f"🚀 Starting Batch Run: {batch_id}")
    module_path = args.script.replace(".py", "").replace("/", ".")
    
    for config_name, details in CONTINUOUS_REGISTRY.items():
        envs = details["envs"]
        print(f"\n# Running {config_name} group: {envs}")
        
        cmd = [
            "python", "-m", module_path,
            "--base-config", config_name,
            "--run_suffix", batch_id,
            "--env_ids"
        ] + envs

        if args.config:
            cmd += ["--config", args.config]
        
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd)

    # --- IMMEDIATE AGGREGATION ---
    print("\n" + "="*50)
    print(f"📊 GENERATING SUMMARY FOR BATCH: {batch_id}")
    print("="*50)

    df = summarize_batch(batch_id, results_root)

    if df is not None and not df.empty:
        df = df.sort_values(["Group", "Mean_Ret"], ascending=[True, False])
        print(df.to_string(index=False))
        
        # NEW: Name the CSV after the algorithm run
        summary_filename = f"{script_base}_summary.csv"
        summary_path = os.path.join(results_root, batch_id, summary_filename)
        
        df.to_csv(summary_path, index=False)
        print(f"\nSaved summary table to: {summary_path}")
        
        try:
            # The filename in the email attachment will now be 'cov_lstd_summary.csv'
            email_results_file(summary_path)
        except Exception as e:
            print(f'failed to email: {e}')

if __name__ == "__main__":
    # Ensure summarize_batch is accessible
    from run_all import summarize_batch # or just define it above run_experiment
    run_experiment()


def summarize_batch(batch_id, results_root):
    all_data = []
    batch_path = os.path.join(results_root, batch_id)
    
    if not os.path.exists(batch_path):
        return None

    envs_found = [d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))]

    for env_name in envs_found:
        try:
            config, out = load_run_data(
                run_folder_name=batch_id,
                env_name=env_name,
                results_base_path=results_root,
            )
            
            # Helper to safely get metrics whether they are at root or inside 'metrics'
            def get_metric(key):
                return out.get(key, out.get("metrics", {}).get(key))

            # --- 1. Undiscounted Returns ---
            rets = get_metric("returned_episode_returns")
            
            if rets is not None:
                final_mean = rets[:, -20:].mean()
                final_std = rets[:, -20:].mean(-1).std()
            else:
                final_mean, final_std = 0, 0

            # --- 2. Discounted Returns ---
            disc_rets = get_metric("returned_discounted_episode_returns")
            
            if disc_rets is not None:
                disc_mean = disc_rets[:, -20:].mean()
                disc_std = disc_rets[:, -20:].mean(-1).std()
            else:
                disc_mean, disc_std = 0, 0

            # Map back to Config Group
            group = next((g for g, d in CONFIG_REGISTRY.items() if env_name in d["envs"]), "other")

            all_data.append({
                "Group": group,
                "Environment": env_name,
                "Mean_Ret": round(float(final_mean), 2),
                "Std_Ret": round(float(final_std), 2),
                "Mean_Disc_Ret": round(float(disc_mean), 2), # New
                "Std_Disc_Ret": round(float(disc_std), 2),   # New
                "Steps": config.get("TOTAL_TIMESTEPS", "n/a")
            })
        except Exception as e:
            print(f"Could not process {env_name}: {e}")

    return pd.DataFrame(all_data)
