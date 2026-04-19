# run_all.py
from core.configs import CONFIG_REGISTRY, FINAL_TESTING, FINAL_EXACT
import subprocess
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
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)  
    # NEW: Allow dynamic registry selection
    parser.add_argument("--registry", type=str, default="FINAL_TESTING", 
                        choices=["CONFIG_REGISTRY", "FINAL_TESTING", "FINAL_EXACT"])
    args = parser.parse_args()

    # Determine which registry to use
    if args.registry == "FINAL_TESTING":
        active_registry = FINAL_TESTING
    elif args.registry == "FINAL_EXACT":
        active_registry = FINAL_EXACT
    else:
        active_registry = CONFIG_REGISTRY

    batch_id = (
        args.suffix
        if args.suffix
        else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    
    script_base = os.path.basename(args.script).replace(".py", "")
    results_root = os.path.join("results", script_base)

    print(f"🚀 Starting Batch Run: {batch_id} (Using {args.registry})")
    module_path = args.script.replace(".py", "").replace("/", ".")

    for config_name, details in active_registry.items():
        envs = details["envs"]
        print(f"\n# Running {config_name} group: {envs}")

        cmd = [
            "python",
            "-m",
            module_path,
            "--base-config",
            config_name,
            "--run_suffix",
            batch_id,
            "--env-ids",
        ] + envs

        if args.config:
            cmd += ["--config", args.config]

        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd)

    # --- IMMEDIATE AGGREGATION ---
    print("\n" + "=" * 50)
    print(f"📊 GENERATING SUMMARY FOR BATCH: {batch_id}")
    print("=" * 50)

    # Pass active_registry so the summarizer maps the groups correctly
    df = summarize_batch(batch_id, results_root, active_registry)

    if df is not None and not df.empty:
        df = df.sort_values(["Group", "Mean_Ret"], ascending=[True, False])
        print(df.to_string(index=False))

        summary_filename = f"{script_base}_summary.csv"
        summary_path = os.path.join(results_root, batch_id, summary_filename)

        df.to_csv(summary_path, index=False)
        print(f"\nSaved summary table to: {summary_path}")

        try:
            email_results_file(summary_path)
        except Exception as e:
            print(f"failed to email: {e}")

def summarize_batch(batch_id, results_root, active_registry):
    all_data = []
    batch_path = os.path.join(results_root, batch_id)

    if not os.path.exists(batch_path):
        return None

    envs_found = [
        d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))
    ]

    for env_name in envs_found:
        try:
            config, out = load_run_data(
                run_folder_name=batch_id,
                env_name=env_name,
                results_base_path=results_root,
            )

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

            # Map back to Config Group dynamically using the passed registry
            group = next(
                (g for g, d in active_registry.items() if env_name in d["envs"]),
                "other",
            )

            all_data.append(
                {
                    "Group": group,
                    "Environment": env_name,
                    "Mean_Ret": round(float(final_mean), 2),
                    "Std_Ret": round(float(final_std), 2),
                    "Mean_Disc_Ret": round(float(disc_mean), 2),
                    "Std_Disc_Ret": round(float(disc_std), 2),
                    "Steps": config.get("TOTAL_TIMESTEPS", "n/a"),
                }
            )
        except Exception as e:
            print(f"Could not process {env_name}: {e}")

    return pd.DataFrame(all_data)

if __name__ == "__main__":
    run_experiment()