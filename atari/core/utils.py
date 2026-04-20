# core.utils.py
# Contains helpers relating to logging, checkpointing, and loading the data.
import os
import yaml
import json
import cloudpickle
import matplotlib.pyplot as plt
from core.networks import *
import numpy as np
import flax.training.checkpoints as checkpoints

def run_experiment_main(make_train, SAVE_DIR):
    import argparse
    import datetime
    import traceback
    import core.helpers as helpers
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run_suffix', type=str, default=run_timestamp)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--env-ids', nargs='+', default=[])
    
    # WandB/Tuning args
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project', type=str, default="lstd-explore")
    
    args = parser.parse_args()

    # 1. Robust Config Loading
    if os.path.isfile(args.config):
        config_path = args.config
    else:
        config_path = os.path.join("core", "configs.py") # Fallback to default
        
    config = helpers.load_config_dict(config_path)
    
    # Apply JSON overrides if --config was passed as a JSON string
    try:
        if args.config.startswith('{'):
            config.update(json.loads(args.config))
    except json.JSONDecodeError:
        pass # It was just a filepath
    
    # 2. Environment overwrite from CLI
    env_list = args.env_ids if args.env_ids else [config.get('ENV_NAME')]

    for i, env_name in enumerate(env_list):
        # Create a clean copy for this specific environment run
        run_config = config.copy()
        run_config['ENV_NAME'] = env_name
        run_config['SEED'] = args.seed
        run_config['THREADS'] = args.threads
        run_config['CONFIG_NAME'] = os.path.basename(config_path)
        
        print(f"\n{'='*50}")
        print(f"RUNNING ENV {i+1}/{len(env_list)}: {env_name} | SEED: {args.seed}")
        print(f"{'='*50}")
        
        rng = jax.random.PRNGKey(run_config['SEED'])
        
        # Optional WandB Initialization
        if args.wandb:
            import wandb
            # Group by config name, name the run by env and seed
            wandb.init(
                project=args.project, 
                config=run_config, 
                name=f"{env_name}_s{args.seed}", 
                group=run_config['CONFIG_NAME']
            )

        try:
            evaluate(run_config, make_train, SAVE_DIR, args, rng)
        except Exception as e:
            print(f"!!! CRITICAL ERROR running {env_name} !!!")
            print(f"Error: {e}")
            traceback.print_exc()
            print("Continuing to next environment...")
            
        if args.wandb:
            wandb.finish()


def parse_config_override(config_str):
    """Parse config override from command line argument."""
    if config_str is None:
        return {}
    
    try:
        # Parse as JSON
        return json.loads(config_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing config override: {e}")
        print("Config override should be valid JSON, e.g.: '{\"LR\": 0.001, \"LAMBDA\": 0.0}'")
        exit(1)


def load_config_dict(file_path: str) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, env_dir):
    config_path = os.path.join(env_dir, f"config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")

def save_results(data, config, env_name, env_dir, save_checkpoint=False):
    os.makedirs(env_dir, exist_ok=True)
    
    # 1. Save Metrics explicitly as a compressed NumPy array
    metrics = jax.device_get(data["metrics"] if "metrics" in data else data)
    np.savez_compressed(os.path.join(env_dir, "metrics.npz"), **metrics)
    
    if save_checkpoint and "runner_state" in data:
        checkpoints.save_checkpoint(env_dir, data["runner_state"], step=config["TOTAL_TIMESTEPS"])
        pass
        
    save_config(config, env_dir)

def save_plot(env_dir, env_name, steps_per_pi, episodic_return, title):
    y = jnp.asarray(episodic_return)
    if y.ndim == 0:
        y = y[None]
    if y.ndim != 1:
        print(f"Skipping plot {title}: expected 1D series, got shape {tuple(y.shape)}")
        return
    if y.shape[0] == 0:
        print(f"Skipping plot {title}: empty series")
        return

    plt.figure()
    x = [i * steps_per_pi for i in range(int(y.shape[0]))]
    plt.plot(x, y, 'o-', label=title)
    plt.xlabel("Step")
    plt.ylabel(f"{title}")
    plt.title(env_name)
    plt.legend()

    # Save plot as a .png file in the environment directory
    plot_path = os.path.join(env_dir, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")


def load_run_data(run_folder_name, env_name, results_base_path="../results"):
    """
    Load the configuration and output data for a run given the run folder and environment.
    Automatically aggregates metrics across all available seeds.
    
    Parameters:
    - run_folder_name (str): The timestamped run folder name, e.g., "dpi_20241110_193658"
    - env_name (str): The environment name, e.g., "Asterix-MinAtar"
    - results_base_path (str): Base path to the results directory.
    
    Returns:
    - config (dict): Loaded JSON configuration.
    - metrics (dict): A dictionary of numpy arrays stacked across the seed dimension.
                      Shape: (num_seeds, num_updates, ...)
    """
    # Construct paths
    run_path = os.path.join(results_base_path, run_folder_name, env_name)
    config_path = os.path.join(run_path, "config.json")
    
    # 1. Load the config
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    
    # 2. Find all metric files for this environment
    metric_files = [f for f in os.listdir(run_path) if f.startswith("seed_") and f.endswith("_metrics.npz")]
    
    if not metric_files:
        print(f"Warning: No metric .npz files found in {run_path}")
        return config, None
        
    # Sort them to ensure consistent ordering (e.g., seed_0, seed_1, seed_2)
    metric_files.sort() 
    
    # 3. Load the first file to get the dictionary keys
    first_data = np.load(os.path.join(run_path, metric_files[0]))
    keys = first_data.files
    
    # Initialize a dictionary of lists to hold the arrays
    aggregated_metrics = {k: [] for k in keys}
    
    # 4. Loop through all seed files and append their arrays
    for mf in metric_files:
        # np.load acts as a context manager/lazy loader, so we extract the data immediately
        with np.load(os.path.join(run_path, mf)) as data:
            for k in keys:
                aggregated_metrics[k].append(data[k])
                
    # 5. Stack the lists into unified numpy arrays along axis 0 (the seed dimension)
    for k in keys:
        aggregated_metrics[k] = np.stack(aggregated_metrics[k], axis=0)
    
    return config, aggregated_metrics

def evaluate(run_config, make_train, SAVE_DIR, args, rng):
    steps_per_pi = run_config["NUM_ENVS"] * run_config["NUM_STEPS"]
    seed = run_config["SEED"]
    
    run_fn = jax.jit(make_train(run_config))
    out = run_fn(rng)
    metrics = out["metrics"]

    print(f"[{run_config['ENV_NAME']}] Mean return: {jnp.mean(metrics['returned_episode_returns']):.1f}")
    print(f"[{run_config['ENV_NAME']}] Max return:  {jnp.max(metrics['returned_episode_returns']):.1f}")

    # Directory structure
    base_env_name = run_config['ENV_NAME']
    env_size = run_config.get("ENV_SIZE")
    full_env_name = f"{base_env_name}-{env_size}" if env_size else base_env_name
    
    if args.output_dir:
        env_dir = os.path.join(args.output_dir, full_env_name)
    else:
        env_dir = os.path.join("results", f"{SAVE_DIR}/{args.run_suffix}", full_env_name)
    
    os.makedirs(env_dir, exist_ok=True)
    print(f"Saving {full_env_name} (Seed {seed}) results to {env_dir}")

    # 1. Save Config (Concurrent safe: they all save the same config)
    with open(os.path.join(env_dir, "config.json"), 'w') as f:
        json.dump(run_config, f, indent=4)
        
    # 2. Save Metrics cleanly as NPZ (prevents SLURM overwrites)
    np_metrics = jax.device_get(metrics)
    np.savez_compressed(os.path.join(env_dir, f"seed_{seed}_metrics.npz"), **np_metrics)
    
    # 3. Handle Checkpointing

    if args.save_checkpoint:
        # runner_state tuple: (train_state [0], lstd_state [1], sigma_state [2], 
        #                      buffer_state [3], rnd_state [4], ...)
        train_state = out["runner_state"]['train_state']
        rnd_state = out["runner_state"]['rnd_state']

        # Pull only the raw parameter dictionaries back to the CPU to avoid OOM
        # and strip away the massive optimizer momentum buffers.
        checkpoint_params = {
            "actor_critic_params": jax.device_get(train_state.params),
            "rnd_params": jax.device_get(rnd_state.params),
            # RNDTrainState includes target_params per your earlier class definition
            "rnd_target_params": jax.device_get(rnd_state.target_params) 
        }

        ckpt_path = os.path.join(env_dir, f"seed_{seed}_final_params.pkl")
        with open(ckpt_path, 'wb') as f:
            cloudpickle.dump(checkpoint_params, f)
            
        print(f"Saved weights to {ckpt_path}")

    # --- Helper for Metrics extraction (Simplified for 1 Seed) ---
    def get_metric(name, slice_idx=0):
        if name not in np_metrics:
            return None
        arr = np_metrics[name]
        
        if arr.ndim == 0:
            return arr[None]
        if arr.ndim == 1:
            return arr[slice_idx:]

        # For grid-like metrics, plot a fixed reference state over time.
        if arr.ndim >= 2:
            if run_config['ENV_NAME'] in {"FourRooms-misc", "FourRoomsCustom-v0"}:
                y_idx = 1 if arr.shape[1] > 1 else 0
                x_idx = 1 if arr.shape[2] > 1 else 0
                idx = [slice(None), y_idx, x_idx]
                if arr.ndim > 3:
                    idx.extend([0] * (arr.ndim - 3))
                return arr[tuple(idx)][slice_idx:]

        idx = [slice(None)]
        if arr.ndim > 1:
            idx.extend([0] * (arr.ndim - 1))
        return arr[tuple(idx)][slice_idx:]

    # --- Post-run Logging & Plotting ---
    log_dict_history = []
    num_updates = np_metrics['returned_episode_returns'].shape[0]

    # 4. Standard Diagnostic Plots
    standard_plots = {
        "returned_episode_returns": 'returned_episode_returns', 
        'bonus_mean': 'i_advantage_mean',
        'bonus_std': 'i_advantage_std',
        'intrinsic_rew_mean': 'intrinsic_rew_mean',
        "mean_rew": "mean_rew",
        "raw_intrinsic_rew_mean": "raw_intrinsic_rew_mean",
        "bellman_residual_non_done": "bellman_residual_non_done",
        "true_v_i_at_goal": "true_v_i_at_goal",
        "rho_on_done": "rho_on_done",
        "v_i_at_done_mean": "v_i_at_done_mean",
        "cond_number_A": "cond_number",
        "num_goals": "num_goals"
    }

    for m_key, save_name in standard_plots.items():
        data = get_metric(m_key, 1) # Note: we use 1 here based on your original code
        if data is not None:
            # Add to wandb history
            if args.wandb:
                for step in range(len(data)):
                    if len(log_dict_history) <= step:
                        log_dict_history.append({"step": (step + 1) * steps_per_pi})
                    log_dict_history[step][m_key] = data[step]
                    
            # Skip plotting massive grids
            if run_config.get('CALC_TRUE_VALUES', False) and m_key in ['vi_pred', 'v_i_pred_opt', 'v_e', 'ri_grid']:
                continue 
            try:
                # Save plot with seed suffix
                plot_title = f"{save_name}_s{seed}"
                save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, data, plot_title)
            except Exception as e:
                print(f"Failed to save plot for {m_key}: {e}")
                
    # 5. Flush to WandB efficiently
    if args.wandb:
        import wandb
        for log_step in log_dict_history:
            wandb.log(log_step)

def save_plot(env_dir, env_name, steps_per_pi, y, title):
    if y.shape[0] == 0:
        return

    plt.figure()
    x = [i * steps_per_pi for i in range(int(y.shape[0]))]
    plt.plot(x, y, 'o-', label=title)
    plt.xlabel("Step")
    plt.ylabel(f"{title}")
    plt.title(f"{env_name} ({title})")
    plt.legend()

    plot_path = os.path.join(env_dir, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()

