# This file contains helpers relating to logging, checkpointing, and loading the data.
import os
import yaml
import json
import cloudpickle
import matplotlib.pyplot as plt
from core.networks import *

def run_experiment_main(make_train, SAVE_DIR):
    import argparse
    import datetime
    import traceback
    import core.helpers as helpers
    import core.configs as configs
    # import warnings; warnings.simplefilter('ignore')
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run_suffix', type=str, default=run_timestamp)
    parser.add_argument('--n-seeds', type=int, default=0)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--base-config', type=str, default='shared', 
                        choices=['shared','mc', 'ds', 'min', 'visual', 'chain'])
    parser.add_argument('--env_ids', nargs='+', default=[])

    args = parser.parse_args()
    config = helpers.load_config(args)

    # Priority: CLI env_ids > Registry Defaults > Config ENV_NAME
    if args.env_ids:
        env_list = args.env_ids
    else:
        registry_item = configs.CONFIG_REGISTRY.get(args.base_config, {})
        env_list = registry_item.get("envs", [config.get('ENV_NAME')])

    for i, env_name in enumerate(env_list):
        print(f"\n{'='*50}")
        print(f"RUNNING ENV {i+1}/{len(env_list)}: {env_name}")
        print(f"Network: {config.get('NETWORK_TYPE')}")
        print(f"{'='*50}")
        
        run_config = config.copy()
        run_config['ENV_NAME'] = env_name
        
        # Override seeds if passed via CLI
        if args.n_seeds > 0:
            run_config['N_SEEDS'] = args.n_seeds
            
        rng = jax.random.PRNGKey(run_config['SEED'])
        
        try:
            evaluate(run_config, make_train, SAVE_DIR, args, rng)
        except Exception as e:
            print(f"!!! CRITICAL ERROR running {env_name} !!!")
            traceback.print_exc()
            print("Continuing to next environment...")

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

def save_results(data, config, env_name, env_dir):
    # Create a subdirectory for the environment within the main run directory
    os.makedirs(env_dir, exist_ok=True)

    # Save the pickle file
    pickle_path = os.path.join(env_dir, "out.pkl")
    with open(pickle_path, 'wb') as f:
        cloudpickle.dump(data, f)
    print(f"Results saved to {pickle_path}")
        
    save_config(config, env_dir)
    print(f"Config saved to {os.path.join(env_dir, f'config.json')}")

    return env_dir

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
    
    Parameters:
    - run_folder_name (str): The timestamped run folder name, e.g., "dpi_20241110_193658"
    - env_name (str): The environment name, e.g., "Asterix-MinAtar"
    - results_base_path (str): Base path to the results directory, default is a sibling "results" directory.
    
    Returns:
    - config (dict): Loaded JSON configuration.
    - results (object): Loaded output data from pickle.
    """
    # Construct paths
    run_path = os.path.join(results_base_path, run_folder_name, env_name)
    config_path = os.path.join(run_path, "config.json")
    results_path = os.path.join(run_path, "out.pkl")
    
    # Load the config
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    
    # Load the results
    with open(results_path, 'rb') as pkl_file:
        results = cloudpickle.load(pkl_file)
    
    
    return config, results

def evaluate(run_config, make_train, SAVE_DIR, args, rng):
    # Setup specific to this run_config
    steps_per_pi = run_config["NUM_ENVS"] * run_config["NUM_STEPS"]
    
    # JIT the train function for this specific config (important if env changes)
    run_fn = jax.jit(jax.vmap(make_train(run_config)))
    
    rngs = jax.random.split(rng, run_config['N_SEEDS'])
    out = run_fn(rngs)
    metrics = out["metrics"]

    print(f"[{run_config['ENV_NAME']}] Mean return: {jnp.mean(metrics['returned_episode_returns']):.4f}")
    print(f"[{run_config['ENV_NAME']}] Max return:  {jnp.max(metrics['returned_episode_returns']):.4f}")

    # Directory structure: results/cov_lstd/timestamp/EnvName/
    run_dir = os.path.join("results", f"{SAVE_DIR}/{args.run_suffix}")
    env_dir = os.path.join(run_dir, run_config['ENV_NAME'])
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(env_dir, exist_ok=True)
    print(f"Saving {run_config['ENV_NAME']} results to {env_dir}")

    if args.save_checkpoint:
        save_results(out, run_config, run_config['ENV_NAME'], env_dir)
    else:
        save_results(metrics, run_config, run_config['ENV_NAME'], env_dir)
    
    # --- Helper for Metrics extraction ---
    def _mean_over_seeds(data):
        arr = jnp.asarray(data)
        if arr.ndim > 0 and arr.shape[0] == run_config['N_SEEDS']:
            arr = arr.mean(0)
        return arr

    def _extract_series(data):
        arr = _mean_over_seeds(data)
        if arr.ndim == 0:
            return arr[None]
        if arr.ndim == 1:
            return arr

        # For grid-like metrics, plot a fixed reference state over time.
        # FourRooms default start is (1, 1), DeepSea default start is (0, 0).
        if arr.ndim >= 3:
            if run_config['ENV_NAME'] in {"FourRooms-misc", "FourRoomsCustom-v0"}:
                y_idx = 1 if arr.shape[1] > 1 else 0
                x_idx = 1 if arr.shape[2] > 1 else 0
                idx = [slice(None), y_idx, x_idx]
                if arr.ndim > 3:
                    idx.extend([0] * (arr.ndim - 3))
                return arr[tuple(idx)]

        idx = [slice(None)]
        if arr.ndim > 1:
            idx.extend([0] * (arr.ndim - 1))
        return arr[tuple(idx)]

    def get_metric(name, slice_idx=0):
        if name not in metrics:
            return None
        series = _extract_series(metrics[name])
        return series[slice_idx:]

    # 1. Main Return Plot
    mean_rets = get_metric('returned_episode_returns', 0)
    if run_config['ENV_NAME'] == "SparseMountainCar-v0":
            mean_rets = get_metric('returned_discounted_episode_returns', 0)
    
    if mean_rets is not None: 
        save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, mean_rets, 'Return')
    
    # 2. Standard Diagnostic Plots
    standard_plots = {
        'bonus_mean': 'i_advantage_mean',
        'bonus_std': 'i_advantage_std',
        'intrinsic_rew_mean': 'intrinsic_rew_mean',
        'vi_pred': 'vi_pred', 
        'v_i_pred_opt': 'v_i_pred_opt',
        'v_e_pred': 'v_e_pred',
        "mean_rew": "mean_rew",
        "raw_intrinsic_rew_mean": "raw_intrinsic_rew_mean"
    }

    for m_key, save_name in standard_plots.items():
        data = get_metric(m_key, 1)
        if data is not None:
            # If we are in True Value mode, 'vi_pred' is a GRID, so we skip standard plotting
            if run_config.get('CALC_TRUE_VALUES', False) and m_key in ['vi_pred', 'v_i_pred_opt', 'v_e', 'ri_grid']:
                    continue 
            try:
                save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, data, save_name)
            except:
                print('failed to save plot for', m_key)

    # 3. Extended / True Value Plots 
    if run_config.get('CALC_TRUE_VALUES', False):
        extended_metrics = ["v_i", "v_e", "v_e_pred", "vi_pred", "v_i_pred_opt", "e_value_error", "i_value_error"]
        for key in extended_metrics:
            if key in metrics:
                series = _extract_series(metrics[key])
                save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, series[1:], key)
