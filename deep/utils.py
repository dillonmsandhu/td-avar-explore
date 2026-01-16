# This file contains helpers relating to logging, checkpointing, and loading the data.
import os
import yaml
import json
import cloudpickle
import matplotlib.pyplot as plt
from networks import *

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
    plt.figure()
    plt.plot([i * steps_per_pi for i in range(len(episodic_return))], episodic_return, 'o-')
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
    def get_metric(name, slice_idx=0):
        if name not in metrics: return None
        data = metrics[name]
        data = data.mean(0) if run_config['N_SEEDS'] > 1 else data
        return data[slice_idx:]

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
                data = metrics[key]
                mean_data = data.mean(0) if run_config['N_SEEDS'] > 1 else data
                
                if "error" in key:
                        save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, mean_data[1:], key)
                else:
                    # Values are Grids (DeepSea), plot initial state (0,0)
                    initial_state_data = mean_data[:, 0, 0] 
                    save_plot(env_dir, run_config['ENV_NAME'], steps_per_pi, initial_state_data[1:], key)
