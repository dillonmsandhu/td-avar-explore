# This file contains helpers relating to logging, checkpointing, and loading the data.
import os
import yaml
import json
import cloudpickle
import matplotlib.pyplot as plt
from core.networks import *

# MAIN function in most algos.
def run_experiment_main(make_train, SAVE_DIR):
    import argparse
    import datetime
    import traceback
    import core.helpers as helpers
    import core.configs as configs
    import jax
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--run_suffix', type=str, default=run_timestamp)
    parser.add_argument('--n-seeds', type=int, default=0)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--registry', type=str, default='FINAL_TESTING', 
                        choices=['CONFIG_REGISTRY', 'FINAL_TESTING', 'FINAL_EXACT'])
    parser.add_argument('--base-config', type=str, default='shared')
    parser.add_argument('--env-ids', nargs='+', default=[])

    args = parser.parse_args()


    target_registry = getattr(configs, args.registry, configs.CONFIG_REGISTRY)

    # 2. Extract Registry Item (e.g., "maze_exact" or "ds_40")
    registry_item = target_registry.get(args.base_config)
    
    if registry_item:
        # Success: Use the specific merged config (e.g., CNN-based visual config)
        config = registry_item["config_dict"].copy()
        env_list = registry_item.get("envs", [])
    else:
        # Fallback: If not in the specific registry, try generic helpers
        print(f"⚠️  Config '{args.base_config}' not found in registry {args.registry}. Falling back to shared.")
        config = helpers.load_config(args)
        env_list = [config.get('ENV_NAME')]

    # 3. Environment Priority (CLI takes precedence)
    if args.env_ids:
        env_list = args.env_ids

    for i, env_name in enumerate(env_list):
        if env_name is None: continue
            
        # Create a clean copy for this specific environment run
        run_config = config.copy()
        run_config['ENV_NAME'] = env_name
        
        # Apply command-line JSON overrides if they exist
        if args.config:
            from core.utils import parse_config_override
            run_config.update(parse_config_override(args.config))
            
        if args.n_seeds > 0:
            run_config['N_SEEDS'] = args.n_seeds

        print(f"\n{'='*50}")
        print(f"RUNNING ENV {i+1}/{len(env_list)}: {env_name}")
        print(f"Registry: {args.registry} | Config: {args.base_config}")
        print(f"Network: {run_config.get('NETWORK_TYPE')}")
        print(f"{'='*50}")
        
        rng = jax.random.PRNGKey(run_config.get('SEED', 42))
        
        try:
            # Note: make_train and evaluate should be defined in your scope
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


import matplotlib.pyplot as plt

def save_chain_diagnostic_plot(env_dir, config, metrics, T_values=[0, 1, 2, 3, 4, 5, 10, 20, 40], SEED=0):
    """Diagnostic plot specifically for the Chain environment."""
    n_rows = len(T_values)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, n_rows * 2.1), sharex=True)
    range_min, range_max = 0, None

    for row_idx, T in enumerate(T_values):
        # Safety check: Ensure T exists in the metrics
        if T >= metrics['visitation_count'].shape[2]: continue

        # --- Row Header ---
        last_10 = int(jnp.sum(metrics['visitation_count'][SEED][T][-11:]))
        axes[row_idx, 0].annotate(f"T={T}\nVis10={last_10}", 
                                    xy=(-0.25, 0.5), xycoords='axes fraction',
                                    fontsize=8, fontweight='bold', ha='right', va='center')
        
        # Plotting logic (Col 1: Visitation, Col 2: Vi, Col 3: Ve)
        # Using [SEED][T] indexing as per your original data structure
        ax1, ax2, ax3 = axes[row_idx]
        
        # Helper to plot twin axes
        def _plot_twin(ax, data1, data2, color, label1, label2, ls2='--'):
            ax_t = ax.twinx()
            p1, = ax.plot(data1, color=color, lw=1.5, label=label1)
            p2, = ax_t.plot(data2, color=color, ls=ls2, alpha=0.6, label=label2)
            ax.tick_params(axis='y', colors=color)
            return p1, p2

        p0, = ax1.plot(metrics['visitation_count'][SEED][T], color='black', ls='--', label='Visits')
        ax1_t = ax1.twinx()
        p1, = ax1_t.plot(metrics['ri_grid'][SEED][T], color='purple', lw=1.2, label='$r_i$')
        
        p2, p3 = _plot_twin(ax2, metrics['v_i'][SEED][T], metrics['vi_pred'][SEED][T], 'blue', 'True $v_i$', 'Pred $v_i$')
        p4, p5 = _plot_twin(ax3, metrics['v_e'][SEED][T], metrics['v_e_pred'][SEED][T], 'red', 'True $v_e$', 'Pred $v_e$')

        if row_idx == 0:
            ax1.set_title("(a) Visitation & $r_i$")
            ax2.set_title("(b) Intrinsic $V_i$")
            ax3.set_title("(c) Extrinsic $V_e$")
            fig.legend(handles=[p0, p1, p2, p3, p4, p5], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=6, frameon=False)

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    plot_path = os.path.join(env_dir, "chain_diagnostic.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close() # Important to close fig in loops to save memory


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

    # Directory structure: results/SAVE_DIR/timestamp/EnvName-Size/
    base_env_name = run_config['ENV_NAME']
    env_size = run_config.get("ENV_SIZE")
    
    # Create the full name (e.g., DeepSea-bsuite-45)
    full_env_name = f"{base_env_name}-{env_size}" if env_size else base_env_name
    
    run_dir = os.path.join("results", f"{SAVE_DIR}/{args.run_suffix}")
    env_dir = os.path.join(run_dir, full_env_name)
    
    os.makedirs(env_dir, exist_ok=True)
    print(f"Saving {full_env_name} results to {env_dir}")

    # Ensure save_results uses the full name for the filename
    if args.save_checkpoint:
        save_results(out, run_config, full_env_name, env_dir)
    else:
        save_results(metrics, run_config, full_env_name, env_dir)
    
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
    discounted_return_envs = ("SparseMountainCar-v0", "FourRooms-misc", "Chain", "FourRoomsCustom-v0")

    mean_rets = get_metric('returned_episode_returns', 0)
    if run_config['ENV_NAME'] in discounted_return_envs: # plot the discoutned return for these envs.
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
        "raw_intrinsic_rew_mean": "raw_intrinsic_rew_mean",
        # Bellman residual diagnostics
        "bellman_residual_non_done": "bellman_residual_non_done",
        "bellman_residual_non_done_std": "bellman_residual_non_done_std",
        # Goal/done diagnostics
        "true_v_i_at_goal": "true_v_i_at_goal",
        "true_next_v_i_on_done": "true_next_v_i_on_done",
        "rho_on_done": "rho_on_done",
        "true_v_i_on_done": "true_v_i_on_done",
        "v_i_at_done_mean": "v_i_at_done_mean",
        "cond_number_A": "cond_number",
        "condition_number": "condition_number",
        "num_goals": "num_goals"
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
        
            if run_config['ENV_NAME'] == "Chain":
                print(f"Generating specialized Chain diagnostic plot for {run_config['ENV_NAME']}...")
                try:
                    # We pass 'metrics' directly from the JAX output
                    save_chain_diagnostic_plot(env_dir, run_config, metrics)
                except Exception as e:
                    print(f"Failed to generate Chain diagnostic: {e}")
                    
