
import os
import json
import cloudpickle
import sys
sys.path.append(os.path.abspath(".."))
import os

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
# from networks import *
from scipy.stats import sem  # For standard error of the mean
import pandas as pd

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

from scipy.stats import sem  # For standard error of the mean

def plot_experiments(envs, experiments, show_sem=True):
    """
    Plots experimental results for multiple environments in a 2x2 grid.
    
    Parameters:
    - envs (list of str): List of environment names.
    - experiments (list of dicts): List of base experiment configurations.
    """
    fig, axes = plt.subplots(1, 4, figsize=(11, 4), sharex=True, constrained_layout=True)
    axes = axes.flatten()  # Flatten for easy iteration
    # Create a list to store custom legend handles
    custom_handles = []
    custom_labels = []
    for ax, env in zip(axes, envs):
        
        # Initialize variables to track min and max returns for scaling
        env_min_return = float('inf')
        env_max_return = -float('inf')
        
        # Plot each experiment
        for exp in experiments:
            ret_name = exp.get('ret_name','test_returned_episode_returns')
            config, metrics = load_run_data(exp['run_dir'], env)
            label = exp['label']
            color = exp.get('color', 'black')
            marker = exp.get('marker', 'o')

            # Extract data
            # global_idx = metrics['global_idx'][0]  # Assume same x-axis across seeds
            global_idx = [i for i in range(len(metrics[ret_name][0]))]
            step = [i * config['NUM_STEPS'] * config['NUM_ENVS'] for i in range(len(metrics[ret_name][0]))]
            test_returns = metrics[ret_name]  # Shape: (num_seeds, num_steps)

            # Compute statistics
            mean_return = test_returns.mean(axis=0) # averaged over seeds
            sem_return = sem(test_returns, axis=0)

            # Update min and max return values for scaling
            env_min_return = min(env_min_return, mean_return.min())
            env_max_return = max(env_max_return, mean_return.max())
            marker_indices = [int(i * (len(step)-1) / 9) for i in range(10)]
            # Plot mean return
            ax.plot(step, mean_return, label=label, color=color)
            # plot the marker
            ax.plot([step[i] for i in marker_indices], 
                    [mean_return[i] for i in marker_indices], 
                    color=color, marker=marker, linestyle='', ms=6)

            # Add to custom legend handles (do this only once for the first environment)
            if env == envs[0]:
                # Create a custom handle that shows both line and marker
                custom_line = plt.Line2D([0], [0], color=color, marker=marker,)
                custom_handles.append(custom_line)
                custom_labels.append(label)

            # Add shaded confidence interval (95% CI)
            if show_sem:
                ax.fill_between(step, mean_return - sem_return, mean_return + sem_return, alpha=0.2, color=color)


        # Set individual y-axis limits for each environment
        ax.set_ylim(env_min_return - 0.1 * (env_max_return - env_min_return), 
                    env_max_return + 0.1 * (env_max_return - env_min_return))

        # Formatting
        ax.set_title(env, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=12, frameon=True, bbox_to_anchor=(0.5, 1.15))\
    fig.legend(custom_handles, custom_labels, loc='upper center', ncol=len(custom_labels), 
               fontsize=12, frameon=True, bbox_to_anchor=(0.5, 1.15))

    # Global formatting
    fig.supxlabel("Timestep", fontsize=12)
    fig.supylabel("Score", fontsize=12)
    # fig.suptitle("Test Return Across Games", fontsize=16, fontweight="bold")

    # Add a shared legend in the last subplot
    # axes[0].legend(loc="upper left")

    plt.show()
    return fig

def make_table(envs, experiments, show_sem=True):
    """
    Collects experimental results for multiple environments and experiments,
    and outputs them as a DataFrame table.

    Parameters:
    - envs (list of str): List of environment names.
    - experiments (list of dicts): List of experiment configs with keys:
        - run_dir: str, base directory
        - label: str, experiment label
        - ret_name: optional str, key in metrics (default: 'returned_episode_returns')
    - show_sem (bool): Whether to include SEM in the final table.
    """
    rows = []

    for env in envs:
        for exp in experiments:
            ret_name = exp.get("ret_name", "returned_episode_returns")
            config, metrics = load_run_data(exp["run_dir"], env)
            label = exp["label"]

            test_returns = np.array(metrics[ret_name])  # shape: (num_seeds, num_steps)
            num_policies = metrics['update_pi'].mean(0).sum()
            # Compute statistics
            mean_curve = test_returns.mean(axis=0)
            sem_curve = sem(test_returns, axis=0)

            final_mean = test_returns[:, -1].mean()
            final_sem = sem(test_returns[:, -1])

            if show_sem:
                row = {
                    "Env": env,
                    "Experiment": label,
                    "Final Return (mean)": f"{final_mean:.1f} ± {final_sem:.1f}",
                    "Num Target Policies": f"{num_policies:.1f}"
                }
            else:
                row = {
                    "Env": env,
                    "Experiment": label,
                    "Final Return (mean)": f"{final_mean:.1f}",
                    "Num Target Policies": f"{num_policies:.1f}"
                }

            rows.append(row)

    df = pd.DataFrame(rows)
    table = df.pivot_table(
        index="Env",
        columns="Experiment",
        values=["Final Return (mean)", "Num Target Policies"],
        aggfunc="first"  # since each env+experiment pair is unique
    )
    # table = df.pivot(index="Env", columns="Experiment", values=("Final Return (mean)", 'Num Target Policies'))
    return table


def plot_chain(run='3_13_lstd_rmax_no_s_prime/3_14_test', T_values=[0, 1,2,3, 4,5,6,7,8, 10,12, 40], SEED=0):
        # does the s' bonus lead to the exploding extrinsic value?
        # Intrinsic Reward

    config, metrics = load_run_data(run, 'Chain')

    n_rows = len(T_values)
    # Width 12 is a good balance for 1.3x linewidth overflow in Overleaf
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, n_rows * 2.1), sharex=True)

    range_min, range_max = 0, None

    for row_idx, T in enumerate(T_values):
        # --- Row Header (Moved closer to the plot: xy changed from -0.5 to -0.25) ---
        last_10 = int(jnp.sum(metrics['visitation_count'][SEED][T][-11:]))
        axes[row_idx, 0].annotate(f"T={T}\nVis10={last_10}", 
                                    xy=(-0.25, 0.5), xycoords='axes fraction',
                                    fontsize=8, fontweight='bold', ha='right', va='center')
        
        # --- Col 1: Visitation & ri ---
        ax1 = axes[row_idx, 0]
        ax1t = ax1.twinx()
        p0, = ax1.plot(metrics['visitation_count'][SEED][T][range_min:range_max], color='black',  ls = '--' , label='Visits')
        p1, = ax1t.plot(metrics['ri_grid'][SEED][T][range_min:range_max], color='purple', lw=1.2, label='$r_i$')
        ax1.tick_params(axis='y', colors='black')
        ax1t.tick_params(axis='y', colors='purple')

        # --- Col 2: vi & vi_pred ---
        ax2 = axes[row_idx, 1]
        ax2t = ax2.twinx()
        p2, = ax2.plot(metrics['v_i'][SEED][T][range_min:range_max], color='blue', lw=1.5, label='True $v_i$')
        p3, = ax2t.plot(metrics['vi_pred'][SEED][T][range_min:range_max], color='blue', ls='--', alpha=0.6, label='Pred $v_i$')
        ax2.tick_params(axis='y', colors='blue')
        ax2t.tick_params(axis='y', colors='blue')

        # --- Col 3: ve & ve_pred ---
        ax3 = axes[row_idx, 2]
        ax3t = ax3.twinx()
        p4, = ax3.plot(metrics['v_e'][SEED][T][range_min:range_max], color='red', lw=1.5, label='True $v_e$')
        p5, = ax3t.plot(metrics['v_e_pred'][SEED][T][range_min:range_max], color='red', ls='--', alpha=0.6, label='Pred $v_e$')
        ax3.tick_params(axis='y', colors='red')
        ax3t.tick_params(axis='y', colors='red')

        # --- Column labels (a), (b), (c) on the top row ---
        if row_idx == 0:
            ax1.set_title("(a) Visitation & $r_i$", pad=10)
            ax2.set_title("(b) Intrinsic $V_i$", pad=10)
            ax3.set_title("(c) Extrinsic $V_e$", pad=10)
            
            # Unified Legend
            fig.legend(handles=[p0, p1, p2, p3, p4, p5], 
                    loc='upper center', bbox_to_anchor=(0.5, 0.98),
                    ncol=6, frameon=False, fontsize=9)

        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.15, ls='-')
            ax.spines['top'].set_visible(False)

    # X-axis label on bottom row
    for ax in axes[-1]:
        ax.set_xlabel('State Index')

    # Reduced the left margin (rect[SEED] changed from 0.08 to 0.05)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])

    # Use bbox_inches='tight' to ensure the PDF is cropped right to the edge of the labels
    # plt.savefig('/Users/dillonsandhu/Documents/Research/bayesian-rl/deep/figures/200-chain-no-warmup.pdf', bbox_inches='tight')
    plt.show()

    for i in range(4):
        plt.plot(metrics['returned_discounted_episode_returns'][i])
    plt.show()
    return fig


def plot_chain_visits_middle(run='3_13_lstd_rmax_no_s_prime/3_14_test', T_values=[0, 1,2,3, 4,5,6,7,8, 10,12, 40], SEED=0):
        # does the s' bonus lead to the exploding extrinsic value?
        # Intrinsic Reward

    config, metrics = load_run_data(run, 'Chain')

    n_rows = len(T_values)
    # Width 12 is a good balance for 1.3x linewidth overflow in Overleaf
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, n_rows * 2.1), sharex=True)

    range_min, range_max = 0, None

    for row_idx, T in enumerate(T_values):
        # --- Row Header (Moved closer to the plot: xy changed from -0.5 to -0.25) ---
        last_10 = int(jnp.sum(metrics['visitation_count'][SEED][T][-11:]))
        axes[row_idx, 0].annotate(f"T={T}\nVis10={last_10}", 
                                    xy=(-0.25, 0.5), xycoords='axes fraction',
                                    fontsize=8, fontweight='bold', ha='right', va='center')
        
        # --- Col 1: Visitation & ri ---
        ax1 = axes[row_idx, 0]
        ax1t = ax1.twinx()
        p0, = ax1.plot(metrics['visitation_count'][SEED][T][range_min:range_max], color='black',  ls = '--' , label='Visits')
        p1, = ax1t.plot(metrics['ri_grid'][SEED][T][range_min:range_max], color='purple', lw=1.2, label='$r_i$')
        ax1.tick_params(axis='y', colors='black')
        ax1t.tick_params(axis='y', colors='purple')


        # --- Col 2: vi & vi_pred (+ visits overlay) ---
        ax2 = axes[row_idx, 1]
        ax2t = ax2.twinx()

        # Third axis for visits
        ax2v = ax2.twinx()
        ax2v.spines["right"].set_position(("outward", 40))  # shift it right
        ax2v.spines["right"].set_visible(True)

        p2, = ax2.plot(metrics['v_i'][SEED][T][range_min:range_max],
                    color='blue', lw=1.5, label='True $v_i$')

        p3, = ax2t.plot(metrics['vi_pred'][SEED][T][range_min:range_max],
                        color='blue', ls='--', alpha=0.6, label='Pred $v_i$')

        # Visits overlay (third axis)
        p6, = ax2v.plot(metrics['visitation_count'][SEED][T][range_min:range_max],
                        color='black', ls='--', alpha=0.5, label='Visits')

        # Axis styling
        ax2.tick_params(axis='y', colors='blue')
        ax2t.tick_params(axis='y', colors='blue')
        ax2v.tick_params(axis='y', colors='black')


        # --- Col 3: ve & ve_pred ---
        ax3 = axes[row_idx, 2]
        ax3t = ax3.twinx()
        p4, = ax3.plot(metrics['v_e'][SEED][T][range_min:range_max], color='red', lw=1.5, label='True $v_e$')
        p5, = ax3t.plot(metrics['v_e_pred'][SEED][T][range_min:range_max], color='red', ls='--', alpha=0.6, label='Pred $v_e$')
        ax3.tick_params(axis='y', colors='red')
        ax3t.tick_params(axis='y', colors='red')

        # --- Column labels (a), (b), (c) on the top row ---
        if row_idx == 0:
            ax1.set_title("(a) Visitation & $r_i$", pad=10)
            ax2.set_title("(b) Intrinsic $V_i$", pad=10)
            ax3.set_title("(c) Extrinsic $V_e$", pad=10)
            
            # Unified Legend
        fig.legend(handles=[p0, p1, p2, p3, p4, p5, p6],
                loc='upper center', bbox_to_anchor=(0.5, 0.98),
                ncol=7, frameon=False, fontsize=9)
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.15, ls='-')
            ax.spines['top'].set_visible(False)

    # X-axis label on bottom row
    for ax in axes[-1]:
        ax.set_xlabel('State Index')

    # Reduced the left margin (rect[SEED] changed from 0.08 to 0.05)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])

    # Use bbox_inches='tight' to ensure the PDF is cropped right to the edge of the labels
    # plt.savefig('/Users/dillonsandhu/Documents/Research/bayesian-rl/deep/figures/200-chain-no-warmup.pdf', bbox_inches='tight')
    plt.show()

    for i in range(4):
        plt.plot(metrics['returned_discounted_episode_returns'][i])
    plt.show()
    return fig