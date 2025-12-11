from typing import NamedTuple
import jax.numpy as jnp
import os
import yaml
import json
import cloudpickle
import matplotlib.pyplot as plt

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

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

def save_plot(env_dir, env_name, steps_per_pi, episodic_return):
    plt.figure()
    plt.plot([i * steps_per_pi for i in range(len(episodic_return))], episodic_return, 'o-')
    plt.xlabel("Step")
    plt.ylabel("Return")
    plt.title(env_name)
    plt.legend()

    # Save plot as a .png file in the environment directory
    plot_path = os.path.join(env_dir, f"plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")