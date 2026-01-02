from typing import NamedTuple
import jax.numpy as jnp
import os
import yaml
import json
import cloudpickle
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from typing import NamedTuple, Dict, Sequence, NamedTuple, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper
from envs.log_wrapper import LogWrapper
from envs.wrappers import NormalizeObservationWrapper, NormalizeRewardWrapper, NormalizeVecObservation, BinaryRewardWrapper
from networks import *
import optax

config = {
    "ENV_NAME": "MountainCar-v0",
    "LR": 5e-4,
    "LR_END": 5e-5,
    "NUM_ENVS": 32,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 250_000,
    "NUM_EPOCHS": 4, 
    "MINIBATCH_SIZE": 256,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.6,
    "CLIP_EPS": 0.2,
    "VF_CLIP": 0.5,
    "ENT_COEF": 0.003,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "SEED": 42,
    "WARMUP": 200, # warmup steps for running mean/std, 200 is one episode.
    "N_SEEDS": 16,
    "RND_TRAIN_FRAC": 0.5,
    "REGULARIZATION": 1e-2,
    "PER_UPDATE_REGULARIZATION": 1e-4,
    "BONUS_SCALE": 1.0,
    "ema_coeff_w": 0.9, #(not actually used)
    "NORMALIZE_FEATURES": False,
    "BINARY_REWARDS": False,
    "NORMALIZE_REWARDS": True,
    "NORMALIZE_OBS": True,
}

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