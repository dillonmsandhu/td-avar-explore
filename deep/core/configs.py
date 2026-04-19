import jax.numpy as jnp
import copy
from typing import NamedTuple

# =============================================================================
# 1. Global Flags & Base Constants
# =============================================================================
NORMALIZE_FEATURES = True  # for LSTD
EPISODIC = True           # RND LSTD
BIAS = True               # for LSTD
NORMALIZE_REWARDS = False  # normalize extrinsic reward

# =============================================================================
# 2. Base Configuration Modules
# =============================================================================

shared = {
    "LR": 5e-4,
    "LR_END": 5e-4,
    "RND_LR": 5e-5,
    "NUM_ENVS": 32,
    "NUM_STEPS": 256,
    "TOTAL_TIMESTEPS": 1e6,
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 256,
    "GAMMA": 0.99,
    "GAMMA_i": 0.99,
    "GAE_LAMBDA": 0.9,
    "GAE_LAMBDA_i": 0.9,
    "CLIP_EPS": 0.2,
    "VF_CLIP": 0.5,
    "ENT_COEF": 0.001,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "SEED": 42,
    # Exploration Specific
    "RND_TRAIN_FRAC": 0.5,
    "NORMALIZE_FEATURES": True,
    "NORMALIZE_OBS": False,  # Default to True for Continuous, overridden for Grids
    "NORMALIZE_REWARDS": False,
    "EPISODIC": False,
    "ABSORBING_TERMINAL_STATE": False, 
    "BONUS_SCALE": 2.0,
    "SCHEDULE_BETA": True,
    "LSTD_L2_REG": 1e-3,
    "RND_FEATURES": 128,
    "NETWORK_TYPE": "mlp",
    "RND_NETWORK_TYPE": "mlp",
    "WARMUP": 1_000,
    "STAGGERED_STARTS": True,
    "BIAS": BIAS,
    "RB_SIZE": 100_000,
    "PERCENT_FIFO": .25,
    "CALC_TRUE_VALUES": False,
    "N_SEEDS": 4,
    "LSTD_PRIOR_SAMPLES": 10,
}

# --- Environment Specific Overrides ---

ds_specific = {
    "ENV_NAME": "DeepSea-bsuite",
    "NORMALIZE_OBS": False,
    "NETWORK_TYPE": "cnn",
    "RND_NETWORK_TYPE": "cnn",
    "N_SEEDS": 4,
    "LSTD_PRIOR_SAMPLES": 1.0,
}

min_specific = {
    "ENV_NAME": "Breakout-MinAtar", # Default placeholder
    "TOTAL_TIMESTEPS": 1e7,
    "LR": 2.5e-3,
    "LR_END": 1e-5,
    "GAE_LAMBDA": 0.8,
    "CLIP_EPS": 0.1,
    "VF_CLIP": 0.2,
    "NORMALIZE_OBS": False,
    "NETWORK_TYPE": "cnn",
    "RND_NETWORK_TYPE": "cnn",
    "N_SEEDS": 4,
}

visual = {
    "NETWORK_TYPE": "cnn",
    "RND_NETWORK_TYPE": "cnn",
    "NORMALIZE_OBS": False,
    "WARMUP": 0,
}

chain = {
    "ENV_NAME": "Chain",
    "RND_NETWORK_TYPE": "identity",
    "NETWORK_TYPE": "mlp",
    "NORMALIZE_OBS": False,
    "NORMALIZE_FEATURES": False,  # tabular
    "NORMALIZE_REWARDS": False,
    "RND_FEATURES": 150,
    "ENV_SIZE": 150,
    "LSTD_FEATURES": 150,
    "CALC_TRUE_VALUES": True,
    "BIAS": False,
    "EPISODIC": EPISODIC,
    "STAGGERED_STARTS": False,
    "N_SEEDS": 4, }


mc_config = shared   # | is the union op. last dict's key takes precedence
ds_config = shared | ds_specific
min_config = shared | min_specific
chain = shared | chain
visual = shared | visual

CONFIG_REGISTRY = {
    # maps from config name to all envs that we can run that use that config.
    "shared": {
        "config_dict": shared,
        "envs": [
            "DiscountingChain-bsuite",
            "BernoulliBandit-misc",
            "GaussianBandit-misc",
            "MetaMaze-misc",
            "CartPole-v1",
            "Acrobot-v1",
            "UmbrellaChain-bsuite",
            "Reacher-misc",
            "PointRobot-misc",
            "Swimmer-misc",
        ],
    },
    "visual": {
        "config_dict": visual,
        "envs": ["Pong-misc", "MNISTBandit-bsuite", "Catch-bsuite","FourRoomsCustom-v0","Maze"],
    },
    "maze": {
        "config_dict": visual,
        "envs": ["Maze"],
    },
    "four_rooms": {
        "config_dict": visual,
        "envs": ["FourRoomsCustom-v0"],
    },
    "mc": {"config_dict": mc_config, "envs": ["SparseMountainCar-v0", "MountainCar-v0"]},
    "ds": {"config_dict": ds_config, "envs": ["DeepSea-bsuite"]},
    "min": {
        "config_dict": min_config,
        "envs": ["SpaceInvaders-MinAtar", "Breakout-MinAtar", "Freeway-MinAtar", "Asterix-MinAtar"],
    },
    "chain": {
        "config_dict": chain,
        "envs": [
            "Chain",
        ],
    },
}


# =============================================================================
# 3. Registry Generator Function
# =============================================================================

def make_final_registries(shared_base, ds_base, min_base, visual_base, chain_base):
    FINAL_TESTING = {}
    FINAL_EXACT = {}

    def merge(d1, d2, overrides):
        res = copy.deepcopy(d1)
        res.update(copy.deepcopy(d2))
        res.update(overrides)
        return res

    # ---------------------------------------------------------------------
    # FINAL_TESTING: No True Value solving, multiple sizes
    # ---------------------------------------------------------------------
    ft_overrides = {"CALC_TRUE_VALUES": False, "N_SEEDS": 5}

    # Standard Benchmarks
    FINAL_TESTING["shared"] = {
        "config_dict": shared_base | ft_overrides,
        "envs": [
            "DiscountingChain-bsuite", "CartPole-v1", "Acrobot-v1", 
            "Reacher-misc", "PointRobot-misc", "Swimmer-misc", 
            "SparseMountainCar-v0", "MountainCar-v0"
        ]
    }

    FINAL_TESTING["visual"] = {
        "config_dict": shared_base | visual_base | ft_overrides,
        "envs": ["Pong-misc"]
    }

    FINAL_TESTING["maze"] = {
        "config_dict": shared_base | visual_base | {"ENV_SIZE": 100, **ft_overrides},
        "envs": ["Maze"]
    }

    FINAL_TESTING["min"] = {
        "config_dict": shared_base | min_base| ft_overrides,
        "envs": ["SpaceInvaders-MinAtar", "Breakout-MinAtar", "Freeway-MinAtar", "Asterix-MinAtar"]
    }

    # DeepSea: 10k episodes per size
    for size in [20, 30, 40, 45, 50]:
        ds_cfg = shared_base | ds_base | {
            "ENV_SIZE": size, 
            "TOTAL_TIMESTEPS": int(size * 10_000), 
            **ft_overrides
        }
        FINAL_TESTING[f"ds_{size}"] = {"config_dict": ds_cfg, "envs": ["DeepSea-bsuite"]}

    # FourRooms: Dual sizes
    for size in [21, 29]:
        fr_cfg = shared_base| visual_base | {"ENV_SIZE": size,**ft_overrides}
        FINAL_TESTING[f"four_rooms_{size}"] = {"config_dict": fr_cfg, "envs": ["FourRoomsCustom-v0"]}

    # Chain: MLP Distillation Proof
    for size in [200, 400]:
        ch_cfg = shared_base | chain_base | {"ENV_SIZE": size,"RND_NETWORK_TYPE": "mlp","RND_FEATURES": 128,"BIAS": True,"NORMALIZE_FEATURES": True,**ft_overrides
        }
        FINAL_TESTING[f"chain_mlp_{size}"] = {"config_dict": ch_cfg, "envs": ["Chain"]}

    # ---------------------------------------------------------------------
    # FINAL_EXACT: Exact solving for analysis/heatmaps
    # ---------------------------------------------------------------------
    fe_overrides = {"CALC_TRUE_VALUES": True, "N_SEEDS": 1}
    
    FINAL_EXACT["chain_mlp_175"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 175, "RND_NETWORK_TYPE": "mlp", "RND_FEATURES": 128, 
            "BIAS": True, "NORMALIZE_FEATURES": True, **fe_overrides
        },
        "envs": ["Chain"]
    }

    FINAL_EXACT["maze_exact"] = {
        "config_dict": shared_base | visual_base | {"ENV_SIZE": 100, **fe_overrides},
        "envs": ["Maze"]
    }

    FINAL_EXACT["ds_40_exact"] = {
        "config_dict": shared_base | ds_base | {"ENV_SIZE": 40, "TOTAL_TIMESTEPS": 400_000, **fe_overrides}, 
        "envs": ["DeepSea-bsuite"]
    }

    FINAL_EXACT["ds_50_exact"] = {
        "config_dict": shared_base | ds_base | {"ENV_SIZE": 50, "TOTAL_TIMESTEPS": 500_000, **fe_overrides}, 
        "envs": ["DeepSea-bsuite"]
    }

    FINAL_EXACT["four_rooms_21_exact"] = {
        "config_dict": shared_base | visual_base | {"ENV_SIZE": 21, **fe_overrides}, 
        "envs": ["FourRoomsCustom-v0"]
    }

    # 150-Chain Ablation: Tabular vs MLP
    FINAL_EXACT["chain_tabular_150_continuing"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 150, "RND_NETWORK_TYPE": "identity", "RND_FEATURES": 150, "CONTINUING": True, "ABSORBING_TERMINAL_STATE": False,
            "BIAS": False, "LSTD_L2_REG": 1e-10, "NORMALIZE_FEATURES": False, **fe_overrides
        },
        "envs": ["Chain"]
    }

    # 150-Chain Ablation: Tabular vs MLP
    FINAL_EXACT["chain_tabular_150_ep"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 140, "RND_NETWORK_TYPE": "identity", "RND_FEATURES": 140, "CONTINUING": False, "ABSORBING_TERMINAL_STATE": False,
            "BIAS": False, "LSTD_L2_REG": 1e-10, "NORMALIZE_FEATURES": False, **fe_overrides
        },
        "envs": ["Chain"]
    }

    # 150-Chain Ablation: Tabular vs MLP
    FINAL_EXACT["chain_tabular_160_absorbing"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 160, "RND_NETWORK_TYPE": "identity", "RND_FEATURES": 160, "CONTINUING": False, "ABSORBING_TERMINAL_STATE": True,
            "BIAS": False, "LSTD_L2_REG": 1e-10, "NORMALIZE_FEATURES": False, **fe_overrides
        },
        "envs": ["Chain"]
    }


    FINAL_EXACT["chain_tabular_50_absorbing"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 50, "RND_NETWORK_TYPE": "identity", "RND_FEATURES": 50, "CONTINUING": False, "ABSORBING_TERMINAL_STATE": True,
            "BIAS": False, "LSTD_L2_REG": 1e-10, "NORMALIZE_FEATURES": False, **fe_overrides
        },
        "envs": ["Chain"]
    }

    return FINAL_TESTING, FINAL_EXACT

# =============================================================================
# 4. Final Initialization
# =============================================================================

FINAL_TESTING, FINAL_EXACT = make_final_registries(
    shared, ds_specific, min_specific, visual, chain
)