import jax.numpy as jnp
import copy
from typing import NamedTuple

# =============================================================================
# 1. Global Flags & Base Constants
# =============================================================================
NORMALIZE_FEATURES = True  # for LSTD
EPISODIC = True           # RND LSTD
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
    "GAMMA": 0.99, # similar to that used by RND
    "GAMMA_i": 0.99,
    "GAE_LAMBDA": 0.9,
    "GAE_LAMBDA_i": 0.9,
    "LSTD_LAMBDA_i": 0.8,
    "CLIP_EPS": 0.2,
    "VF_CLIP": 0.5,
    "ENT_COEF": 0.001,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "SEED": 42,
    # Exploration Specific
    "RND_TRAIN_FRAC": 0.5,
    "NORMALIZE_RHO_FEATURES": True, # can be set to false seems to have little repercussion either way.
    "NORMALIZE_LSTD_FEATURES": True,
    "EPISODIC": True,
    "ABSORBING_GOAL_STATE": True,  # death states with a positive reward are considerd goals
    "USE_ABSORBING_OVERWRITE": True, # if absorbing goal state is true, overwrite the value 
    "BONUS_SCALE": 2.0,
    "SCHEDULE_BETA": False, # decays Beta from BONUS_SCALE to 0 during learning.
    "LSTD_L2_REG": 1e-4, # is multiplied by N (~1e5) so will be ~1e-2 in the end.
    "NETWORK_TYPE": "mlp",
    "RND_NETWORK_TYPE": "mlp", # used for the (Static) LSTD network and Rho network
    "LSTD_NETWORK_TYPE": "mlp", # used for the (Static) LSTD network and Rho network
    "LSTD_FEATURES": 128,
    "WARMUP": 1_000,
    "STAGGERED_STARTS": True,
    "BIAS": False,
    "LSTD_BIAS": True,
    "RB_SIZE": 100_000,
    "PERCENT_FIFO": .25,
    "CALC_TRUE_VALUES": False,
    "N_SEEDS": 2,
    # not tested over:
    "NORMALIZE_OBS": False, 
    "NORMALIZE_REWARDS": False,
    # pretrained mode: offline feature cache lookup (e.g. DINOv2 ViT-S/14)
    "PRETRAINED_CACHE_PATH": None,       # path to .npz from precompute script
    "PRETRAINED_MODEL_TAG": "dinov2_vits14",  # informational only
    "GLOBAL_ADVANTAGE_CENTERING": False,
    "ALPHA_LSTD": 1.0, # 1 = hard update (default). 
}

# --- Environment Specific Overrides ---
ds_specific = {
    "ENV_NAME": "DeepSea-bsuite",
    "NORMALIZE_OBS": False,
    "NETWORK_TYPE": "cnn",
    "RND_NETWORK_TYPE": "cnn",
    "LSTD_NETWORK_TYPE": "cnn_pooling",
    "N_SEEDS": 4,
    "ENV_SIZE": 50,
    "ABSORBING_GOAL_STATE": False,
    "BONUS_SCALE": 0.5, # very sparse reward but does have negative reward of 1/N
    "NORMALIZE_RHO_FEATURES": False, 
    "RB_SIZE": 200_000, # prevent forgetting after convergence
    "CLIP_EPS": 0.1, # prevent the policy from too rapidly veering away from the correct path after finding it due to "distractor" rewards.
    "VF_CLIP": 0.1,
    "LSTD_L2_REG": 1e-8,
}

min_specific = {
    "ENV_NAME": "Breakout-MinAtar", # Default placeholder
    "TOTAL_TIMESTEPS": 1e7,
    "LR": 2.5e-3,
    "LR_END": 1e-5,
    "GAE_LAMBDA": 0.8,
    "NORMALIZE_OBS": False,
    "NETWORK_TYPE": "cnn",
    "RND_NETWORK_TYPE": "cnn",
    "LSTD_NETWORK_TYPE": "cnn_pooling",
    "N_SEEDS": 4,
    "ENT_COEF": 0.001,
    "LSTD_L2_REG": 1e-7,
    }

visual = {
    "NETWORK_TYPE": "cnn",
    "RND_NETWORK_TYPE": "cnn",
    "LSTD_NETWORK_TYPE": "cnn_pooling", # used for the (Static) LSTD network and Rho network
    "NORMALIZE_OBS": False,
    "WARMUP": 0,
    "ENT_COEF": 0.01,
    "BONUS_SCALE": 0.5,
    "LSTD_FEATURES": 128,
    "RHO_FEATURES": 128,
    "LSTD_L2_REG": 1e-7,
}

chain = {
    "ENV_NAME": "Chain",
    "RND_NETWORK_TYPE": "identity",
    "NETWORK_TYPE": "cnn_1d",
    "NORMALIZE_OBS": False,
    "NORMALIZE_FEATURES": False,  # tabular
    "NORMALIZE_REWARDS": False,
    "RHO_FEATURES": 150,
    "ENV_SIZE": 150,
    "LSTD_FEATURES": 150,
    "LSTD_BIAS": False,
    "CALC_TRUE_VALUES": True,
    "NORMALIZE_LSTD_FEATURES": False,
    "BIAS": False,
    "EPISODIC": EPISODIC,
    "ABSORBING_GOAL_STATE": True,
    "STAGGERED_STARTS": False,
    "LSTD_LAMBDA_i": 0.0,
    "N_SEEDS": 4, 
    }


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
    "mc": {"config_dict": mc_config, "envs": ["SparseMountainCar-v0"]},
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
            "SparseMountainCar-v0"
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
    for size in [20, 30, 40, 45, 50, 60, 70, 80]:
        ds_cfg = shared_base | ds_base | {
            "ENV_SIZE": size, 
            "TOTAL_TIMESTEPS": int(size * 5_000), 
            "NUM_ENVS": 100,
            "NUM_STEPS": size, # exactly 100 episodes at a time for 50 rounds.
            **ft_overrides
        }
        FINAL_TESTING[f"ds_{size}"] = {"config_dict": ds_cfg, "envs": ["DeepSea-bsuite"]}

    # FourRooms: Dual sizes
    for size in [21, 31, 41]:
        fr_cfg = shared_base| visual_base | {"ENV_SIZE": size,**ft_overrides}
        FINAL_TESTING[f"four_rooms_{size}"] = {"config_dict": fr_cfg, "envs": ["FourRoomsCustom-v0"]}

    # Chain: cnn_1d Distillation Proof
    for size in [200, 400, 600]:
        ch_cfg = shared_base | chain_base | {"ENV_SIZE": size,"RND_NETWORK_TYPE": "mlp", "RHO_FEATURES": 128,"BIAS": True,"NORMALIZE_FEATURES": True, "LSTD_FEATURES": 128, **ft_overrides
        }
        FINAL_TESTING[f"chain_mlp_{size}"] = {"config_dict": ch_cfg, "envs": ["Chain"]}

    # ---------------------------------------------------------------------
    # FINAL_EXACT: Exact solving for analysis/heatmaps
    # ---------------------------------------------------------------------
    fe_overrides = {"CALC_TRUE_VALUES": True, "N_SEEDS": 1}
    
    FINAL_EXACT["chain_mlp_175"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 175, "RND_NETWORK_TYPE": "mlp", "RHO_FEATURES": 128, "LSTD_FEATURES": 128,
            "BIAS": True, "NORMALIZE_FEATURES": True, **fe_overrides
        },
        "envs": ["Chain"]
    }

    FINAL_EXACT["maze_exact"] = {
        "config_dict": shared_base | visual_base | {"ENV_SIZE": 100, **fe_overrides},
        "envs": ["Maze"]
    }

    FINAL_EXACT["ds_40_exact"] = {
        "config_dict": shared_base | ds_base | {"ENV_SIZE": 40, "TOTAL_TIMESTEPS": 200_000, "NUM_ENVS": 64, "NUM_STEPS": 40, **fe_overrides}, 
        "envs": ["DeepSea-bsuite"]
    }

    FINAL_EXACT["ds_50_exact"] = {
        "config_dict": shared_base | ds_base | {"ENV_SIZE": 50, "TOTAL_TIMESTEPS": 250_000, "NUM_ENVS": 64, "NUM_STEPS": 50, **fe_overrides}, 
        "envs": ["DeepSea-bsuite"]
    }

    FINAL_EXACT["ds_60_exact"] = {
        "config_dict": shared_base | ds_base | {"ENV_SIZE": 60, "TOTAL_TIMESTEPS": 300_000, "NUM_ENVS": 64, "NUM_STEPS": 60,  **fe_overrides}, 
        "envs": ["DeepSea-bsuite"]
    }

    FINAL_EXACT["ds_70_exact"] = {
        "config_dict": shared_base | ds_base | {"ENV_SIZE": 70, "TOTAL_TIMESTEPS": 350_000, "NUM_ENVS": 64, "NUM_STEPS": 70, **fe_overrides}, 
        "envs": ["DeepSea-bsuite"]
    }

    FINAL_EXACT["four_rooms_21_exact"] = {
        "config_dict": shared_base | visual_base | {"ENV_SIZE": 21, **fe_overrides}, 
        "envs": ["FourRoomsCustom-v0"]
    }

    FINAL_EXACT["four_rooms_31_exact"] = {
        "config_dict": shared_base | visual_base | {"ENV_SIZE": 31, **fe_overrides}, 
        "envs": ["FourRoomsCustom-v0"]
    }

    FINAL_EXACT["four_rooms_41_exact"] = {
        "config_dict": shared_base | visual_base | {"ENV_SIZE": 41, **fe_overrides}, 
        "envs": ["FourRoomsCustom-v0"]
    }

    FINAL_EXACT["four_rooms_51_exact"] = {
        "config_dict": shared_base | visual_base | {"ENV_SIZE": 51, **fe_overrides}, 
        "envs": ["FourRoomsCustom-v0"]
    }

    # 150-Chain Ablation: Tabular vs cnn_1d
    FINAL_EXACT["chain_tabular_150_continuing"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 150, "RND_NETWORK_TYPE": "identity", "RHO_FEATURES": 150, "EPISODIC": False, "ABSORBING_GOAL_STATE": False,
            "BIAS": False, "LSTD_L2_REG": 1e-10, "NORMALIZE_FEATURES": False, "LSTD_FEATURES": 150, **fe_overrides
        },
        "envs": ["Chain"]
    }

    # 150-Chain Ablation: Tabular vs cnn_1d
    FINAL_EXACT["chain_tabular_150_ep"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 140, "RND_NETWORK_TYPE": "identity", "RHO_FEATURES": 140, "EPISODIC": True, "ABSORBING_GOAL_STATE": False,
            "BIAS": False, "LSTD_L2_REG": 1e-10, "NORMALIZE_FEATURES": False, "LSTD_FEATURES": 140, **fe_overrides
        },
        "envs": ["Chain"]
    }

    # 150-Chain Ablation: Tabular vs cnn_1d
    FINAL_EXACT["chain_tabular_160_absorbing"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 160, "RND_NETWORK_TYPE": "identity", "RHO_FEATURES": 160, "EPISODIC": True, "ABSORBING_GOAL_STATE": True,
            "BIAS": False, "LSTD_L2_REG": 1e-10, "NORMALIZE_FEATURES": False, "LSTD_FEATURES": 160, **fe_overrides
        },
        "envs": ["Chain"]
    }

    FINAL_EXACT["chain_mlp_165_absorbing"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 160, "RND_NETWORK_TYPE": "mlp", "RHO_FEATURES": 128, "EPISODIC": True, "ABSORBING_GOAL_STATE": True,
            "BIAS": False, "LSTD_BIAS": True ,"LSTD_L2_REG": 1e-10, "NORMALIZE_FEATURES": True, "LSTD_FEATURES": 128, **fe_overrides
        },
        "envs": ["Chain"]
    }

    FINAL_EXACT["chain_cnn_170_absorbing"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 160, "RND_NETWORK_TYPE": "cnn_1d", "RHO_FEATURES": 128, "EPISODIC": True, "ABSORBING_GOAL_STATE": True,
            "BIAS": False, "LSTD_BIAS": True ,"LSTD_L2_REG": 1e-10, "NORMALIZE_FEATURES": True, "LSTD_FEATURES": 128, **fe_overrides
        },
        "envs": ["Chain"]
    }

    FINAL_EXACT["chain_tabular_50_cont"] = {
        "config_dict": shared_base | chain_base | {
            "ENV_SIZE": 50, "RND_NETWORK_TYPE": "identity", "RHO_FEATURES": 50, "EPISODIC": False, "ABSORBING_GOAL_STATE": False,
            "BIAS": False, "LSTD_L2_REG": 1e-10, "NORMALIZE_FEATURES": False, "LSTD_FEATURES": 50, **fe_overrides
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