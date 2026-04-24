NORMALIZE_FEATURES = True  # for LSTD.
EPISODIC = True  # RND continuous. RND LSTD try both.
BIAS = True  # for LSTD
NORMALIZE_REWARDS = False # normalize extrinsic reward.
# for RND: bias, episodic, and normalize feawtures are all false.
# for covariance based: all true

mc_specific = {
    "ENV_NAME": "MountainCar-v0",
    "NORMALIZE_FEATURES": NORMALIZE_FEATURES,
    "NORMALIZE_REWARDS": NORMALIZE_REWARDS,
}

ds_specific = {
    "ENV_NAME": "DeepSea-bsuite",
    "TOTAL_TIMESTEPS": 1e5 * 5,
    "NORMALIZE_OBS": False,
    "NORMALIZE_FEATURES": NORMALIZE_FEATURES,
    "ENV_SIZE": 45,
    "WARMUP": 20_000,  # warmup steps for running mean/std
    "NETWORK_TYPE": "cnn",
    "RND_NETWORK_TYPE": "cnn",
    "CALC_TRUE_VALUES": True,
    "NORMALIZE_REWARDS": False,
    "N_SEEDS": 4,
    "LSTD_PRIOR_SAMPLES": 1.0,
    "BONUS_SCALE": 0.5,
}

min_specific = {
    "ENV_NAME": "Breakout-MinAtar",
    "TOTAL_TIMESTEPS": 1e7,
    "LR": 2.5e-3,
    "LR_END": 1e-5,
    "NUM_ENVS": 32,
    "NUM_STEPS": 128,
    "GAE_LAMBDA": 0.8,
    "CLIP_EPS": 0.1,
    "VF_CLIP": 0.2,
    "NORMALIZE_FEATURES": NORMALIZE_FEATURES,
    "NORMALIZE_OBS": False,
    "WARMUP": 20_000,  # warmup steps for running mean/std
    "NETWORK_TYPE": "cnn",
    "RND_NETWORK_TYPE": "cnn",
    "NORMALIZE_REWARDS": False,
    "N_SEEDS": 4,
}

shared = {
    "LR": 5e-4,
    "LR_END": 5e-4,
    "RND_LR": 5e-5,
    "NUM_ENVS": 32,
    "NUM_STEPS": 256,
    "TOTAL_TIMESTEPS": 1e6,  # will be adjusted up
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 256,
    "GAMMA": 0.99,  # extrinsic Gamma
    "GAMMA_i": 0.99,  # Intrinsic Gamma
    "GAE_LAMBDA": 0.9,
    "GAE_LAMBDA_i": 0.9,
    "CLIP_EPS": 0.2,
    "VF_CLIP": 0.5,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "SEED": 42,
    # FOR RND
    "RND_TRAIN_FRAC": 0.5,
    "NORMALIZE_FEATURES": NORMALIZE_FEATURES,
    "NORMALIZE_OBS": True,
    "NORMALIZE_REWARDS": NORMALIZE_REWARDS,
    # FOR Covariance Based Reward
    "BONUS_SCALE": 0.5,
    "SCHEDULE_BETA": False,
    "A_REGULARIZATION_PER_STEP": 1e-2,
    "A_REGULARIZATION": 1e-2,
    "GRAM_REG": 1e-2,
    "N_SEEDS": 8,
    "EPISODIC": EPISODIC,
    "RND_FEATURES": 128,
    "NETWORK_TYPE": "mlp",
    "RND_NETWORK_TYPE": "mlp",
    "WARMUP": 20_000,
    "ALPHA_SCHEDULE": "inv_t",
    # "MIN_COV_LR": 1 / 100,
    "MIN_LSTD_LR": 1 / 100,
    # "MIN_LSTD_LR_RI": 1 / 100,  # LSTD for intrinsic reward: faster forgetting of intrinsic reward.
    "ADAPTIVE_BETA": False,
    "LSTD_PRIOR_SAMPLES": 50.0,
    "STAGGERED_STARTS": True,
    "BIAS": BIAS,
    "CLIP_REWARD": False,
    # for the LSPI variant
    "LSPI_NUM_ITERS": 5,
    "ABSORBING_GOAL_STATE": True, 
    "RB_SIZE": 250_000,
    "PERCENT_FIFO": 0.5
}

visual = {
    "NETWORK_TYPE": "cnn",
    "RND_NETWORK_TYPE": "cnn",
    "NORMALIZE_OBS": False,
    "WARMUP": 0,
    "CALC_TRUE_VALUES": False,
}
four_rooms = {
    "NETWORK_TYPE": "cnn",
    "RND_NETWORK_TYPE": "cnn",
    "NORMALIZE_OBS": False,
    "WARMUP": 0,
    "CALC_TRUE_VALUES": True,
    "N_SEEDS": 4,
}

maze = {
    "NETWORK_TYPE": "cnn",
    "RND_NETWORK_TYPE": "cnn",
    "NORMALIZE_OBS": False,
    "WARMUP": 0,
    "CALC_TRUE_VALUES": True,
    "ENV_SIZE": 100,
    "N_SEEDS": 4,
}

continuous = {
    "LR": 1e-3,
    "LR_END": 5e-4,
}
chain = {
    "ENV_NAME": "Chain",
    "RND_NETWORK_TYPE": "identity",
    "NETWORK_TYPE": "mlp",
    "NORMALIZE_OBS": False,
    "NORMALIZE_FEATURES": False,  # tabular
    "NORMALIZE_REWARDS": False,
    "RND_FEATURES": 200,
    "ENV_SIZE": 200,
    "LSTD_FEATURES": 200,
    "CALC_TRUE_VALUES": True,
    "BIAS": False,
    "EPISODIC": EPISODIC,
    "STAGGERED_STARTS": False,
    "N_SEEDS": 4, 
    # "ALPHA_SCHEDULE": "inv_t",
    # "MIN_COV_LR": 1 / 100,
    # "MIN_LSTD_LR": 1 / 100,
    # "MIN_LSTD_LR_RI": 1 / 100,  # LSTD for intrinsic reward: faster forgetting of intrinsic reward.
}

if chain["RND_NETWORK_TYPE"] == "identity":
    chain["RND_FEATURES"] = chain["ENV_SIZE"]

mc_config = shared | mc_specific  # | is the union op. last dict's key takes precedence
ds_config = shared | ds_specific
min_config = shared | min_specific
four_rooms_config = shared | four_rooms
maze_config = shared | maze
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
    "four_rooms": {
        "config_dict": four_rooms_config,
        # "envs": ["FourRoomsPixels", "FourRoomsCustom-v0", ],
        "envs": ["FourRoomsCustom-v0", ],
    },
    "maze": {
        "config_dict": maze_config,
        # "envs": ["FourRoomsPixels", "FourRoomsCustom-v0", ],
        "envs": ["Maze", ],
    },
    "visual": {
        "config_dict": visual,
        "envs": ["Pong-misc", "MNISTBandit-bsuite", "Catch-bsuite"],
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

DEBUG_REGISTRY = {
    "chain": {
        "config_dict": chain,
        "envs": [
            "Chain",
        ],
    },
    "four_rooms": {
        "config_dict": four_rooms_config,
        "envs": ["FourRoomsCustom-v0"],
    },
    "ds": {"config_dict": ds_config, "envs": ["DeepSea-bsuite"]},
    "maze": {
        "config_dict": maze_config,
        # "envs": ["FourRoomsPixels", "FourRoomsCustom-v0", ],
        "envs": ["Maze", ],
    },
}

DISCRETE_REGISTRY = {
    # maps from config name to all envs that we can run that use that config.
    "shared": {
        "config_dict": shared,
        "envs": [
            "DiscountingChain-bsuite",
            "BernoulliBandit-misc",
            "GaussianBandit-misc",
            "MetaMaze-misc",
            "CartPole-v1",
            "UmbrellaChain-bsuite",
        ],
    },
    "four_rooms": {
        "config_dict": four_rooms_config,
        "envs": ["FourRoomsCustom-v0"],
    },
    "visual": {
        "config_dict": visual,
        "envs": ["Pong-misc", "MNISTBandit-bsuite", "Catch-bsuite"],
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
    "maze": {
        "config_dict": maze_config,
        # "envs": ["FourRoomsPixels", "FourRoomsCustom-v0", ],
        "envs": ["Maze", ],
    },
}

CONTINUOUS_REGISTRY = {
    "shared": {
        "config_dict": shared,
        "envs": [
            "Acrobot-v1",
            "UmbrellaChain-bsuite",
            "Reacher-misc",
            "PointRobot-misc",
            "Swimmer-misc",
        ],
    }
}


EXACT_REGISTRY = {
    "four_rooms": {
        "config_dict": four_rooms_config,
        "envs": ["FourRoomsCustom-v0"],
    },
    "ds": {"config_dict": ds_config, "envs": ["DeepSea-bsuite"]},
    "chain": {"config_dict": chain,"envs": ["Chain",]},
    "maze": {
        "config_dict": maze_config,
        # "envs": ["FourRoomsPixels", "FourRoomsCustom-v0", ],
        "envs": ["Maze", ],
    },
}


import copy

def make_final_registries(shared_base, ds_base, four_rooms_base, min_base, visual_base, maze_base, chain_base):
    """
    Generates FINAL_TESTING (no exact solving) and FINAL_EXACT (exact solving) 
    without mutating the original dictionaries.
    """
    FINAL_TESTING = {}
    FINAL_EXACT = {}

    def merge(d1, d2, overrides):
        res = copy.deepcopy(d1)
        res.update(copy.deepcopy(d2))
        res.update(overrides)
        return res

    # =====================================================================
    # 1. FINAL TESTING (NO TRUE VALUE SOLVING)
    # =====================================================================
    ft_overrides = {"CALC_TRUE_VALUES": False, "N_SEEDS": 5}

    # 1. Shared (Continuous/Standard Grid)
    FINAL_TESTING["shared"] = {
        "config_dict": merge(shared_base, {}, ft_overrides),
        "envs": [
            "DiscountingChain-bsuite", "CartPole-v1", "Acrobot-v1", 
            "Reacher-misc", "PointRobot-misc", "Swimmer-misc", 
            "SparseMountainCar-v0", "MountainCar-v0"
        ]
    }

    # 2. Visual / MinAtar / Maze
    FINAL_TESTING["visual"] = {
        "config_dict": merge(shared_base, visual_base, ft_overrides),
        "envs": ["Pong-misc"]
    }
    FINAL_TESTING["maze"] = {
        "config_dict": merge(shared_base, maze_base, ft_overrides),
        "envs": ["Maze"]
    }
    FINAL_TESTING["min"] = {
        "config_dict": merge(shared_base, min_base, ft_overrides),
        "envs": ["SpaceInvaders-MinAtar", "Breakout-MinAtar", "Freeway-MinAtar", "Asterix-MinAtar"]
    }

    # 3. DeepSea Sizes (10k episodes = 10,000 * size steps)
    for size in [20, 30, 40, 45, 50]:
        ds_cfg = merge(shared_base, ds_base, {
            "ENV_SIZE": size, 
            "TOTAL_TIMESTEPS": int(size * 10_000), 
            **ft_overrides
        })
        FINAL_TESTING[f"ds_{size}"] = {"config_dict": ds_cfg, "envs": ["DeepSea-bsuite"]}

    # 4. FourRooms Sizes
    for size in [21, 28]:
        fr_cfg = merge(shared_base, four_rooms_base, {
            "ENV_SIZE": size,
            **ft_overrides
        })
        FINAL_TESTING[f"four_rooms_{size}"] = {"config_dict": fr_cfg, "envs": ["FourRoomsCustom-v0"]}

    # 5. Chain (MLP setup for RND)
    for size in [200, 400]:
        ch_cfg = merge(shared_base, chain_base, {
            "ENV_SIZE": size,
            "RND_NETWORK_TYPE": "mlp",    # Ensure it uses network distillation
            "RND_FEATURES": 128,          # Standard network feature size
            "BIAS": True,                 # LSTD Bias enabled
            "NORMALIZE_FEATURES": True,   # Important for NN features
            **ft_overrides
        })
        FINAL_TESTING[f"chain_mlp_{size}"] = {"config_dict": ch_cfg, "envs": ["Chain"]}


    # =====================================================================
    # 2. FINAL EXACT (WITH TRUE VALUE SOLVING)
    # =====================================================================
    fe_overrides = {"CALC_TRUE_VALUES": True, "N_SEEDS": 1}

    FINAL_EXACT["maze"] = {
        "config_dict": merge(shared_base, maze_base, fe_overrides),
        "envs": ["Maze"]
    }

    FINAL_EXACT["ds_40"] = {
        "config_dict": merge(shared_base, ds_base, {"ENV_SIZE": 40, "TOTAL_TIMESTEPS": 400_000, **fe_overrides}), 
        "envs": ["DeepSea-bsuite"]
    }

    FINAL_EXACT["four_rooms_21"] = {
        "config_dict": merge(shared_base, four_rooms_base, {"ENV_SIZE": 21, **fe_overrides}), 
        "envs": ["FourRoomsCustom-v0"]
    }

    # 150-Chain: Tabular Baseline
    FINAL_EXACT["chain_tabular_150"] = {
        "config_dict": merge(shared_base, chain_base, {
            "ENV_SIZE": 150,
            "RND_NETWORK_TYPE": "identity", 
            "RND_FEATURES": 150, 
            "BIAS": False, 
            "NORMALIZE_FEATURES": False,
            **fe_overrides
        }),
        "envs": ["Chain"]
    }

    # 150-Chain: MLP (Network) Baseline
    FINAL_EXACT["chain_mlp_150"] = {
        "config_dict": merge(shared_base, chain_base, {
            "ENV_SIZE": 150,
            "RND_NETWORK_TYPE": "mlp",
            "RND_FEATURES": 128,
            "BIAS": True,
            "NORMALIZE_FEATURES": True,
            **fe_overrides
        }),
        "envs": ["Chain"]
    }

    return FINAL_TESTING, FINAL_EXACT

# Execute the function to create the global dictionaries
FINAL_TESTING, FINAL_EXACT = make_final_registries(
    shared, ds_specific, four_rooms, min_specific, visual, maze, chain
)