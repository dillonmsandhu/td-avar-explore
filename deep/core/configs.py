mc_specific = {
    "ENV_NAME": "SparseMountainCar-v0",
    "NORMALIZE_FEATURES": False,
    "NORMALIZE_REWARDS": False,
    "BONUS_SCALE": 1.96,
    "ALPHA_SCHEDULE": 'constant',
}

ds_specific = {
    "ENV_NAME": "DeepSea-bsuite",
    "TOTAL_TIMESTEPS": 1e5 * 50,
    "NORMALIZE_OBS": False,
    "NORMALIZE_FEATURES": False,
    "DEEPSEA_SIZE": 50,
    "WARMUP": 0, # warmup steps for running mean/std
    "NETWORK_TYPE": 'cnn',
    "CALC_TRUE_VALUES": False,
    "BONUS_SCALE": 0.5,
    "NORMALIZE_REWARDS": False,
}

min_specific = {
    "ENV_NAME": "Breakout-MinAtar",
    "TOTAL_TIMESTEPS": 1e7,
    "LR": 2.5e-3,
    "LR_END": 1e-5,
    "NUM_ENVS": 128,
    "NUM_STEPS": 64,
    "GAE_LAMBDA": 0.8,
    "CLIP_EPS": 0.1,
    "VF_CLIP": 0.2,
    "NORMALIZE_FEATURES": False,
    "NORMALIZE_OBS": False,
    "WARMUP": 200, # warmup steps for running mean/std
    "NETWORK_TYPE": 'cnn',
    "NORMALIZE_REWARDS": False,
}

shared = {    
    "LR": 5e-4,
    "LR_END": 5e-4,
    "RND_LR": 1e-5, # very slow - learns features
    "NUM_ENVS": 32,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 250_000, # will be adjusted up
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 256,
    "GAMMA": 0.999, # extrinsic Gamma
    "GAMMA_i": 0.99, # Intrinsic Gamma
    "GAE_LAMBDA": 0.9,
    "CLIP_EPS": 0.2,
    "VF_CLIP": 0.5,
    "ENT_COEF": 0.0001,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "SEED": 42,
    # FOR RND
    "RND_TRAIN_FRAC": 0.5,
    "NORMALIZE_FEATURES": False,
    "NORMALIZE_OBS": True,
    "NORMALIZE_REWARDS": False,
    # FOR Covariance Based Reward
    "BONUS_SCALE": 1.96,
    "A_REGULARIZATION_PER_STEP": 1e-3,
    "A_REGULARIZATION": 1e-2,
    "GRAM_REG": 1e-3,
    "EFFECTIVE_VISITS_TO_REMAIN_OPT": 10,
    # For LSTD Avar
    "PRIOR_N": 1, # strength of prior: number of transitions where the "prior" (max) td error was "observed".
    "N_SEEDS": 4,
    "EPISODIC": True,
    "RND_FEATURES": 128,
    "NETWORK_TYPE": 'mlp',
    "WARMUP": 20_000,
    "ALPHA_SCHEDULE": 'constant',
    "MIN_LSTD_LR": 1/10,
    "MIN_COV_LR": 1/20,
    "STANDARDIZE_RHO": False,
    "STANDARDIZE_I_GAE": True,
    "STANDARDIZE_E_GAE": False
}

visual = {
    'NETWORK_TYPE': 'cnn',
    "FOURROOMS_SIZE": 21,
    "NORMALIZE_OBS": False,
}
continuous = {
    "LR": 1e-3,
    "LR_END": 5e-4,
}
chain={
    'ENV_NAME': 'Chain',
    "TOTAL_TIMESTEPS": 20_000, # will be adjusted up
    # 'RND_NETWORK_TYPE': 'cnn_1d',
    'RND_NETWORK_TYPE': 'identity',
    'NETWORK_TYPE': 'mlp',
    'NORMALIZE_OBS': False,
    'NORMALIZE_FEATURES': True,
    "RND_FEATURES": 100,
    # "RND_FEATURES": 64,
    "CHAIN_LENGTH": 30,
    "CALC_TRUE_VALUES": True,
    "BIAS": False,
    # "BIAS": True,
    "ALPHA_SCHEDULE": 'inv_t',
    # "ALPHA_SCHEDULE": 'constant',
    "A_REGULARIZATION_PER_STEP": 1e-12,
    "A_REGULARIZATION": 1e-3,
    "MIN_COV_LR": 1/20,
    "MIN_LSTD_LR": 1/20,
    "MIN_LSTD_LR_RI": 1/10,
    "NUM_ENVS": 32,
    "NUM_STEPS": 128,
    "LSTD_PRIOR_SAMPLES": 10.0,
    "STAGGERED_STARTS": False,
    "EPISODIC": True,
    "GRAM_REG": 1e-2,
    "GAMMA": 0.99, # extrinsic Gamma
    "GAMMA_i": 0.99, # extrinsic Gamma
    "GAE_LAMBDA": 0.9,
    "GAE_LAMBDA_i": 0.9,
    "WARMUP": 0,
    "CLIP_EPS": 0.05,
    "ENT_COEF": 0.01,
    "ADAPTIVE_BETA": True,
}

if chain['RND_NETWORK_TYPE'] == 'identity':
    chain["RND_FEATURES"] = chain['CHAIN_LENGTH']

mc_config = shared | mc_specific # | is the union op. last dict's key takes precedence
ds_config = shared | ds_specific
min_config = shared | min_specific
visual = shared | visual
chain = shared | chain

CONFIG_REGISTRY = {
    # maps from config name to all envs that we can run that use that config.
    "shared": {"config_dict": shared, 
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
                        "Swimmer-misc"]},
    "visual": {"config_dict": visual, 
                "envs": [
                    "Pong-misc", 
                    "FourRooms-misc", 
                    "MNISTBandit-bsuite", 
                    "Catch-bsuite"]},
    "mc":     {"config_dict": mc_config, 
                "envs": ["SparseMountainCar-v0"]},
    "ds":     {"config_dict": ds_config, 
                "envs": ["DeepSea-bsuite"]},
    "min":    {"config_dict": min_config, 
                "envs": 
                ["SpaceInvaders-MinAtar", 
                "Breakout-MinAtar", 
                "Freeway-MinAtar", 
                "Asterix-MinAtar"],}, 
    'chain':    {"config_dict": chain, "envs": ['Chain']}
}