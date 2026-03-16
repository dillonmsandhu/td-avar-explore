NORMALIZE_FEATURES = False # for LSTD.
EPISODIC = True # RND continuous. RND LSTD try both. 
BIAS = True # for LSTD 
NORMALIZE_REWARDS = False
# for RND: bias, episodic, and normalize feawtures are all false.
# for covariance based: all true

mc_specific = {
    "ENV_NAME": "SparseMountainCar-v0",
    "NORMALIZE_FEATURES": NORMALIZE_FEATURES,
    "NORMALIZE_REWARDS": NORMALIZE_REWARDS,
}

ds_specific = {
    "ENV_NAME": "DeepSea-bsuite",
    "TOTAL_TIMESTEPS": 1e5 * 50,
    "NORMALIZE_OBS": False,
    "NORMALIZE_FEATURES": NORMALIZE_FEATURES,
    "DEEPSEA_SIZE": 50,
    "WARMUP": 0, # warmup steps for running mean/std
    "NETWORK_TYPE": 'cnn',
    "RND_NETWORK_TYPE": 'cnn',
    "CALC_TRUE_VALUES": False,
    "NORMALIZE_REWARDS": NORMALIZE_REWARDS,
    "N_SEEDS": 4,
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
    "NORMALIZE_FEATURES": NORMALIZE_FEATURES,
    "NORMALIZE_OBS": False,
    "WARMUP": 20_000, # warmup steps for running mean/std
    "NETWORK_TYPE": 'cnn',
    "NORMALIZE_REWARDS": NORMALIZE_REWARDS,
    "N_SEEDS": 4,
}

shared = {    
    "LR": 5e-4,
    "LR_END": 5e-4,
    "RND_LR": 5e-5,
    "NUM_ENVS": 32,
    "NUM_STEPS": 256,
    "TOTAL_TIMESTEPS": 500_000, # will be adjusted up
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 256,
    "GAMMA": 0.99, # extrinsic Gamma
    "GAMMA_i": 0.99, # Intrinsic Gamma
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
    "BONUS_SCALE": 1.0,
    "A_REGULARIZATION_PER_STEP": 1e-8,
    "A_REGULARIZATION": 1e-3,
    "GRAM_REG": 1e-3,
    "N_SEEDS": 8,
    "EPISODIC": EPISODIC,
    "RND_FEATURES": 128,
    "NETWORK_TYPE": 'mlp',
    "RND_NETWORK_TYPE": 'mlp',
    "WARMUP": 20_000,
    "ALPHA_SCHEDULE": 'constant',
    "MIN_COV_LR": 1/20,
    "MIN_LSTD_LR": 1/20,
    "MIN_LSTD_LR_RI": 1/10, # LSTD for intrinsic reward: faster forgetting of intrinsic reward.
    "ADAPTIVE_BETA": True,
    "LSTD_PRIOR_SAMPLES": 100.0,
    "STAGGERED_STARTS": False,
    "BIAS": BIAS,
}

visual = {
    'NETWORK_TYPE': 'cnn',
    "RND_NETWORK_TYPE": 'cnn',
    "NORMALIZE_OBS": False,
    "WARMUP": 0,
}
continuous = {
    "LR": 1e-3,
    "LR_END": 5e-4,
}
chain={
    'ENV_NAME': 'Chain',
    'RND_NETWORK_TYPE': 'identity',
    'NETWORK_TYPE': 'mlp',
    'NORMALIZE_OBS': False,
    'NORMALIZE_FEATURES': NORMALIZE_FEATURES,
    'NORMALIZE_REWARDS': NORMALIZE_REWARDS,
    "RND_FEATURES": 100,
    "CHAIN_LENGTH": 100,
    "CALC_TRUE_VALUES": True,
    "BIAS": False,
    "EPISODIC": EPISODIC,
    "STAGGERED_STARTS": False,
    "ALPHA_SCHEDULE": 'inv_t',
    "MIN_COV_LR": 1/100,
    "MIN_LSTD_LR": 1/100,
    "MIN_LSTD_LR_RI": 1/100, # LSTD for intrinsic reward: faster forgetting of intrinsic reward.
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
                    "FourRoomsCustom-v0",
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
    "chain":    {"config_dict": chain, 
                "envs": 
                ["Chain",],}

    
}
