mc_specific = {
    "ENV_NAME": "SparseMountainCar-v0",
    "NORMALIZE_OBS": True,
    "NORMALIZE_FEATURES": False,
    "WARMUP": 200, # warmup steps for running mean/std

}

ds_specific = {
    "ENV_NAME": "DeepSea-bsuite",
    "NORMALIZE_OBS": False,
    "NORMALIZE_FEATURES": True,
    "DEEPSEA_SIZE": 20,
    "WARMUP": 0, # warmup steps for running mean/std
}

shared = {    
    "LR": 5e-4,
    "LR_END": 5e-4,
    "RND_LR": 1e-5, # very slow - learns features
    "NUM_ENVS": 32,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 120_000, # will be adjusted up
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 256,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.6,
    "CLIP_EPS": 0.2,
    "VF_CLIP": 0.5,
    "ENT_COEF": 0.003,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "SEED": 42,
    "NORMALIZE_REWARDS": False,
    # FOR RND
    "RND_TRAIN_FRAC": 0.5,
    # FOR Covariance Based Reward
    "BONUS_SCALE": 1.96,
    "A_REGULARIZATION_PER_STEP": 1e-4,
    "A_REGULARIZATION": 1e-2,
    "GRAM_REG": 1e-3,
    # For LSTD Avar
    "PRIOR_N": 1_000, # strength of prior: number of transitions where the "prior" (max) td error was "observed".
    "N_SEEDS": 4,
    "EPISODIC": True,
    "OPTIMISTIC_INIT": True,
    "RND_FEATURES": 128
}

mc_config = shared | mc_specific # | is the union op.
ds_config = shared | ds_specific