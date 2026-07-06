shared_config = { # for simple pong testing.
    "TOTAL_TIMESTEPS": int(1e7),
    "NUM_ENVS": 128, # CleanRL RND: 128
    "NUM_STEPS": 128, # CleanRL RND: 128, Clean
    "GAMMA": 0.999, # CleanRL RND is 0.999
    "GAE_LAMBDA": 0.95, # CleanRL RND
    "CLIP_EPS": 0.1, # CleanRL RND
    "VF_CLIP": 0.1, # Seperate, based on Museli's high clipping and other claims that vf clipping doesn't help
    "ENT_COEF": 0.001, # CleanRL 0.01, CleanRL RND: 0.001.
    "ENV_NAME": "Pong-v5",
    'LR': 1e-4, # CleanRL: 2e-4, museli 3e-4
    'LR_END': 1e-6, # CleanRL: 0.0
    'MAX_GRAD_NORM': 0.5,
    "LR_SCHEDULE": "linear",
    "NUM_EPOCHS": 4, # CleanRL: 4
    "MINIBATCH_SIZE": 4096, # CleanRL: 256 (= 8 * 128 / 4)
    "VF_COEF": 0.5, # museli
    "ENV_KWARGS": {
        "episodic_life": True,
        "reward_clip": True,
        "repeat_action_probability": 0.25,
        "frame_skip": 4,
        "noop_max": 30,
    },
    "CNN_TORSO": "CNN",
    "SEED": 42,
    
    # Exploration Specific
    "GAMMA_i": 0.99, # CleanRL RND
    "GAE_LAMBDA_i": 0.95, # CleanRL RND
    "LSTD_LAMBDA_i": 0.8, # New
    "EPISODIC": False,  # CleanRL does continuing
    "ABSORBING_GOAL_STATE": False,
    
    # Rho
    "NORMALIZE_RHO_OBS": True,
    "RND_FEATURES": 512,
    "NORMALIZE_RHO_FEATURES": False, # New
    "BIAS": False,
    "COV_LEAK": 0.999999,
    "BONUS_SCALE": 0.5, # CleanRL RND starts at 2 and keeps it at 2. Due to reward normalization exploration never decays
    "SCHEDULE_BETA": False, # New
    

    # LSTD
    "LSTD_FEATURES": 512, # 384. DinoV2 small has this many so let's see how that does.
    "LSTD_L2_REG": 1e-8,
    "POOL_LSTD_NET": False,
    "LSTD_DINO": False, # DinoV2 features plus a bias, unnormalized
    "NORMALIZE_LSTD_OBS": True,
    "NORMALIZE_LSTD_FEATURES": True, # New
    "RB_SIZE": 350_000,
    "PERCENT_FIFO": .5, # higher means more on-policy, but can lead to forgetting after convergence.
}
