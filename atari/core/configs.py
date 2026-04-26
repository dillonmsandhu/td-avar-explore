# ClearRL RND: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
shared_config = {
    "TOTAL_TIMESTEPS": int(5e7),
    "NUM_ENVS": 128, # CleanRL RND: 128
    "NUM_STEPS": 128, # CleanRL RND: 128, Clean
    "GAMMA": 0.995, # CleanRL RND is 0.999
    "GAE_LAMBDA": 0.95, # CleanRL RND
    "CLIP_EPS": 0.1, # CleanRL RND
    "VF_CLIP": 0.5, # Seperate, based on Museli's high clipping and other claims that vf clipping doesn't help
    "ENT_COEF": 0.001, # CleanRL 0.01, CleanRL RND: 0.001.
    "ENV_NAME": "Pong-v5",
    "LAYER_NORM" : True,
    'LR': 3e-4, # CleanRL: 2e-4, museli 3e-4
    'LR_END': 1e-6, # CleanRL: 0.0
    'MAX_GRAD_NORM': 1.0,
    "LR_SCHEDULE": "linear",
    "NUM_EPOCHS": 4, # CleanRL: 4
    "MINIBATCH_SIZE": 1024, # CleanRL: 256 (= 8 * 128 / 4)
    "VF_COEF": 0.25, # museli
    "ENV_KWARGS": {
        "episodic_life": False,
        "reward_clip": True,
        "repeat_action_probability": 0.25,
        "frame_skip": 4,
        "noop_max": 30,
    },
    # Exploration Specific
    "GAMMA_i": 0.99, # CleanRL RND
    "GAE_LAMBDA_i": 0.95, # CleanRL RND
    "LSTD_LAMBDA_i": 0.8, # New
    "NORMALIZE_RHO_FEATURES": True, # New
    "NORMALIZE_LSTD_FEATURES": True, # New
    "EPISODIC": True,  # CleanRL does continuous
    "ABSORBING_GOAL_STATE": True,
    "BONUS_SCALE": 2.0, # CleanRL RND starts at 2 and keeps it at 2. Due to reward normalization exploration never decays
    "SCHEDULE_BETA": True, # New
    "LSTD_L2_REG": 1e-6,
    "RND_FEATURES": 256,
    "LSTD_FEATURES": 256, # 384. DinoV2 small has this many so let's see how that does.
    "BIAS": False,
    "RB_SIZE": 200_000,
    "PERCENT_FIFO": .1,
    "SEED": 42,

}
