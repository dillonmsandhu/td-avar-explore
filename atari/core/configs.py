# ClearRL RND: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
shared_config = {
    "TOTAL_TIMESTEPS": int(5e7),
    "NUM_ENVS": 128, # CleanRL: 8, museli 768(?)
    "NUM_STEPS": 128, # CleanRL: 128
    "GAMMA": 0.999,
    "GAE_LAMBDA": 0.95, # CleanRL: 0.95
    "CLIP_EPS": 0.1, # CleanRL: 0.1
    "ENT_COEF": 0.001, # CleanRL 0.01
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
        "episodic_life": True,
        "reward_clip": True,
        "repeat_action_probability": 0.25,
        "frame_skip": 4,
        "noop_max": 30,
    },
    # Exploration Specific
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
    "BIAS": True,
    "RB_SIZE": 100_000,
    "PERCENT_FIFO": .25,

}
