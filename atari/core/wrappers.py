# Adapted from from purejaxql: https://github.com/mttga/purejaxql/blob/main/purejaxql/utils/atari_wrapper.py
import jax
import jax.numpy as jnp
from flax import struct
import numpy as np
from functools import partial

import gym
from packaging import version

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")
assert is_legacy_gym, "Current version supports only gym<=0.23.1"

# (random,human)
ATARI_SCORES = {
    "Alien-v5": (227.8, 7127.7),
    "Amidar-v5": (5.8, 1719.5),
    "Assault-v5": (222.4, 742.0),
    "Asterix-v5": (210.0, 8503.3),
    "Asteroids-v5": (719.1, 47388.7),
    "Atlantis-v5": (12850.0, 29028.1),
    "BankHeist-v5": (14.2, 753.1),
    "BattleZone-v5": (2360.0, 37187.5),
    "BeamRider-v5": (363.9, 16926.5),
    "Berzerk-v5": (123.7, 2630.4),
    "Bowling-v5": (23.1, 160.7),
    "Boxing-v5": (0.1, 12.1),
    "Breakout-v5": (1.7, 30.5),
    "Centipede-v5": (2090.9, 12017.0),
    "ChopperCommand-v5": (811.0, 7387.8),
    "CrazyClimber-v5": (10780.5, 35829.4),
    "Defender-v5": (2874.5, 18688.9),
    "DemonAttack-v5": (152.1, 1971.0),
    "DoubleDunk-v5": (-18.6, -16.4),
    "Enduro-v5": (0.0, 860.5),
    "FishingDerby-v5": (-91.7, -38.7),
    "Freeway-v5": (0.0, 29.6),
    "Frostbite-v5": (65.2, 4334.7),
    "Gopher-v5": (257.6, 2412.5),
    "Gravitar-v5": (173.0, 3351.4),
    "Hero-v5": (1027.0, 30826.4),
    "IceHockey-v5": (-11.2, 0.9),
    "Jamesbond-v5": (29.0, 302.8),
    "Kangaroo-v5": (52.0, 3035.0),
    "Krull-v5": (1598.0, 2665.5),
    "KungFuMaster-v5": (258.5, 22736.3),
    "MontezumaRevenge-v5": (0.0, 4753.3),
    "MsPacman-v5": (307.3, 6951.6),
    "NameThisGame-v5": (2292.3, 8049.0),
    "Phoenix-v5": (761.4, 7242.6),
    "Pitfall-v5": (-229.4, 6463.7),
    "Pong-v5": (-20.7, 14.6),
    "PrivateEye-v5": (24.9, 69571.3),
    "Qbert-v5": (163.9, 13455.0),
    "Riverraid-v5": (1338.5, 17118.0),
    "RoadRunner-v5": (11.5, 7845.0),
    "Robotank-v5": (2.2, 11.9),
    "Seaquest-v5": (68.4, 42054.7),
    "Skiing-v5": (-17098.1, -4336.9),
    "Solaris-v5": (1236.3, 12326.7),
    "SpaceInvaders-v5": (148.0, 1668.7),
    "StarGunner-v5": (664.0, 10250.0),
    "Surround-v5": (-10.0, 6.5),
    "Tennis-v5": (-23.8, -8.3),
    "TimePilot-v5": (3568.0, 5229.2),
    "Tutankham-v5": (11.4, 167.6),
    "UpNDown-v5": (533.4, 11693.2),
    "Venture-v5": (0.0, 1187.5),
    "VideoPinball-v5": (16256.9, 17667.9),
    "WizardOfWor-v5": (563.5, 4756.5),
    "YarsRevenge-v5": (3092.9, 54576.9),
    "Zaxxon-v5": (32.5, 9173.3),
}

@struct.dataclass
class LogEnvState:
    handle: jnp.array
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array
    lives: jnp.array  # Added to track life decrements across JAX steps
    was_done: jnp.array 
    was_goal: jnp.array 

class JaxEnvPoolWrapper(gym.Wrapper):
    """
    Wraps EnvPool via XLA to provide logging AND terminal state injections 
    (real_next_obs, is_goal) for RND/bootstrapped traces.
    """
    def __init__(self, env, config, reset_info=True):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.env_name = env.name
        
        # Load scores
        if self.env_name in ATARI_SCORES:
            self.env_random_score, self.env_human_score = ATARI_SCORES[self.env_name]
        else:
            self.env_random_score, self.env_human_score = 0.0, 1.0

        self.reset_info = reset_info
        
        # XLA setup
        handle, recv, send, step = env.xla()
        self.init_handle = handle
        self.step_f = step

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        env_state = LogEnvState(
            handle=jnp.array(self.init_handle),
            episode_returns=jnp.zeros(self.num_envs, dtype=jnp.float32), 
            episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.float32), 
            returned_episode_returns=jnp.zeros(self.num_envs, dtype=jnp.float32), 
            returned_episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.float32), 
            lives=jnp.zeros(self.num_envs, dtype=jnp.int32), # Initialize lives
            was_done=jnp.zeros(self.num_envs, dtype=jnp.bool_),
            was_goal=jnp.zeros(self.num_envs, dtype=jnp.bool_),
        )
        return observations, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # 1. Step the underlying EnvPool XLA environment
        new_handle, (observations, raw_rewards, dones, infos) = self.step_f(
            state.handle, action
        )

        # 2. Terminal & Goal Logic Injection
        terminated = infos.get("terminated", dones) 
        truncated = infos.get("TimeLimit.truncated", jnp.zeros_like(dones, dtype=jnp.bool_))
        current_lives = infos.get("lives", jnp.zeros_like(dones, dtype=jnp.int32))
        
        # Capture the previous step's flags BEFORE we update the state!
        is_dummy = state.was_done
        was_goal = state.was_goal
        
        # Life loss heuristic: if lives strictly decreased from the previous step
        life_loss = current_lives < state.lives
        
        # Goal logic: Episodic termination (no timeout), reward > 0, and NOT a life-loss event
        terminal_no_timeout = terminated & ~truncated
        is_goal = terminal_no_timeout & (raw_rewards > 0) & ~life_loss

        # 3. Logging Logic
        new_episode_return = state.episode_returns + infos["reward"]
        new_episode_length = state.episode_lengths + 1
        
        # Mask out resets
        mask = 1 - (terminated | truncated)
        
        state = state.replace(
            handle=new_handle,
            episode_returns=new_episode_return * mask,
            episode_lengths=new_episode_length * mask,
            returned_episode_returns=jnp.where(
                terminated | truncated,
                new_episode_return,
                state.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                terminated | truncated,
                new_episode_length,
                state.returned_episode_lengths,
            ),
            lives=current_lives,  
            was_done=(terminated | truncated),
            was_goal=is_goal, # Save current step's goal status for the next step
        )

        # 4. Package standard infos + custom injections
        if self.reset_info:
            elapsed_steps = infos["elapsed_step"]
            episode_done = terminated | truncated
            infos = {} # Clear underlying dict to prevent bloat
            infos["elapsed_step"] = elapsed_steps
            infos["returned_episode"] = episode_done

        # Re-inject transition dynamics
        infos["is_goal"] = is_goal # currently transitioned into goal
        infos["is_dummy"] = is_dummy # transitioned from S_T -> S_0 (dummy step)
        infos['was_goal'] = was_goal  # is_goal was true for the last timestep (i.e. is a dummy after goal)

        # Inject metrics
        normalize_score = lambda x: (x - self.env_random_score) / (
            self.env_human_score - self.env_random_score + 1e-8
        )
        infos["returned_episode_returns"] = state.returned_episode_returns
        infos["normalized_returned_episode_returns"] = normalize_score(state.returned_episode_returns)
        infos["returned_episode_lengths"] = state.returned_episode_lengths

        return (
            observations, 
            state,
            raw_rewards, 
            dones,
            infos,
        )

# info["lives"]: The current number of lives remaining.
# info["reward"]: The raw, unclipped reward (crucial if you enable reward_clip=True in the EnvPool config, as the standard reward array will be clipped to [-1, 1], but this info key retains the true score).
# info["terminated"]: A boolean flag directly mapped to the emulator's game_over() signal.
# info["ram"]: A 128-byte array representing the raw RAM state of the Atari 2600.
