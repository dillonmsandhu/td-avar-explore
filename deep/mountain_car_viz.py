"""
Visualization for Mountain Car.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networks


def visualize_mountain_car(
    runner_state,
    config,
    method,
    vis_type="intrinsic_value",
    seed_idx=0,
    n_pos=100,
    n_vel=100,
    save_path=None,
    figsize=(12, 8),
    title=None,
):
    valid_methods = ["rnd", "rnd_lstd", "cov_lstd", "cov_net"]
    valid_vis_types = ["intrinsic_value", "intrinsic_reward"]

    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method}")
    if vis_type not in valid_vis_types:
        raise ValueError(f"vis_type must be one of {valid_vis_types}, got {vis_type}")

    if method == "rnd":
        if len(runner_state) == 7:
            train_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms, idx = runner_state
        else:  # old format without idx
            train_state, rnd_state, env_state, last_obs, rng, rnd_ret_rms = runner_state
            idx = None
        lstd_state = None
        sigma_state = None
    elif method == "rnd_lstd":
        if len(runner_state) == 8:
            train_state, rnd_state, lstd_state, env_state, last_obs, rng, rnd_ret_rms, idx = runner_state
        else:  # old format without idx
            train_state, rnd_state, lstd_state, env_state, last_obs, rng, rnd_ret_rms = runner_state
            idx = None
        sigma_state = None
    elif method == "cov_lstd":
        if len(runner_state) == 8:
            train_state, lstd_state, sigma_state, rnd_state, env_state, last_obs, rng, idx = runner_state
        elif len(runner_state) == 7:
            # Old format: no rnd_state, but has idx
            train_state, lstd_state, sigma_state, env_state, last_obs, rng, idx = runner_state
            rnd_state = None
        elif len(runner_state) == 6:
            # Very old format: no rnd_state, no idx
            train_state, lstd_state, sigma_state, env_state, last_obs, rng = runner_state
            rnd_state = None
            idx = None
        else:
            raise ValueError(f"Unexpected runner_state length for cov_lstd: {len(runner_state)}. "
                           f"Types: {[type(x).__name__ for x in runner_state]}")
        rnd_ret_rms = None
    elif method == "cov_net":
        if len(runner_state) == 7:
            train_state, sigma_state, rnd_state, env_state, last_obs, rng, idx = runner_state
        else:  # old format without idx
            train_state, sigma_state, rnd_state, env_state, last_obs, rng = runner_state
            idx = None
        lstd_state = None
        rnd_ret_rms = None

    # --- SET SEED ---
    if seed_idx is not None:
        train_params = jax.tree.map(lambda x: x[seed_idx], train_state.params)
        if rnd_state is not None:
            target_params = jax.tree.map(lambda x: x[seed_idx], rnd_state.target_params)
            pred_params = jax.tree.map(lambda x: x[seed_idx], rnd_state.params)
        else:
            target_params = None
            pred_params = None

        env_state_seed = jax.tree.map(lambda x: x[seed_idx], env_state)
        obs_mean = env_state_seed.mean_std.mean[0]
        obs_var = env_state_seed.mean_std.var[0]

        if lstd_state is not None:
            if method == "rnd_lstd":
                w_int = lstd_state.get('w_int', lstd_state.get('w'))[seed_idx] if isinstance(lstd_state, dict) else None
            elif method == "cov_lstd":
                # Try different possible key names for weights
                if isinstance(lstd_state, dict):
                    if 'w' in lstd_state:
                        w = lstd_state['w'][seed_idx]
                    elif 'weights' in lstd_state:
                        w = lstd_state['weights'][seed_idx]
                    else:
                        raise KeyError(f"lstd_state doesn't contain 'w' or 'weights'. Available keys: {list(lstd_state.keys())}")
                else:
                    raise TypeError(f"Expected lstd_state to be a dict, got {type(lstd_state)}")
        else:
            w_int = None
            w = None

        if sigma_state is not None:
            S = sigma_state['S'][seed_idx]
            N = sigma_state['N'][seed_idx]
        else:
            S = None
            N = None

        if rnd_ret_rms is not None:
            rnd_ret_var = rnd_ret_rms.var[seed_idx] if hasattr(rnd_ret_rms.var, '__getitem__') else rnd_ret_rms.var
        else:
            rnd_ret_var = None
    else:
        # Use non-indexed versions
        train_params = train_state.params
        if rnd_state is not None:
            target_params = rnd_state.target_params
            pred_params = rnd_state.params
        else:
            target_params = None
            pred_params = None

        obs_mean = env_state.mean_std.mean[0]
        obs_var = env_state.mean_std.var[0]

        if lstd_state is not None:
            if method == "rnd_lstd":
                w_int = lstd_state.get('w_int', lstd_state.get('w')) if isinstance(lstd_state, dict) else None
            elif method == "cov_lstd":
                # Try different possible key names for weights
                if isinstance(lstd_state, dict):
                    if 'w' in lstd_state:
                        w = lstd_state['w']
                    elif 'weights' in lstd_state:
                        w = lstd_state['weights']
                    else:
                        raise KeyError(f"lstd_state doesn't contain 'w' or 'weights'. Available keys: {list(lstd_state.keys())}")
                else:
                    raise TypeError(f"Expected lstd_state to be a dict, got {type(lstd_state)}")
        else:
            w_int = None
            w = None

        if sigma_state is not None:
            S = sigma_state['S']
            N = sigma_state['N']
        else:
            S = None
            N = None

        if rnd_ret_rms is not None:
            rnd_ret_var = rnd_ret_rms.var
        else:
            rnd_ret_var = None

    # Mountain car state bounds
    pos_min, pos_max = -1.2, 0.6
    vel_min, vel_max = -0.07, 0.07

    # Grid of states
    positions = np.linspace(pos_min, pos_max, n_pos)
    velocities = np.linspace(vel_min, vel_max, n_vel)
    pos_grid, vel_grid = np.meshgrid(positions, velocities)
    states = np.stack([pos_grid, vel_grid], axis=-1)

    if method in ["rnd", "rnd_lstd", "cov_lstd", "cov_net"]:
        rnd_net = networks.RND_Net(
            network_type=config.get('NETWORK_TYPE', 'mlp'),
            k=128,
            normalize=config.get('NORMALIZE_FEATURES', False)
        )

        # For old checkpoints that don't have target_params in runner_state
        if target_params is None and method in ["cov_lstd", "cov_net"]:
            raise ValueError(
                f"Old checkpoint format detected: runner_state doesn't contain RND parameters.\n"
                f"For cov_lstd/cov_net visualization, target_params are required.\n"
                f"Checkpoint has {len(runner_state)} elements in runner_state.\n"
                f"Please re-run training to generate a checkpoint with the current format."
            )

    def normalize_obs(obs):
        return jnp.clip((obs - obs_mean) / jnp.sqrt(obs_var + 1e-8), -10.0, 10.0)

    if vis_type == "intrinsic_reward":
        if method in ["rnd", "rnd_lstd"]:
            # RND-based intrinsic reward: mean((pred - target)^2) / sqrt(var)
            def compute_value_for_state(state):
                norm_state = normalize_obs(state)
                rnd_obs = jnp.clip(norm_state, -5, 5)
                target_embedding = rnd_net.apply(target_params, rnd_obs)
                pred_embedding = rnd_net.apply(pred_params, rnd_obs)
                intrinsic_reward_raw = jnp.mean((pred_embedding - target_embedding)**2)
                intrinsic_reward = intrinsic_reward_raw / (jnp.sqrt(rnd_ret_var) + 1e-8)
                return jnp.squeeze(intrinsic_reward)

        elif method in ["cov_lstd", "cov_net"]:
            # Covariance-based intrinsic reward: sqrt(phi^T Sigma^{-1} phi / N)
            bonus_scale = config.get('BONUS_SCALE', 1.0)
            gram_reg = config.get('GRAM_REG', 1e-4)

            def compute_value_for_state(state):
                norm_state = normalize_obs(state)
                rnd_obs = jnp.clip(norm_state, -5, 5)
                phi = rnd_net.apply(target_params, rnd_obs)
                phi = jnp.atleast_1d(jnp.squeeze(phi))

                # Compute Sigma^{-1}
                Sigma_inv = jnp.linalg.solve(
                    S + gram_reg * jnp.eye(S.shape[0]),
                    jnp.eye(S.shape[0])
                )
                bonus_sq = jnp.einsum('i,ij,j->', phi, Sigma_inv, phi) / jnp.maximum(1.0, N)
                bonus = bonus_scale * jnp.sqrt(jnp.maximum(bonus_sq, 0.0))
                return jnp.squeeze(bonus)

    else:  # vis_type == "intrinsic_value"
        if method == "rnd":
            # RND with learned intrinsic value network
            def compute_value_for_state(state):
                norm_state = normalize_obs(state)
                _, _, i_value = train_state.apply_fn(train_params, norm_state)
                return jnp.squeeze(i_value)

        elif method == "rnd_lstd":
            # RND with LSTD intrinsic value: phi @ w_int
            def compute_value_for_state(state):
                norm_state = normalize_obs(state)
                rnd_obs = jnp.clip(norm_state, -5, 5)
                phi = rnd_net.apply(target_params, rnd_obs)
                phi = jnp.atleast_1d(jnp.squeeze(phi))
                intrinsic_value = jnp.dot(phi, w_int)
                return jnp.squeeze(intrinsic_value)

        elif method == "cov_lstd":
            # Covariance with LSTD intrinsic value: phi @ w
            def compute_value_for_state(state):
                norm_state = normalize_obs(state)
                rnd_obs = jnp.clip(norm_state, -5, 5)
                phi = rnd_net.apply(target_params, rnd_obs)
                phi = jnp.atleast_1d(jnp.squeeze(phi))
                intrinsic_value = jnp.dot(phi, w)
                return jnp.squeeze(intrinsic_value)

        elif method == "cov_net":
            # Covariance with learned intrinsic value network
            def compute_value_for_state(state):
                norm_state = normalize_obs(state)
                _, _, i_value = train_state.apply_fn(train_params, norm_state)
                return jnp.squeeze(i_value)

    # Compute values over the grid
    compute_value_batch = jax.vmap(jax.vmap(compute_value_for_state))
    value_grid = compute_value_batch(jnp.array(states))
    value_grid = np.array(value_grid)

    print(f"{method} - {vis_type} grid shape: {value_grid.shape}")
    print(f"  Min: {value_grid.min():.6f}, Max: {value_grid.max():.6f}, Mean: {value_grid.mean():.6f}")

    if value_grid.ndim != 2:
        raise ValueError(f"Value grid should be 2D but has shape {value_grid.shape}")

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(pos_grid, vel_grid, value_grid, shading='auto', cmap='viridis')

    cbar_label = 'Intrinsic Value' if vis_type == 'intrinsic_value' else 'Intrinsic Reward'
    cbar = plt.colorbar(im, ax=ax, label=cbar_label)

    # Mountain car terrain
    def height(xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    ax2 = ax.twinx()
    terrain_heights = height(positions)
    terrain_line, = ax2.plot(positions, terrain_heights, 'w--', linewidth=2.5, label='Terrain')
    ax2.set_ylim(0, 1.5)
    ax2.set_yticks([])

    goal_line = ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2.5, label='Goal')

    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Velocity', fontsize=12)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    if title is None:
        method_names = {
            'rnd': 'RND',
            'rnd_lstd': 'RND LSTD',
            'cov_lstd': 'Cov LSTD',
            'cov_net': 'Cov Net'
        }
        vis_names = {
            'intrinsic_value': 'Intrinsic Value',
            'intrinsic_reward': 'Intrinsic Reward'
        }
        title = f'{method_names[method]} - {vis_names[vis_type]}'

    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return fig, ax, value_grid
