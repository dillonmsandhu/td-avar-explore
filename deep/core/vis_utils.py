import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def visualize_bonus(runner_state, config, seed_idx=0, n_pos=100, n_vel=100,
                    figsize=(12, 8)):
    """
    Visualize exploration bonus across Mountain Car state space.
    """
    train_state, lstd_state, rnd_state, env_state, last_obs, rng, idx = runner_state

    if seed_idx is not None:
        target_params = jax.tree.map(lambda x: x[seed_idx], rnd_state.target_params)
        Sandwich = lstd_state['Sandwich'][seed_idx]
    else:
        target_params = rnd_state.target_params
        Sandwich = lstd_state['Sandwich']

    # Mountain car state bounds
    pos_min, pos_max = -1.2, 0.6
    vel_min, vel_max = -0.07, 0.07

    # Create grid of states
    positions = np.linspace(pos_min, pos_max, n_pos)
    velocities = np.linspace(vel_min, vel_max, n_vel)
    pos_grid, vel_grid = np.meshgrid(positions, velocities)

    # Create states array (shape: n_vel x n_pos x 2)
    states = np.stack([pos_grid, vel_grid], axis=-1)

    rnd_apply_fn = rnd_state.apply_fn
    bonus_scale = config['BONUS_SCALE']

    def compute_bonus_for_state(state):
        """Compute bonus for a single state"""
        phi = rnd_apply_fn(target_params, state)
        variance = jnp.einsum('i,ij,j->', phi, Sandwich, phi)
        bonus = bonus_scale * jnp.sqrt(jnp.maximum(variance, 1e-6))
        return bonus

    # Compute bonus over the grid
    compute_bonus_batch = jax.vmap(jax.vmap(compute_bonus_for_state))
    bonus_grid = compute_bonus_batch(jnp.array(states))
    bonus_grid = np.array(bonus_grid)

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(pos_grid, vel_grid, bonus_grid, shading='auto', cmap='viridis')
    cbar = plt.colorbar(im, ax=ax, label='Exploration Bonus')

    # Mountain car height function
    def height(xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    ax2 = ax.twinx()
    terrain_heights = height(positions)
    terrain_line, = ax2.plot(positions, terrain_heights, 'w--', linewidth=2.5, label='Terrain')
    ax2.set_ylim(0, 1.5)
    goal_line = ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2.5, label='Goal')

    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Velocity', fontsize=12)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    num_updates = config['NUM_UPDATES']
    ax.set_title(f'Exploration Bonus in Mountain Car State Space\n(After {num_updates} Training Updates)', fontsize=14)

    plt.tight_layout()
    plt.show()

    return fig, ax, bonus_grid
