# DINOv2-backed LSTD variant of 4_16_lstd.py.
# The LSTD basis phi(s) comes from an offline-extracted DINOv2 feature cache
# (PRETRAINED_CACHE_PATH .npz) instead of a random frozen CNN. Keyed on
# (agent_idx, goal_idx) one-hots; FourRooms-only.
#
# Required config keys:
#   PRETRAINED_CACHE_PATH : path to .npz produced by scripts/precompute_pretrained_features.py
#   NORMALIZE_FEATURES    : L2-normalize features after lookup
#   BIAS                  : append a constant bias column
from core.imports import *
import numpy as np
import core.helpers as helpers
import core.networks as networks
from core.buffer import LSTDBufferState, FeatureTraceBufferManager
from core.lstd import solve_lstd_lambda_from_buffer

SAVE_DIR = "4_16_lstd_dino"


def _build_dino_feature_pipeline(config, obs_shape):
    """Load a pretrained-feature .npz and return (get_features_fn, k).

    get_features_fn(obs) -> phi, single-argument signature matching the
    4_16_lstd.py convention (vmapped as `jax.vmap(get_features_fn)`).
    """
    cache_path = config.get("PRETRAINED_CACHE_PATH")
    if cache_path is None:
        raise ValueError("PRETRAINED_CACHE_PATH must be set for the dino LSTD variant")
    env_name = config.get("ENV_NAME", "")
    if env_name not in {"FourRooms-misc", "FourRoomsCustom-v0"}:
        raise ValueError(
            f"Pretrained (dino) features are only wired up for FourRooms envs, got {env_name!r}"
        )
    if len(obs_shape) != 3 or obs_shape[-1] != 3:
        raise ValueError(f"Pretrained lookup expects visual (H,W,3) obs, got {obs_shape}")

    data = np.load(cache_path)
    feats_np = np.asarray(data["features"], dtype=np.float32)
    obs_np = np.asarray(data["obs_stack"], dtype=np.float32)
    N, H, W, _ = obs_np.shape
    if (H, W) != tuple(obs_shape[:2]):
        raise ValueError(f"Cache spatial dims {(H, W)} != env obs_shape {obs_shape[:2]}")
    cells = H * W

    agent_keys = obs_np[..., 1].reshape(N, cells).argmax(axis=-1)
    goal_keys = obs_np[..., 2].reshape(N, cells).argmax(axis=-1)
    keys = agent_keys * cells + goal_keys
    lookup_np = np.zeros(cells * cells, dtype=np.int32)
    lookup_np[keys] = np.arange(N, dtype=np.int32)

    feature_table = jnp.asarray(feats_np)
    lookup = jnp.asarray(lookup_np)

    k_raw = int(feats_np.shape[-1])
    normalize = config.get("NORMALIZE_FEATURES", False)
    use_bias = config.get("BIAS", True)
    k = k_raw + (1 if use_bias else 0)

    def get_features_fn(obs):
        agent = jnp.argmax(obs[..., 1].reshape(*obs.shape[:-3], cells), axis=-1)
        goal = jnp.argmax(obs[..., 2].reshape(*obs.shape[:-3], cells), axis=-1)
        row = lookup[agent * cells + goal]
        phi = feature_table[row]
        if normalize:
            phi = phi / (jnp.linalg.norm(phi, axis=-1, keepdims=True) + 1e-8)
        if use_bias:
            bias = jnp.ones((*phi.shape[:-1], 1))
            phi = jnp.concatenate([phi, bias], axis=-1)
        return phi

    return get_features_fn, k


class Transition(NamedTuple):
    done: jnp.ndarray
    goal: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    i_value: jnp.ndarray
    next_i_val: jnp.ndarray
    reward: jnp.ndarray
    intrinsic_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    # Episodic / Continuing / Absorbing
    is_episodic = config.get("EPISODIC", True)
    is_continuing = ~is_episodic
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    assert is_episodic or (is_continuing and not is_absorbing), 'Cannot be continuing and absorbing'

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size

    # Env (built first so obs_shape is available for the feature pipeline)
    env, env_params = helpers.make_env(config)
    obs_shape = env.observation_space(env_params).shape
    evaluator = helpers.initialize_evaluator(config)

    # Feature pipeline: DINOv2 pretrained lookup. k depends on cache + bias.
    get_features_fn, k_lstd = _build_dino_feature_pipeline(config, obs_shape)
    batch_get_features = jax.vmap(get_features_fn)

    # Replay Buffer (needs k_lstd from the pipeline)
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    config['CHUNK_SIZE'] = 100_000 + batch_size  # chunking for LSTD solver
    buffer_manager = FeatureTraceBufferManager(config, k_lstd, BUFFER_CAPACITY, EXTENDED_CAPACITY, config['CHUNK_SIZE'])
    config['NUM_CHUNKS'] = buffer_manager.padded_capacity // config['CHUNK_SIZE']
    config['PADDED_CAPACITY'] = buffer_manager.padded_capacity

    if config.get('SCHEDULE_BETA', False):
        beta_sch = helpers.make_triangle_schedule(total_updates=config['NUM_UPDATES'], max_beta=config['BONUS_SCALE'], peak_at=0.01)
    else:
        beta_sch = lambda x: config['BONUS_SCALE']

    def _compile_metrics(network, batch_get_features, traj_batch, next_phi, loss_info, gaes, targets, rho_scale, Sigma_inv, lstd_state, train_state):
            metric = {k: v.mean() for k, v in traj_batch.info.items() if k not in ["real_next_obs", "real_next_state"]}
            metric.update({
                "ppo_loss": loss_info[0],
                "rnd_loss": loss_info[1],
                "feat_norm": jnp.linalg.norm(next_phi, axis=-1).mean(),
                "bonus_mean": gaes[1].mean(),
                "bonus_std": gaes[1].std(),
                "bonus_max": gaes[1].max(),
                "lambda_ret_mean": targets[0].mean(),
                "lambda_ret_std": targets[0].std(),
                "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
                "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
                "mean_rew": traj_batch.reward.mean(),
                "rho_scale": rho_scale,
                "num_goals": jnp.sum(traj_batch.goal)
            })

            if evaluator is None:
                metric.update({
                    "vi_pred": traj_batch.i_value.mean(),
                    "v_e_pred": traj_batch.value.mean(),
                })
            else:
                int_rew_from_state = lambda s: helpers.get_scale_free_bonus(Sigma_inv, batch_get_features(s)) * rho_scale
                get_vi = lambda obs: batch_get_features(obs) @ lstd_state["w"] * rho_scale

                metric = helpers.add_values_to_metric(
                    config, metric, int_rew_from_state, evaluator, rho_scale,
                    network, train_state, traj_batch, get_vi,
                )
            return metric

    def train(rng):
        initial_lstd_state = {"w": jnp.zeros(k_lstd), }
        initial_buffer_state = buffer_manager.init_state()
        initial_sigma_state = {"S": jnp.eye(k_lstd, dtype=jnp.float64)}

        # Dummy RND scaffolding: initialize_flax_train_states requires rnd_net
        # and params, even though the frozen target is never queried here
        # (features come from the dino cache).
        rnd_rng, rng = jax.random.split(rng)
        rnd_net, rnd_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["RND_NETWORK_TYPE"], config["NORMALIZE_FEATURES"], config["BIAS"],
            int(config.get("RND_FEATURES", 128)),
        )
        target_params = rnd_params

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, env, env_params, config, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rnd_net, network_params, rnd_params, target_params
        )
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        (env_state, obsv, rng) = helpers.warmup_env(rng, env, env_params, config)

        def _update_step(runner_state, unused):
            train_state, lstd_state, sigma_state, buffer_state, rnd_state, env_state, last_obs, rng, idx = runner_state

            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                is_goal = info['is_goal']
                target_next_obs = info["real_next_obs"].reshape(last_obs.shape)
                next_val = network.apply(train_state.params, target_next_obs, method=network.value)

                intrinsic_reward = jnp.zeros_like(reward)
                i_val = jnp.zeros_like(reward)
                next_i_val = jnp.zeros_like(reward)

                transition = Transition(
                    done, is_goal, action, value, next_val, i_val, next_i_val, reward, intrinsic_reward, log_prob, last_obs, target_next_obs, info
                )
                return (train_state, rnd_state, env_state, obsv, rng), transition

            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            phi = batch_get_features(traj_batch.obs)
            next_phi = batch_get_features(traj_batch.next_obs)
            terminals = jnp.where(not is_continuing, traj_batch.done, 0)
            absorb_masks = jnp.where(is_absorbing, traj_batch.goal, 0)
            traces = helpers.calculate_traces(traj_batch, phi, config["GAMMA_i"], config["GAE_LAMBDA_i"], is_continuing)

            sigma_state = helpers.update_cov(traj_batch, sigma_state, phi, next_phi)

            buffer_batch = LSTDBufferState(traces, phi, next_phi, terminals, absorb_masks, size=jnp.array(batch_size))
            buffer_state = buffer_manager.update_buffer(buffer_state, buffer_batch)

            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"])
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_lstd))

            lstd_state = solve_lstd_lambda_from_buffer(buffer_state, Sigma_inv, lstd_state, config)

            rng, prb_rng = jax.random.split(rng)
            buffer_state = buffer_manager.evict_buffer(buffer_state, prb_rng)

            rho = helpers.get_scale_free_bonus(Sigma_inv, next_phi)

            v_i = phi @ lstd_state["w"]
            next_v_i = next_phi @ lstd_state["w"]

            V_max_raw = 1.0 / (1.0 - config['GAMMA_i'])
            v_i, next_v_i = jax.tree.map(lambda x: jnp.clip(x, 0, V_max_raw), (v_i, next_v_i))

            traj_batch = traj_batch._replace(
                i_value=v_i,
                intrinsic_reward=rho,
                next_i_val=next_v_i
            )

            gaes, targets = helpers.calculate_gae(
                traj_batch,
                config["GAMMA"],
                config["GAE_LAMBDA"],
                is_continuing,
                γi=config["GAMMA_i"],
                λi=config["GAE_LAMBDA_i"]
            )
            gae_e, gae_i = gaes

            rho_scale = beta_sch(idx)
            advantages = gae_e + (rho_scale * gae_i)
            extrinsic_target = targets[0]

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers._loss_fn, has_aux=True)
                    (total_loss, _), grads = grad_fn(
                        train_state.params, network, traj_batch, advantages, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), total_loss

            initial_update_state = (train_state, traj_batch, advantages, extrinsic_target, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state

            metric = _compile_metrics(
                network, batch_get_features, traj_batch, next_phi, loss_info, gaes, targets,
                rho_scale, Sigma_inv, lstd_state, train_state
            )

            runner_state = (train_state, lstd_state, sigma_state, buffer_state, rnd_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, initial_sigma_state, initial_buffer_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    from core.utils import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)
