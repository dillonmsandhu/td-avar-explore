"""COV-LSTD with online DINOv2 features for Atari.

Mirrors `atari/algos/cov_lstd.py` but swaps the random-CNN LSTD feature branch
for a frozen DINOv2 ViT-S/14 called once per rollout (post-scan, host-side) via
DLPack zero-copy JAX<->torch transfer. The rho/RND novelty branch is unchanged.

The outer training loop is a Python `for` (not `jax.lax.scan` over updates) so
the DINO torch call can sit between two jit'd halves of each update step:
`_collect_rollout` (env scan -> trajectory) and `_update_post_phi`
(cov / traces / buffer / LSTD solve / GAE / PPO).

Because `train(rng)` is no longer trace-compatible end-to-end (the torch call
breaks tracing), this file ships a custom main that bypasses the outer
`jax.jit(make_train(...))` wrap used by `core.utils.evaluate`.
"""
from core.imports import *
import core.helpers as helpers
import core.networks as networks
from core.buffer import FeatureTraceBufferManager, LSTDBufferState
from core.lstd import solve_lstd_lambda_from_buffer
from core.helpers import Transition
from core.dino_features import DinoFeatureExtractor

SAVE_DIR = "cov_lstd_dino"


def make_train(config):
    config.setdefault("DINO_MODEL", "dinov2_vits14")
    config.setdefault("DINO_DTYPE", "fp16")
    config.setdefault("DINO_BATCH_SIZE", 4096)
    config.setdefault("DINO_FRAME_STRATEGY", "last")

    k_lstd = int(config.get("LSTD_FEATURES", 384))
    k_rho = config.get("RND_FEATURES", 128)
    is_episodic = config.get("EPISODIC", True)
    is_continuing = (not is_episodic)
    is_absorbing = config.get("ABSORBING_TERMINAL_STATE", True)
    assert is_episodic or (is_continuing and not is_absorbing), 'Cannot be continuing and absorbing'

    def define_trace_logic(terminals, is_dummy, is_goal, was_goal):
        if is_episodic:
            cut_trace = terminals | is_dummy
            absorb_mask = jnp.zeros_like(terminals, dtype=jnp.bool_)
        elif is_continuing:
            cut_trace = jnp.zeros_like(terminals, dtype=jnp.bool_)
            absorb_mask = jnp.zeros_like(terminals, dtype=jnp.bool_)
        elif is_absorbing:
            death = terminals & ~is_goal
            cut_trace = death | is_dummy
            absorb_mask = was_goal
        continue_mask = jnp.logical_not(cut_trace)
        return cut_trace, continue_mask, absorb_mask

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    BUFFER_CAPACITY = config.get('RB_SIZE', 100_000)
    EXTENDED_CAPACITY = BUFFER_CAPACITY + batch_size
    config['CHUNK_SIZE'] = 100_000 + batch_size
    buffer_manager = FeatureTraceBufferManager(
        config, k_lstd, k_rho, BUFFER_CAPACITY, EXTENDED_CAPACITY, config['CHUNK_SIZE']
    )
    config['NUM_CHUNKS'] = buffer_manager.padded_capacity // config['CHUNK_SIZE']
    config['PADDED_CAPACITY'] = buffer_manager.padded_capacity

    env = helpers.make_env(config)
    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n

    if config.get('SCHEDULE_BETA', False):
        beta_sch = helpers.make_triangle_schedule(
            total_updates=config['NUM_UPDATES'], max_beta=config['BONUS_SCALE'], peak_at=0.01
        )
    else:
        beta_sch = lambda x: config['BONUS_SCALE']

    def _compile_metrics(traj_batch, loss_info, gaes, targets, rho_scale):
        metric = {k: v.mean() for k, v in traj_batch.info.items() if k not in ["real_next_obs", "real_next_state"]}
        metric.update({
            "ppo_loss": loss_info[0],
            "rnd_loss": loss_info[1],
            "feat_norm": jnp.linalg.norm(traj_batch.next_phi, axis=-1).mean(),
            "bonus_mean": gaes[1].mean(),
            "bonus_std": gaes[1].std(),
            "bonus_max": gaes[1].max(),
            "lambda_ret_mean": targets[0].mean(),
            "lambda_ret_std": targets[0].std(),
            "intrinsic_rew_mean": traj_batch.intrinsic_reward.mean(),
            "intrinsic_rew_std": traj_batch.intrinsic_reward.std(),
            "mean_rew": traj_batch.reward.mean(),
            "rho_scale": rho_scale,
            "num_goals": jnp.sum(traj_batch.info.get('is_goal', jnp.zeros_like(traj_batch.done))),
            "vi_pred": traj_batch.i_value.mean(),
            "v_e_pred": traj_batch.value.mean(),
        })
        return metric

    def train(rng):
        dino = DinoFeatureExtractor(config)
        assert dino.k == k_lstd, (
            f"DINO model embed_dim ({dino.k}) != LSTD_FEATURES ({k_lstd}). "
            "Set LSTD_FEATURES in config to match the chosen DINO model."
        )

        initial_lstd_state = {"w": jnp.zeros(k_lstd)}
        initial_buffer_state = buffer_manager.init_state()
        initial_sigma_state = {"S": jnp.eye(k_rho, dtype=jnp.float64)}

        rnd_rng, rng = jax.random.split(rng)
        rho_net, rho_params = networks.initialize_rnd_network(
            rnd_rng, obs_shape, config["NORMALIZE_FEATURES"], bias=True, k=k_rho
        )

        def get_rho_feats(obs):
            return rho_net.apply(rho_params, obs)

        network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, n_heads=2)
        train_state, rnd_state = networks.initialize_flax_train_states(
            config, network, rho_net, network_params, rho_params
        )

        obsv, env_state = env.reset()
        obsv = jnp.asarray(obsv)
        initial_phi = dino(obsv)
        initial_rho_feat = get_rho_feats(obsv)

        T = config["NUM_STEPS"]
        B = config["NUM_ENVS"]
        phi_placeholder = jnp.zeros((B, k_lstd), dtype=jnp.float32)

        @jax.jit
        def _collect_rollout(carry):
            train_state = carry["train_state"]
            env_state = carry["env_state"]
            last_obs = carry["last_obs"]
            last_rho_feat = carry["last_rho_feat"]
            rng = carry["rng"]

            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_rho_feat, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                b, value = network.apply(train_state.params, last_obs)
                action = b.sample(seed=_rng)
                log_prob = b.log_prob(action)

                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(env_state, action)
                next_val = network.apply(train_state.params, obsv, method=network.value)

                next_rho_feat = get_rho_feats(obsv)
                dummy = jnp.zeros_like(reward)

                transition = Transition(
                    done, action, value, next_val, dummy, dummy, reward, dummy, log_prob,
                    last_obs, obsv, info,
                    phi=phi_placeholder, next_phi=phi_placeholder,
                    rho_feats=last_rho_feat, next_rho_feats=next_rho_feat,
                )

                runner_state = (train_state, env_state, obsv, next_rho_feat, rng)
                return runner_state, transition

            init = (train_state, env_state, last_obs, last_rho_feat, rng)
            (train_state, env_state, last_obs, last_rho_feat, rng), traj_batch = jax.lax.scan(
                _env_step, init, None, T
            )
            new_carry = {
                **carry,
                "train_state": train_state,
                "env_state": env_state,
                "last_obs": last_obs,
                "last_rho_feat": last_rho_feat,
                "rng": rng,
            }
            return new_carry, traj_batch

        @jax.jit
        def _update_post_phi(carry, traj_batch):
            train_state = carry["train_state"]
            lstd_state = carry["lstd_state"]
            sigma_state = carry["sigma_state"]
            buffer_state = carry["buffer_state"]
            rnd_state = carry["rnd_state"]
            rng = carry["rng"]
            idx = carry["idx"]

            sigma_state = helpers.update_cov(sigma_state, traj_batch.rho_feats)
            cho_S = jax.scipy.linalg.cho_factor(sigma_state["S"])
            Sigma_inv = jax.scipy.linalg.cho_solve(cho_S, jnp.eye(k_rho))

            terminals = traj_batch.done
            is_dummy = traj_batch.info.get("is_dummy", jnp.zeros_like(terminals))
            is_goal = traj_batch.info.get("is_goal", jnp.zeros_like(terminals))
            was_goal = traj_batch.info.get("was_goal", jnp.zeros_like(terminals))
            cut_trace, continue_mask, absorb_mask = define_trace_logic(terminals, is_dummy, is_goal, was_goal)

            traces = helpers.calculate_traces(traj_batch.phi, cut_trace, config["GAMMA_i"], config["LSTD_LAMBDA_i"])
            buffer_batch = LSTDBufferState(
                traces=traces,
                features=traj_batch.phi,
                next_features=traj_batch.next_phi,
                rho_features=traj_batch.rho_feats,
                next_rho_features=traj_batch.next_rho_feats,
                continue_masks=continue_mask,
                absorb_masks=absorb_mask,
                size=jnp.array(batch_size),
            )
            buffer_state = buffer_manager.update_buffer(buffer_state, buffer_batch)
            lstd_state = solve_lstd_lambda_from_buffer(buffer_state, Sigma_inv, config)

            rng, prb_rng = jax.random.split(rng)
            buffer_state = buffer_manager.evict_buffer(buffer_state, prb_rng)

            rho_feats_final = jnp.where(absorb_mask[..., None], traj_batch.rho_feats, traj_batch.next_rho_feats)
            rho = helpers.get_scale_free_bonus(Sigma_inv, rho_feats_final)

            v_i = traj_batch.phi @ lstd_state["w"]
            next_v_i = traj_batch.next_phi @ lstd_state["w"]

            V_max_raw = 1.0 / (1.0 - config['GAMMA_i'])
            v_i = jnp.clip(v_i, 0, V_max_raw)
            next_v_i = jnp.clip(next_v_i, 0, V_max_raw)

            traj_batch = traj_batch._replace(i_value=v_i, intrinsic_reward=rho, next_i_val=next_v_i)

            gaes, targets = helpers.calculate_gae(
                traj_batch, config["GAMMA"], config["GAE_LAMBDA"],
                cut_trace, absorb_mask, γi=config["GAMMA_i"], λi=config["GAE_LAMBDA_i"],
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

            metric = _compile_metrics(traj_batch, loss_info, gaes, targets, rho_scale)

            new_carry = {
                **carry,
                "train_state": train_state,
                "rng": rng,
                "lstd_state": lstd_state,
                "rnd_state": rnd_state,
                "sigma_state": sigma_state,
                "buffer_state": buffer_state,
                "idx": idx + 1,
            }
            return new_carry, metric

        rng, _rng = jax.random.split(rng)
        runner_state = {
            "train_state": train_state,
            "env_state": env_state,
            "last_obs": obsv,
            "rng": _rng,
            "lstd_state": initial_lstd_state,
            "rnd_state": rnd_state,
            "sigma_state": initial_sigma_state,
            "buffer_state": initial_buffer_state,
            "idx": 1,
            "last_phi": initial_phi,
            "last_rho_feat": initial_rho_feat,
        }

        try:
            from tqdm import trange
            iterator = trange(config["NUM_UPDATES"], desc=SAVE_DIR, dynamic_ncols=True)
        except ImportError:
            iterator = range(config["NUM_UPDATES"])

        metrics_list = []
        for _ in iterator:
            runner_state, traj_batch = _collect_rollout(runner_state)

            obs_flat = traj_batch.next_obs.reshape(T * B, *obs_shape)
            next_phi_flat = dino(obs_flat)
            next_phi = next_phi_flat.reshape(T, B, k_lstd)

            last_phi = runner_state["last_phi"]
            phi = jnp.concatenate([last_phi[None], next_phi[:-1]], axis=0)

            traj_batch = traj_batch._replace(phi=phi, next_phi=next_phi)
            runner_state = {**runner_state, "last_phi": next_phi[-1]}

            runner_state, metric = _update_post_phi(runner_state, traj_batch)
            metrics_list.append(metric)

        metrics = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *metrics_list)
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def _custom_main():
    """Custom entry point that bypasses the default `jax.jit(make_train(...))`
    wrap in `core.utils.evaluate`. The DINO torch call inside `train` cannot
    be traced as a whole; only `_collect_rollout` and `_update_post_phi` are
    individually jit'd inside `train`.
    """
    import argparse, copy, datetime, json, os, traceback
    import numpy as np
    from core.configs import shared_config

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="{}")
    parser.add_argument('--run-suffix', type=str, default=timestamp)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--envs', nargs='+', default=[])
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project', type=str, default="lstd-explore")
    args = parser.parse_args()

    config = copy.deepcopy(shared_config)
    config_name = "shared_config"
    if args.config and args.config.startswith('{'):
        try:
            overrides = json.loads(args.config)
            config.update(overrides)
            if overrides:
                config_name = "shared_config_custom"
        except json.JSONDecodeError as e:
            print(f"Warning: failed to parse --config JSON: {e}")

    env_list = args.envs if args.envs else [config.get('ENV_NAME')]
    for i, env_name in enumerate(env_list):
        run_config = config.copy()
        run_config['ENV_NAME'] = env_name
        run_config['SEED'] = args.seed
        run_config['THREADS'] = args.threads
        run_config['CONFIG_NAME'] = config_name

        print(f"\n{'='*50}\nRUNNING ENV {i+1}/{len(env_list)}: {env_name} | SEED: {args.seed}\n{'='*50}")
        rng = jax.random.PRNGKey(run_config['SEED'])

        if args.wandb:
            import wandb
            wandb.init(
                project=args.project,
                config=run_config,
                name=f"{env_name}_s{args.seed}",
                group=run_config['CONFIG_NAME'],
            )

        try:
            train_fn = make_train(run_config)
            out = train_fn(rng)
            metrics = out["metrics"]
            print(f"[{env_name}] Mean return: {jnp.mean(metrics['returned_episode_returns']):.1f}")
            print(f"[{env_name}] Max return:  {jnp.max(metrics['returned_episode_returns']):.1f}")

            base_env_name = run_config['ENV_NAME']
            env_size = run_config.get("ENV_SIZE")
            full_env_name = f"{base_env_name}-{env_size}" if env_size else base_env_name
            if args.output_dir:
                env_dir = os.path.join(args.output_dir, full_env_name)
            else:
                env_dir = os.path.join("results", f"{SAVE_DIR}/{args.run_suffix}", full_env_name)
            os.makedirs(env_dir, exist_ok=True)

            with open(os.path.join(env_dir, "config.json"), 'w') as f:
                json.dump(run_config, f, indent=4)
            np_metrics = jax.device_get(metrics)
            np.savez_compressed(os.path.join(env_dir, f"seed_{args.seed}_metrics.npz"), **np_metrics)
            print(f"Saved {env_name} (seed {args.seed}) to {env_dir}")

            if args.wandb:
                steps_per_pi = run_config["NUM_ENVS"] * run_config["NUM_STEPS"]
                num_updates = np_metrics['returned_episode_returns'].shape[0]
                for step in range(num_updates):
                    log_dict = {"step": (step + 1) * steps_per_pi}
                    for k, v in np_metrics.items():
                        if hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] > step:
                            arr = v[step]
                            if hasattr(arr, "ndim") and arr.ndim == 0:
                                log_dict[k] = float(arr)
                    wandb.log(log_dict)

        except Exception as e:
            print(f"!!! CRITICAL ERROR running {env_name} !!!\nError: {e}")
            traceback.print_exc()
            print("Continuing to next environment...")

        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    _custom_main()
