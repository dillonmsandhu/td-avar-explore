# LSTD-based Critic PPO (Extrinsic Reward Only) with Exact Value Logging
from imports import *
import helpers
import networks
import numpy as np
SAVE_DIR = 'lstd_exact_val'

# --- Exact Value Evaluator ---
class DeepSeaExactValue:
    def __init__(self, size: int, unscaled_move_cost: float = 0.01, gamma: float = 0.99, episodic: bool = False):
        self.N = size
        self.cost = unscaled_move_cost
        self.gamma = gamma
        self.episodic = episodic
        self.num_grid_states = size * size
        self.num_total_states = self.num_grid_states + 1  # +1 for Absorbing Terminal
        self.terminal_idx = self.num_grid_states
        self.reachable_mask = jnp.tril(jnp.ones((size,size)))
        self.num_actions = 2
        
        # 1. Pre-compute Observations Stack (N^2 x N x N)
        self.obs_stack = self._create_obs_stack()
        
        # 2. Pre-compute Transition Matrices
        self.P, self.R_extrinsic = self._build_env_dynamics()
        self.P_cont = self._build_env_dynamics_continuing()

    def _create_obs_stack(self):
        obs_stack = np.zeros((self.num_grid_states, self.N, self.N), dtype=np.float32)
        idx = 0
        for r in range(self.N):
            for c in range(self.N):
                obs_stack[idx, r, c] = 1.0
                idx += 1
        return jnp.array(obs_stack)[...,None]

    def _build_env_dynamics(self):
        num_states = self.num_total_states
        num_actions = self.num_actions
        
        P = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions))
        
        for r in range(self.N):
            for c in range(self.N):
                curr_idx = r * self.N + c
                
                # --- Action 0: Left ---
                next_r = r + 1
                next_c = max(0, c - 1)
                next_idx = self.terminal_idx if next_r >= self.N else next_r * self.N + next_c
                P[curr_idx, 0, next_idx] = 1.0
                R[curr_idx, 0] = 0.0 

                # --- Action 1: Right ---
                next_r_right = r + 1
                next_c_right = min(self.N - 1, c + 1)
                next_idx_right = self.terminal_idx if next_r_right >= self.N else next_r_right * self.N + next_c_right
                P[curr_idx, 1, next_idx_right] = 1.0
                R[curr_idx, 1] = -(self.cost / self.N)

        # Episodic Goal Reward
        bottom_right_idx = (self.N - 1) * self.N + (self.N - 1)
        R[bottom_right_idx, 1] += 1.0 

        # Terminal Loop
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        R[self.terminal_idx, :] = 0.0
        return jnp.array(P), jnp.array(R)

    def _build_env_dynamics_continuing(self):
        num_states = self.num_total_states
        num_actions = self.num_actions
        P = np.zeros((num_states, num_actions, num_states))
        start_idx = 0
        for r in range(self.N):
            for c in range(self.N):
                curr_idx = r * self.N + c
                # Action 0 Left
                next_r = r + 1
                next_c = max(0, c - 1)
                next_idx = start_idx if next_r >= self.N else next_r * self.N + next_c
                P[curr_idx, 0, next_idx] = 1.0
                # Action 1 Right
                next_r_right = r + 1
                next_c_right = min(self.N - 1, c + 1)
                next_idx_right = start_idx if next_r_right >= self.N else next_r_right * self.N + next_c_right
                P[curr_idx, 1, next_idx_right] = 1.0
        P[self.terminal_idx, :, self.terminal_idx] = 1.0
        return jnp.array(P)

    def solve_linear_system(self, pi: jax.Array, P_env: jax.Array, R_env: jax.Array):
        P_pi = jnp.einsum('sa, sam -> sm', pi, P_env)
        R_pi = jnp.einsum('sa, sa -> s', pi, R_env)
        I = jnp.eye(self.num_total_states)
        A_mat = I - self.gamma * P_pi
        return jnp.linalg.solve(A_mat, R_pi)

    def get_value_grid(self, V_flat: jax.Array) -> jax.Array:
        return V_flat[:self.num_grid_states].reshape((self.N, self.N))


# --- Main Training Code ---
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray 
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    
    calc_true_values = config.get('CALC_TRUE_VALUES', False)

    env, env_params = helpers.make_env(config)
    n_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape

    # Initialize Evaluator if requested
    if calc_true_values:
        evaluator = DeepSeaExactValue(
            size=config['DEEPSEA_SIZE'], 
            unscaled_move_cost=0.01, 
            gamma=config['GAMMA'], 
            episodic=config['EPISODIC']
        )

    k = config.get('RND_FEATURES', 128)

    def calculate_gae(traj_batch, last_val, gamma, gae_lambda):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + gamma * next_value * (1 - done) - value
            gae = delta + gamma * gae_lambda * (1 - done) * gae
            return (gae, value), (gae, value + gae)

        _, (advantages, targets) = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, targets

    def cross_cov_full(current_features, next_features, done, γ):
        td_features = current_features - γ  * next_features * (1-done)
        A_sample = jnp.outer(current_features, td_features)
        return A_sample
    
    cross_cov = lambda phi, phi_prime, done: cross_cov_full(phi, phi_prime, done, config['GAMMA'])

    def lstd_batch_update(lstd_state: Dict, transitions, features, next_features):        
        batch_axes = tuple(range(transitions.done.ndim))
        N = transitions.done.size + lstd_state['N']
        t = lstd_state['t']
        α = jnp.maximum(config.get('MIN_COV_LR', 1/10), 1/t)
        α = 1.0
        
        A_update = jax.vmap(jax.vmap(cross_cov))(features, next_features, transitions.done)
        A_b = A_update.mean(axis=batch_axes)
        
        # Standard unscaled reward
        b_sample = features * transitions.reward[..., None]
        b_b = b_sample.mean(axis=batch_axes)
        
        def EMA(α, x_start, x_sample):
            return (1-α) * x_start + α * x_sample
        
        A = EMA(α, lstd_state['A'], A_b)
        b = EMA(α, lstd_state['b'], b_b)
        
        εI = config['A_REGULARIZATION_PER_STEP'] * jnp.eye(A.shape[0])
        w = jnp.linalg.solve(A + εI, b)
        
        return {'A': A, 'b': b, 'w': w, 'N': N, 't': t+1}

    def train(rng):
        rnd_rng, rng = jax.random.split(rng)
        target_rng, rng = jax.random.split(rng)
        rnd_net, _ = networks.initialize_rnd_network(rnd_rng, obs_shape, config, k)
        _, target_params = networks.initialize_rnd_network(target_rng, obs_shape, config, k)
        
        network, network_params = networks.initialize_actor_critic(rng, obs_shape, n_actions, config, n_heads = 2)
        train_state, rnd_state = networks.initialize_flax_train_states(config, network, rnd_net, network_params, {}, target_params)
        
        get_features_fn = lambda obs: rnd_net.apply(target_params, obs)
        batch_get_features = jax.vmap(get_features_fn)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        initial_lstd_state = {
            'A': jnp.eye(k) * config['A_REGULARIZATION'], 
            'b': jnp.zeros(k), 
            'w': jnp.zeros(k),
            'N': 0, 't': 1,
        }

        # WARMUP
        def _warmup_step(runner_state, unused):
            env_state, last_obs, rng = runner_state
            rng, _rng = jax.random.split(rng)
            rng_action = jax.random.split(_rng, config["NUM_ENVS"])
            action = jax.vmap(env.action_space(env_params).sample)(rng_action)
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                rng_step, env_state, action, env_params
            )
            return (env_state, obsv, rng), None

        warmup_runner_state = (env_state, obsv, rng)
        (env_state, obsv, rng), _ = jax.lax.scan(
            _warmup_step, warmup_runner_state, None, config["WARMUP"]
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            
            train_state, lstd_state, rnd_state, env_state, last_obs, rng, idx = runner_state
            
            # 1. COLLECT
            def _env_step(env_scan_state, unused):
                train_state, rnd_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, _ = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                
                transition = Transition(
                    done, action, jnp.zeros_like(reward), reward, log_prob, last_obs, obsv, info, 
                )
                runner_state = (train_state, rnd_state, env_state, obsv, rng)
                return runner_state, transition
            
            env_step_state = (train_state, rnd_state, env_state, last_obs, rng)
            (_, _, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state , None, config["NUM_STEPS"]
            )
            
            # 2. CRITIC UPDATE
            next_phi = batch_get_features(traj_batch.next_obs)
            phi = batch_get_features(traj_batch.obs)
            lstd_state = lstd_batch_update(lstd_state, traj_batch, phi, next_phi)
            lstd_values = phi @ lstd_state["w"]
            traj_batch = traj_batch._replace(value=lstd_values)

            # 3. GAE
            last_phi = get_features_fn(last_obs)
            last_val_lstd = last_phi @ lstd_state["w"]
            advantages, targets = calculate_gae(traj_batch, last_val_lstd, config["GAMMA"], config["GAE_LAMBDA"])
            
            # 4. POLICY UPDATE
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
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            initial_update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, _, _, _, rng = update_state
            
            # 5. METRICS
            metric = {k: v.mean() for k, v in traj_batch.info.items()}
            metric.update({
                "ppo_loss": loss_info[0], 
                "lstd_val_mean": lstd_values.mean(),
                "mean_rew": traj_batch.reward.mean(),
                "lambda_ret_mean": targets.mean(),
            })

            if calc_true_values:
                # --- EXACT VALUE LOGGING (Extrinsic Only) ---
                
                # 1. Get Current Policy
                out = network.apply(train_state.params, evaluator.obs_stack)
                # Handle possible tuple return types from network wrapper
                if isinstance(out, tuple):
                    pi_probs = out[0]
                else:
                    pi_probs = out
                
                # Append dummy terminal policy [1, 0]
                terminal_policy = jnp.array([[1.0, 0.0]])
                pi_full = jnp.vstack([pi_probs.probs, terminal_policy])
                
                # 2. Compute TRUE Extrinsic Value Grid
                v_e_flat = evaluator.solve_linear_system(pi_full, evaluator.P, evaluator.R_extrinsic)
                v_e_grid = evaluator.get_value_grid(v_e_flat)
                
                # 3. Compute LSTD Predicted Value Grid
                all_phi = get_features_fn(evaluator.obs_stack)
                v_pred_flat = all_phi @ lstd_state['w']
                v_pred_grid = evaluator.get_value_grid(v_pred_flat)
                
                # 4. Extrinsic Reward Grid (Visualize "Best" reward per state, i.e., Right action)
                r_ext_flat = evaluator.R_extrinsic[:, 1]
                r_grid = evaluator.get_value_grid(r_ext_flat)

                metric.update({
                    "v_true_grid": v_e_grid,
                    "v_pred_grid": v_pred_grid,
                    "r_grid": r_grid,
                    "value_error": jnp.mean(evaluator.reachable_mask * (v_e_grid - v_pred_grid)**2),
                })

            runner_state = (train_state, lstd_state, rnd_state, env_state, last_obs, rng, idx+1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, initial_lstd_state, rnd_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
    
def main():
    import warnings; warnings.simplefilter('ignore')
    from utils import parse_config_override, evaluate
    import datetime
    import argparse
    import configs
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Run LSTD-Critic PPO with Exact Values')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON string to override config values')
    parser.add_argument('--run_suffix', type=str, default=run_timestamp,
                       help=f'saves to {SAVE_DIR}/args.run_suffix/' )
    parser.add_argument('--n-seeds', type=int, default=0)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--base-config', type = str, default = 'mc', choices = ['mc', 'ds', 'min'])
    parser.add_argument('--env_ids', nargs='+', default=[], 
                       help='Optional list of envs to run sequentially.')

    args = parser.parse_args()
    
    if args.base_config == 'mc':
        config = configs.mc_config.copy()
    elif args.base_config == 'ds':
        config = configs.ds_config.copy()
    elif args.base_config  == 'min':
        config = configs.min_config.copy()

    config_override = parse_config_override(args.config)
    config.update(config_override)

    env_list = args.env_ids if args.env_ids else [config['ENV_NAME']]

    for i, env_name in enumerate(env_list):
        print(f"\n{'='*50}")
        print(f"RUNNING ENV {i+1}/{len(env_list)}: {env_name}")
        print(f"{'='*50}")
        
        run_config = config.copy()
        run_config['ENV_NAME'] = env_name
        
        rng = jax.random.PRNGKey(run_config['SEED'])
        
        try:
            evaluate(run_config, make_train, SAVE_DIR, args, rng)
        except Exception as e:
            print(f"!!! CRITICAL ERROR running {env_name} !!!")
            print(e)
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()