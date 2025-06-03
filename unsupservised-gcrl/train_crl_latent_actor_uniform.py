import os
import jax
import flax
import tyro
import time
import optax
import wandb
import pickle
import random
import wandb_osh
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from etils import epath
from dataclasses import dataclass
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling

from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue
from envs.wrappers import EpisodeWrapper, VmapWrapper, AutoResetWrapper

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "exploration"
    wandb_entity: str = 'user'
    wandb_mode: str = 'online'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    capture_video: bool = False
    checkpoint: bool = False

    #environment specific arguments
    env_id: str = "ant"
    episode_length: int = 1000
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0
    
    # Algorithm specific arguments
    total_env_steps: int = 50000000
    num_epochs: int = 50
    num_envs: int = 1024
    num_eval_envs: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    rep_size: int = 64
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    small_networks: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    
    unroll_length: int  = 62

    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState

class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    action: jnp.ndarray
    extras: jnp.ndarray = ()  

def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)

def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))
                   
if __name__ == "__main__":

    args = tyro.cli(Args)

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    args.num_prefill_actor_steps = np.ceil(args.min_replay_size / args.unroll_length)
    args.num_training_steps_per_epoch = (args.total_env_steps - args.num_prefill_env_steps) // (args.num_epochs * args.env_steps_per_actor_step)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:

        if args.wandb_group ==  '.':
            args.wandb_group = 'normal'
            
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()
        
    if args.checkpoint:
        from pathlib import Path
        save_path = Path(args.wandb_dir) / Path(run_name)
        os.mkdir(path=save_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)
    
    # Environment setup  
    if args.env_id == "ant":
        from envs.ant import Ant
        env = Ant(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=False,
        )

        args.obs_dim = 29
        args.goal_start_idx = 0
        args.goal_end_idx = 2

    elif "maze" in args.env_id and "ant" in args.env_id:
        from envs.ant_maze import AntMaze
        env = AntMaze(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=False,
            maze_layout_name=args.env_id[4:]
        )

        args.obs_dim = 29
        args.goal_start_idx = 0
        args.goal_end_idx = 2

    else:
        raise NotImplementedError

    env = EpisodeWrapper(env, args.episode_length, action_repeat=1)
    env = VmapWrapper(env)
    env = AutoResetWrapper(env)

    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, args.num_envs)
    env.step = jax.jit(env.step)
    env.reset = jax.jit(env.reset)
    env_state = env.reset(env_keys)

    if args.small_networks:
        from models import small_Actor as Actor, small_SA_encoder as SA_encoder, small_G_encoder as G_encoder
    else:
        from models import Actor, SA_encoder, G_encoder
    
    # Network setup
    # Actor
    actor = Actor(action_size=action_size)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, args.obs_dim]), np.ones([1, args.rep_size])),
        tx=optax.adam(learning_rate=args.actor_lr)
    )

    # Critic
    sa_encoder = SA_encoder(rep_size=args.rep_size)
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))
    g_encoder = G_encoder(rep_size=args.rep_size)
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    # Entropy coefficient
    target_entropy = -0.5 * action_size
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )

    actor.apply = jax.jit(actor.apply)
    sa_encoder.apply = jax.jit(sa_encoder.apply)
    g_encoder.apply = jax.jit(g_encoder.apply)
    
    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )

    #Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))

    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        extras={
            "state_extras": {
                "seed": 0.0,
            }        
        },
    )

    def jit_wrap(buffer):
        buffer.insert = jax.jit(buffer.insert)
        buffer.sample = jax.jit(buffer.sample)
        return buffer
    
    replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=args.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=args.batch_size,
                num_envs=args.num_envs,
                sequence_length=args.num_envs+1, #this is greater than the episode length, but that's fine.
            )
        )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        """Function to collect data during evaluation. Used in evaluator.py"""

        obs = env_state.obs
        state = obs[:, :args.obs_dim]
        goal = obs[:, args.obs_dim:]
        g_repr = g_encoder.apply(training_state.critic_state.params["g_encoder"], goal)
        
        means, _ = actor_state.apply_fn(training_state.actor_state.params, state, g_repr)
        actions = nn.tanh( means )

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            extras={
                'state_extras': state_extras
            })

    def actor_step(training_state, env, env_state, explore_goal_rep, key, extra_fields):
        state = env_state.obs[:, :args.obs_dim]
        means, log_stds = actor.apply(training_state.actor_state.params, state, explore_goal_rep)
        stds = jnp.exp(log_stds)
        actions = nn.tanh( means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype) )

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        
        return training_state, nstate, Transition(
                                            observation=env_state.obs,
                                            action=actions,
                                            extras={"state_extras": state_extras},
                                        )

    @jax.jit
    def get_experience(training_state, transitions, env_state, buffer_state, key):    
        @jax.jit
        def f(carry, unused_t):
            training_state, env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            training_state, env_state, transition = actor_step(training_state, env, env_state, explore_goal_reps, current_key, extra_fields=("seed",))
            return (training_state, env_state, next_key), transition
        
        random_g_key, key = jax.random.split(key, 2)
                
        permutation = jax.random.permutation(random_g_key, len(transitions.extras['goal_obs']))
        random_goals = transitions.extras['goal_obs'][:, args.goal_start_idx : args.goal_end_idx][permutation][:args.num_envs]
        explore_goal_reps = g_encoder.apply(g_encoder_params, random_goals)

        (training_state, env_state, _), data = jax.lax.scan(f, (training_state, env_state, key), (), length=args.unroll_length)

        buffer_state = replay_buffer.insert(buffer_state, data)
        return training_state, env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, key = carry
            key, current_key = jax.random.split(key)

            obs = env_state.obs
            state = obs[:, :args.obs_dim]
            goal = obs[:, args.obs_dim:]
            g_repr = g_encoder.apply(training_state.critic_state.params["g_encoder"], goal)
            means, log_stds = actor.apply(training_state.actor_state.params, state, g_repr)
            stds = jnp.exp(log_stds)
            actions = nn.tanh( means + stds * jax.random.normal(current_key, shape=means.shape, dtype=means.dtype) )

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in ("seed",)}

            transition = Transition(
                observation=env_state.obs,
                action=actions,
                extras={"state_extras": state_extras},
            )
 
            return (training_state, nstate, key), (transition)

        (training_state, env_state, _), data =  jax.lax.scan(f, (training_state, env_state, key), (), length=args.unroll_length * args.num_prefill_actor_steps)
        
        buffer_state = replay_buffer.insert(buffer_state, data)
        training_state = training_state.replace(env_steps=training_state.env_steps + args.env_steps_per_actor_step * args.num_prefill_actor_steps)
        
        return training_state, env_state, buffer_state
    
    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            state = transitions.extras["state"]
            goal = transitions.extras['goal_obs'][:, args.goal_start_idx : args.goal_end_idx]
            sa_encoder_params, g_encoder_params = jax.lax.stop_gradient(critic_params["sa_encoder"]), jax.lax.stop_gradient(critic_params["g_encoder"])
            
            g_repr = g_encoder.apply(g_encoder_params, goal)

            means, log_stds = actor.apply(actor_params, state, g_repr)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)           # shape = B

            sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))      # shape = B

            actor_loss = jnp.mean( jnp.exp(log_alpha) * log_prob - (qf_pi) )

            return actor_loss, log_prob

        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))
            return jnp.mean(alpha_loss)
        
        (actorloss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(training_state.actor_state.params, training_state.critic_state.params, training_state.alpha_state.params['log_alpha'], transitions, key)
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
            "alpha_aloss": alphaloss,   
            "log_alpha": training_state.alpha_state.params["log_alpha"],
        }

        return training_state, metrics
    
    @jax.jit
    def update_critic(transitions, training_state, key):
        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            
            state = transitions.extras["state"]
            goal = transitions.extras['goal_obs'][:, args.goal_start_idx : args.goal_end_idx]
            action = transitions.action
            
            sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
            g_repr = g_encoder.apply(g_encoder_params, goal)
            
            # backward InfoNCE
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)) #shape = BxB
            critic_loss = -jnp.mean( jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0) )

            # logsumexp regularisation
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

            I = jnp.eye(logits.shape[0])
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

            return critic_loss, (logsumexp, correct, logits_pos, logits_neg)
            
        (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(critic_loss, has_aux=True)(training_state.critic_state.params, transitions, key)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state = new_critic_state)

        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
        }

        return training_state, metrics
    
    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)

        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)

        training_state = training_state.replace(gradient_steps = training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        
        return (training_state, key,), metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key):
        experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)
        
        # sample actor-step worth of transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys
        )
        
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )

        permutation = jax.random.permutation(experience_key2, len(transitions.action))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

        # update buffer
        training_state, env_state, buffer_state = get_experience(
            training_state,
            transitions,
            env_state,
            buffer_state,
            experience_key1,
        )
        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )
        
        # take actor-step worth of training-step
        (training_state, _,), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)
        
        return (training_state, env_state, buffer_state,), metrics

    @jax.jit
    def training_epoch(
        training_state,
        env_state,
        buffer_state,
        key,
    ):  
        @jax.jit
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs,), metrics = training_step(ts, es, bs, train_key)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_training_steps_per_epoch)
        
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics
    
    '''Setting up evaluator'''
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        env,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        key=eval_env_key,
    )

    print('prefilling replay buffer....')
    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    training_walltime = 0
    print('starting training....')
    for ne in range(args.num_epochs):
        
        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, epoch_key)
        
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }

        metrics = evaluator.run_evaluation(training_state, metrics)

        print(metrics)


        if args.checkpoint:
            # Save current policy and critic params.
            params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
            path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
            save_params(path, params)
        
        if args.track:
            wandb.log(metrics, step=ne)

            if args.wandb_mode == 'offline':
                trigger_sync()
    
    if args.checkpoint:
        # Save current policy and critic params.
        params = (training_state.alpha_state.params, training_state.actor_state.params, training_state.critic_state.params)
        path = f"{save_path}/final.pkl"
        save_params(path, params)