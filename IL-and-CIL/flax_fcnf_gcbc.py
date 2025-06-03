import os
os.environ["MUJOCO_GL"] = "egl"

import time
import tqdm
import tyro
import copy
import json
import wandb
import random
import ogbench
from tqdm import trange
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
from etils import epath
from dataclasses import dataclass, asdict
from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
jax.config.update("jax_default_matmul_precision", "float32")

from utils.log_utils import CsvLogger, get_wandb_video, setup_torch_wandb
from utils.flax_utils import count_parameters, save_params
from utils.flax_fcnf import RealNVP, RealNVPEncoder, create_prior
from utils.torch_datasets import ocbc_dataset
from utils.torch_evaluations import make_env

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    #logging
    track: bool = False
    wandb_project_name: str = "OGBench"
    wandb_entity: str = 'raj19'
    wandb_mode: str = 'online'
    wandb_dir: str = 'exp/'
    save_dir: str = 'exp/'
    wandb_group: str = 'Debug'
    wandb_name_tag: str = None

    log_interval: int = 5000
    eval_interval: int = 100000
    save_interval: int = 1000000

    eval_tasks: int = None
    eval_episodes: int = 20
    video_episodes: int = 1
    video_frame_skip: int = 3

    #environment specific arguments
    env_name: str = "cube-single-play-v0"
    frame_stack: int = None
    dataset_dir: str = None

    #algorithm specific arguments
    num_trainsteps: int = 1000000
    actor_lr: float = 3e-4
    weight_decay: float = 1e-6
    
    action_len_pred: int = None
    action_len_exec: int = None
    denoise_action: bool = True

    batch_size: int = 256
    rep_size: int = 64
    gamma: float = 0.99
    noise_std: float = 0.1

    #bc achitecture
    bc_channels: int = 256
    bc_blocks: int = 6
    bc_rep_size: int = 64

if __name__ == "__main__":

    args = tyro.cli(Args)

    # Set up logger.
    exp_name = f"{args.wandb_name_tag + '__' if args.wandb_name_tag is not None else ''}{args.env_name}__fcnf_gcbc__{args.bc_channels}-{args.bc_blocks}-{args.bc_channels}__{args.wandb_group}__{int(time.time())}"

    if args.track:
        _, trigger_sync = setup_torch_wandb(entity='raj19', project=args.wandb_project_name, group=args.wandb_group, name=exp_name, wandb_output_dir=args.wandb_dir, mode=args.wandb_mode, config=asdict(args))

    args.save_dir = os.path.join(args.wandb_dir, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    flag_dict = asdict(args)
    with open(os.path.join(args.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    random.seed(args.seed)
    np.random.seed(args.seed)

    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(args.env_name, compact_dataset=True, dataset_dir=args.dataset_dir if args.dataset_dir is not None else '~/.ogbench/data')
    envs = DummyVecEnv([make_env(args.env_name, i, args.seed) for i in range(args.eval_episodes)])

    train_dataset = ocbc_dataset(train_dataset, args.batch_size, args.gamma, args.frame_stack, action_len=args.action_len_pred)
    val_dataset = ocbc_dataset(val_dataset, args.batch_size, args.gamma, args.frame_stack, action_len=args.action_len_pred)

    # Setup GCBC agent
    # Create agent.
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, encoder_key = jax.random.split(key, 3)

    encoder = RealNVPEncoder(
        input_size = env.observation_space.shape[0]*2,
        rep_size = args.bc_rep_size
    )
    encoder_params = encoder.init(encoder_key, x = jnp.zeros(shape = [1, env.observation_space.shape[0]*2]) ) 

    nf_model = RealNVP(
        num_blocks = args.bc_blocks,   
        in_channels=env.action_space.shape[0],
        cond_channels=args.bc_rep_size, 
        channels = args.bc_channels,
    )
    nf_model_params = nf_model.init(actor_key, x = jnp.zeros(shape=(1, env.action_space.shape[0])), y = jnp.zeros(shape=(1, args.bc_rep_size), dtype=jnp.float32))

    combined_params = {
        'model' : nf_model_params,
        'encoder' : encoder_params,
    }
    for k in combined_params.keys():
        print(f'{k} has {count_parameters(combined_params[k])} number of parameters.')
    
    lr_scheduler = optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=args.actor_lr,
                warmup_steps=500, decay_steps=args.num_trainsteps,
                end_value=1e-6,
    )
    optimizer = optax.adamw(learning_rate=lr_scheduler, weight_decay=args.weight_decay)
    actor_state = train_state.TrainState.create(
        apply_fn=None,  # or your combined apply function if needed
        params=combined_params,
        tx=optimizer
    )

    prior = create_prior(env.action_space.shape[0])

    # Train agent.
    train_logger = CsvLogger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(args.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    @jax.jit
    def get_entropy(state, b_actions, b_labels, sample_key):
        prior_sample = prior.sample(sample_shape=(b_labels.shape[0],), seed=sample_key)
        p_actions, p_logdets = nf_model.apply(
            state.params['model'], 
            x=prior_sample, 
            y=encoder.apply(state.params['encoder'], b_labels), 
            reverse=True, 
            )
        entropy = (prior.log_prob(prior_sample) - p_logdets).mean() 
        mse = ((p_actions - b_actions ) **2).mean()
        return entropy, mse

    @jax.jit
    def get_loss(state, b_actions, b_labels):
        z, logdets = nf_model.apply(state.params['model'], x=b_actions, y=encoder.apply(state.params['encoder'], b_labels))
        loss = - (prior.log_prob(z) + logdets).mean()
        return loss, z, logdets
    
    @jax.jit
    def train_step(state, b_actions, b_labels):
        def loss_fn(params):
            z, logdets = nf_model.apply(params['model'], x=b_actions, y=encoder.apply(params['encoder'], b_labels))
            loss = - (prior.log_prob(z) + logdets).mean()
            return loss, (z, logdets)

        # Compute the loss and gradients.
        (loss, (z, logdets)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # Update the state with the computed gradients.
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, z, logdets

    @partial(jax.jit, static_argnames=['num_eval_episodes'])
    def get_action(state, observation, goal, sample_key, num_eval_episodes):
        prior_sample = prior.sample(sample_shape=(num_eval_episodes,), seed=sample_key)
        observation_goal = jnp.concatenate([observation, goal], axis=-1).astype(jnp.float32)
        observation_goal_z = encoder.apply(state.params['encoder'], observation_goal)
        action, _ = nf_model.apply(
            state.params['model'], 
            x=prior_sample, 
            y=observation_goal_z, 
            reverse=True, 
            )
        return action

    @partial(jax.jit, static_argnames=['num_eval_episodes'])
    def get_denoised_action(state, observation, goal, sample_key, num_eval_episodes):
         
        def log_prob_fn(x, y):
            z, logdets = nf_model.apply(state.params['model'], x=x, y=y)
            logprob = prior.log_prob(z) + logdets
            return logprob.sum()
        
        prior_sample = prior.sample(sample_shape=(num_eval_episodes,), seed=sample_key)
        observation_goal = jnp.concatenate([observation, goal], axis=-1).astype(jnp.float32)
        observation_goal_z = encoder.apply(state.params['encoder'], observation_goal)
        action, _ = nf_model.apply(
            state.params['model'], 
            x=prior_sample, 
            y=observation_goal_z, 
            reverse=True, 
            )
        
        action = jax.lax.stop_gradient(action)
        action_score = jax.grad(log_prob_fn)(action, observation_goal_z)
        action = action + args.noise_std**2 * action_score
        return action

    def evaluate(
        seed,
        eval_key,
        envs,
        task_infos,
        num_tasks,
        actor_state,
        denoise_action,
        action_len_pred=None,
        action_len_exec=None,
        num_eval_episodes=50,
    ):
        
        stats = defaultdict(lambda: np.array([]))

        if action_len_pred is not None: 
            assert action_len_pred >= action_len_exec
        
        for task_id in trange(1, num_tasks + 1):
            task_name = task_infos[task_id - 1]['task_name']

            results = []
            for idx, env in enumerate(envs.envs):
                results.append( env.reset(seed=seed+idx, options=dict(task_id=task_id, render_goal=False)) )
        
            done = np.zeros((num_eval_episodes,), dtype=bool)
            success = np.zeros((num_eval_episodes,))

            obs_list, info_list = zip(*results)
            goal_list = []
            for info in info_list:
                goal_list.append( info.get('goal') )
            observation = np.stack(obs_list, axis=0)
            goal = np.stack(goal_list, axis=0)

            step = 0
            while not np.all(done):
                action_sample_key, eval_key = jax.random.split(eval_key)
                if action_len_pred is not None:
                    raise NotImplementedError
                else:
                    if denoise_action:
                        action = get_denoised_action(actor_state, observation, goal, action_sample_key, num_eval_episodes)
                    else:
                        action = get_action(actor_state, observation, goal, action_sample_key, num_eval_episodes)

                    action = action
                    action = np.clip(jax.device_get(action), -1, 1)
                
                next_observation, reward, next_done, info = envs.step(action)
                step += 1
                step_success = []
                for i in info:
                    step_success.append( i.get('success') )
                
                success = (1-done) * np.array(step_success) + done * success

                done = np.logical_or(next_done, done) 
                observation = next_observation
            
            stats[f'evaluation/{task_name}_success'] = success
            stats[f'evaluation/overall_success'] = np.append(stats[f'evaluation/overall_success'], success)
        
        return stats

    for i in tqdm.tqdm(range(1, args.num_trainsteps + 1), smoothing=0.1, dynamic_ncols=True):
        sample_key, key = jax.random.split(key)

        # Update agent.
        b_obs, b_actions, b_goals = train_dataset.sample(args.batch_size)
        b_labels =  jnp.concatenate([b_obs, b_goals], axis=-1).astype(jnp.float32)
        
        eps = args.noise_std * jax.random.normal(sample_key, shape=b_actions.shape)
        b_actions = b_actions + eps

        actor_state, b_loss, b_z, b_logdets = train_step(actor_state, b_actions, b_labels)

        metrics = {
            'actor/loss' : b_loss.item(),
            'actor/logdets' : b_logdets.mean().item(),
            'actor/norms/layer__0' : jnp.square( b_z ).mean().item(),
            'actor/learning_rate': lr_scheduler(i),
        }

        # Log metrics.
        if i % args.log_interval == 0:
            sample_entropy_key, sample_entropy_val_key, sample_key, key = jax.random.split(key, 4)      
            metrics['actor/entropy'], metrics['actor/mse'] = get_entropy(actor_state, b_actions, b_labels, sample_entropy_key)
            train_metrics = {f'training/{k}': v for k, v in metrics.items()}
            
            if val_dataset is not None:
                vb_obs, vb_actions, vb_goals = val_dataset.sample(args.batch_size)
                vb_labels = jnp.concatenate([vb_obs, vb_goals], axis=-1).astype(jnp.float32)
                eps = args.noise_std * jax.random.normal(sample_key, shape=vb_actions.shape)
                vb_actions = vb_actions + eps
                vb_loss, vb_z, vb_logdets = get_loss(actor_state, vb_actions, vb_labels)
                
                val_info = {
                    'actor/loss' : vb_loss.item(),
                    'actor/logdets' : vb_logdets.mean().item(),
                    'actor/norms/layer__0' : jnp.square( vb_z ).mean().item()
                }
                val_info['actor/entropy'], val_info['actor/mse'] = get_entropy(actor_state, vb_actions, vb_labels, sample_entropy_val_key)

                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / args.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()

            if args.track:
                wandb.log(train_metrics, step=i)
                if args.wandb_mode == 'offline':
                    trigger_sync()

            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % args.eval_interval == 0:            
            eval_metrics = {}
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = args.eval_tasks if args.eval_tasks is not None else len(task_infos)
            eval_key, key = jax.random.split(key)
            eval_info = evaluate(
                    args.seed,
                    eval_key,
                    envs,
                    task_infos,
                    num_tasks,
                    actor_state,
                    denoise_action=args.denoise_action,
                    num_eval_episodes=args.eval_episodes,
                )

            for eval_key in eval_info:
                eval_metrics[eval_key] = np.mean(eval_info[eval_key])

            if args.track:
                wandb.log(eval_metrics, step=i)
                if args.wandb_mode == 'offline':
                    trigger_sync()
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % args.save_interval == 0:             
            save_params(actor_state.params, args.save_dir, i)

    env.close()
    envs.close()
    train_logger.close()
    eval_logger.close()