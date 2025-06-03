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
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
from etils import epath
from dataclasses import dataclass
from collections import defaultdict
from dataclasses import asdict

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions

from utils.log_utils import CsvLogger, get_wandb_video, setup_torch_wandb
from utils.torch_datasets import ocbc_dataset
from utils.torch_networks import RealNVP, GEncoder
from utils.torch_evaluations import evaluate_parallel, make_env
from utils.torch_utils import save_agent, count_parameters

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    #logging
    track: bool = False
    wandb_project_name: str = "OGBench"
    wandb_entity: str = 'user'
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

    #achitecture
    channels: int = 256
    blocks: int = 4
    layers_per_block : int = 4
    head_dim: int = 64
    expansion: int = 4
    rep_dim: int = 64

if __name__ == "__main__":

    args = tyro.cli(Args)

    # Set up logger.
    exp_name = f"{args.wandb_name_tag + '__' if args.wandb_name_tag is not None else ''}{args.env_name}__gcnfbc__{args.channels}-{args.blocks}-{args.layers_per_block}__{args.wandb_group}__{int(time.time())}"

    if args.track:
        _, trigger_sync = setup_torch_wandb(entity='user', project=args.wandb_project_name, group=args.wandb_group, name=exp_name, wandb_output_dir=args.wandb_dir, mode=args.wandb_mode, config=asdict(args))

    args.save_dir = os.path.join(args.wandb_dir, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    flag_dict = asdict(args)
    with open(os.path.join(args.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu' # if mps not available
        
    print(f'using device {device}')

    random.seed(args.seed)
    np.random.seed(args.seed)

    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(args.env_name, compact_dataset=True, dataset_dir=args.dataset_dir)
    envs = DummyVecEnv([make_env(args.env_name, i, args.seed) for i in range(args.eval_episodes)])

    train_dataset = ocbc_dataset(train_dataset, args.batch_size, args.gamma, args.frame_stack, action_len=args.action_len_pred)
    val_dataset = ocbc_dataset(val_dataset, args.batch_size, args.gamma, args.frame_stack, action_len=args.action_len_pred)

    input_dims_nf = env.action_space.shape[0] * ( args.action_len_pred if args.action_len_pred is not None else 1)
    prior = distributions.MultivariateNormal(torch.zeros(input_dims_nf).to(device), torch.eye(input_dims_nf).to(device))
    model = RealNVP(input_dims_nf, args.channels, args.rep_dim, args.head_dim, args.blocks, args.layers_per_block, prior).to(device)
    encoder = GEncoder(input_size=env.observation_space.shape[0]*2, rep_size=args.rep_dim).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=args.actor_lr, weight_decay=args.weight_decay)

    print(f'number of parameters in realnvp: {count_parameters(model)}, and number of parameters in encoder are {count_parameters(encoder)}')

    # Train agent.
    train_logger = CsvLogger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(args.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    for i in tqdm.tqdm(range(1, args.num_trainsteps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        b_obs, b_actions, b_goals = train_dataset.sample(args.batch_size)

        b_labels = torch.from_numpy(np.concatenate([b_obs, b_goals], axis=-1)).to(device).unsqueeze(1)
        b_actions = torch.from_numpy(b_actions).to(device).unsqueeze(-1)
        eps = args.noise_std * torch.randn_like(b_actions)
        b_actions = b_actions + eps

        b_z, b_outputs, b_logdets = model(b_actions, encoder(b_labels))
        b_loss = - ( model.prior.log_prob(b_z.squeeze(-1)) + b_logdets ).mean()

        optimizer.zero_grad()
        b_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metrics = {
            'actor/loss' : b_loss.item(),
            'actor/logdets' : b_logdets.mean().item()
        }

        for b in range(args.blocks):
            metrics[f"actor/norms/layer__{b}"] = b_outputs[b].squeeze(-1).pow(2).mean().item()

        # Log metrics.
        if i % args.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in metrics.items()}
            
            if val_dataset is not None:
                vb_obs, vb_actions, vb_goals = val_dataset.sample(args.batch_size)
                
                vb_labels = torch.from_numpy(np.concatenate([vb_obs, vb_goals], axis=-1)).to(device).unsqueeze(1)
                vb_actions = torch.from_numpy(vb_actions).to(device).unsqueeze(-1)
                eps = args.noise_std * torch.randn_like(vb_actions)
                vb_actions = vb_actions + eps
                vb_z, vb_outputs, vb_logdets = model(vb_actions, encoder(vb_labels))
                vb_loss = - ( model.prior.log_prob(vb_z.squeeze(-1)) + vb_logdets ).mean()

                val_info = {
                    'actor/loss' : vb_loss.item(),
                    'actor/logdets' : vb_logdets.mean().item()
                }

                for b in range(args.blocks):
                    val_info[f"actor/norms/layer__{b}"] = vb_outputs[b].squeeze(-1).pow(2).mean().item()

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

            eval_info = evaluate_parallel(
                    args.seed,
                    envs,
                    task_infos,
                    num_tasks,
                    model,
                    encoder,
                    denoise_action=args.denoise_action,
                    noise_std=args.noise_std,
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
            save_dict = save_params = {
                'model': model.state_dict(),
                'encoder': encoder.state_dict(),
            }
                        
            save_agent(save_dict, args.save_dir, i)

    env.close()
    envs.close()
    train_logger.close()
    eval_logger.close()
