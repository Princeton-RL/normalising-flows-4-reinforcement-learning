import os
os.environ["MUJOCO_GL"] = "egl"

import sys
sys.path.append("notebooks/")

import os
import time
import gym
import tqdm
import tyro
import copy
import json
import wandb
import random
import numpy as np
from etils import epath
from dataclasses import dataclass
from collections import defaultdict
from dataclasses import asdict
from stable_baselines3.common.vec_env import DummyVecEnv

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions
from torch.utils.data import DataLoader

from kitchen_env import KitchenWrapper
from vqdet_dataset import get_relay_kitchen_train_val

from utils.log_utils import CsvLogger, get_wandb_video, setup_torch_wandb
from utils.torch_fcnf import RealNVP, GEncoder
from utils.torch_utils import save_agent, count_parameters, CosineLRSchedule

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    #logging
    track: bool = False
    wandb_project_name: str = "BC"
    wandb_entity: str = 'user'
    wandb_mode: str = 'online'
    wandb_dir: str = 'exp/'
    save_dir: str = 'exp/'
    wandb_group: str = 'Debug'
    wandb_name_tag: str = None

    eval_interval: int = 5
    save_interval: int = 50

    num_eval_episodes: int = 20
    # video_episodes: int = 1

    #environment specific arguments
    dataset_dir: str = '/n/fs/url-rg/vqbet-data/vqbet_datasets_for_release/relay_kitchen/'

    #algorithm specific arguments
    num_epochs: int = 100
    actor_lr: float = 3e-4
    weight_decay: float = 1e-6

    num_workers: int = 4
    batch_size: int = 256
    noise_std: float = 0.1

    window_size: int = 10
    goal_window_size: int = 10
    action_window_size: int = 1
    eval_action_window_size: int = 1

    #achitecture
    channels: int = 512
    blocks: int = 12
    rep_dim: int = 1024

if __name__ == "__main__":

    args = tyro.cli(Args)

    # Set up logger.
    exp_name = f"{args.wandb_name_tag + '__' if args.wandb_name_tag is not None else ''}__kitchen__fcnf_gcbc__{args.channels}-{args.blocks}-{args.rep_dim}__{args.wandb_group}__{int(time.time())}"

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

    obs_dim = 60
    act_dim = 9

    train_data, test_data = get_relay_kitchen_train_val(
        data_directory=args.dataset_dir,
        goal_conditional='future',
        window_size=args.window_size,
        future_seq_len=args.goal_window_size,
        action_window_size=args.action_window_size,
        visual_input=False,
        vqbet_get_future_action_chunk=True,
        device='cpu'
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=False
    )

    # stats = {}
    # stats['obs'] = {}
    # stats['actions'] = {}
    # stats['obs']['max'], stats['obs']['min'] = torch.max( train_data.dataset.dataset.tensors[0].reshape(-1, obs_dim), dim=0 ).values, torch.min( train_data.dataset.dataset.tensors[0].reshape(-1, obs_dim), dim=0 ).values
    # stats['actions']['max'], stats['actions']['min'] = torch.max( train_data.dataset.dataset.actions.reshape(-1, act_dim), dim=0 ).values, torch.min( train_data.dataset.dataset.actions.reshape(-1, act_dim), dim=0 ).values 

    # def normalize_data(data, stats):
    #     # nomalize to [0,1]
    #     ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    #     # normalize to [-1, 1]
    #     ndata = ndata * 2 - 1
    #     return ndata

    # def unnormalize_data(ndata, stats):
    #     ndata = (ndata + 1) / 2
    #     data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    #     return data
    

    input_dims_nf = act_dim * args.action_window_size
    prior = distributions.MultivariateNormal(torch.zeros(input_dims_nf).to(device), torch.eye(input_dims_nf).to(device))
    model = RealNVP(input_dims_nf, args.channels, args.rep_dim, args.blocks, prior).to(device)
    encoder = GEncoder(input_size=obs_dim*args.window_size, rep_size=args.rep_dim).to(device)

    from timeit import default_timer as timer
    obs_stack = torch.zeros([1, args.window_size, 60]).reshape(-1, obs_dim * args.window_size).to(device)
    prior_sample = model.prior.sample((1,))

    start = timer()
    obsz = encoder(obs_stack)
    action = model.reverse(prior_sample, obsz)
    end = timer()

    print(f"Time taken: {end - start:.9f} seconds")

    optimizer = torch.optim.AdamW(
        params=list(model.parameters()) + list(encoder.parameters()),
        lr=args.actor_lr, weight_decay=args.weight_decay)

    # Cosine LR schedule with linear warmup
    lr_scheduler = CosineLRSchedule(optimizer, warmup_steps=500, total_steps = args.num_epochs * len(train_loader), min_lr=1e-6, max_lr=args.actor_lr)

    print(f'number of parameters in realnvp: {count_parameters(model)}, and number of parameters in encoder are {count_parameters(encoder)}')

    # Train agent.
    train_logger = CsvLogger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(args.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    best_score = -np.inf

    def make_kitchen_envs(rank: int, seed: int = 0):
        def _init():
            env = KitchenWrapper(
                env = gym.make('kitchen-v0'),
                id = 'kitchen-v0',
                visual_input=False
            )
            env.seed(seed = seed + rank)
            return env

        return _init

    envs = DummyVecEnv([make_kitchen_envs(i, args.seed) for i in range(args.num_eval_episodes)])


    def evaluate_parallel_bc():
        done = np.zeros((args.num_eval_episodes,), dtype=bool)
        success = np.zeros((args.num_eval_episodes,))
        rewards = np.zeros((args.num_eval_episodes,))

        step = 0

        init_obs = []
        for idx, env in enumerate(envs.envs):
            init_obs.append( env.reset() )

        init_obs = np.stack(init_obs, axis=0)

        obs_stack = np.repeat(init_obs[:, None, :], args.window_size, axis=1)

        sample_new_action_count = args.eval_action_window_size

        while not np.all(done):

            if sample_new_action_count == args.eval_action_window_size:
                sample_new_action_count = 0

                obs = torch.from_numpy( obs_stack )
                obs = obs.to(device, dtype=torch.float32)
                
                #sample action with denoising
                with torch.no_grad():
                    obsz = encoder(obs.reshape(-1, obs_dim * args.window_size))
                    prior_sample = model.prior.sample((args.num_eval_episodes,))
                    action = model.reverse(prior_sample, obsz)
                
                action = torch.clone(action).detach()
                action.requires_grad = True
                z, outputs, logdets = model(action, obsz)
                logprob = model.prior.log_prob(z.squeeze(-1)) + logdets
                grad = torch.autograd.grad(logprob.sum(), [action])[0]
                action.data = action.data + args.noise_std**2 * grad

                action = action.reshape(args.num_eval_episodes, args.action_window_size, -1).detach()
                action = action.to('cpu').numpy()
            
            curr_action = action[:, sample_new_action_count]
            sample_new_action_count += 1

            curr_obs = []
            curr_rews = []
            curr_done = []
            curr_success = []
            for idx, env in enumerate(envs.envs):
                eo, er, ed, ei = env.step(curr_action[idx]) 

                curr_obs.append(eo)
                curr_rews.append(er)
                curr_done.append(ed)
                curr_success.append(len(ei['all_completions_ids']))

            curr_obs = np.stack(curr_obs, axis=0)
            curr_rews = np.stack(curr_rews, axis=0)
            curr_success = np.stack(curr_success, axis=0)
            curr_done = np.stack(curr_done, axis=0)

            success = (1-done) * curr_success + done * success
            rewards = (1-done) * (rewards + curr_rews) + done * rewards
            
            obs_stack = np.concatenate( [ obs_stack[:, 1:], np.expand_dims( curr_obs, axis=1) ], axis=1)
            done = np.logical_or(curr_done, done) 
            
            step += 1
        
        return {'evaluation/rewards' : np.mean(rewards)}

    for epoch_idx in range(1, args.num_epochs+1):
        epoch_loss = []
        # batch loop
        for nbatch in train_loader:

            # nobs = normalize_data(nbatch[0], stats['obs']).to(device)
            # naction = normalize_data(nbatch[1], stats['actions']).to(device)
            
            nobs = nbatch[0].to(device)
            naction = nbatch[1].to(device) 

            naction = naction[:, -args.action_window_size:, :]

            nobs = nobs.reshape(-1, obs_dim * args.window_size)
            naction = naction.reshape(-1, act_dim * args.action_window_size)
            
            eps = args.noise_std * torch.randn_like(naction)
            naction = naction + eps

            z, outputs, logdets = model(naction, encoder(nobs))
            
            loss = - (model.prior.log_prob(z) + logdets).mean()
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch_loss.append(loss.item())
            
            if len(epoch_loss)%100 == 0:
                print(f'{len(epoch_loss) / len(train_loader)}% epoch is completed.')
            

        train_metrics = {
            'training/actor/loss' : np.mean(epoch_loss).item(),
            'training/actor/logdets' : logdets.mean().item()
        }
        for b in range(args.blocks):
            train_metrics[f"training/actor/norms/layer__{b}"] = outputs[b].pow(2).mean().item()


        with torch.no_grad():
            val_epoch_loss = []
            for nbatch in test_loader:

                # nobs = normalize_data(nbatch[0], stats['obs']).to(device)
                # naction = normalize_data(nbatch[1], stats['actions']).to(device) 

                nobs = nbatch[0].to(device)
                naction = nbatch[1].to(device) 
                
                naction = naction[:, -args.action_window_size:, :]

                nobs = nobs.reshape(-1, obs_dim * args.window_size)
                naction = naction.reshape(-1, act_dim * args.action_window_size)
                
                eps = args.noise_std * torch.randn_like(naction)
                naction = naction + eps

                z, outputs, logdets = model(naction, encoder(nobs))
                
                loss = - (model.prior.log_prob(z) + logdets).mean()
                        
                val_epoch_loss.append(loss.item())

        val_metrics = {
            'validation/actor/loss' : np.mean(val_epoch_loss).item(),
            'validation/actor/logdets' : logdets.mean().item()
        }
        for b in range(args.blocks):
            val_metrics[f"validation/actor/norms/layer__{b}"] = outputs[b].pow(2).mean().item()

        train_metrics.update(val_metrics)

        train_metrics['time/epoch_time'] = (time.time() - last_time)
        train_metrics['time/total_time'] = time.time() - first_time
        last_time = time.time()

        if args.track:
            wandb.log(train_metrics, step=epoch_idx)
            if args.wandb_mode == 'offline':
                trigger_sync()

        train_logger.log(train_metrics, step=epoch_idx)
        

        print(f"epoch {epoch_idx} lr {optimizer.param_groups[0]['lr']:.6f} train loss {np.mean(epoch_loss):.4f} val loss {np.mean(val_epoch_loss):.4f}")
        print('Validation layer norms', ' '.join([f'{z.pow(2).mean():.4f}' for z in outputs]))
        
        # Evaluate agent.
        if epoch_idx == 1 or epoch_idx % args.eval_interval == 0:
            eval_metrics = evaluate_parallel_bc()
            
            print(eval_metrics)

            if best_score <= eval_metrics['evaluation/rewards']:
                best_score = eval_metrics['evaluation/rewards']

                print('saving agent with best score', best_score)
                
                save_dict = save_params = {
                    'model': model.state_dict(),
                    'encoder': encoder.state_dict(),
                }
                            
                save_agent(save_dict, args.save_dir, 'best-score')

            if args.track:
                wandb.log(eval_metrics, step=epoch_idx)
                if args.wandb_mode == 'offline':
                    trigger_sync()

            eval_logger.log(eval_metrics, step=epoch_idx)
        
        # Save agent.
        if epoch_idx % args.save_interval == 0:
            save_dict = save_params = {
                'model': model.state_dict(),
                'encoder': encoder.state_dict(),
            }
                        
            save_agent(save_dict, args.save_dir, epoch_idx)

    envs.close()
    train_logger.close()
    eval_logger.close()