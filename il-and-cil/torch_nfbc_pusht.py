import sys
sys.path.append("notebooks/")

import os
import time
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

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions
from torch.utils.data import DataLoader

from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_wrapper import VideoWrapper
from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply

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

    eval_tasks: int = None
    eval_episode_len: int = 300
    eval_env_seed: int = 100000
    eval_episodes: int = 20
    video_episodes: int = 1

    #environment specific arguments
    dataset_dir: str = '/n/fs/url-rg/data/pusht/pusht_cchi_v7_replay.zarr'

    #algorithm specific arguments
    num_epochs: int = 100
    actor_lr: float = 1e-3
    weight_decay: float = 1e-6
    val_ratio: float = 0.1
    max_train_episodes: int = 90

    obs_len_input: int = 2
    action_len_pred: int = 16
    action_len_exec: int = 8
    denoise_action: bool = True

    num_workers: int = 1
    batch_size: int = 256
    gamma: float = 0.99
    noise_std: float = 0.1

    #achitecture
    channels: int = 512
    blocks: int = 12
    rep_dim: int = 256

if __name__ == "__main__":

    args = tyro.cli(Args)

    # Set up logger.
    exp_name = f"{args.wandb_name_tag + '__' if args.wandb_name_tag is not None else ''}keypoint-pusht__fcnf_gcbc__{args.channels}-{args.blocks}-{args.rep_dim}__{args.wandb_group}__{int(time.time())}"

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

    env = MultiStepWrapper(
                    VideoWrapper(
                        PushTKeypointsEnv(
                            legacy=True,
                            keypoint_visible_rate=1.0,
                            agent_keypoints=False,
                            **PushTKeypointsEnv.genenerate_keypoint_manager_params()
                        ),
                    ),
                    n_obs_steps=args.obs_len_input,
                    n_action_steps=args.action_len_exec,
                    max_episode_steps=args.eval_episode_len
                )
    env_obs_dim = 20 # 9*2 keypoints + 2 state
    env_action_dim = 2
    env_keypoint_dim = 2

    env.seed(args.eval_env_seed)

    dataset = PushTLowdimDataset(
        args.dataset_dir, 
        horizon=args.action_len_pred,
        pad_before=args.obs_len_input - 1,
        pad_after=args.action_len_exec - 1,
        obs_key='keypoint',
        state_key='state',
        action_key='action',
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_train_episodes=args.max_train_episodes
    )

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, persistent_workers=False)
    normalizer = dataset.get_normalizer()

    # configure validation dataset
    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, persistent_workers=False)

    input_dims_nf = env_action_dim * args.action_len_pred

    prior = distributions.MultivariateNormal(torch.zeros(input_dims_nf).to(device), torch.eye(input_dims_nf).to(device))
    model = RealNVP(input_dims_nf, args.channels, args.rep_dim, args.blocks, prior).to(device)
    encoder = GEncoder(input_size=env_obs_dim*args.obs_len_input, rep_size=args.rep_dim).to(device)

    optimizer = torch.optim.AdamW(
        params=list(model.parameters()) + list(encoder.parameters()),
        lr=args.actor_lr, weight_decay=args.weight_decay)

    # Cosine LR schedule with linear warmup
    lr_scheduler = CosineLRSchedule(optimizer, warmup_steps=500, total_steps = args.num_epochs * len(train_dataloader), min_lr=1e-9, max_lr=args.actor_lr)

    print(f'number of parameters in realnvp: {count_parameters(model)}, and number of parameters in encoder are {count_parameters(encoder)}')

    # Train agent.
    train_logger = CsvLogger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(args.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    best_score = -np.inf

    def pusht_evaluate(
            model,
            encoder,
            env,
            normalizer,
            env_obs_dim,
            obs_len_input,
            action_len_pred,
            action_len_exec,
            max_steps,
            noise_std=0.1,
            num_episodes=50,
            device='cuda',
        ):

        episode_max_performance = []
        episode_final_performance = []
        episode_max_coverage = []
        episode_final_coverage = []
        
        for t in range(num_episodes):
            obs = env.reset()
                    
            done = False
            step_idx = 0
            max_coverage = 0
            
            while step_idx < max_steps:
                nobs = normalizer['obs'].normalize(obs[:,:env_obs_dim])
                nobs = nobs.to(device, dtype=torch.float32)
                
                #sample action with denoising
                with torch.no_grad():
                    nobsz = encoder(nobs.reshape(-1, env_obs_dim * obs_len_input))
                    prior_sample = model.prior.sample().unsqueeze(0)
                    naction = model.reverse(prior_sample, nobsz)
                
                naction = torch.clone(naction).detach()
                naction.requires_grad = True
                z, outputs, logdets = model(naction, nobsz)
                logprob = model.prior.log_prob(z) + logdets
                grad = torch.autograd.grad(logprob, [naction])[0]
                naction.data = naction.data + noise_std**2 * grad
                naction = naction.reshape(1, action_len_pred, -1).detach().to('cpu')
            
                naction = normalizer['action'].unnormalize(naction)
                
                start = obs_len_input - 1
                end = start + action_len_exec
                
                action = naction.numpy()[:,0:][0,start:end]
            
                obs, reward, done, info = env.step(action)

                max_coverage = np.max(info['coverage']) if np.max(info['coverage']) > max_coverage else max_coverage
            
                if done:
                    break
                    
                step_idx += action.shape[0]

            # print('Evaluation episode', 0, 'Max Score: ', max(env.get_attr('reward')), 'Final Score: ', env.get_attr('reward')[-1])
            # print('Evaluation episode', 0, 'Max Coverage: ', max_coverage, 'Final coverage: ', info['coverage'][-1])
                
            episode_max_performance.append(max(env.get_attr('reward')))
            episode_final_performance.append(env.get_attr('reward')[-1])

            episode_max_coverage.append(max_coverage)
            episode_final_coverage.append(info['coverage'][-1])

        eval_info = {
            'evaluation/episode_max_performance' : np.mean(episode_max_performance),
            'evaluation/episode_final_performance' : np.mean(episode_final_performance),
            'evaluation/episode_max_coverage' : np.mean(episode_max_coverage),
            'evaluation/episode_final_coverage' :np.mean( episode_final_coverage),
        }
            
        return eval_info, env.render()

    for epoch_idx in range(1, args.num_epochs+1):
        epoch_loss = []
        for nbatch in train_dataloader:

            nbatch = normalizer.normalize(nbatch)
            nbatch = dict_apply(nbatch, lambda x: x.to(device))
                
            nobs = nbatch['obs'][:,:args.obs_len_input,:]
            naction = nbatch['action']
            
            nobs = nobs.reshape(-1, env_obs_dim * args.obs_len_input) #.unsqueeze(1)
            naction = naction.reshape(-1, env_action_dim * args.action_len_pred) #.unsqueeze(-1)
            
            eps = args.noise_std * torch.randn_like(naction)
            naction = naction + eps

            z, outputs, logdets = model(naction, encoder(nobs))
            
            loss = - (model.prior.log_prob(z) + logdets).mean()
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch_loss.append(loss.item())

        train_metrics = {
            'training/actor/loss' : np.mean(epoch_loss).item(),
            'training/actor/logdets' : logdets.mean().item()
        }
        for b in range(args.blocks):
            train_metrics[f"training/actor/norms/layer__{b}"] = outputs[b].pow(2).mean().item()

        val_epoch_loss = []
        for nbatch in val_dataloader:

            nbatch = normalizer.normalize(nbatch)
            nbatch = dict_apply(nbatch, lambda x: x.to(device))
                
            nobs = nbatch['obs'][:,:args.obs_len_input,:]
            naction = nbatch['action']
            
            nobs = nobs.reshape(-1, env_obs_dim * args.obs_len_input) #.unsqueeze(1)
            naction = naction.reshape(-1, env_action_dim * args.action_len_pred) #.unsqueeze(-1)
            
            eps = args.noise_std * torch.randn_like(naction)
            naction = naction + eps

            with torch.no_grad():
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

        print(f"epoch {epoch_idx} lr {optimizer.param_groups[0]['lr']:.6f} train loss {np.mean(epoch_loss):.4f} val loss {np.mean(val_epoch_loss):.4f}")
        print('Validation layer norms', ' '.join([f'{z.pow(2).mean():.4f}' for z in outputs]))

        train_metrics['time/epoch_time'] = (time.time() - last_time)
        train_metrics['time/total_time'] = time.time() - first_time
        last_time = time.time()

        if args.track:
            wandb.log(train_metrics, step=epoch_idx)
            if args.wandb_mode == 'offline':
                trigger_sync()

        train_logger.log(train_metrics, step=epoch_idx)
    
        # Evaluate agent.
        if epoch_idx == 1 or epoch_idx % args.eval_interval == 0:
            eval_metrics, renders = pusht_evaluate(
                    model=model,
                    encoder=encoder,
                    env=env,
                    normalizer=normalizer,
                    env_obs_dim=env_obs_dim,
                    obs_len_input=args.obs_len_input,
                    action_len_pred=args.action_len_pred,
                    action_len_exec=args.action_len_exec,
                    max_steps=args.eval_episode_len,
                    noise_std=args.noise_std,
                    num_episodes=args.eval_episodes,
                    device=device,
                )
            
            print(eval_metrics)

            if best_score <= eval_metrics['evaluation/episode_final_coverage']:
                best_score = eval_metrics['evaluation/episode_final_coverage']

                print('saving agent with best score', best_score)
                
                save_dict = save_params = {
                    'model': model.state_dict(),
                    'encoder': encoder.state_dict(),
                }
                            
                save_agent(save_dict, args.save_dir, 'best-score')

            if args.track:
                # if args.video_episodes > 0:
                #     video = wandb.Video(renders, fps=15, format='mp4')
                #     eval_metrics['video'] = video

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