import ogbench
from collections import defaultdict

import numpy as np
from tqdm import trange

import torch

def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def make_env(env_id: str, rank: int, seed: int = 0):

    def _init():
        env = ogbench.make_env_and_datasets(env_id, env_only=True)
        env.reset(seed = seed + rank)
        return env

    return _init

def evaluate_parallel(
    seed,
    envs,
    task_infos,
    num_tasks,
    model,
    encoder,
    denoise_action,
    noise_std=0.0,
    action_len_pred=None,
    action_len_exec=None,
    num_eval_episodes=50,
):
    
    device = next(model.parameters()).device

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
            if action_len_pred is not None:
                raise NotImplementedError
            else:
                if denoise_action:
                    with torch.no_grad():
                        prior_sample = model.prior.sample((num_eval_episodes,)).unsqueeze(-1)
                        observation_goal = torch.from_numpy(np.concatenate([observation, goal], axis=-1)).unsqueeze(1).float().to(device)
                        observation_goal_z = encoder(observation_goal)
                        action = model.reverse(prior_sample, observation_goal_z)

                    action = torch.clone(action).detach()
                    action.requires_grad = True
                    z, _, logdets = model(action, observation_goal_z)
                    logprob = model.prior.log_prob(z.squeeze(-1)) + logdets
                    grad = torch.autograd.grad(logprob.sum(), [action])[0]
                    action.data = action.data + noise_std**2 * grad
                    action = action.squeeze(-1).detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        prior_sample = model.prior.sample((num_eval_episodes,)).unsqueeze(-1)
                        observation_goal = torch.from_numpy(np.concatenate([observation, goal], axis=-1)).unsqueeze(1).float().to(device)
                        observation_goal_z = encoder(observation_goal)
                        action = model.reverse(prior_sample, observation_goal_z).squeeze(-1).detach().cpu().numpy()
                                        
                action = np.clip(action, -1, 1)
            
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

def evaluate(
    model,
    encoder,
    env,
    denoise_action,
    noise_std=0.0,
    action_len_pred=None,
    action_len_exec=None,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """

    if action_len_pred is not None: 
        assert action_len_pred >= action_len_exec

    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []

        if action_len_pred is not None:
            sample_new_action_count = action_len_exec

        while not done:
            if action_len_pred is not None:
                if sample_new_action_count == action_len_exec:
                    if denoise_action:
                        raise NotImplementedError
                    else:
                        with torch.no_grad():
                            prior_sample = model.prior.sample().unsqueeze(0).unsqueeze(-1)
                            observation_goal = torch.from_numpy(np.concatenate([observation, goal], axis=-1)).unsqueeze(0).unsqueeze(0).float().to('cuda')
                            action_sequence = model.reverse(prior_sample, encoder(observation_goal)).squeeze(-1).squeeze(0).cpu().numpy().reshape(action_len_pred, -1)
                    sample_new_action_count = 0

                action = np.clip(action_sequence[sample_new_action_count], -1, 1)
                sample_new_action_count += 1
            
            else:
                if denoise_action:
                    with torch.no_grad():
                        prior_sample = model.prior.sample().unsqueeze(0).unsqueeze(-1)
                        observation_goal = torch.from_numpy(np.concatenate([observation, goal], axis=-1)).unsqueeze(0).unsqueeze(0).float().to('cuda')
                        observation_goal_z = encoder(observation_goal)
                        action = model.reverse(prior_sample, observation_goal_z)

                    action = torch.clone(action).detach()
                    action.requires_grad = True
                    z, _, logdets = model(action, observation_goal_z)
                    logprob = model.prior.log_prob(z.squeeze(-1)) + logdets
                    grad = torch.autograd.grad(logprob, [action])[0]
                    action.data = action.data + noise_std**2 * grad
                    action = action.squeeze(-1).squeeze(0).detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        prior_sample = model.prior.sample().unsqueeze(0).unsqueeze(-1)
                        observation_goal = torch.from_numpy(np.concatenate([observation, goal], axis=-1)).unsqueeze(0).unsqueeze(0).float().to('cuda')
                        action = model.reverse(prior_sample, encoder(observation_goal)).squeeze(-1).squeeze(0).cpu().numpy()
                    
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders

# def vq_det_evaluate()

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
                nobsz = encoder(nobs.reshape(-1, env_obs_dim * obs_len_input).unsqueeze(1))
                prior_sample = model.prior.sample().unsqueeze(0).unsqueeze(-1)
                naction = model.reverse(prior_sample, nobsz)
            
            naction = torch.clone(naction).detach()
            naction.requires_grad = True
            z, outputs, logdets = model(naction, nobsz)
            logprob = model.prior.log_prob(z.squeeze(-1)) + logdets
            grad = torch.autograd.grad(logprob, [naction])[0]
            naction.data = naction.data + noise_std**2 * grad
            naction = naction.squeeze(-1).reshape(1, action_len_pred, -1).detach().to('cpu')
        
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