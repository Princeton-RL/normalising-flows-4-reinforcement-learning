import os
os.environ["MUJOCO_GL"] = "egl"

import platform

import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from flax.core import FrozenDict
from absl import app, flags
from ml_collections import config_flags
from stable_baselines3.common.vec_env import DummyVecEnv

from agents import agents
from envs.env_utils import make_env_and_datasets, make_env
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, evaluate_parallel, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

from utils.flax_utils import count_parameters, load_params

jax.config.update("jax_default_matmul_precision", "float32")

FLAGS = flags.FLAGS

flags.DEFINE_string('wandb_dir', 'exp/', 'Wandb dir.')
flags.DEFINE_string('wandb_name_tag', '', 'Wandb dir.')
flags.DEFINE_string('wandb_entity', 'raj19', 'Wandb entity.')
flags.DEFINE_string('wandb_mode', 'offline', 'Wandb mode.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('dataset_dir', '~/.ogbench/data', 'Directory for ogbench dataset.')

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('buffer_size', 1000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)    
    exp_name = f"{FLAGS.wandb_name_tag}{FLAGS.env_name}__{FLAGS.run_group}__{exp_name}"

    _, trigger_sync = setup_wandb(entity=FLAGS.wandb_entity, project='fql-orl', group=FLAGS.run_group, name=exp_name, wandb_output_dir=FLAGS.wandb_dir, mode=FLAGS.wandb_mode)

    FLAGS.save_dir = os.path.join(FLAGS.wandb_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack, dataset_dir=FLAGS.dataset_dir)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    envs = DummyVecEnv([make_env(FLAGS.env_name, i, FLAGS.seed) for i in range(FLAGS.eval_episodes)])

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)

    # Use the training dataset as the replay buffer.
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
    
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] in ['rebrac']:
                dataset.return_next_actions = True

    # Create agent.
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    expl_metrics = dict()
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):

        # Offline RL.
        batch = train_dataset.sample(config['batch_size'])

        agent, update_info = agent.update(batch)
        
        # Log metrics.a
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)

                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            if FLAGS.wandb_mode == 'offline':
                trigger_sync()
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):            
            eval_metrics = {}
            eval_info = evaluate_parallel(
                agent=agent,
                seed=FLAGS.seed,
                envs=envs,
                num_eval_episodes=FLAGS.eval_episodes,
            )

            for k, v in eval_info.items():
                eval_metrics[f'parallel_evaluation/{k}'] = np.mean(v) 
            
            if FLAGS.video_episodes > 0:
                renders = []

                _, _, cur_renders = evaluate(
                    agent=agent,
                    env=eval_env,
                    config=config,
                    num_eval_episodes=0,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                )
                renders.extend(cur_renders)

                video = get_wandb_video(renders=renders)    
                eval_metrics['video'] = video
    
            wandb.log(eval_metrics, step=i)
            if FLAGS.wandb_mode == 'offline':
                trigger_sync()
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)