# Offline RL

## 🛠 Installation

Create the conda environment with all required packages:

```bash
conda create --name nfs_orl python=3.10 \
  jax==0.4.26 "jaxlib==0.4.26=cuda120*" \
  flax==0.8.3 optax==0.2.2 distrax==0.1.5 \
  numpy==1.26.4 scipy==1.12 \
  -c conda-forge -c nvidia

conda activate nfs_orl

pip install \
  gymnasium==1.0.0 \
  ogbench==1.1.0 \
  tensorflow-probability==0.20.1 \
  tyro==0.9.14 \
  tqdm==4.67.1 \
  stable_baselines3==2.5.0 \
  ml_collections==1.0.0 \
  matplotlib==3.10.0 \
  imageio==2.37.0 \
  wandb==0.19.1 \
  wandb_osh==1.2.2
```
## 🚀 Running Experiments

> [!Note]
> Because we use stable baselines' multiprocessing for fast parallel evaluation of the BC policy, you should either use 16 CPUs (8 CPUs also suffice for many environments) or reduce the `eval_episodes` argument.

Use the following commands to run all the offline RL experiments from the paper. To use wandb add your wandb_entity in main.py. You'd also need to set the values of wandb_dir and data_dir.


```bash

WANDB_DIR="/path/to/wandb"
DATA_DIR="/path/to/data"

# puzzle-3x3-play-singletask
python main.py --seed=1 --env_name=puzzle-3x3-play-singletask-task1-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$$DATA_DIR
python main.py --seed=1 --env_name=puzzle-3x3-play-singletask-task2-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=puzzle-3x3-play-singletask-task3-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=puzzle-3x3-play-singletask-task4-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=puzzle-3x3-play-singletask-task5-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR

# scene-play-singletask
python main.py --seed=1 --env_name=scene-play-singletask-task1-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=scene-play-singletask-task2-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=scene-play-singletask-task3-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=scene-play-singletask-task4-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=scene-play-singletask-task5-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR

# antmaze-large-navigate-singletask
python main.py --seed=1 --env_name=antmaze-large-navigate-singletask-task1-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=antmaze-large-navigate-singletask-task2-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=antmaze-large-navigate-singletask-task3-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=antmaze-large-navigate-singletask-task4-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=antmaze-large-navigate-singletask-task5-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR

# cube-single-play-singletask
python main.py --seed=1 --env_name=cube-single-play-singletask-task1-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=cube-single-play-singletask-task1-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=cube-single-play-singletask-task1-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=cube-single-play-singletask-task1-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=cube-single-play-singletask-task1-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=10 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR

# humanoidmaze-medium-navigate
python main.py --seed=1 --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=humanoidmaze-medium-navigate-singletask-task3-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=humanoidmaze-medium-navigate-singletask-task4-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=humanoidmaze-medium-navigate-singletask-task5-v0 --agent=agents/vinf.py --agent.q_agg=min --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR

# antsoccer-arena-navigate-singletask
python main.py --seed=1 --env_name=antsoccer-arena-navigate-singletask-task1-v0 --agent=agents/vinf.py --agent.q_agg=mean --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=antsoccer-arena-navigate-singletask-task2-v0 --agent=agents/vinf.py --agent.q_agg=mean --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=antsoccer-arena-navigate-singletask-task3-v0 --agent=agents/vinf.py --agent.q_agg=mean --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=antsoccer-arena-navigate-singletask-task4-v0 --agent=agents/vinf.py --agent.q_agg=mean --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
python main.py --seed=1 --env_name=antsoccer-arena-navigate-singletask-task5-v0 --agent=agents/vinf.py --agent.q_agg=mean --agent.discount=0.995 --agent.alpha_actor=1 --wandb_mode=online --wandb_dir=$WANDB_DIR --dataset_dir=$DATA_DIR
```
