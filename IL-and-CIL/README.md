# Conditional Imitation Learning

## 🛠 Installation

Create the conda environment with all required packages:

```bash
conda create --name nfs_il python=3.10 \
  jax==0.4.26 "jaxlib==0.4.26=cuda120*" \
  flax==0.8.3 optax==0.2.2 distrax==0.1.5 \
  numpy==1.26.4 scipy==1.12 \
  -c conda-forge -c nvidia

conda activate nfs_il

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

Use the following commands to run all the conditional imitation learning experiments from the paper. To use wandb set the `wandb_entity` argument in main.py. You'd also need to set the values of `wandb_dir` and `data_dir` arguments.

```bash
WANDB_DIR="/path/to/wandb"
DATA_DIR="/path/to/data"

# antmaze-large-navigate
python flax_fcnf_gcbc.py --seed 1 --gamma 0.97 --env_name antmaze-large-navigate-v0 --eval_episodes=50 --num_trainsteps 1000000 --eval_interval 50000 --save_interval 100000 --track --wandb_mode online --dataset_dir $DATA_DIR --wandb_dir $WANDB_DIR 

# antsoccer-arena-navigate
python flax_fcnf_gcbc.py --seed 1 --gamma 0.97 --env_name antsoccer-arena-navigate-v0 --eval_episodes=50 --num_trainsteps 1000000 --eval_interval 50000 --save_interval 100000 --track --wandb_mode online --dataset_dir $DATA_DIR --wandb_dir $WANDB_DIR 

# antsoccer-arena-stitch
python flax_fcnf_gcbc.py --seed 1 --gamma 0.97 --env_name antsoccer-arena-stitch-v0 --eval_episodes=50 --num_trainsteps 1000000 --eval_interval 50000 --save_interval 100000 --track --wandb_mode online --dataset_dir $DATA_DIR --wandb_dir $WANDB_DIR 

# antmaze-medium-stitch
python flax_fcnf_gcbc.py --seed 1 --gamma 0.97 --env_name antmaze-medium-stitch-v0 --eval_episodes=50 --num_trainsteps 1000000 --eval_interval 50000 --save_interval 100000 --track --wandb_mode online --dataset_dir $DATA_DIR --wandb_dir $WANDB_DIR

# humanoidmaze-medium-navigate
python flax_fcnf_gcbc.py --seed 1 --gamma 0.97 --env_name humanoidmaze-medium-navigate-v0 --eval_episodes=50 --num_trainsteps 1000000 --eval_interval 50000 --save_interval 100000 --track --wandb_mode online --dataset_dir $DATA_DIR --wandb_dir $WANDB_DIR

# cube-single-play
python flax_fcnf_gcbc.py --seed 1 --gamma 0.97 --env_name cube-single-play-v0 --eval_episodes=50 --num_trainsteps 1000000 --eval_interval 50000 --save_interval 100000 --track --wandb_mode online --dataset_dir $DATA_DIR --wandb_dir $WANDB_DIR 

# cube-double-play
python flax_fcnf_gcbc.py --seed 1 --gamma 0.97 --env_name cube-double-play-v0 --eval_episodes=50 --num_trainsteps 1000000 --eval_interval 50000 --save_interval 100000 --track --wandb_mode online --dataset_dir $DATA_DIR --wandb_dir $WANDB_DIR 

# scene-play
python flax_fcnf_gcbc.py --seed 1 --gamma 0.97 --env_name scene-play-v0 --eval_episodes=50 --num_trainsteps 1000000 --eval_interval 50000 --save_interval 100000 --track --wandb_mode online --dataset_dir $DATA_DIR --wandb_dir $WANDB_DIR 

# puzzle-3x3-play
python flax_fcnf_gcbc.py --seed 1 --gamma 0.97 --env_name puzzle-3x3-play-v0 --eval_episodes=50 --num_trainsteps 1000000 --eval_interval 50000 --save_interval 100000 --track --wandb_mode online --dataset_dir $DATA_DIR --wandb_dir $WANDB_DIR  
```

---


# Imitation Learning

Unlike all other experiments, the code for this experiment uses torch rather than jax. 

## 🛠 Installation and 📥 Downloading data
The environments and the datasets for these experiments are taken from the [VQ-BeT](https://github.com/jayLEE0301/vq_bet_official/tree/main) paper. Checkout their repository for the instructions to install the required [packages](https://github.com/jayLEE0301/vq_bet_official/tree/main?tab=readme-ov-file#installation) and download the [datasets](https://github.com/jayLEE0301/vq_bet_official/tree/main?tab=readme-ov-file#step-0-download-dataset-and-set-dataset-path--saving-path). We will add more detailed installation instructions soon.

## 🚀 Running Experiments

Use the following commands to run all the imitation learning experiments from the paper. To use wandb add your wandb_entity in main.py. You'd also need to set the values of wandb_dir and dataset_dir. 

```bash
WANDB_DIR="/path/to/wandb"
DATA_DIR="/path/to/data"

python torch_fcnf_gcbc_pusht.py --seed 1 --blocks 24 --action_len_pred 8 --action_len_exec 4 --eval_episodes 50 --num_epochs 100 --track --wandb_dir $WANDB_DIR --dataset_dir DATA_DIR

python torch_fcnf_gcbc_ur3.py --seed 1 --wandb_name_tag final_longer_12_blocks_fcnf --num_eval_episodes 50 --num_epochs 300 --track --wandb_dir $WANDB_DIR --dataset_dir DATA_DIR

python torch_fcnf_gcbc_ant.py --seed 1 --action_window_size 8 --eval_action_window_size 8 --num_eval_episodes 50 --num_epochs 500 --track --wandb_dir $WANDB_DIR --dataset_dir DATA_DIR

python torch_fcnf_gcbc_kitchen.py --seed 1 --num_eval_episodes 50 --num_epochs 1000 --track --wandb_dir $WANDB_DIR --dataset_dir DATA_DIR
```
