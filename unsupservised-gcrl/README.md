## 🛠 Installation

Create the conda environment with all required packages:

```bash
conda create --name nfs_url python=3.10 \
  jax==0.4.26 "jaxlib==0.4.26=cuda120*" \
  flax==0.8.3 optax==0.2.2 distrax==0.1.5 \
  numpy==1.26.4 scipy==1.12 \
  -c conda-forge -c nvidia

conda activate nfs_url

pip install \
  brax==0.10.5 \
  tensorflow-probability==0.20.1 \
  tyro==0.9.14 \
  ml_collections==1.0.0 \
  matplotlib==3.10.0 \
  imageio==2.37.0 \
  wandb==0.19.1 \
  wandb_osh==1.2.2
```

## 🚀 Running Experiments

Use the following commands to run all the unsupervised goal sampling experiments from the paper. To use wandb add your wandb_entity in main.py and set the value of wandb_dir.

```bash

WANDB_DIR="/path/to/wandb"

# ant_u_maze
python train_auto_NF_rl.py --seed 1 --batch_size 4096 --env-id ant_u_maze --total_env_steps 100000000 --track --wandb_dir $WANDB_DIR

# ant_big_maze
python train_auto_NF_rl.py --seed 1 --batch_size 4096 --env-id ant_big_maze --total_env_steps 100000000 --track --wandb_dir $WANDB_DIR

# ant_hardest_maze
python train_auto_NF_rl.py --seed 1 --batch_size 4096 --env-id ant_hardest_maze --total_env_steps 100000000 --track --wandb_dir $WANDB_DIR
```

Use the following commands to run the baselines.

```bash

# crl oracle
python train_crl_latent_actor.py --seed 1 --batch_size 4096 --env-id ant_u_maze --total_env_steps 100000000 --track --wandb_dir $WANDB_DIR

# crl minmax
python train_crl_latent_actor_minmax.py --seed 1 --batch_size 4096 --env-id ant_u_maze --total_env_steps 100000000 --track --wandb_dir $WANDB_DIR

# crl uniform
python train_crl_latent_actor_uniform --seed 1 --batch_size 4096 --env-id ant_u_maze --total_env_steps 100000000 --track --wandb_dir $WANDB_DIR
```