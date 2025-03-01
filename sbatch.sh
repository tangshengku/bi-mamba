#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4

# Data location, please modify accordingly 
train_data_dir=/path/to/your/data

# srun python train_bimamba.py --tag mamba2_1.3b --model_size 1.3B --train_data_dir $train_data_dir --use_kd 1 --n_nodes 1 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 4 --run_wandb

# srun python train_bimamba.py  --tag mamba2_780m --model_size 780M --train_data_dir $train_data_dir --use_kd 1 --n_nodes 1 --n_devices_per_node 4 --per_device_batch_size 8 --w_bits 1 --accumulate_grad_batches 8 --run_wandb

# srun python train_bimamba.py  --tag mamba2_2.7b --model_size 2.7B  --train_data_dir $train_data_dir --use_kd 1 --n_nodes 1 --n_devices_per_node 8 --per_device_batch_size 8 --w_bits 1 --accumulate_grad_batches 4 --run_wandb


srun python teacher.py  --tag mamba2_780m --model_size 780M  --train_data_dir data/Slimpajama --use_kd 1 --n_nodes 2 --n_devices_per_node 4 --per_device_batch_size 2 --w_bits 1 --accumulate_grad_batches 32