#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --reservation=training

export WANDB_API_KEY=d21feb3a65169296eb2a8f0d26e09021ca2c9090

# srun python fbi_mamba.py --tag mamba2_3b --model_size 3B --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 1 --n_nodes 4 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 1 --run_wandb

# srun python fbi_mamba.py --tag mamba2_3b_ar --model_size 3B --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 0 --n_nodes 4 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 1 --run_wandb

# srun python fbi_mamba.py --tag mamba_3b --model_size 3B --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 1 --n_nodes 4 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 1 --run_wandb

srun python fbi_mamba.py --tag mamba2_1.3b --model_size 1.3B --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 1 --n_nodes 1 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 4 --run_wandb

# srun python fbi_mamba.py --tag mamba2_780m --model_size 780M --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 1 --n_nodes 2 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 2 --run_wandb
