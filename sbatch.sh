#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --reservation=training


# srun python fbi_mamba.py --tag mamba2_3b --model_size 2.7B --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 1 --n_nodes 4 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 1 --run_wandb

# srun python fbi_mamba.py --tag mamba2_3b_ar --model_size 2.7B  --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 0 --n_nodes 4 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 1 --run_wandb

# srun python fbi_mamba.py --tag mamba_3b --model_size 2.7B  --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 1 --n_nodes 4 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 1 --run_wandb

srun python fbi_mamba.py --tag mamba2_1.3b --model_size 1.3B --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 1 --n_nodes 1 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 4 --run_wandb

srun python fbi_mamba.py --tag mamba2_780m --model_size 780M --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 1 --n_nodes 2 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 2 --run_wandb

srun python fbi_mamba.py --tag mamba2_3b_true --model_size 2.7B  --train_data_dir /lustre/scratch/shared-folders/llm_project/liqun/AmberDatasets/train --use_kd 1 --n_nodes 1 --n_devices_per_node 8 --per_device_batch_size 8 --w_bits 1 --accumulate_grad_batches 4 --run_wandb
