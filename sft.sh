#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --reservation=training

export WANDB_API_KEY=d21feb3a65169296eb2a8f0d26e09021ca2c9090

# srun python sft_mamba.py --tag sft_from_scratch --model_size 3B --n_nodes 1 --n_devices_per_node 4 --per_device_batch_size 16 --accumulate_grad_batches 2 --run_wandb

srun python sft_mamba.py --tag final --model_size 3B  --model_path fully_qat_record/mamba2_3b_1_3B_1bit_amber/ckpt-32 --n_nodes 1 --n_devices_per_node 4 --per_device_batch_size 16 --accumulate_grad_batches 2 --run_wandb
