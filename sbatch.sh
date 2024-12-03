#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:8
#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=1000G
#SBATCH --no-requeue
#Define the number of hours the job should run.
#SBATCH --time=200:00:00


export WANDB_API_KEY=d21feb3a65169296eb2a8f0d26e09021ca2c9090
export TRITON_LIBCUDA_PATH=/mnt/nfs/clustersw/shared/cuda/12.1.0/targets/x86_64-linux/lib/stubs
# srun python fbi_mamba.py --tag mamba2_3b --model_size 3B --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 1 --n_nodes 4 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 1 --run_wandb

# srun python fbi_mamba.py --tag mamba2_3b_ar --model_size 3B --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 0 --n_nodes 4 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 1 --run_wandb

# srun python fbi_mamba.py --tag mamba_3b --model_size 3B --train_data_dir /lustre/scratch/shared-folders/llm_project/AmberDataset/train --use_kd 1 --n_nodes 4 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 1 --run_wandb

# srun python fbi_mamba.py --tag mamba2_1.3b --model_size 1.3B --train_data_dir /nfs/scistore19/alistgrp/stang/data/amber/train --use_kd 1 --n_nodes 1 --n_devices_per_node 4 --per_device_batch_size 16 --w_bits 1 --accumulate_grad_batches 4 --run_wandb

srun python fbi_mamba.py --tag mamba2_780m_phi-3.5-mini-instruct --model_size 780M --train_data_dir /nfs/scistore19/alistgrp/stang/data/amber/train --use_kd 1 --n_nodes 1 --n_devices_per_node 8 --per_device_batch_size 8 --w_bits 1 --accumulate_grad_batches 4 --run_wandb



# python fbi_mamba.py --tag mamba2_3b_true --model_size 3B --train_data_dir /rampart-stor/liqun/AmberDatasets/train --use_kd 1 --n_nodes 1 --n_devices_per_node 8 --per_device_batch_size 8 --w_bits 1 --accumulate_grad_batches 4 --run_wandb
