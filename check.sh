#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --reservation=training

export WANDB_API_KEY=d21feb3a65169296eb2a8f0d26e09021ca2c9090

# CUDA_VISIBLE_DEVICES=0 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 0,1 &
# CUDA_VISIBLE_DEVICES=1 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 2,3 &
# CUDA_VISIBLE_DEVICES=2 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 4,5 &
# CUDA_VISIBLE_DEVICES=3 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 6,7 &
# wait

# CUDA_VISIBLE_DEVICES=0 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 8,9 &
# CUDA_VISIBLE_DEVICES=1 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 10,11 &
# CUDA_VISIBLE_DEVICES=2 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 12,13 &
# CUDA_VISIBLE_DEVICES=3 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 14,15 &
# wait

# CUDA_VISIBLE_DEVICES=0 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 16,17 &
# CUDA_VISIBLE_DEVICES=1 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 18,19 &
# CUDA_VISIBLE_DEVICES=2 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 20,21 &
# CUDA_VISIBLE_DEVICES=3 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 22,23 &
# wait

CUDA_VISIBLE_DEVICES=0 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 24,25 &
CUDA_VISIBLE_DEVICES=1 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 26,27 &
CUDA_VISIBLE_DEVICES=2 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 28 &
# CUDA_VISIBLE_DEVICES=3 python generate.py --model_size 3B --ckpt_dir fully_qat_record/mamba2_3b_1_3B_1bit_amber --ckpt_ids 22,23 &
wait
