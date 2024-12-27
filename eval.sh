#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4


CUDA_VISIBLE_DEVICES=0 python eval_mamba.py --path pretrained/mamba2-1.3b --batch_size 16 --eval_open_src

CUDA_VISIBLE_DEVICES=0 python eval_mamba.py --path fully_qat_record/mamba2_1.3b_1_1.3B_1bit_amber --exist_extra_para --ckpt_ids 30 --batch_size 16 --model_size 1.3B

CUDA_VISIBLE_DEVICES=0 python eval_mamba.py --path gptq_mamba/mamba2_1.3b_3bit_seq --eval_open_src --batch_size 16 