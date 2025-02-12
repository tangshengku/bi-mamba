#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4

full_precision_weight_path=/path/
bimamba_weight_path=/path/
gpt_weight=/path/

# Full-precision evaluation
CUDA_VISIBLE_DEVICES=0 python eval_bimamba.py --path $full_precision_weight_path --batch_size 16 --eval_open_src
# Binarized model evaluation
CUDA_VISIBLE_DEVICES=0 python eval_bimamba.py --path $bimamba_weight_path --exist_extra_para --batch_size 16 --model_size 1.3B
# gptq model evaluation
CUDA_VISIBLE_DEVICES=0 python eval_bimamba.py --path $gpt_weight --eval_open_src --batch_size 16 