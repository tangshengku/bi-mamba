#!/bin/bash
#SBATCH --job-name=gptq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4


CUDA_VISIBLE_DEVICES=0 python gptq.py pretrained/mamba2-780m c4 --wbits 3 --true-sequential --act-order --save gptq_mamba/mamba2_780M_3bit_seq/pytorch_model.bin  &
CUDA_VISIBLE_DEVICES=1 python gptq.py pretrained/mamba2-1.3b c4 --wbits 3 --true-sequential --act-order --save gptq_mamba/mamba2_1.3b_3bit_seq/pytorch_model.bin  &
CUDA_VISIBLE_DEVICES=2 python gptq.py pretrained/mamba2-2.7b c4 --wbits 3 --true-sequential --act-order --save gptq_mamba/mamba2_3b_3bit_seq/pytorch_model.bin  &


CUDA_VISIBLE_DEVICES=3 python gptq.py pretrained/mamba2-780m c4 --wbits 2 --true-sequential --act-order --save gptq_mamba/mamba2_780M_2bit_seq/pytorch_model.bin  &
CUDA_VISIBLE_DEVICES=4 python gptq.py pretrained/mamba2-1.3b c4 --wbits 2 --true-sequential --act-order --save gptq_mamba/mamba2_1.3b_2bit_seq/pytorch_model.bin  &
CUDA_VISIBLE_DEVICES=5 python gptq.py pretrained/mamba2-2.7b c4 --wbits 2 --true-sequential --act-order --save gptq_mamba/mamba2_3b_2bit_seq/pytorch_model.bin  &
