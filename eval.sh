#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --reservation=training

export WANDB_API_KEY=d21feb3a65169296eb2a8f0d26e09021ca2c9090

CUDA_VISIBLE_DEVICES=0 python eval_mamba.py --path gptq_mamba/mamba2_780M_3bit_seq --eval_open_src --batch_size 16 &
CUDA_VISIBLE_DEVICES=1 python eval_mamba.py --path gptq_mamba/mamba2_1.3b_3bit_seq --eval_open_src --batch_size 16 &
CUDA_VISIBLE_DEVICES=2 python eval_mamba.py --path gptq_mamba/mamba2_3b_3bit_seq --eval_open_src --batch_size 16 &
CUDA_VISIBLE_DEVICES=3 python eval_mamba.py --path gptq_mamba/mamba2_780M_2bit_seq --eval_open_src --batch_size 16 &
wait

CUDA_VISIBLE_DEVICES=1 python eval_mamba.py --path gptq_mamba/mamba2_1.3b_2bit_seq --eval_open_src --batch_size 16 &
CUDA_VISIBLE_DEVICES=2 python eval_mamba.py --path gptq_mamba/mamba2_3b_2bit_seq --eval_open_src --batch_size 16 &
wait

# CUDA_VISIBLE_DEVICES=0 python eval_mamba.py --path BiLLM/output/mamba2-780m_c4_braq_128_hessian --eval_open_src --batch_size 16 &
# CUDA_VISIBLE_DEVICES=1 python eval_mamba.py --path BiLLM/output/mamba2-1.3b_c4_braq_128_hessian --eval_open_src --batch_size 16 &
# CUDA_VISIBLE_DEVICES=2 python eval_mamba.py --path BiLLM/output/mamba2-2.7b_c4_braq_128_hessian --eval_open_src --batch_size 16 &
# wait

# CUDA_VISIBLE_DEVICES=0 python eval_mamba.py --path bi_part/mamba2-2-in_proj.weight --eval_open_src --batch_size 16 &
# CUDA_VISIBLE_DEVICES=1 python eval_mamba.py --path bi_part/mamba2-2-in_proj.weight --eval_open_src --batch_size 16 &
# CUDA_VISIBLE_DEVICES=2 python eval_mamba.py --path bi_part/mamba2-2-mixer.A_log --eval_open_src --batch_size 16  &
# CUDA_VISIBLE_DEVICES=3 python eval_mamba.py --path bi_part/mamba2-2-mixer.D --eval_open_src --batch_size 16  &
# wait

# CUDA_VISIBLE_DEVICES=3 python eval_mamba.py --path fully_qat_record/mamba2_3b_1_3B_1bit_amber --exist_extra_para --ckpt_ids 29,30,31 --batch_size 16 --model_size 3B

# CUDA_VISIBLE_DEVICES=0 python eval_mamba.py --path fully_qat_record/mamba2_3b_1_3B_1bit_amber --exist_extra_para --ckpt_ids 23,24 --batch_size 16 --model_size 3B &
# CUDA_VISIBLE_DEVICES=1 python eval_mamba.py --path fully_qat_record/mamba2_3b_1_3B_1bit_amber --exist_extra_para --ckpt_ids 25 --batch_size 16 --model_size 3B &
# CUDA_VISIBLE_DEVICES=2 python eval_mamba.py --path fully_qat_record/mamba2_3b_1_3B_1bit_amber --exist_extra_para --ckpt_ids 26 --batch_size 16 --model_size 3B &
# CUDA_VISIBLE_DEVICES=3 python eval_mamba.py --path fully_qat_record/mamba2_3b_1_3B_1bit_amber --exist_extra_para --ckpt_ids 27 --batch_size 16 --model_size 3B &
# wait

# CUDA_VISIBLE_DEVICES=0 python eval_mamba.py --path fully_qat_record/mamba2_780m_1_780M_1bit_amber --exist_extra_para --ckpt_ids 6 --batch_size 16 --model_size 780M
# 
# CUDA_VISIBLE_DEVICES=0 python eval_mamba.py --path bi_part/mamba2-2-conv1d.weight --eval_open_src --batch_size 16  &
# CUDA_VISIBLE_DEVICES=1 python eval_mamba.py --path bi_part/mamba2-2-in_proj.weight --eval_open_src --batch_size 16 &
# CUDA_VISIBLE_DEVICES=2 python eval_mamba.py --path bi_part/mamba2-2-mixer.A_log --eval_open_src --batch_size 16  &
# CUDA_VISIBLE_DEVICES=3 python eval_mamba.py --path bi_part/mamba2-2-mixer.D --eval_open_src --batch_size 16  &
# wait

# python eval_mamba.py --path pretrained/mamba2-2.7b --eval_open_src --batch_size 16

# CUDA_VISIBLE_DEVICES=0 python eval_mamba.py --path bi_part/mamba2-2-mixer.dt_bias --eval_open_src --batch_size 32  &
# CUDA_VISIBLE_DEVICES=1 python eval_mamba.py --path bi_part/mamba2-2-out_proj.weight --eval_open_src --batch_size 32 &
# wait


# CUDA_VISIBLE_DEVICES=3 python eval_mamba.py --path fully_qat_record/mamba2_780m_1_780M_1bit_amber --exist_extra_para --ckpt_ids 6 --batch_size 16 --model_size 780M
# CUDA_VISIBLE_DEVICES=1 python eval_mamba.py --path pretrained/mamba2-2.7b --batch_size 16 --eval_open_src &
# CUDA_VISIBLE_DEVICES=2 python eval_mamba.py --path pretrained/mamba2-1.3b --batch_size 16 --eval_open_src &
# CUDA_VISIBLE_DEVICES=3 python eval_mamba.py --path pretrained/mamba2-780m --batch_size 16 --eval_open_src &
# wait