## Train Bi-Mamba
``` sbatch sbatch.sh ```

## SFT Bi-Mamba
``` sft.sh ```

## Train Dataset
``` /lustre/scratch/shared-folders/llm_project/AmberDataset/train ```

## GPTQ Mamba2
Before using GPTQ to quantize Mamba2, please Modify the code on line 53 of file a to: ``` use_mem_eff_path=False ``` 

base: ``` gptq.py ```

column-wise random quantize: ``` gptq_rand.py ```

## BiLLM Mamba2
``` BiLLM/run_mamba.sh ```